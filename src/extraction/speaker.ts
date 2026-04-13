/**
 * Speaker-dependent audio feature extraction.
 *
 * Extracts features that characterize HOW someone speaks (prosody, vocal physiology)
 * rather than WHAT they say (phonetic content). These features are stable across
 * different utterances from the same speaker.
 *
 * Output: 44 values
 *   F0 statistics (5) + F0 delta (4) + jitter (4) + shimmer (4) +
 *   HNR statistics (5) + formant ratios (8) + LTAS (8) + voicing ratio (1) +
 *   amplitude statistics (5)
 */
import type { AudioCapture } from "../sensor/types";
import { condense, entropy } from "./statistics";
import { extractFormantRatios } from "./lpc";

/**
 * Compute frame size adaptive to actual sample rate.
 * YIN pitch detection requires: frameSize >= 4 * sampleRate / minF0
 * (because YIN halves the buffer twice internally).
 * Returns next power of 2 for FFT compatibility.
 */
function getFrameSize(sampleRate: number): number {
  const MIN_F0 = 50; // lowest detectable pitch (Hz)
  const minSize = Math.ceil(4 * sampleRate / MIN_F0);
  let size = 512; // minimum
  while (size < minSize) size *= 2;
  return size;
}

/**
 * Compute hop size as ~10ms at the given sample rate.
 */
function getHopSize(sampleRate: number): number {
  return Math.max(1, Math.round(sampleRate * 0.01));
}

const SPEAKER_FEATURE_COUNT = 44;

// Dynamic imports for browser compatibility
let pitchDetector: ((buf: Float32Array) => number | null) | null = null;
let pitchDetectorRate = 0;
let meydaModule: any = null;

async function getPitchDetector(sampleRate: number): Promise<(buf: Float32Array) => number | null> {
  if (!pitchDetector || pitchDetectorRate !== sampleRate) {
    const PitchFinder = await import("pitchfinder");
    pitchDetector = PitchFinder.YIN({ sampleRate, threshold: 0.15 });
    pitchDetectorRate = sampleRate;
  }
  return pitchDetector;
}

async function getMeyda(): Promise<any> {
  if (!meydaModule) {
    try {
      meydaModule = await import("meyda");
    } catch {
      return null;
    }
  }
  return meydaModule.default ?? meydaModule;
}

/**
 * Detect F0 (fundamental frequency) contour and amplitude peaks per frame.
 */
async function detectF0Contour(
  samples: Float32Array,
  sampleRate: number
): Promise<{ f0: number[]; amplitudes: number[]; periods: number[] }> {
  const detect = await getPitchDetector(sampleRate);
  const frameSize = getFrameSize(sampleRate);
  const hopSize = getHopSize(sampleRate);
  const f0: number[] = [];
  const amplitudes: number[] = [];
  const periods: number[] = [];
  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;

  if (sampleRate !== 16000) {
    console.warn(`[IAM SDK] Audio captured at ${sampleRate}Hz (requested 16kHz). Frame size adjusted to ${frameSize}.`);
  }

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    const frame = samples.slice(start, start + frameSize);

    // F0 detection
    const pitch = detect(frame);
    if (pitch && pitch > 50 && pitch < 600) {
      f0.push(pitch);
      periods.push(1 / pitch);
    } else {
      f0.push(0); // unvoiced frame
    }

    // RMS amplitude per frame
    let sum = 0;
    for (let j = 0; j < frame.length; j++) {
      sum += (frame[j] ?? 0) * (frame[j] ?? 0);
    }
    amplitudes.push(Math.sqrt(sum / frame.length));
  }

  return { f0, amplitudes, periods };
}

/**
 * Compute jitter measures from pitch period contour.
 * Jitter = cycle-to-cycle perturbation of the fundamental period.
 */
function computeJitter(periods: number[]): number[] {
  const voiced = periods.filter((p) => p > 0);
  if (voiced.length < 3) return [0, 0, 0, 0];

  const meanPeriod = voiced.reduce((a, b) => a + b, 0) / voiced.length;
  if (meanPeriod === 0) return [0, 0, 0, 0];

  // Jitter (local): average absolute difference between consecutive periods
  let localSum = 0;
  for (let i = 1; i < voiced.length; i++) {
    localSum += Math.abs(voiced[i]! - voiced[i - 1]!);
  }
  const jitterLocal = localSum / (voiced.length - 1) / meanPeriod;

  // RAP: Relative Average Perturbation (3-point running average)
  let rapSum = 0;
  for (let i = 1; i < voiced.length - 1; i++) {
    const avg3 = (voiced[i - 1]! + voiced[i]! + voiced[i + 1]!) / 3;
    rapSum += Math.abs(voiced[i]! - avg3);
  }
  const jitterRAP = voiced.length > 2 ? rapSum / (voiced.length - 2) / meanPeriod : 0;

  // PPQ5: Five-Point Period Perturbation Quotient
  let ppq5Sum = 0;
  let ppq5Count = 0;
  for (let i = 2; i < voiced.length - 2; i++) {
    const avg5 = (voiced[i - 2]! + voiced[i - 1]! + voiced[i]! + voiced[i + 1]! + voiced[i + 2]!) / 5;
    ppq5Sum += Math.abs(voiced[i]! - avg5);
    ppq5Count++;
  }
  const jitterPPQ5 = ppq5Count > 0 ? ppq5Sum / ppq5Count / meanPeriod : 0;

  // DDP: Difference of Differences of Periods
  let ddpSum = 0;
  for (let i = 1; i < voiced.length - 1; i++) {
    const d1 = voiced[i]! - voiced[i - 1]!;
    const d2 = voiced[i + 1]! - voiced[i]!;
    ddpSum += Math.abs(d2 - d1);
  }
  const jitterDDP = voiced.length > 2 ? ddpSum / (voiced.length - 2) / meanPeriod : 0;

  return [jitterLocal, jitterRAP, jitterPPQ5, jitterDDP];
}

/**
 * Compute shimmer measures from amplitude peaks.
 * Shimmer = cycle-to-cycle amplitude perturbation.
 */
function computeShimmer(amplitudes: number[], f0: number[]): number[] {
  // Use amplitudes only at voiced frames
  const voicedAmps = amplitudes.filter((_, i) => f0[i]! > 0);
  if (voicedAmps.length < 3) return [0, 0, 0, 0];

  const meanAmp = voicedAmps.reduce((a, b) => a + b, 0) / voicedAmps.length;
  if (meanAmp === 0) return [0, 0, 0, 0];

  // Shimmer (local)
  let localSum = 0;
  for (let i = 1; i < voicedAmps.length; i++) {
    localSum += Math.abs(voicedAmps[i]! - voicedAmps[i - 1]!);
  }
  const shimmerLocal = localSum / (voicedAmps.length - 1) / meanAmp;

  // APQ3: 3-point Amplitude Perturbation Quotient
  let apq3Sum = 0;
  for (let i = 1; i < voicedAmps.length - 1; i++) {
    const avg3 = (voicedAmps[i - 1]! + voicedAmps[i]! + voicedAmps[i + 1]!) / 3;
    apq3Sum += Math.abs(voicedAmps[i]! - avg3);
  }
  const shimmerAPQ3 = voicedAmps.length > 2 ? apq3Sum / (voicedAmps.length - 2) / meanAmp : 0;

  // APQ5
  let apq5Sum = 0;
  let apq5Count = 0;
  for (let i = 2; i < voicedAmps.length - 2; i++) {
    const avg5 = (voicedAmps[i - 2]! + voicedAmps[i - 1]! + voicedAmps[i]! + voicedAmps[i + 1]! + voicedAmps[i + 2]!) / 5;
    apq5Sum += Math.abs(voicedAmps[i]! - avg5);
    apq5Count++;
  }
  const shimmerAPQ5 = apq5Count > 0 ? apq5Sum / apq5Count / meanAmp : 0;

  // DDA: Difference of Differences of Amplitudes
  let ddaSum = 0;
  for (let i = 1; i < voicedAmps.length - 1; i++) {
    const d1 = voicedAmps[i]! - voicedAmps[i - 1]!;
    const d2 = voicedAmps[i + 1]! - voicedAmps[i]!;
    ddaSum += Math.abs(d2 - d1);
  }
  const shimmerDDA = voicedAmps.length > 2 ? ddaSum / (voicedAmps.length - 2) / meanAmp : 0;

  return [shimmerLocal, shimmerAPQ3, shimmerAPQ5, shimmerDDA];
}

/**
 * Compute Harmonic-to-Noise Ratio per frame using autocorrelation.
 */
function computeHNR(
  samples: Float32Array,
  sampleRate: number,
  f0Contour: number[]
): number[] {
  const frameSize = getFrameSize(sampleRate);
  const hopSize = getHopSize(sampleRate);
  const hnr: number[] = [];
  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;

  for (let i = 0; i < numFrames && i < f0Contour.length; i++) {
    const f0 = f0Contour[i]!;
    if (f0 <= 0) continue; // Skip unvoiced frames

    const start = i * hopSize;
    const frame = samples.slice(start, start + frameSize);
    const period = Math.round(sampleRate / f0);

    if (period <= 0 || period >= frame.length) continue;

    // Autocorrelation at the fundamental period
    let num = 0;
    let den = 0;
    for (let j = 0; j < frame.length - period; j++) {
      num += (frame[j] ?? 0) * (frame[j + period] ?? 0);
      den += (frame[j] ?? 0) * (frame[j] ?? 0);
    }

    if (den > 0) {
      const r = num / den;
      const clampedR = Math.max(0.001, Math.min(0.999, r));
      hnr.push(10 * Math.log10(clampedR / (1 - clampedR)));
    }
  }

  return hnr;
}

/**
 * Compute LTAS (Long-Term Average Spectrum) features using Meyda.
 * Returns 8 values: spectral centroid, rolloff, flatness, spread — each mean + variance.
 */
async function computeLTAS(
  samples: Float32Array,
  sampleRate: number
): Promise<number[]> {
  const frameSize = getFrameSize(sampleRate);
  const hopSize = getHopSize(sampleRate);
  const Meyda = await getMeyda();
  if (!Meyda) return new Array(8).fill(0);

  const centroids: number[] = [];
  const rolloffs: number[] = [];
  const flatnesses: number[] = [];
  const spreads: number[] = [];
  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    const frame = samples.slice(start, start + frameSize);
    const paddedFrame = new Float32Array(frameSize);
    paddedFrame.set(frame);

    const features = Meyda.extract(
      ["spectralCentroid", "spectralRolloff", "spectralFlatness", "spectralSpread"],
      paddedFrame,
      { sampleRate, bufferSize: frameSize }
    );

    if (features) {
      if (Number.isFinite(features.spectralCentroid)) centroids.push(features.spectralCentroid);
      if (Number.isFinite(features.spectralRolloff)) rolloffs.push(features.spectralRolloff);
      if (Number.isFinite(features.spectralFlatness)) flatnesses.push(features.spectralFlatness);
      if (Number.isFinite(features.spectralSpread)) spreads.push(features.spectralSpread);
    }
  }

  const m = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const v = (arr: number[]) => {
    if (arr.length < 2) return 0;
    const mu = m(arr);
    return arr.reduce((sum, x) => sum + (x - mu) * (x - mu), 0) / (arr.length - 1);
  };

  return [
    m(centroids), v(centroids),
    m(rolloffs), v(rolloffs),
    m(flatnesses), v(flatnesses),
    m(spreads), v(spreads),
  ];
}

/**
 * Compute derivative (frame-to-frame differences) of a time series.
 */
function derivative(values: number[]): number[] {
  const d: number[] = [];
  for (let i = 1; i < values.length; i++) {
    d.push(values[i]! - values[i - 1]!);
  }
  return d;
}

/**
 * Extract speaker-dependent audio features.
 *
 * Captures physiological vocal characteristics (F0, jitter, shimmer, HNR, formant
 * ratios) that are stable across different utterances from the same speaker.
 * Content-independent by design — different phrases produce similar feature values.
 *
 * Returns 44 values.
 */
export async function extractSpeakerFeatures(audio: AudioCapture): Promise<number[]> {
  const { samples, sampleRate } = audio;

  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || samples.length === 0) {
    console.warn("[IAM SDK] Invalid audio data. Speaker features will be zeros.");
    return new Array(SPEAKER_FEATURE_COUNT).fill(0);
  }

  const frameSize = getFrameSize(sampleRate);
  const hopSize = getHopSize(sampleRate);

  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;
  if (numFrames < 5) {
    console.warn(`[IAM SDK] Too few audio frames (${numFrames}). Speaker features will be zeros.`);
    return new Array(SPEAKER_FEATURE_COUNT).fill(0);
  }

  // Peak-normalize audio for robust pitch detection.
  // Raw mic input (especially desktop without AGC) can be very quiet,
  // causing autocorrelation-based pitch detectors to fail.
  // All relative features (jitter, shimmer, HNR, F0) are unaffected
  // since they measure ratios, not absolute levels.
  // Absolute amplitude is computed from the original samples below.
  let peakAmp = 0;
  for (let i = 0; i < samples.length; i++) {
    const abs = Math.abs(samples[i] ?? 0);
    if (abs > peakAmp) peakAmp = abs;
  }

  const normalizedSamples = peakAmp > 1e-6
    ? new Float32Array(samples.map((s) => (s / peakAmp) * 0.9))
    : samples;

  // 1. F0 detection + amplitude contour (on normalized audio)
  const { f0, amplitudes: normalizedAmplitudes, periods } = await detectF0Contour(normalizedSamples, sampleRate);

  // Compute amplitude from ORIGINAL samples (pre-normalization) for biometric consistency
  const amplitudes: number[] = [];
  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    let sum = 0;
    const end = Math.min(start + frameSize, samples.length);
    for (let j = start; j < end; j++) {
      sum += (samples[j] ?? 0) * (samples[j] ?? 0);
    }
    amplitudes.push(Math.sqrt(sum / (end - start)));
  }

  const voicedF0 = f0.filter((v) => v > 0);
  const voicedRatio = voicedF0.length / f0.length;

  // 2. F0 statistics (5 values)
  const f0Stats = condense(voicedF0);
  const f0Entropy = entropy(voicedF0);
  const f0Features = [f0Stats.mean, f0Stats.variance, f0Stats.skewness, f0Stats.kurtosis, f0Entropy];

  // 3. F0 delta statistics (4 values)
  const f0Delta = derivative(voicedF0);
  const f0DeltaStats = condense(f0Delta);
  const f0DeltaFeatures = [f0DeltaStats.mean, f0DeltaStats.variance, f0DeltaStats.skewness, f0DeltaStats.kurtosis];

  // 4. Jitter (4 values)
  const jitterFeatures = computeJitter(periods);

  // 5. Shimmer (4 values)
  const shimmerFeatures = computeShimmer(amplitudes, f0);

  // 6. HNR statistics (5 values)
  const hnrValues = computeHNR(normalizedSamples, sampleRate, f0);
  const hnrStats = condense(hnrValues);
  const hnrEntropy = entropy(hnrValues);
  const hnrFeatures = [hnrStats.mean, hnrStats.variance, hnrStats.skewness, hnrStats.kurtosis, hnrEntropy];

  // 7. Formant ratios (8 values)
  const { f1f2, f2f3 } = extractFormantRatios(normalizedSamples, sampleRate, frameSize, hopSize);
  const f1f2Stats = condense(f1f2);
  const f2f3Stats = condense(f2f3);
  const formantFeatures = [
    f1f2Stats.mean, f1f2Stats.variance, f1f2Stats.skewness, f1f2Stats.kurtosis,
    f2f3Stats.mean, f2f3Stats.variance, f2f3Stats.skewness, f2f3Stats.kurtosis,
  ];

  // 8. LTAS (8 values)
  const ltasFeatures = await computeLTAS(samples, sampleRate);

  // 9. Voicing ratio (1 value)
  const voicingFeatures = [voicedRatio];

  // 10. Amplitude statistics (5 values)
  const ampStats = condense(amplitudes);
  const ampEntropy = entropy(amplitudes);
  const ampFeatures = [ampStats.mean, ampStats.variance, ampStats.skewness, ampStats.kurtosis, ampEntropy];

  const features = [
    ...f0Features,        // 5
    ...f0DeltaFeatures,   // 4
    ...jitterFeatures,    // 4
    ...shimmerFeatures,   // 4
    ...hnrFeatures,       // 5
    ...formantFeatures,   // 8
    ...ltasFeatures,      // 8
    ...voicingFeatures,   // 1
    ...ampFeatures,       // 5
  ]; // = 44

  return features;
}

export { SPEAKER_FEATURE_COUNT };
