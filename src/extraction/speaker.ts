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
import { condense, entropy, mean as meanOf, variance as varianceOf } from "./statistics";
import { extractLpcAnalysis } from "./lpc";
import { extractMfccFeatures, MFCC_FEATURE_COUNT } from "./mfcc";
import {
  extractVoiceQualityFeatures,
  VOICE_QUALITY_FEATURE_COUNT,
} from "./voice-quality";
import { pitchContourShape, PITCH_CONTOUR_SHAPE_FEATURE_COUNT } from "./dct";
import { yieldToMainThread } from "../yield";
import { sdkWarn } from "../log";

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

// Existing legacy 44 features (preserved at the start of the audio block so
// the validator's named sub-range constants — JITTER, SHIMMER,
// LTAS_FLATNESS_VAR, VOICING_RATIO, etc. — keep pointing at the same
// indices and the TTS detector's threshold checks remain unchanged).
const LEGACY_SPEAKER_FEATURE_COUNT = 44;

// LPC coefficient statistics: 12 coefficients × {mean, variance} per
// coefficient time series.
const LPC_COEFFICIENT_STATS = 12 * 2;

// Formant absolute values + dynamics: F1/F2/F3 absolute (6) + F1/F2/F3
// derivative (6) + F1/F2 bandwidth (4) = 16. F3 bandwidth omitted because
// LPC pole bandwidth on the 3rd formant is consistently noisier than the
// underlying frequency signal — keeping it would dilute the block.
const FORMANT_TRAJECTORY_FEATURE_COUNT = 16;

const SPEAKER_FEATURE_COUNT =
  LEGACY_SPEAKER_FEATURE_COUNT +
  MFCC_FEATURE_COUNT +
  LPC_COEFFICIENT_STATS +
  FORMANT_TRAJECTORY_FEATURE_COUNT +
  VOICE_QUALITY_FEATURE_COUNT +
  PITCH_CONTOUR_SHAPE_FEATURE_COUNT;
// = 44 + 72 + 24 + 16 + 9 + 5 = 170. See
// docs/master/BLUEPRINT-feature-pipeline-v2.md §2.1.
// (Audio block shrank 176 → 170 in v3 when MFCC[0] was dropped — see
// mfcc.ts::MFCC_DROP_LEADING.)

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
 * Frame batch size between cooperative yields inside `detectF0Contour`.
 * YIN at 16 kHz / 1024-sample frame is ~1 ms per frame on commodity
 * hardware, so 16 frames ≈ one 60 fps paint budget — yielding here keeps
 * the verify spinner repainting smoothly through the dominant block of
 * extraction work (which the user perceives as the freeze right after
 * the 12 s capture ends, before any other yield site fires).
 */
const F0_YIELD_EVERY_N_FRAMES = 16;

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
    sdkWarn(`[Entros SDK] Audio captured at ${sampleRate}Hz (requested 16kHz). Frame size adjusted to ${frameSize}.`);
  }

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    // `subarray` is a zero-copy view — `detect()` and the RMS loop below
    // are read-only, so we save ~1.2k Float32Array allocations per session
    // (and matching GC pressure on Hermes).
    const frame = samples.subarray(start, start + frameSize);

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

    // Cooperative yield every N frames so the host UI gets a paint frame
    // mid-loop. Skipped on frame 0 (no work done yet) and the last frame
    // (the function returns immediately after).
    if (i > 0 && i < numFrames - 1 && (i % F0_YIELD_EVERY_N_FRAMES) === 0) {
      await yieldToMainThread();
    }
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
    // Read-only autocorrelation — view is safe.
    const frame = samples.subarray(start, start + frameSize);
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

  // Meyda.extract's third argument is `previousSignal`, NOT options — so
  // passing `{ sampleRate, bufferSize: frameSize }` there is silently
  // ignored. Without setting the globals here, Meyda would use whatever
  // bufferSize the previous extractor in the pipeline left behind (mfcc /
  // voice-quality both set 2048; if they ran first, this function would
  // see 2048; if they didn't run yet, it would see the default 512). That
  // call-order dependency makes the per-frame spectral extractors
  // non-deterministic across pipeline orderings — the exact bug the v2
  // pentest replay assertion (distance must be 0) caught. Set the globals
  // so this extractor is order-independent.
  Meyda.bufferSize = frameSize;
  Meyda.sampleRate = sampleRate;

  const centroids: number[] = [];
  const rolloffs: number[] = [];
  const flatnesses: number[] = [];
  const spreads: number[] = [];
  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;
  // Pre-allocate the buffer Meyda receives — frames are exactly `frameSize`
  // (the loop bounds guarantee no overrun), so we can overwrite each
  // iteration via `.set()` instead of allocating ~1.2k Float32Arrays.
  const paddedFrame = new Float32Array(frameSize);

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    paddedFrame.set(samples.subarray(start, start + frameSize), 0);

    const features = Meyda.extract(
      ["spectralCentroid", "spectralRolloff", "spectralFlatness", "spectralSpread"],
      paddedFrame,
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
 * Captures physiological vocal characteristics that are stable across
 * different utterances from the same speaker. Content-independent by
 * design — different phrases produce similar feature values.
 *
 * Returns 44 values.
 */
/**
 * Extracts 44 speaker features AND the raw F0 contour.
 * The F0 contour is surfaced so server-side analysis can pair it with
 * the motion time-series. Feature vector shape and semantics are
 * unchanged.
 */
export async function extractSpeakerFeaturesDetailed(
  audio: AudioCapture,
): Promise<{ features: number[]; f0Contour: number[] }> {
  const { samples, sampleRate } = audio;

  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || samples.length === 0) {
    sdkWarn("[Entros SDK] Invalid audio data. Speaker features will be zeros.");
    return { features: new Array(SPEAKER_FEATURE_COUNT).fill(0), f0Contour: [] };
  }

  const frameSize = getFrameSize(sampleRate);
  const hopSize = getHopSize(sampleRate);

  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;
  if (numFrames < 5) {
    sdkWarn(`[Entros SDK] Too few audio frames (${numFrames}). Speaker features will be zeros.`);
    return { features: new Array(SPEAKER_FEATURE_COUNT).fill(0), f0Contour: [] };
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

  // Single-allocation normalisation. Order of operations preserved exactly
  // (`(s / peakAmp) * 0.9`) so the output is bit-identical to the previous
  // `samples.map(...)` form.
  let normalizedSamples: Float32Array;
  if (peakAmp > 1e-6) {
    normalizedSamples = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      normalizedSamples[i] = (samples[i]! / peakAmp) * 0.9;
    }
  } else {
    normalizedSamples = samples;
  }

  // 1. F0 detection + amplitude contour (on normalized audio)
  const { f0, amplitudes: normalizedAmplitudes, periods } = await detectF0Contour(normalizedSamples, sampleRate);
  // YIN pitch detection is the heaviest single block in the audio path
  // (~120 frames × O(frameSize²)). Yield before HNR + formant extraction
  // so the verify UI can repaint the "Extracting features..." spinner.
  await yieldToMainThread();

  // Compute per-frame RMS amplitude from the peak-normalized buffer.
  // After the SDK's capture-time RMS normalization, the input `samples`
  // carries stable RMS = TARGET_CAPTURE_RMS for every speaker, so per-
  // frame mean amplitude computed off `samples` would converge to that
  // same value across all users — a wasted feature slot. The peak-
  // normalized `normalizedSamples` (peak = 0.9, variable RMS) preserves
  // dynamic-range identity instead: monotone speakers have higher mean
  // amplitude (more time near peak), expressive speakers have lower mean
  // (more time in valleys). Variance / skewness / kurtosis are scale-
  // invariant either way; switching the source affects only mean.
  const amplitudes: number[] = [];
  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    let sum = 0;
    const end = Math.min(start + frameSize, normalizedSamples.length);
    for (let j = start; j < end; j++) {
      sum += (normalizedSamples[j] ?? 0) * (normalizedSamples[j] ?? 0);
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
  // HNR autocorrelation is per-frame and synchronous. Yield before LPC
  // formant extraction so a paint frame can land between the two heaviest
  // synchronous stages.
  await yieldToMainThread();

  // 7. Formant analysis — single LPC pass surfaces ratios (legacy
  // 8 features), LPC coefficients (24 new), absolute formants + dynamics
  // + bandwidths (16 new). All derived from one autocorrelate +
  // Levinson-Durbin per frame; CPU cost identical to the legacy
  // extractFormantRatios call but the output is much richer.
  const lpc = extractLpcAnalysis(normalizedSamples, sampleRate, frameSize, hopSize);
  const f1f2Stats = condense(lpc.f1f2);
  const f2f3Stats = condense(lpc.f2f3);
  const formantFeatures = [
    f1f2Stats.mean, f1f2Stats.variance, f1f2Stats.skewness, f1f2Stats.kurtosis,
    f2f3Stats.mean, f2f3Stats.variance, f2f3Stats.skewness, f2f3Stats.kurtosis,
  ];
  await yieldToMainThread();

  // 8. LTAS (8 values)
  const ltasFeatures = await computeLTAS(samples, sampleRate);

  // 9. Voicing ratio (1 value)
  const voicingFeatures = [voicedRatio];

  // 10. Amplitude statistics (5 values)
  const ampStats = condense(amplitudes);
  const ampEntropy = entropy(amplitudes);
  const ampFeatures = [ampStats.mean, ampStats.variance, ampStats.skewness, ampStats.kurtosis, ampEntropy];

  // ----- v2 feature pipeline additions -----
  // Each new block appended to the end of the audio vector so the existing
  // 44-feature layout (and the validator's named sub-range constants) stays
  // pinned at indices 0..44.

  // 11. MFCC + delta-MFCC stats (72 values total: 12×4 + 12×2 — MFCC[0] dropped)
  await yieldToMainThread();
  const mfccFeatures = await extractMfccFeatures(
    normalizedSamples,
    sampleRate,
    frameSize,
    hopSize,
  );

  // 12. LPC coefficient statistics (24 values: 12 coefficients × mean, variance)
  // Derived from the single LPC pass that already ran for the formant block.
  const lpcStats: number[] = [];
  for (let c = 0; c < 12; c++) {
    const track = lpc.lpcCoefficients[c] ?? [];
    const mu = meanOf(track);
    lpcStats.push(mu, varianceOf(track, mu));
  }

  // 13. Formant trajectories (16 values):
  //   F1/F2/F3 absolute mean+var (6) + derivative mean+var (6) + B1/B2 mean+var (4)
  const f1Stats = { mean: meanOf(lpc.f1), var: varianceOf(lpc.f1) };
  const f2Stats = { mean: meanOf(lpc.f2), var: varianceOf(lpc.f2) };
  const f3Stats = { mean: meanOf(lpc.f3), var: varianceOf(lpc.f3) };
  const f1Delta = derivative(lpc.f1);
  const f2Delta = derivative(lpc.f2);
  const f3Delta = derivative(lpc.f3);
  const f1DeltaMu = meanOf(f1Delta);
  const f2DeltaMu = meanOf(f2Delta);
  const f3DeltaMu = meanOf(f3Delta);
  const b1Mu = meanOf(lpc.b1);
  const b2Mu = meanOf(lpc.b2);
  const formantTrajectoryFeatures = [
    f1Stats.mean, f1Stats.var,
    f2Stats.mean, f2Stats.var,
    f3Stats.mean, f3Stats.var,
    f1DeltaMu, varianceOf(f1Delta, f1DeltaMu),
    f2DeltaMu, varianceOf(f2Delta, f2DeltaMu),
    f3DeltaMu, varianceOf(f3Delta, f3DeltaMu),
    b1Mu, varianceOf(lpc.b1, b1Mu),
    b2Mu, varianceOf(lpc.b2, b2Mu),
  ];

  // 14. Voice quality (9 values: CPP, tilt, H1-H2 each mean+var, plus low/mid/high band ratios)
  await yieldToMainThread();
  const voiceQualityFeatures = await extractVoiceQualityFeatures(
    normalizedSamples,
    sampleRate,
    frameSize,
    hopSize,
    f0,
  );

  // 15. Pitch contour shape (5 DCT coefficients of the F0 contour)
  // Strict dimensionality reduction: ~1200 voiced frames → 5 coefficients.
  // Captures prosodic shape (rising/falling/modulated) that the existing
  // F0_STATS block doesn't reach.
  const pitchShapeFeatures = pitchContourShape(f0, PITCH_CONTOUR_SHAPE_FEATURE_COUNT);

  const features = [
    ...f0Features,                 // 5     [0..5]    F0_STATS
    ...f0DeltaFeatures,            // 4     [5..9]    F0_DELTA
    ...jitterFeatures,             // 4     [9..13]   JITTER
    ...shimmerFeatures,            // 4     [13..17]  SHIMMER
    ...hnrFeatures,                // 5     [17..22]  HNR
    ...formantFeatures,            // 8     [22..30]  FORMANT_RATIOS
    ...ltasFeatures,               // 8     [30..38]  LTAS
    ...voicingFeatures,            // 1     [38]      VOICING_RATIO
    ...ampFeatures,                // 5     [39..44]  AMPLITUDE
    ...mfccFeatures,               // 72    [44..116] MFCC + delta-MFCC (MFCC[0] dropped)
    ...lpcStats,                   // 24    [116..140] LPC coefficient stats
    ...formantTrajectoryFeatures,  // 16    [140..156] Formant absolutes + dynamics + bandwidths
    ...voiceQualityFeatures,       // 9     [156..165] Voice quality
    ...pitchShapeFeatures,         // 5     [165..170] Pitch contour shape DCT
  ]; // = 170

  return { features, f0Contour: f0 };
}

/**
 * Extracts 44 speaker features. Backward-compatible wrapper that discards
 * the F0 contour; use `extractSpeakerFeaturesDetailed` when the contour is
 * needed (e.g. for server-side analysis).
 */
export async function extractSpeakerFeatures(audio: AudioCapture): Promise<number[]> {
  const { features } = await extractSpeakerFeaturesDetailed(audio);
  return features;
}

export { SPEAKER_FEATURE_COUNT };
