/**
 * Voice quality feature extraction.
 *
 * Captures identity-bearing voice characteristics that the existing 44
 * audio features and the new MFCC / LPC blocks don't directly target:
 *
 *   - **Cepstral Peak Prominence (CPP)** — strongest single measure of
 *     voice quality (breathy ↔ pressed phonation). Computed as the height
 *     of the cepstral peak in the typical-F0 quefrency band, relative to
 *     a linear regression baseline.
 *   - **Spectral tilt** — slope of the log-magnitude spectrum vs log
 *     frequency. Captures vocal "brightness" / spectral balance, an
 *     individual-speaker signature.
 *   - **H1-H2** — amplitude difference (in dB) between the first two
 *     harmonics of the voice. Identifies phonation type (modal /
 *     breathy / pressed) and is identity-bearing.
 *   - **Sub-band energy ratios** — fraction of total energy in three
 *     bands (low <1kHz, mid 1-3kHz, high 3-8kHz). Captures gross
 *     spectral balance complementary to spectral tilt.
 *
 * Output: 9 floats (see `VOICE_QUALITY_FEATURE_COUNT`):
 *   [cppMean, cppVar, tiltMean, tiltVar, h1h2Mean, h1h2Var,
 *    lowBandRatio, midBandRatio, highBandRatio]
 *
 * @privacyGuarantee Each metric is a per-frame scalar; the output is
 * statistical aggregates over the per-frame time series. Same privacy
 * posture as the existing audio features: aggregates of aggregates of
 * frequency-domain transforms of on-device-captured audio. Cannot
 * reconstruct intelligible speech.
 */
import { mean as meanOf, variance as varianceOf } from "./statistics";
import { sdkWarn } from "../log";

export const VOICE_QUALITY_FEATURE_COUNT = 9;

// Sub-band frequency boundaries in Hz. Tuned to capture distinct
// spectral regions of human speech: low band covers F0 + first
// formant region, mid covers F2-F3 (vowel quality), high covers
// fricative + breathiness energy.
const LOW_BAND_HZ = 1000;
const MID_BAND_HZ = 3000;
const HIGH_BAND_HZ = 8000;

// Quefrency band for CPP peak search (in samples). Corresponds to
// typical fundamental-frequency range 60–400 Hz: quefrency q maps
// to frequency sampleRate/q, so q in [sampleRate/400, sampleRate/60].
function cppQuefrencyRange(sampleRate: number): { qMin: number; qMax: number } {
  return {
    qMin: Math.max(2, Math.floor(sampleRate / 400)),
    qMax: Math.floor(sampleRate / 60),
  };
}

/**
 * Lazy-import meyda mirroring speaker.ts::getMeyda. Meyda's published
 * types don't surface the runtime API on the default export cleanly
 * across bundler interop, so the `any` typing matches the existing
 * pattern.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let meydaModule: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
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
 * Compute Cepstral Peak Prominence for a single frame's power spectrum.
 *
 * CPP = peak_value − regression_baseline_at_peak_quefrency
 *
 * Where the cepstrum is computed as DCT-II of the log power spectrum
 * (mathematically equivalent to the real cepstrum for an even-symmetric
 * input, and faster than IFFT for our use case). The peak is searched
 * in the quefrency band corresponding to typical human F0 (60-400 Hz),
 * and the baseline is a linear regression fit to the cepstrum across
 * the search band.
 *
 * Returns 0 if the spectrum is too small or degenerate.
 */
function cepstralPeakProminence(
  powerSpectrum: Float32Array | number[],
  sampleRate: number,
): number {
  const N = powerSpectrum.length;
  if (N < 8) return 0;

  const { qMin, qMax } = cppQuefrencyRange(sampleRate);
  if (qMax >= N || qMax <= qMin) return 0;

  // Take log of power spectrum with a floor to avoid log(0).
  const FLOOR = 1e-12;
  const logPower: number[] = new Array(N);
  for (let i = 0; i < N; i++) {
    const p = Math.max(powerSpectrum[i]!, FLOOR);
    const l = Math.log(p);
    if (!Number.isFinite(l)) return 0;
    logPower[i] = l;
  }

  // Cepstrum via DCT-II of the log spectrum, computed ONLY over the
  // quefrency band we need. Naive `dctII(logPower, N)` is O(N²) ≈ 1M ops
  // per frame at N=1024; restricting the output range to [qMin..qMax]
  // (≈ 200 bins for typical F0 60-400 Hz at 16 kHz) drops the cost to
  // O(N × (qMax - qMin + 1)) ≈ 200k ops per frame — a 5× speedup. The
  // baseline regression and peak-finding still run only over the band,
  // so the full-N DCT was wasted work in the original implementation.
  const bandLen = qMax - qMin + 1;
  const cepstrumBand: number[] = new Array(bandLen);
  const piOverN = Math.PI / N;
  for (let bIdx = 0; bIdx < bandLen; bIdx++) {
    const k = qMin + bIdx;
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += logPower[n]! * Math.cos(piOverN * (n + 0.5) * k);
    }
    cepstrumBand[bIdx] = sum;
  }

  // Find peak in the band.
  let peakBIdx = 0;
  let peakVal = cepstrumBand[0]!;
  for (let bIdx = 1; bIdx < bandLen; bIdx++) {
    if (cepstrumBand[bIdx]! > peakVal) {
      peakVal = cepstrumBand[bIdx]!;
      peakBIdx = bIdx;
    }
  }
  const peakQuefrency = qMin + peakBIdx;

  // Linear regression baseline: simple least-squares fit y = a + b × q
  // over the same band. CPP = peak − baseline_at_peak_q.
  const M = bandLen;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let bIdx = 0; bIdx < bandLen; bIdx++) {
    const x = qMin + bIdx;
    const y = cepstrumBand[bIdx]!;
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const denom = M * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return 0;
  const slope = (M * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / M;
  const baselineAtPeak = intercept + slope * peakQuefrency;

  return peakVal - baselineAtPeak;
}

/**
 * Compute spectral tilt for a single frame: linear regression slope of
 * `log(power[k])` vs `log(frequency[k])` over the analysis band.
 *
 * Higher-magnitude (more negative) slope indicates a darker / breathier
 * voice; flatter slope indicates pressed / clearer voice. Identity-
 * bearing within a speaker's range.
 *
 * Returns 0 on degenerate input. Bands below 100 Hz are excluded
 * because they're dominated by DC + room noise.
 */
function spectralTilt(
  powerSpectrum: Float32Array | number[],
  sampleRate: number,
): number {
  const N = powerSpectrum.length;
  if (N < 8) return 0;
  // Frequency at bin k (assuming standard FFT bin spacing): k × (sampleRate / 2) / (N - 1)
  // for half-spectrum N. Meyda's amplitudeSpectrum/powerSpectrum has length
  // bufferSize/2, so binHz = k × sampleRate / bufferSize. Caller may pass
  // either; we assume standard interpretation: binHz = k × (sampleRate / 2) / (N - 1).
  // Mathematically the slope is unaffected by frequency-axis scaling
  // (since both x and ln-x scale together).
  const FLOOR = 1e-12;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  let count = 0;
  // Skip k=0 (DC) and the lowest bins (< 100 Hz) where noise dominates.
  const minBin = Math.max(1, Math.floor((100 * 2 * (N - 1)) / sampleRate));
  for (let k = minBin; k < N; k++) {
    const p = powerSpectrum[k]!;
    if (p < FLOOR) continue;
    const x = Math.log(k); // log-frequency proxy
    const y = Math.log(p); // log-power
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
    count++;
  }
  if (count < 4) return 0;
  const denom = count * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return 0;
  return (count * sxy - sx * sy) / denom;
}

/**
 * Compute H1-H2 (amplitude in dB of first vs second harmonic) for a
 * single frame given the power spectrum and the F0 estimate.
 *
 * Looks for the local peak in a small window around k_F0 (first
 * harmonic) and around k_2F0 (second harmonic), takes the max within
 * each window, converts to dB, returns difference.
 *
 * Returns 0 if F0 is invalid (≤ 0) or harmonics fall outside the
 * spectrum.
 */
function h1MinusH2(
  powerSpectrum: Float32Array | number[],
  sampleRate: number,
  f0: number,
): number {
  if (!Number.isFinite(f0) || f0 <= 0) return 0;
  const N = powerSpectrum.length;
  if (N < 8) return 0;

  // Bin index for frequency f: k = f × (N - 1) / (sampleRate / 2)
  const binPerHz = (2 * (N - 1)) / sampleRate;
  const k1 = Math.round(f0 * binPerHz);
  const k2 = Math.round(2 * f0 * binPerHz);

  // ±2-bin search window around each harmonic.
  const window = 2;

  function peakNear(k: number): number {
    let best = -Infinity;
    for (let i = k - window; i <= k + window; i++) {
      if (i <= 0 || i >= N) continue;
      const p = powerSpectrum[i]!;
      if (p > best) best = p;
    }
    return best;
  }

  const h1 = peakNear(k1);
  const h2 = peakNear(k2);
  if (!Number.isFinite(h1) || !Number.isFinite(h2) || h1 <= 0 || h2 <= 0) return 0;
  // dB difference of amplitudes: 10 × (log10(h1) - log10(h2)) since these
  // are powers (squared amplitudes); equivalent to 20 × log10(amp ratio).
  return 10 * Math.log10(h1 / h2);
}

/**
 * Compute sub-band energy ratios from the power spectrum. Returns
 * [low, mid, high] each in [0, 1] summing to ≤ 1 (can be < 1 if some
 * energy is above HIGH_BAND_HZ).
 */
function subbandRatios(
  powerSpectrum: Float32Array | number[],
  sampleRate: number,
): [number, number, number] {
  const N = powerSpectrum.length;
  if (N < 4) return [0, 0, 0];
  // Bin per Hz mapping: k = f × (N - 1) / (sampleRate / 2)
  const binPerHz = (2 * (N - 1)) / sampleRate;
  const lowBin = Math.min(N - 1, Math.round(LOW_BAND_HZ * binPerHz));
  const midBin = Math.min(N - 1, Math.round(MID_BAND_HZ * binPerHz));
  const highBin = Math.min(N - 1, Math.round(HIGH_BAND_HZ * binPerHz));

  let total = 0;
  let low = 0;
  let mid = 0;
  let high = 0;
  // Skip k=0 (DC) — never carries voice energy.
  for (let k = 1; k < N; k++) {
    const p = powerSpectrum[k]!;
    if (!Number.isFinite(p) || p < 0) continue;
    total += p;
    if (k <= lowBin) low += p;
    else if (k <= midBin) mid += p;
    else if (k <= highBin) high += p;
  }
  if (total < 1e-12) return [0, 0, 0];
  return [low / total, mid / total, high / total];
}

/**
 * Extract voice quality features from an audio capture.
 *
 * Per-frame: compute power spectrum (via Meyda), then derive CPP,
 * spectral tilt, H1-H2 (using the per-frame F0), and sub-band ratios.
 * Aggregate per-frame metrics over the session.
 *
 * Returns 9 floats in the order documented at the top of this module.
 * Returns all-zeros on invalid input or when Meyda is unavailable.
 *
 * @param samples — Float32Array of audio samples (peak-normalized
 *   recommended; matches the speaker.ts pre-processing step).
 * @param sampleRate — sample rate in Hz.
 * @param frameSize — frame size in samples (must be a power of two for
 *   Meyda's FFT).
 * @param hopSize — hop size in samples.
 * @param f0PerFrame — F0 (Hz) for each frame, indexed 0..numFrames-1.
 *   Frames where F0 ≤ 0 are treated as unvoiced and skipped for H1-H2
 *   only (CPP, tilt, sub-bands run on every frame). Length should match
 *   the frame iteration in this function (numFrames = floor((samples.length
 *   - frameSize) / hopSize) + 1).
 */
export async function extractVoiceQualityFeatures(
  samples: Float32Array,
  sampleRate: number,
  frameSize: number,
  hopSize: number,
  f0PerFrame: number[],
): Promise<number[]> {
  if (
    !Number.isFinite(sampleRate) ||
    sampleRate <= 0 ||
    samples.length === 0 ||
    frameSize <= 0 ||
    hopSize <= 0
  ) {
    return new Array(VOICE_QUALITY_FEATURE_COUNT).fill(0);
  }

  const Meyda = await getMeyda();
  if (!Meyda) {
    sdkWarn("[Entros SDK] Meyda unavailable; voice quality features will be zeros.");
    return new Array(VOICE_QUALITY_FEATURE_COUNT).fill(0);
  }

  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;
  if (numFrames < 5) {
    return new Array(VOICE_QUALITY_FEATURE_COUNT).fill(0);
  }

  const cppValues: number[] = [];
  const tiltValues: number[] = [];
  const h1h2Values: number[] = [];
  const lowRatios: number[] = [];
  const midRatios: number[] = [];
  const highRatios: number[] = [];

  // Reusable frame buffer (matches speaker.ts::computeLTAS).
  const frame = new Float32Array(frameSize);

  // Meyda.extract's third argument is `previousSignal`, NOT options — passing
  // `{ sampleRate, bufferSize }` there is silently ignored, leaving Meyda on
  // its default bufferSize=512 / sampleRate=44100. That truncates the visible
  // spectrum to bufferSize/2 = 256 bins regardless of input frame size, and
  // miscomputes any frequency-domain mapping. Set the globals before
  // extracting so the returned spectrum matches the input frame's actual
  // FFT geometry. (See speaker.ts::computeLTAS, which has the same misuse
  // pattern but only consumes scalar features so the bug is invisible there;
  // a follow-up commit aligns it.)
  Meyda.bufferSize = frameSize;
  Meyda.sampleRate = sampleRate;

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    frame.set(samples.subarray(start, start + frameSize), 0);

    const features = Meyda.extract("powerSpectrum", frame);
    const power = features as Float32Array | undefined | null;
    if (!power || power.length === 0) continue;

    const cpp = cepstralPeakProminence(power, sampleRate);
    if (Number.isFinite(cpp)) cppValues.push(cpp);

    const tilt = spectralTilt(power, sampleRate);
    if (Number.isFinite(tilt)) tiltValues.push(tilt);

    const f0 = f0PerFrame[i] ?? 0;
    if (f0 > 0) {
      const h1h2 = h1MinusH2(power, sampleRate, f0);
      if (Number.isFinite(h1h2)) h1h2Values.push(h1h2);
    }

    const [low, mid, high] = subbandRatios(power, sampleRate);
    lowRatios.push(low);
    midRatios.push(mid);
    highRatios.push(high);
  }

  const cppMean = meanOf(cppValues);
  const cppVar = varianceOf(cppValues, cppMean);
  const tiltMean = meanOf(tiltValues);
  const tiltVar = varianceOf(tiltValues, tiltMean);
  const h1h2Mean = meanOf(h1h2Values);
  const h1h2Var = varianceOf(h1h2Values, h1h2Mean);
  const lowMean = meanOf(lowRatios);
  const midMean = meanOf(midRatios);
  const highMean = meanOf(highRatios);

  return [
    cppMean,
    cppVar,
    tiltMean,
    tiltVar,
    h1h2Mean,
    h1h2Var,
    lowMean,
    midMean,
    highMean,
  ];
}
