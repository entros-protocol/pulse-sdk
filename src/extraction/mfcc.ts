/**
 * MFCC (Mel-Frequency Cepstral Coefficient) feature extraction.
 *
 * MFCCs are the industry-standard speaker-recognition feature: they encode
 * the SHAPE of the vocal tract via cepstral coefficients on a perceptual
 * mel-frequency scale. Two adult humans speaking the same word produce
 * different MFCC trajectories the same way two violins produce different
 * timbres of the same note. The original 44-feature audio block omitted
 * MFCCs entirely — this is the largest single discriminative-power gap
 * the v2 feature pipeline closes (see
 * `docs/master/BLUEPRINT-feature-pipeline-v2.md` §1.1).
 *
 * Output of `extractMfccFeatures` is 78 statistical aggregates over the
 * per-frame MFCC time-series captured during a 12-second session:
 *
 *   - 13 MFCC coefficients × 4 stats (mean, var, skewness, kurtosis) = 52
 *   - 13 delta-MFCC coefficients × 2 stats (mean, var) = 26
 *
 * The deltas (first-order temporal derivatives via 9-frame regression
 * window) capture how the vocal tract shape CHANGES during articulation —
 * a complementary identity signal to the static MFCCs.
 *
 * @privacyGuarantee MFCCs are themselves a dimensionality reduction (FFT →
 * mel filter bank → log → DCT, keeping the first 13 coefficients). The
 * statistics across frames add a second reduction layer. Aggregated
 * MFCC stats cannot reconstruct intelligible audio without a separate
 * vocoder model, and even with one only a coarse approximation is
 * possible. This is the same privacy posture as the existing 44-feature
 * audio block: statistical aggregates of on-device-computed signals.
 */
import { entropy } from "./statistics";
import { sdkWarn } from "../log";

const NUM_MFCC_COEFFICIENTS = 13;
/** Half-width of the regression window used to compute delta-MFCCs. The
 *  standard speech-recognition value is 2 (window of 9 frames total: 4 on
 *  each side plus the center). Larger N smooths more but lags behind
 *  rapid articulation; 2 balances responsiveness against noise. */
const DELTA_REGRESSION_HALF_WIDTH = 2;

/**
 * Total feature count produced by `extractMfccFeatures`. Imported by
 * speaker.ts when assembling the final audio feature vector.
 */
export const MFCC_FEATURE_COUNT =
  NUM_MFCC_COEFFICIENTS * 4 + // mean, var, skew, kurt per coefficient
  NUM_MFCC_COEFFICIENTS * 2; // mean, var per delta coefficient
// = 52 + 26 = 78

/**
 * Compute mean, variance, skewness, kurtosis of a 1D numeric array.
 *
 * Local helper rather than reusing `condense()` from statistics.ts because
 * `condense` returns 4 values [mean, var, skew, kurt] but we want to use
 * different combinations per channel (4 for MFCCs, 2 for delta-MFCCs).
 * Inlining keeps the per-channel call sites minimal.
 */
function moments(values: number[]): { mean: number; var_: number; skew: number; kurt: number } {
  const n = values.length;
  if (n === 0) return { mean: 0, var_: 0, skew: 0, kurt: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return { mean, var_: 0, skew: 0, kurt: 0 };
  let m2 = 0;
  let m3 = 0;
  let m4 = 0;
  for (const v of values) {
    const d = v - mean;
    const d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  const var_ = m2 / (n - 1);
  if (var_ < 1e-12) return { mean, var_: 0, skew: 0, kurt: 0 };
  const std = Math.sqrt(var_);
  const skew = m3 / n / (std * std * std);
  const kurt = m4 / n / (var_ * var_) - 3; // excess kurtosis (Gaussian = 0)
  return { mean, var_, skew, kurt };
}

/**
 * Compute first-order delta (temporal derivative) of a per-frame time
 * series using a regression window. This is the standard delta formula
 * from speech-recognition pipelines (Furui 1986):
 *
 *   delta_t = Σ_{n=1..N} n × (x_{t+n} - x_{t-n}) / (2 × Σ_{n=1..N} n²)
 *
 * Equivalent to a least-squares fit of a line to the surrounding 2N+1
 * frames. Edge frames use truncated windows (no padding), preserving
 * the time-series length.
 */
function computeDelta(series: number[], halfWidth: number): number[] {
  const n = series.length;
  const out: number[] = new Array(n);
  // Pre-compute the denominator: 2 × Σ_{i=1..N} i² = 2 × N(N+1)(2N+1)/6
  const fullDenom = (2 * halfWidth * (halfWidth + 1) * (2 * halfWidth + 1)) / 6 * 2;
  for (let t = 0; t < n; t++) {
    let num = 0;
    let denom = fullDenom;
    let edgeAdjust = false;
    for (let k = 1; k <= halfWidth; k++) {
      const tPlus = t + k;
      const tMinus = t - k;
      if (tPlus >= n || tMinus < 0) {
        // Edge: truncate the window. Drop k contribution from numerator AND
        // adjust denominator so the regression remains a least-squares fit
        // over the surviving offsets — otherwise edge deltas are biased
        // toward zero.
        edgeAdjust = true;
        denom -= 2 * k * k;
        continue;
      }
      num += k * (series[tPlus]! - series[tMinus]!);
    }
    if (edgeAdjust && denom <= 0) {
      // Pathological case (very short series); deliver zero rather than NaN.
      out[t] = 0;
      continue;
    }
    out[t] = num / denom;
  }
  return out;
}

/**
 * Loaded lazily so SDK consumers that don't need audio extraction don't pay
 * the Meyda bundle cost. Mirrors the pattern in speaker.ts::getMeyda
 * exactly, including the `any` typing — Meyda's published types don't
 * surface the runtime `.extract` method on the module-default export
 * cleanly across bundler interop, and matching speaker.ts's existing
 * pragma keeps the integration consistent.
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
 * Extract MFCC and delta-MFCC statistical features from an audio capture.
 *
 * Computes MFCCs frame-by-frame using Meyda's built-in extractor, then
 * applies regression-based delta to capture temporal dynamics. Aggregates
 * each per-coefficient time-series to a small set of moments suitable for
 * fingerprinting (resistant to phrase content; sensitive to vocal tract
 * shape).
 *
 * Returns 78 floats (see MFCC_FEATURE_COUNT above) in stable order:
 *   [mean(c0), var(c0), skew(c0), kurt(c0),
 *    mean(c1), var(c1), skew(c1), kurt(c1),
 *    ...
 *    mean(c12), var(c12), skew(c12), kurt(c12),
 *    mean(d0), var(d0),
 *    mean(d1), var(d1),
 *    ...
 *    mean(d12), var(d12)]
 *
 * On invalid input (zero-length samples, non-finite sample rate, or Meyda
 * unavailable) returns a zero vector of the correct length so the caller
 * can concatenate without conditional logic.
 */
export async function extractMfccFeatures(
  samples: Float32Array,
  sampleRate: number,
  frameSize: number,
  hopSize: number,
): Promise<number[]> {
  if (
    !Number.isFinite(sampleRate) ||
    sampleRate <= 0 ||
    samples.length === 0 ||
    frameSize <= 0 ||
    hopSize <= 0
  ) {
    return new Array(MFCC_FEATURE_COUNT).fill(0);
  }

  const Meyda = await getMeyda();
  if (!Meyda) {
    sdkWarn("[Entros SDK] Meyda unavailable; MFCC features will be zeros.");
    return new Array(MFCC_FEATURE_COUNT).fill(0);
  }

  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;
  if (numFrames < 5) {
    return new Array(MFCC_FEATURE_COUNT).fill(0);
  }

  // Per-coefficient time series: mfccTracks[i][t] is the i-th MFCC at frame t.
  const mfccTracks: number[][] = Array.from(
    { length: NUM_MFCC_COEFFICIENTS },
    () => [],
  );

  // Reusable frame buffer to avoid allocating per frame (matches the
  // pre-allocation pattern in speaker.ts::computeLTAS).
  const frame = new Float32Array(frameSize);

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    frame.set(samples.subarray(start, start + frameSize), 0);

    const result = Meyda.extract("mfcc", frame, { sampleRate, bufferSize: frameSize }) as
      | number[]
      | null
      | undefined;

    if (!Array.isArray(result) || result.length !== NUM_MFCC_COEFFICIENTS) {
      // Skip frames where Meyda failed to extract MFCCs (typically silent
      // or pathologically small frames). Keeping per-coefficient track
      // lengths in sync — a frame is either added to ALL tracks or NONE.
      continue;
    }

    let allFinite = true;
    for (let c = 0; c < NUM_MFCC_COEFFICIENTS; c++) {
      if (!Number.isFinite(result[c]!)) {
        allFinite = false;
        break;
      }
    }
    if (!allFinite) continue;

    for (let c = 0; c < NUM_MFCC_COEFFICIENTS; c++) {
      mfccTracks[c]!.push(result[c]!);
    }
  }

  // Aggregate per-coefficient track to 4 moments.
  const out: number[] = [];
  out.length = MFCC_FEATURE_COUNT;
  let writeIdx = 0;

  for (let c = 0; c < NUM_MFCC_COEFFICIENTS; c++) {
    const m = moments(mfccTracks[c]!);
    out[writeIdx++] = m.mean;
    out[writeIdx++] = m.var_;
    out[writeIdx++] = m.skew;
    out[writeIdx++] = m.kurt;
  }

  // Compute delta tracks and aggregate to 2 moments each.
  for (let c = 0; c < NUM_MFCC_COEFFICIENTS; c++) {
    const delta = computeDelta(mfccTracks[c]!, DELTA_REGRESSION_HALF_WIDTH);
    const m = moments(delta);
    out[writeIdx++] = m.mean;
    out[writeIdx++] = m.var_;
  }

  return out;
}

// `entropy` import retained for parity with the wider extraction module's
// shared statistical vocabulary; not needed for MFCCs (entropy on raw MFCC
// values is dominated by their bin distribution, which the moments already
// capture cleanly). Re-exported as a no-op import-elision guard so a future
// commit that DOES need entropy (e.g., entropy of mel-band energies) doesn't
// re-add the import line.
export { entropy as _entropyImportGuard };
