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
 * Output of `extractMfccFeatures` is 72 statistical aggregates over the
 * per-frame MFCC time-series captured during a 12-second session
 * (MFCC[0] is dropped as a cepstral DC term — see MFCC_DROP_LEADING):
 *
 *   - 12 used MFCC coefficients × 4 stats (mean, var, skewness, kurtosis) = 48
 *   - 12 used delta-MFCC coefficients × 2 stats (mean, var) = 24
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
import { condense, mean as meanOf, variance as varianceOf } from "./statistics";
import { sdkWarn } from "../log";

const NUM_MFCC_COEFFICIENTS = 13;
/** Drop the leading MFCC[0] (cepstral DC term, log-energy proxy). It's
 *  dominated by frame energy and mic gain rather than vocal-tract
 *  identity, sits at 1-3 orders of magnitude larger than MFCC[1..12], and
 *  was a contributing outlier to the per-modality z-score collapse the
 *  validator-side winsorize defends against. Standard Kaldi / sidekit
 *  speaker-recognition pipelines drop C0; we follow the convention. */
const MFCC_DROP_LEADING = 1;
/** Coefficients actually retained for downstream stats (12 = 13 - 1). */
const NUM_USED_MFCC = NUM_MFCC_COEFFICIENTS - MFCC_DROP_LEADING;
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
  NUM_USED_MFCC * 4 + // mean, var, skew, kurt per coefficient
  NUM_USED_MFCC * 2; // mean, var per delta coefficient
// = 48 + 24 = 72 (down from 78 after dropping MFCC[0])

/**
 * Apply standard pre-emphasis filter `H(z) = 1 - 0.97 z^-1` to the raw
 * sample stream before MFCC framing. Boosts the high-frequency content
 * where speaker-individual differences (F2/F3 formants, fricatives,
 * breathiness) live; without it MFCCs are dominated by the F1 vocal-tract
 * resonance which is less speaker-discriminative. Standard Kaldi/sidekit
 * pipelines apply this; Meyda's MFCC extractor does not.
 *
 * Filter coefficient 0.97 is the conventional speech-recognition value
 * (Furui 1981, ETSI ES 201 108). Returns a new buffer; does not mutate
 * the input.
 */
function applyPreEmphasis(samples: Float32Array): Float32Array {
  const out = new Float32Array(samples.length);
  if (samples.length === 0) return out;
  out[0] = samples[0]!;
  for (let i = 1; i < samples.length; i++) {
    out[i] = samples[i]! - 0.97 * samples[i - 1]!;
  }
  return out;
}

/**
 * Compute first-order delta (temporal derivative) of a per-frame time
 * series using a regression window. Standard speech-recognition formula
 * (Furui 1986):
 *
 *   delta_t = Σ_{n=1..N} n × (x_{t+n} - x_{t-n}) / (2 × Σ_{n=1..N} n²)
 *
 * Equivalent to a least-squares fit of a line to the surrounding 2N+1
 * frames. The numerator is symmetric: contributions from `n=1..N` on each
 * side are weighted by `n`. The denominator `2 × Σ_{i=1..N} i² =
 * N(N+1)(2N+1)/3` normalizes so a series with constant slope produces a
 * delta equal to that slope (verified by the linear-input test in
 * mfcc.test.ts).
 *
 * Edge frames use truncated windows: when offset `k` would land outside
 * `[0, n)` on either side, the entire ±k pair is dropped from the
 * numerator AND `2k²` is subtracted from the denominator. This keeps the
 * result a valid (less precise, but unbiased toward zero) least-squares
 * estimate over the surviving symmetric offsets.
 */
function computeDelta(series: number[], halfWidth: number): number[] {
  const n = series.length;
  const out: number[] = new Array(n);
  // Σ_{i=-N..N} i² = 2 × N(N+1)(2N+1)/6 = N(N+1)(2N+1)/3.
  // Computed once per series; constant across all frames except for
  // edge-truncation adjustments below.
  const fullDenom = (halfWidth * (halfWidth + 1) * (2 * halfWidth + 1)) / 3;
  for (let t = 0; t < n; t++) {
    let num = 0;
    let denom = fullDenom;
    for (let k = 1; k <= halfWidth; k++) {
      const tPlus = t + k;
      const tMinus = t - k;
      if (tPlus >= n || tMinus < 0) {
        // Edge: drop the symmetric pair from numerator and remove the
        // matching 2k² from the denominator. Without the adjustment edge
        // deltas would be biased toward zero (smaller numerator over an
        // unchanged denominator).
        denom -= 2 * k * k;
        continue;
      }
      num += k * (series[tPlus]! - series[tMinus]!);
    }
    if (denom <= 0) {
      // Pathological case — series too short for any symmetric window.
      // Deliver zero rather than NaN so downstream stats stay finite.
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
 * Returns 72 floats (see MFCC_FEATURE_COUNT above) in stable order over
 * coefficients C1..C12 (C0 is intentionally dropped — see MFCC_DROP_LEADING):
 *   [mean(c1), var(c1), skew(c1), kurt(c1),
 *    mean(c2), var(c2), skew(c2), kurt(c2),
 *    ...
 *    mean(c12), var(c12), skew(c12), kurt(c12),
 *    mean(d1), var(d1),
 *    mean(d2), var(d2),
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

  // Per-coefficient time series: mfccTracks[i][t] is the (i + MFCC_DROP_LEADING)-th
  // MFCC at frame t. Indexed by the USED coefficient slot, not Meyda's raw
  // [0..NUM_MFCC_COEFFICIENTS) range.
  const mfccTracks: number[][] = Array.from(
    { length: NUM_USED_MFCC },
    () => [],
  );

  // Reusable frame buffer to avoid allocating per frame (matches the
  // pre-allocation pattern in speaker.ts::computeLTAS).
  const frame = new Float32Array(frameSize);

  // Meyda.extract's third argument is `previousSignal`, NOT options — passing
  // `{ sampleRate, bufferSize }` silently goes to that slot and is ignored.
  // Configure Meyda's globals before extracting so the mel filter bank is
  // built for the correct sample rate and frame size. Without this, Meyda
  // uses default sampleRate=44100 / bufferSize=512 — producing MFCCs that
  // are still self-consistent (same input → same output) but not aligned
  // with the actual frequency content of our 16 kHz / 2048-sample frames,
  // and not parity-comparable to librosa or other reference implementations.
  Meyda.bufferSize = frameSize;
  Meyda.sampleRate = sampleRate;

  // Apply pre-emphasis once over the whole capture before framing. Boosts
  // high-frequency content where F2/F3 + fricative differences between
  // speakers live; without it MFCCs are dominated by F1 resonance which
  // is less speaker-discriminative. Cheap (one O(n) pass), allocates one
  // Float32Array of the input length.
  const emphasized = applyPreEmphasis(samples);

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    frame.set(emphasized.subarray(start, start + frameSize), 0);

    const result = Meyda.extract("mfcc", frame) as number[] | null | undefined;

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

    // Skip MFCC[0..MFCC_DROP_LEADING) — the cepstral DC term carries
    // mic-energy noise rather than vocal-tract identity. Read from
    // `result[c + MFCC_DROP_LEADING]` so track index `c` corresponds to
    // the c-th USED coefficient (C1..C12 with the default drop).
    for (let c = 0; c < NUM_USED_MFCC; c++) {
      mfccTracks[c]!.push(result[c + MFCC_DROP_LEADING]!);
    }
  }

  // Aggregate per-coefficient track using the existing repo statistics
  // helpers. Reusing condense/mean/variance keeps the moment formulas
  // (sample variance, statistically-correct skewness, excess-kurtosis-
  // corrected kurtosis) consistent with the existing 44 audio features
  // computed elsewhere in the speaker block.
  const out: number[] = [];
  out.length = MFCC_FEATURE_COUNT;
  let writeIdx = 0;

  // 12 × 4 = 48 features (mean, variance, skewness, kurtosis per used coefficient).
  for (let c = 0; c < NUM_USED_MFCC; c++) {
    const stats = condense(mfccTracks[c]!);
    out[writeIdx++] = stats.mean;
    out[writeIdx++] = stats.variance;
    out[writeIdx++] = stats.skewness;
    out[writeIdx++] = stats.kurtosis;
  }

  // 12 × 2 = 24 features (mean, variance per delta of each used coefficient).
  for (let c = 0; c < NUM_USED_MFCC; c++) {
    const delta = computeDelta(mfccTracks[c]!, DELTA_REGRESSION_HALF_WIDTH);
    const muDelta = meanOf(delta);
    out[writeIdx++] = muDelta;
    out[writeIdx++] = varianceOf(delta, muDelta);
  }

  return out;
}
