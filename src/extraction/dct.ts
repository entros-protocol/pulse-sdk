/**
 * Discrete Cosine Transform Type-II (DCT-II) and pitch-contour shape
 * encoding.
 *
 * Used by the v2 feature pipeline to capture the SHAPE of the per-session
 * F0 (pitch) contour as a small number of coefficients. Identity-bearing
 * because individual speakers have characteristic prosodic patterns —
 * rising/falling/modulated melodies — that the existing F0 statistics
 * (mean, variance, skew, kurtosis) don't reach. Closes the
 * "no-pitch-trajectory-shape" gap documented in
 * `docs/master/BLUEPRINT-feature-pipeline-v2.md` §1.1.
 *
 * @privacyGuarantee The output is a small fixed number of coefficients
 * (5 by default) capturing the lowest-frequency components of the pitch
 * contour. This is a strict dimensionality reduction (1200 frames → 5
 * coefficients) and cannot reconstruct the original contour, let alone
 * the underlying audio.
 */

/**
 * Compute the first `numCoefficients` Type-II DCT coefficients of a 1D
 * signal:
 *
 *   X_k = Σ_{n=0..N-1} x_n × cos((π/N) × (n + 0.5) × k),  k = 0..K-1
 *
 * No orthonormalization factor applied — caller can divide by
 * `sqrt(N)` (or equivalent) for length-invariance if comparing across
 * variable-length inputs. For length-fixed contexts (like the per-session
 * F0 contour) the un-normalized form is fine.
 *
 * Direct O(N × K) implementation is faster than FFT-DCT for the small K
 * values (≤ 16) we use in the feature pipeline. For N=1200 frames and
 * K=5, the total cost is 6000 cosine evaluations + multiply-adds —
 * sub-millisecond on modern CPUs.
 *
 * Returns an array of length exactly `numCoefficients`. If `numCoefficients`
 * exceeds N, trailing positions are zero. If N is 0 or numCoefficients ≤ 0,
 * returns a zero-padded array of the requested length.
 */
export function dctII(input: number[], numCoefficients: number): number[] {
  const N = input.length;
  const K = Math.max(0, numCoefficients);
  const output = new Array(K).fill(0);
  if (N === 0 || K === 0) return output;

  const upper = Math.min(K, N);
  const piOverN = Math.PI / N;

  for (let k = 0; k < upper; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += input[n]! * Math.cos(piOverN * (n + 0.5) * k);
    }
    output[k] = sum;
  }
  return output;
}

/**
 * Encode the shape of a pitch contour as a small number of DCT
 * coefficients suitable for fingerprinting.
 *
 * Pre-processing:
 * 1. Strip unvoiced frames (F0 = 0). Pitch detectors mark frames
 *    without detectable voicing as 0; including them in the DCT mixes
 *    silence with prosody and dilutes the shape signal.
 * 2. Mean-center the surviving voiced frames so the DC coefficient (X_0)
 *    captures shape deviation from the speaker's mean pitch, not the
 *    speaker's absolute pitch (which is already encoded in the F0_STATS
 *    block of the existing 44 features).
 * 3. Length-normalize the DCT output by `1/sqrt(N)` so two recordings of
 *    different durations produce comparable shape vectors.
 *
 * Returns exactly `numCoefficients` floats. If too few voiced frames
 * exist (< 2 × numCoefficients), returns all zeros — short voiced
 * segments produce noisy shape estimates that would hurt
 * discrimination more than help.
 */
export function pitchContourShape(
  contour: number[],
  numCoefficients: number = 5,
): number[] {
  if (numCoefficients <= 0) return [];
  const zero = () => new Array(numCoefficients).fill(0);

  // 1. Filter to voiced frames (F0 > 0) and finite values.
  const voiced: number[] = [];
  for (const v of contour) {
    if (Number.isFinite(v) && v > 0) voiced.push(v);
  }

  // Require enough voiced frames that the lowest few DCT coefficients are
  // statistically meaningful. 2 × numCoefficients is a coarse heuristic;
  // tighter would risk dropping legitimate short utterances.
  if (voiced.length < numCoefficients * 2) return zero();

  // 2. Mean-center.
  let sum = 0;
  for (const v of voiced) sum += v;
  const mu = sum / voiced.length;
  const centered = voiced.map((v) => v - mu);

  // 3. DCT and length-normalize.
  const N = centered.length;
  const norm = 1 / Math.sqrt(N);
  return dctII(centered, numCoefficients).map((c) => c * norm);
}

/**
 * Total feature count produced by `pitchContourShape` with the default
 * `numCoefficients = 5`. Imported by speaker.ts when assembling the
 * final audio feature vector.
 */
export const PITCH_CONTOUR_SHAPE_FEATURE_COUNT = 5;
