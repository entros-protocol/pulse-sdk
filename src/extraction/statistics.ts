import type { StatsSummary } from "./types";

export function mean(values: number[]): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

export function variance(values: number[], mu?: number): number {
  if (values.length < 2) return 0;
  const m = mu ?? mean(values);
  let sum = 0;
  for (const v of values) sum += (v - m) ** 2;
  return sum / (values.length - 1);
}

// Cap on the absolute value of skewness emitted to the feature vector.
// Real human behavioral signals (voice prosody, motion, touch) produce
// skewness in [-5, +5]; values beyond ±20 indicate near-zero variance
// pathological frames (silent windows, near-constant signals) where the
// standardized third moment becomes numerically unstable. Unbounded
// outliers in this position were the root cause of the May-2026
// cross-person fingerprint collapse: a single feature with raw skewness
// ≈ 7,000 dominates the per-modality z-score in the validator and forces
// ~half the audio bits to a deterministic value across all users.
const SKEWNESS_BOUND = 20;

// Cap on kurtosis. Real human kurtosis is typically [0, 15]; values
// above 50 indicate extreme outliers in low-variance signals that don't
// carry identity-bearing information. Empirical: production LTAS
// kurtosis on a near-silent capture hit 153,881 — well outside any
// physically meaningful range — and dominated the audio block z-score.
const KURTOSIS_LOWER = 0;
const KURTOSIS_UPPER = 50;

export function skewness(values: number[]): number {
  if (values.length < 3) return 0;
  const n = values.length;
  const m = mean(values);
  const s = Math.sqrt(variance(values, m));
  if (s === 0) return 0;
  let sum = 0;
  for (const v of values) sum += ((v - m) / s) ** 3;
  const raw = (n / ((n - 1) * (n - 2))) * sum;
  return Math.max(-SKEWNESS_BOUND, Math.min(SKEWNESS_BOUND, raw));
}

export function kurtosis(values: number[]): number {
  if (values.length < 4) return 0;
  const n = values.length;
  const m = mean(values);
  const s2 = variance(values, m);
  if (s2 === 0) return 0;
  let sum = 0;
  for (const v of values) sum += ((v - m) ** 4) / s2 ** 2;
  const k =
    ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum -
    (3 * (n - 1) ** 2) / ((n - 2) * (n - 3));
  return Math.max(KURTOSIS_LOWER, Math.min(KURTOSIS_UPPER, k));
}

export function condense(values: number[]): StatsSummary {
  const m = mean(values);
  return {
    mean: m,
    variance: variance(values, m),
    skewness: skewness(values),
    kurtosis: kurtosis(values),
  };
}

/**
 * Shannon entropy over histogram bins. Measures information density.
 * Real human data has moderate entropy (varied but structured).
 * Synthetic data is either too uniform (high entropy) or too structured (low entropy).
 */
export function entropy(values: number[], bins: number = 16): number {
  if (values.length < 2) return 0;
  let min = values[0]!;
  let max = values[0]!;
  for (let i = 1; i < values.length; i++) {
    if (values[i]! < min) min = values[i]!;
    if (values[i]! > max) max = values[i]!;
  }
  if (min === max) return 0;

  const counts = new Array(bins).fill(0);
  const range = max - min;
  for (const v of values) {
    const idx = Math.min(Math.floor(((v - min) / range) * bins), bins - 1);
    counts[idx]++;
  }

  let h = 0;
  for (const c of counts) {
    if (c > 0) {
      const p = c / values.length;
      h -= p * Math.log2(p);
    }
  }
  return h;
}

/**
 * Autocorrelation at a given lag. Detects periodic synthetic patterns.
 * Real human data has low autocorrelation at most lags (chaotic/noisy).
 * Synthetic data often has high autocorrelation (periodic/smooth).
 */
export function autocorrelation(values: number[], lag: number = 1): number {
  if (values.length <= lag) return 0;
  const m = mean(values);
  const v = variance(values, m);
  if (v === 0) return 0;

  let sum = 0;
  for (let i = 0; i < values.length - lag; i++) {
    sum += (values[i]! - m) * (values[i + lag]! - m);
  }
  return sum / ((values.length - lag) * v);
}

/**
 * Normalize a feature group to zero mean and unit variance.
 * Ensures each modality (audio, motion, touch) contributes equally
 * to SimHash hyperplane projections regardless of raw magnitude scale.
 */
/**
 * Z-score normalize a feature group to zero mean and unit variance.
 * Ensures each modality contributes equally to SimHash hyperplane
 * projections regardless of raw magnitude scale.
 */
export function normalizeGroup(features: number[]): number[] {
  if (features.length === 0) return features;

  // Sanitize NaN/Infinity to 0 before computing stats.
  // Meyda spectral features can produce NaN on near-silent frames (0/0),
  // and a single NaN would poison the entire modality group.
  const clean = features.map((v) => (Number.isFinite(v) ? v : 0));

  let sum = 0;
  for (const v of clean) sum += v;
  const mean = sum / clean.length;

  let sqSum = 0;
  for (const v of clean) sqSum += (v - mean) * (v - mean);
  const std = Math.sqrt(sqSum / clean.length);

  if (std < 1e-8) return clean.map(() => 0);
  return clean.map((v) => (v - mean) / std);
}

/**
 * Concatenate raw features without normalization.
 * Used for server-side validation where physical units matter.
 */
export function fuseRawFeatures(
  audio: number[],
  motion: number[],
  touch: number[]
): number[] {
  const sanitize = (v: number) => (Number.isFinite(v) ? v : 0);
  return [...audio.map(sanitize), ...motion.map(sanitize), ...touch.map(sanitize)];
}

/**
 * Normalize and concatenate features for SimHash computation.
 */
export function fuseFeatures(
  audio: number[],
  motion: number[],
  touch: number[]
): number[] {
  return [...normalizeGroup(audio), ...normalizeGroup(motion), ...normalizeGroup(touch)];
}
