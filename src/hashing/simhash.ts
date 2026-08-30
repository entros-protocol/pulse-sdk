import { FINGERPRINT_BITS, LEGACY_SIMHASH_SEED } from "../config";
import { SPEAKER_FEATURE_COUNT } from "../extraction/speaker";
import {
  MOTION_FEATURE_COUNT,
  TOUCH_FEATURE_COUNT,
} from "../extraction/kinematic";
import { sdkWarn } from "../log";
import type { TemporalFingerprint } from "./types";
import { publicProjectionCoefficients } from "./hyperplanes";
import { getProjectionDefinition } from "../projection";

const hyperplaneCache = new Map<string, Float64Array>();

function legacyMulberry32(seed: number): () => number {
  let state = seed | 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}

function legacySeed(value: string): number {
  let hash = 0;
  for (const character of value) {
    hash = ((hash << 5) - hash + character.charCodeAt(0)) | 0;
  }
  return hash;
}

function legacyProjectionCoefficients(dimension: number): Float64Array {
  const random = legacyMulberry32(legacySeed(LEGACY_SIMHASH_SEED));
  return Float64Array.from(
    { length: FINGERPRINT_BITS * dimension },
    () => random() * 2 - 1,
  );
}

function getHyperplanes(
  dimension: number,
  projectionVersion: number,
): Float64Array {
  const cacheKey = `${projectionVersion}:${dimension}`;
  const cached = hyperplaneCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const definition = getProjectionDefinition(projectionVersion);
  const hyperplanes =
    definition.hyperplanes.family === "legacy"
      ? legacyProjectionCoefficients(dimension)
      : publicProjectionCoefficients(
          dimension,
          definition.hyperplanes.transcriptVersion,
        );
  hyperplaneCache.set(cacheKey, hyperplanes);
  return hyperplanes;
}

/**
 * Compute a 256-bit SimHash fingerprint from a feature vector.
 * Uses deterministic random hyperplanes seeded from the protocol constant.
 * Similar feature vectors produce fingerprints with low Hamming distance.
 */
// All supported feature schemas contain the same per-modality counts.
// Projection 0 uses schema 3 legacy extraction. Projection 1 uses schema 4
// corrected extraction. Projection 2 uses schema 5 normalized touch capture.
//   - Speaker: 44 legacy + 72 MFCC (12×4 + 12×2, MFCC[0] dropped)
//     + 24 LPC + 16 formant trajectories + 9 voice quality
//     + 5 pitch DCT = 170.
//   - Motion: 54 legacy + 27 v2 (cross-axis covariance,
//     FFT band energy, tremor peak, direction-reversal stats, motion
//     autocorrelation) = 81.
//   - Touch: 36 legacy + 21 v2 (pressure derivative, contact
//     geometry, curvature, velocity autocorrelation, gap distribution,
//     path efficiency) = 57.
// Total: 308. The constant is a soft warning gate. A dimension change must
// ship under a new projection version and migrate through authenticated
// rebaseline before the client stores the replacement baseline.
const EXPECTED_FEATURE_DIMENSION =
  SPEAKER_FEATURE_COUNT + MOTION_FEATURE_COUNT + TOUCH_FEATURE_COUNT;

function validateFeatureVector(
  features: number[],
  projectionVersion: number,
): void {
  const invalidIndex = features.findIndex(
    (feature) => !Number.isFinite(feature),
  );
  if (invalidIndex >= 0) {
    throw new Error(
      `Feature vector contains a non-finite value at ${invalidIndex}`,
    );
  }

  if (
    getProjectionDefinition(projectionVersion).hyperplanes.family ===
      "public" &&
    features.length !== EXPECTED_FEATURE_DIMENSION
  ) {
    throw new Error(
      `Projection version ${projectionVersion} requires exactly ${EXPECTED_FEATURE_DIMENSION} features`,
    );
  }

  if (features.length !== 0 && features.length !== EXPECTED_FEATURE_DIMENSION) {
    sdkWarn(
      `[Entros SDK] Feature vector has ${features.length} dimensions, expected ${EXPECTED_FEATURE_DIMENSION}. ` +
        `Fingerprint quality may be degraded.`,
    );
  }
}

function dotProductForPlane(
  features: number[],
  planes: Float64Array,
  planeIndex: number,
): number {
  const planeOffset = planeIndex * features.length;
  let dot = 0;
  for (let featureIndex = 0; featureIndex < features.length; featureIndex++) {
    dot +=
      (features[featureIndex] ?? 0) * (planes[planeOffset + featureIndex] ?? 0);
  }
  return dot;
}

/**
 * Return the signed projection value behind each SimHash bit.
 *
 * This source-only diagnostic lets parity tests distinguish feature drift from
 * a bit flip near zero. The package root does not export it.
 */
export function simhashDotProducts(
  features: number[],
  projectionVersion = 0,
): number[] {
  validateFeatureVector(features, projectionVersion);
  if (features.length === 0) {
    return new Array(FINGERPRINT_BITS).fill(0);
  }

  const planes = getHyperplanes(features.length, projectionVersion);
  const dotProducts = new Array<number>(FINGERPRINT_BITS);

  for (let i = 0; i < FINGERPRINT_BITS; i++) {
    dotProducts[i] = dotProductForPlane(features, planes, i);
  }

  return dotProducts;
}

export function simhash(
  features: number[],
  projectionVersion = 0,
): TemporalFingerprint {
  validateFeatureVector(features, projectionVersion);

  if (features.length === 0) {
    return new Array(FINGERPRINT_BITS).fill(0);
  }

  const planes = getHyperplanes(features.length, projectionVersion);
  const fingerprint: TemporalFingerprint = [];

  for (let i = 0; i < FINGERPRINT_BITS; i++) {
    const dot = dotProductForPlane(features, planes, i);
    fingerprint.push(dot >= 0 ? 1 : 0);
  }

  return fingerprint;
}

/**
 * Compute Hamming distance between two fingerprints.
 */
export function hammingDistance(
  a: TemporalFingerprint,
  b: TemporalFingerprint,
): number {
  let distance = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) distance++;
  }
  return distance;
}
