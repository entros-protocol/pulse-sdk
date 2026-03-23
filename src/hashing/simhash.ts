import { FINGERPRINT_BITS, SIMHASH_SEED } from "../config";
import type { TemporalFingerprint } from "./types";

// Mulberry32 PRNG: deterministic, fast, good distribution
function mulberry32(seed: number): () => number {
  let state = seed | 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Derive a numeric seed from the protocol seed string
function deriveSeed(seedStr: string): number {
  let hash = 0;
  for (let i = 0; i < seedStr.length; i++) {
    const ch = seedStr.charCodeAt(i);
    hash = ((hash << 5) - hash + ch) | 0;
  }
  return hash;
}

let cachedHyperplanes: number[][] | null = null;
let cachedDimension = 0;

function getHyperplanes(dimension: number): number[][] {
  if (cachedHyperplanes && cachedDimension === dimension) {
    return cachedHyperplanes;
  }

  const rng = mulberry32(deriveSeed(SIMHASH_SEED));
  const planes: number[][] = [];

  for (let i = 0; i < FINGERPRINT_BITS; i++) {
    const plane: number[] = [];
    for (let j = 0; j < dimension; j++) {
      // Random value in [-1, 1]
      plane.push(rng() * 2 - 1);
    }
    planes.push(plane);
  }

  cachedHyperplanes = planes;
  cachedDimension = dimension;
  return planes;
}

/**
 * Compute a 256-bit SimHash fingerprint from a feature vector.
 * Uses deterministic random hyperplanes seeded from the protocol constant.
 * Similar feature vectors produce fingerprints with low Hamming distance.
 */
const EXPECTED_FEATURE_DIMENSION = 134; // 44 speaker + 54 motion/mouse + 36 touch

export function simhash(features: number[]): TemporalFingerprint {
  if (features.length === 0) {
    return new Array(FINGERPRINT_BITS).fill(0);
  }

  if (features.length !== EXPECTED_FEATURE_DIMENSION) {
    console.warn(
      `[IAM SDK] Feature vector has ${features.length} dimensions, expected ${EXPECTED_FEATURE_DIMENSION}. ` +
      `Fingerprint quality may be degraded.`
    );
  }

  const planes = getHyperplanes(features.length);
  const fingerprint: TemporalFingerprint = [];

  for (let i = 0; i < FINGERPRINT_BITS; i++) {
    const plane = planes[i];
    let dot = 0;
    for (let j = 0; j < features.length; j++) {
      dot += (features[j] ?? 0) * (plane?.[j] ?? 0);
    }
    fingerprint.push(dot >= 0 ? 1 : 0);
  }

  return fingerprint;
}

/**
 * Compute Hamming distance between two fingerprints.
 */
export function hammingDistance(
  a: TemporalFingerprint,
  b: TemporalFingerprint
): number {
  let distance = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) distance++;
  }
  return distance;
}
