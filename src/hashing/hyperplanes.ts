import { sha256 } from "@noble/hashes/sha2.js";
import {
  CLIENT_PROJECTION_VERSION,
  SIMHASH_PUBLIC_SEED_HEX,
} from "../config";

const DOMAIN = Uint8Array.from(
  "entros-simhash-hyperplanes-v1\0",
  (character) => character.charCodeAt(0)
);
const WORD_SCALE = 0x1_0000_0000;
const MAX_PROJECTION_DIMENSION = 308;
const WORDS_PER_HYPERPLANE = 256;

export const PROJECTION_PURPOSE = {
  public: 0x00,
  private: 0x01,
} as const;

export type ProjectionPurpose =
  (typeof PROJECTION_PURPOSE)[keyof typeof PROJECTION_PURPOSE];

function hexToBytes(hex: string): Uint8Array {
  if (!/^[0-9a-f]{64}$/i.test(hex)) {
    throw new Error("Projection seed must contain 32 bytes of hexadecimal data");
  }

  const bytes = new Uint8Array(32);
  for (let index = 0; index < bytes.length; index++) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return bytes;
}

function writeUint16Le(target: Uint8Array, offset: number, value: number): void {
  new DataView(target.buffer, target.byteOffset, target.byteLength).setUint16(
    offset,
    value,
    true
  );
}

function writeUint32Le(target: Uint8Array, offset: number, value: number): void {
  new DataView(target.buffer, target.byteOffset, target.byteLength).setUint32(
    offset,
    value,
    true
  );
}

/**
 * Generate the deterministic word stream that defines a projection.
 * Each SHA-256 digest contributes eight little-endian unsigned words.
 */
export function generateProjectionWords(
  seed: Uint8Array,
  purpose: ProjectionPurpose,
  version: number,
  dimension: number,
  wordCount: number
): Uint32Array {
  if (seed.length !== 32) {
    throw new Error("Projection seed must contain exactly 32 bytes");
  }
  if (purpose !== PROJECTION_PURPOSE.public && purpose !== PROJECTION_PURPOSE.private) {
    throw new Error("Projection purpose must be public or private");
  }
  if (!Number.isInteger(version) || version < 0 || version > 0xffff) {
    throw new Error("Projection version must fit in an unsigned 16-bit integer");
  }
  if (!Number.isInteger(dimension) || dimension <= 0 || dimension > 0xffff_ffff) {
    throw new Error("Projection dimension must fit in a positive unsigned 32-bit integer");
  }
  if (!Number.isSafeInteger(wordCount) || wordCount < 0) {
    throw new Error("Projection word count must be a non-negative safe integer");
  }
  if (dimension > MAX_PROJECTION_DIMENSION) {
    throw new Error(`Projection dimension must not exceed ${MAX_PROJECTION_DIMENSION}`);
  }
  const maximumWordCount = WORDS_PER_HYPERPLANE * dimension;
  if (wordCount > maximumWordCount) {
    throw new Error(
      `Projection word count must not exceed ${maximumWordCount} for dimension ${dimension}`
    );
  }

  const prefixLength = DOMAIN.length + 1 + 2 + 4 + seed.length;
  const transcript = new Uint8Array(prefixLength + 4);
  transcript.set(DOMAIN, 0);

  let offset = DOMAIN.length;
  transcript[offset] = purpose;
  offset += 1;
  writeUint16Le(transcript, offset, version);
  offset += 2;
  writeUint32Le(transcript, offset, dimension);
  offset += 4;
  transcript.set(seed, offset);

  const blockOffset = prefixLength;
  const words = new Uint32Array(wordCount);
  let outputIndex = 0;
  let block = 0;

  while (outputIndex < wordCount) {
    writeUint32Le(transcript, blockOffset, block);
    const digest = sha256(transcript);
    const view = new DataView(digest.buffer, digest.byteOffset, digest.byteLength);
    for (let digestOffset = 0; digestOffset < digest.length && outputIndex < wordCount; digestOffset += 4) {
      words[outputIndex] = view.getUint32(digestOffset, true);
      outputIndex += 1;
    }
    block += 1;
  }

  return words;
}

export function projectionCoefficients(
  seed: Uint8Array,
  purpose: ProjectionPurpose,
  version: number,
  dimension: number,
  count: number
): Float64Array {
  const words = generateProjectionWords(seed, purpose, version, dimension, count);
  return Float64Array.from(words, (word) => (word / WORD_SCALE) * 2 - 1);
}

export function publicProjectionCoefficients(dimension: number): Float64Array {
  return projectionCoefficients(
    hexToBytes(SIMHASH_PUBLIC_SEED_HEX),
    PROJECTION_PURPOSE.public,
    CLIENT_PROJECTION_VERSION,
    dimension,
    256 * dimension
  );
}
