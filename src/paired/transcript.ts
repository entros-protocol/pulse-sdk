/**
 * Canonical transcript for paired rounds.
 *
 * Every digest carries the session nonce and the round index, so a commitment
 * from one session or one round cannot stand in for another. Every field is
 * length-prefixed with a four-byte big-endian length. Without the prefix,
 * `("entros", "paired")` and `("entrospaired", "")` would hash alike and a
 * caller could move bytes across a field boundary.
 *
 * The validator, the native app and an independent generator hold the same
 * formulas, and all of them check against one shared vector file.
 */

import { sha256 } from "@noble/hashes/sha2.js";

const ascii = (text: string): Uint8Array => new TextEncoder().encode(text);

export const DOMAINS = {
  session: ascii("entros/paired-round/v1/session\0"),
  attempt: ascii("entros/paired-round/v1/attempt\0"),
  challenge: ascii("entros/paired-round/v1/challenge\0"),
  audio: ascii("entros/paired-round/v1/audio\0"),
  path: ascii("entros/paired-round/v1/path\0"),
  round: ascii("entros/paired-round/v1/round\0"),
  commitRequest: ascii("entros/paired-round/v1/commit-request\0"),
  final: ascii("entros/paired-round/v1/final\0"),
  attestation: ascii("entros/attestation/v1\0"),
} as const;

/** The one audio format. Its label enters every audio digest. */
export const AUDIO_FORMAT = "pcm_s16le_16000_mono";
export const SCHEMA_VERSION = 1;
export const MIN_WAYPOINTS = 3;
export const MAX_WAYPOINTS = 5;
export const MIN_PATH_POINTS = 8;
export const MAX_PATH_POINTS = 64;
/** Coordinates lie on an integer grid over the interaction surface. */
export const COORDINATE_MAX = 1000;

export type Digest = Uint8Array;

export interface GridPoint {
  x: number;
  y: number;
}

export type EncodingFailure =
  | "waypoint_count_out_of_range"
  | "point_count_out_of_range"
  | "coordinate_out_of_range"
  | "malformed";

export class PairedEncodingError extends Error {
  constructor(readonly reason: EncodingFailure) {
    super(reason);
    this.name = "PairedEncodingError";
  }
}

function u32(value: number): Uint8Array {
  const out = new Uint8Array(4);
  new DataView(out.buffer).setUint32(0, value, false);
  return out;
}

function u16(value: number): Uint8Array {
  const out = new Uint8Array(2);
  new DataView(out.buffer).setUint16(0, value, false);
  return out;
}

function u64(value: number): Uint8Array {
  const out = new Uint8Array(8);
  new DataView(out.buffer).setBigUint64(0, BigInt(value), false);
  return out;
}

/** Length-prefixed concatenation. Exposed so the vectors can check the encoding itself. */
export function encode(fields: readonly Uint8Array[]): Uint8Array {
  const total = fields.reduce((sum, field) => sum + 4 + field.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const field of fields) {
    out.set(u32(field.length), offset);
    out.set(field, offset + 4);
    offset += 4 + field.length;
  }
  return out;
}

function digest(fields: readonly Uint8Array[]): Digest {
  const hash = sha256.create();
  for (const field of fields) {
    hash.update(u32(field.length));
    hash.update(field);
  }
  return hash.digest();
}

function checkGrid(points: readonly GridPoint[]): void {
  for (const point of points) {
    if (
      !Number.isInteger(point.x) ||
      !Number.isInteger(point.y) ||
      point.x < 0 ||
      point.y < 0 ||
      point.x > COORDINATE_MAX ||
      point.y > COORDINATE_MAX
    ) {
      throw new PairedEncodingError("coordinate_out_of_range");
    }
  }
}

function writePoints(out: Uint8Array, offset: number, points: readonly GridPoint[]): void {
  const view = new DataView(out.buffer, out.byteOffset, out.byteLength);
  points.forEach((point, index) => {
    view.setUint16(offset + index * 4, point.x, false);
    view.setUint16(offset + index * 4 + 2, point.y, false);
  });
}

/** `version || waypoint_count || (x, y) ...` */
export function encodePathTarget(waypoints: readonly GridPoint[]): Uint8Array {
  if (waypoints.length < MIN_WAYPOINTS || waypoints.length > MAX_WAYPOINTS) {
    throw new PairedEncodingError("waypoint_count_out_of_range");
  }
  checkGrid(waypoints);
  const out = new Uint8Array(2 + waypoints.length * 4);
  out[0] = SCHEMA_VERSION;
  out[1] = waypoints.length;
  writePoints(out, 2, waypoints);
  return out;
}

export function decodePathTarget(bytes: Uint8Array): GridPoint[] {
  if (bytes.length < 2 || bytes[0] !== SCHEMA_VERSION) {
    throw new PairedEncodingError("malformed");
  }
  const count = bytes[1]!;
  if (count < MIN_WAYPOINTS || count > MAX_WAYPOINTS) {
    throw new PairedEncodingError("waypoint_count_out_of_range");
  }
  if (bytes.length !== 2 + count * 4) throw new PairedEncodingError("malformed");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const points = Array.from({ length: count }, (_, index) => ({
    x: view.getUint16(2 + index * 4, false),
    y: view.getUint16(4 + index * 4, false),
  }));
  checkGrid(points);
  return points;
}

/** `version || point_count || (x, y) ...` */
export function encodeCoarsePath(points: readonly GridPoint[]): Uint8Array {
  if (points.length < MIN_PATH_POINTS || points.length > MAX_PATH_POINTS) {
    throw new PairedEncodingError("point_count_out_of_range");
  }
  checkGrid(points);
  const out = new Uint8Array(3 + points.length * 4);
  out[0] = SCHEMA_VERSION;
  new DataView(out.buffer).setUint16(1, points.length, false);
  writePoints(out, 3, points);
  return out;
}

export function attemptBindingDigest(serverAttemptId: Uint8Array, challengeNonce: Uint8Array): Digest {
  return digest([DOMAINS.attempt, serverAttemptId, challengeNonce]);
}

export function challengeDigest(
  sessionNonce: Uint8Array,
  roundIndex: number,
  roundNonce: Uint8Array,
  word: string,
  pathTarget: Uint8Array,
): Digest {
  return digest([DOMAINS.challenge, sessionNonce, u32(roundIndex), roundNonce, ascii(word), pathTarget]);
}

export function audioDigest(
  sessionNonce: Uint8Array,
  roundIndex: number,
  challenge: Digest,
  audioFormat: string,
  segment: Uint8Array,
): Digest {
  return digest([DOMAINS.audio, sessionNonce, u32(roundIndex), challenge, ascii(audioFormat), segment]);
}

export function pathDigest(
  sessionNonce: Uint8Array,
  roundIndex: number,
  challenge: Digest,
  coarsePath: Uint8Array,
): Digest {
  return digest([DOMAINS.path, sessionNonce, u32(roundIndex), challenge, coarsePath]);
}

/** `C_0`. The chain starts at the session, so no round commitment stands alone. */
export function sessionCommitment(
  sessionNonce: Uint8Array,
  attemptBinding: Digest,
  rounds: number,
  sessionExpiryUnixMs: number,
): Digest {
  return digest([DOMAINS.session, sessionNonce, attemptBinding, u32(rounds), u64(sessionExpiryUnixMs)]);
}

export interface RoundCommitmentFields {
  sessionNonce: Uint8Array;
  roundIndex: number;
  roundNonce: Uint8Array;
  challenge: Digest;
  previous: Digest;
  audioFormat: string;
  audioByteLength: number;
  audio: Digest;
  pathPointCount: number;
  path: Digest;
}

/** `C_k`. It carries the previous commitment, so the chain fixes the order. */
export function roundCommitment(fields: RoundCommitmentFields): Digest {
  return digest([
    DOMAINS.round,
    fields.sessionNonce,
    u32(fields.roundIndex),
    fields.roundNonce,
    fields.challenge,
    fields.previous,
    ascii(fields.audioFormat),
    u32(fields.audioByteLength),
    fields.audio,
    u32(fields.pathPointCount),
    fields.path,
  ]);
}

/** The server computes this itself. Clients use it to check their own bodies. */
export function commitRequestDigest(
  fields: Omit<RoundCommitmentFields, "audio" | "path"> & { current: Digest },
): Digest {
  return digest([
    DOMAINS.commitRequest,
    fields.sessionNonce,
    u32(fields.roundIndex),
    fields.roundNonce,
    fields.challenge,
    fields.previous,
    fields.current,
    ascii(fields.audioFormat),
    u32(fields.audioByteLength),
    u32(fields.pathPointCount),
  ]);
}

/** Audio and path digests in round order. Fixed width, so no separator is needed. */
export function evidenceManifest(entries: readonly { audio: Digest; path: Digest }[]): Uint8Array {
  const out = new Uint8Array(entries.length * 64);
  entries.forEach((entry, index) => {
    out.set(entry.audio, index * 64);
    out.set(entry.path, index * 64 + 32);
  });
  return out;
}

export function finalDigest(
  sessionNonce: Uint8Array,
  lastCommitment: Digest,
  rounds: number,
  manifest: Uint8Array,
): Digest {
  return digest([DOMAINS.final, sessionNonce, lastCommitment, u32(rounds), manifest]);
}

/**
 * The digest a native vendor attestation binds as its request hash. It covers
 * the protocol version, the session, the attempt, every committed round
 * through `finalDigest`, and the projection.
 */
export function attestationDigest(
  protocolVersion: number,
  sessionNonce: Uint8Array,
  attemptBinding: Digest,
  final: Digest,
  projectionVersion: number,
): Digest {
  return digest([
    DOMAINS.attestation,
    u16(protocolVersion),
    sessionNonce,
    attemptBinding,
    final,
    u16(projectionVersion),
  ]);
}

export function toHex(bytes: Uint8Array): string {
  let out = "";
  for (const byte of bytes) out += byte.toString(16).padStart(2, "0");
  return out;
}

/** Strict lowercase hex of an exact length, or null. */
export function fromHex(hex: string, byteLength?: number): Uint8Array | null {
  if (hex.length % 2 !== 0 || !/^[0-9a-f]*$/.test(hex)) return null;
  if (byteLength !== undefined && hex.length !== byteLength * 2) return null;
  const out = new Uint8Array(hex.length / 2);
  for (let index = 0; index < out.length; index++) {
    out[index] = parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return out;
}

export function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index++) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}
