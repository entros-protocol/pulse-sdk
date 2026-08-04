// Poseidon backend: `poseidon-lite/poseidon3`, pure JS with zero dependencies.
//
// This replaced `circomlibjs.buildPoseidon()`, and the reason is dependency
// reach rather than taste. `circomlibjs` pulls `ethers@^5.5.1`, which pulls the
// whole `@ethersproject/*` tree, and that single edge accounted for **fifteen
// of the twenty-eight** advisories against `entros.io` — including a high on
// `ws` and the long-standing `elliptic` finding that had been recorded as
// unpatchable. It also put an Ethereum wallet stack in the bundle of a Solana
// protocol, and 26.9 MB of installed dependencies behind one hash function.
//
// Two other costs went with it. `buildPoseidon` compiles a WASM module,
// measured at **381 ms** on first call, landing between feature extraction and
// the commitment on the user's critical path.
//
//   circomlibjs      first call 381 ms   warm 0.149 ms/hash   26.9 MB installed
//   poseidon-lite    first call 0.4 ms   warm 0.209 ms/hash   788 KB, 0 deps
//
// The 26.9 MB is installed size. What the browser actually downloaded is a
// smaller and separate question, unmeasured at the time of writing: the old
// backend was reached through a dynamic `import()`, so bundlers code-split it
// into a chunk fetched when a commitment was first needed rather than on route
// load. Measure the `/verify` chunk before and after if the shipped figure is
// ever quoted anywhere.
//
// Per hash it is 0.06 ms slower, across roughly three hashes per verification,
// against 381 ms of startup removed.
//
// **The swap is bit-exact and that is not an assumption.** Both libraries
// implement iden3-parity round constants and MDS matrix over BN254. Verified
// across 300 random fingerprints and salts — including 0, 1, 2^128 and the
// field maximum — with zero differences. `entros-mobile` has run this backend
// in production since before the swap, and its `hashing/__tests__/parity.test.ts`
// pins byte-equality against the web SDK from the other side.
//
// The golden vectors in `test/poseidon.test.ts` are the gate here. They were
// generated against `circomlibjs` and must keep passing unchanged: every
// commitment ever written on chain is a function of this output, so a divergence
// would strand every existing identity.
//
// PRIVACY: this module holds the last reference to the 256-bit fingerprint.
// Callers must drop theirs after `generateTBH` returns and forward only the
// 32-byte commitment and salt.

import { poseidon3 } from "poseidon-lite/poseidon3";

import { BN254_SCALAR_FIELD, FINGERPRINT_BITS } from "../config";
import type { PackedFingerprint, TBH, TemporalFingerprint } from "./types";

/**
 * Pack 256-bit fingerprint into two 128-bit field elements.
 * Little-endian bit ordering within each chunk (matches circuit's Bits2Num).
 */
export function packBits(fingerprint: TemporalFingerprint): PackedFingerprint {
  let lo = BigInt(0);
  for (let i = 0; i < 128; i++) {
    if (fingerprint[i] === 1) {
      lo += BigInt(1) << BigInt(i);
    }
  }

  let hi = BigInt(0);
  for (let i = 0; i < 128; i++) {
    if (fingerprint[128 + i] === 1) {
      hi += BigInt(1) << BigInt(i);
    }
  }

  return { lo, hi };
}

/**
 * Compute Poseidon commitment: Poseidon(pack_lo, pack_hi, salt).
 * Matches the circuit's CommitmentCheck template exactly.
 */
export async function computeCommitment(
  fingerprint: TemporalFingerprint,
  salt: bigint
): Promise<bigint> {
  const { lo, hi } = packBits(fingerprint);
  // `poseidon3` returns the field element directly, where `circomlibjs` returned
  // a Montgomery-form buffer needing `F.toObject`. The value is identical.
  // Kept `async` deliberately: it is public API, and callers across the SDK,
  // `entros.io` and `entros-mobile` all await it.
  return poseidon3([lo, hi, salt]);
}

/**
 * Generate a random salt within the BN254 scalar field.
 */
export function generateSalt(): bigint {
  const bytes = new Uint8Array(31);
  crypto.getRandomValues(bytes);
  let val = BigInt(0);
  for (let i = 0; i < bytes.length; i++) {
    val = (val << BigInt(8)) + BigInt(bytes[i] ?? 0);
  }
  return val % BN254_SCALAR_FIELD;
}

/**
 * Convert a BigInt to a 32-byte big-endian Uint8Array.
 */
export function bigintToBytes32(n: bigint): Uint8Array {
  const bytes = new Uint8Array(32);
  let val = n;
  for (let i = 31; i >= 0; i--) {
    bytes[i] = Number(val & BigInt(0xff));
    val >>= BigInt(8);
  }
  return bytes;
}

/**
 * Generate a complete TBH from a fingerprint.
 */
export async function generateTBH(
  fingerprint: TemporalFingerprint,
  salt?: bigint
): Promise<TBH> {
  const s = salt ?? generateSalt();
  const commitment = await computeCommitment(fingerprint, s);
  return {
    fingerprint,
    salt: s,
    commitment,
    commitmentBytes: bigintToBytes32(commitment),
  };
}
