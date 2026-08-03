import { describe, it, expect } from "vitest";
import {
  computeCommitment,
  generateSalt,
  packBits,
  bigintToBytes32,
  generateTBH,
} from "../src/hashing/poseidon";
import { bytesToHex } from "../src/submit/receipt";
import { FINGERPRINT_BITS } from "../src/config";

describe("poseidon", () => {
  const testFingerprint = Array.from({ length: FINGERPRINT_BITS }, (_, i) =>
    i % 3 === 0 ? 1 : 0
  );
  const testSalt = BigInt("12345678901234567890");

  it("packs bits into two 128-bit field elements (little-endian)", () => {
    const bits = new Array(FINGERPRINT_BITS).fill(0);
    bits[0] = 1; // bit 0 → lo = 1
    bits[128] = 1; // bit 128 → hi = 1

    const { lo, hi } = packBits(bits);
    expect(lo).toBe(BigInt(1));
    expect(hi).toBe(BigInt(1));
  });

  it("packs complex bit patterns correctly", () => {
    const bits = new Array(FINGERPRINT_BITS).fill(0);
    bits[0] = 1;
    bits[1] = 1;
    bits[7] = 1;
    // lo should be 1 + 2 + 128 = 131
    const { lo } = packBits(bits);
    expect(lo).toBe(BigInt(131));
  });

  it("computes deterministic commitment", async () => {
    const c1 = await computeCommitment(testFingerprint, testSalt);
    const c2 = await computeCommitment(testFingerprint, testSalt);
    expect(c1).toBe(c2);
  });

  it("different salts produce different commitments", async () => {
    const c1 = await computeCommitment(testFingerprint, testSalt);
    const c2 = await computeCommitment(testFingerprint, testSalt + BigInt(1));
    expect(c1).not.toBe(c2);
  });

  it("different fingerprints produce different commitments", async () => {
    const fp2 = [...testFingerprint];
    fp2[0] = fp2[0] === 1 ? 0 : 1;
    const c1 = await computeCommitment(testFingerprint, testSalt);
    const c2 = await computeCommitment(fp2, testSalt);
    expect(c1).not.toBe(c2);
  });

  it("generates salt within BN254 scalar field", () => {
    const salt = generateSalt();
    expect(salt).toBeGreaterThan(BigInt(0));
    expect(salt).toBeLessThan(
      BigInt(
        "21888242871839275222246405745257275088548364400416034343698204186575808495617"
      )
    );
  });

  it("converts bigint to 32-byte big-endian", () => {
    const bytes = bigintToBytes32(BigInt(256));
    expect(bytes[30]).toBe(1);
    expect(bytes[31]).toBe(0);
    expect(bytes.length).toBe(32);
  });

  it("generates complete TBH", async () => {
    const tbh = await generateTBH(testFingerprint);
    expect(tbh.fingerprint).toEqual(testFingerprint);
    expect(tbh.salt).toBeGreaterThan(BigInt(0));
    expect(tbh.commitment).toBeGreaterThan(BigInt(0));
    expect(tbh.commitmentBytes.length).toBe(32);
  });
});

/**
 * Golden vectors for the commitment pipeline.
 *
 * IF ONE OF THESE FAILS, DO NOT UPDATE THE EXPECTED VALUE.
 *
 * A failure means `Poseidon(pack_lo, pack_hi, salt)` now returns something
 * different for an input it has already been asked about in production. Every
 * commitment ever written is a function of this output:
 *
 *   - the witness fed to `circuits/circom/entros_hamming.circom`
 *   - `public_inputs[0]` and `[1]` verified on chain by `entros-verifier`
 *   - `IdentityState.current_commitment` on every existing account
 *   - the 32-byte plaintext inside every encrypted baseline blob
 *
 * A changed value strands every existing identity with no route back except a
 * manual reset. That is the failure mode master-list #215 exists to prevent,
 * and it would arrive here disguised as a dependency bump.
 *
 * The values below were produced by the implementation in place on 2026-08-03,
 * which resolves Poseidon through `circomlibjs@0.1.7`. They are the contract
 * that any replacement has to satisfy before it can ship.
 *
 * The other tests in this file check properties (determinism, difference,
 * range) and every one of them passes against a Poseidon with different
 * parameters. Only these vectors catch that.
 */
describe("poseidon golden vectors", () => {
  const zeros = () => new Array(FINGERPRINT_BITS).fill(0);
  const withBit = (index: number) => {
    const bits = zeros();
    bits[index] = 1;
    return bits;
  };
  const vectors: Array<{
    name: string;
    bits: number[];
    salt: bigint;
    lo: bigint;
    hi: bigint;
    commitment: bigint;
    hex: string;
  }> = [
    {
      name: "all-zero bits, salt 0",
      bits: zeros(),
      salt: BigInt(0),
      lo: BigInt(0),
      hi: BigInt(0),
      commitment: BigInt(
        "5317387130258456662214331362918410991734007599705406860481038345552731150762"
      ),
      hex: "0bc188d27dcceadc1dcfb6af0a7af08fe2864eecec96c5ae7cee6db31ba599aa",
    },
    {
      name: "bit 0 only, salt 1",
      bits: withBit(0),
      salt: BigInt(1),
      lo: BigInt(1),
      hi: BigInt(0),
      commitment: BigInt(
        "19374975721259875597650302716689543547647001662517455822229477759190533109280"
      ),
      hex: "2ad5d8ff25aca9eb83aa08d35a9f6b882014e4e61b4a106c14f89c96991b7620",
    },
    {
      name: "bit 255 only, salt 1",
      bits: withBit(255),
      salt: BigInt(1),
      lo: BigInt(0),
      hi: BigInt("170141183460469231731687303715884105728"),
      commitment: BigInt(
        "17644347761648545681682997521384669051793140027481664359963980382622982655987"
      ),
      hex: "270258d06c7aa1d2cff2852d05f1632ca9996ecd0743f62fae0e1f1d4264cff3",
    },
    {
      name: "every third bit, salt 12345678901234567890",
      bits: Array.from({ length: FINGERPRINT_BITS }, (_, i) =>
        i % 3 === 0 ? 1 : 0
      ),
      salt: BigInt("12345678901234567890"),
      lo: BigInt("97223533405982418132392744980505203273"),
      hi: BigInt("194447066811964836264785489961010406546"),
      commitment: BigInt(
        "2580289765604261883697346907572201508384344210776783208497831974345038124587"
      ),
      hex: "05b4646ab0fc6698e3a5ab72a2860d5d3887c64c4ba8f5ca39056cd64c60562b",
    },
    {
      name: "all-one bits, salt 2^128",
      bits: new Array(FINGERPRINT_BITS).fill(1),
      salt: BigInt(1) << BigInt(128),
      lo: BigInt("340282366920938463463374607431768211455"),
      hi: BigInt("340282366920938463463374607431768211455"),
      commitment: BigInt(
        "17669011500249618667333297656535391305497061293013161677504289535263510874035"
      ),
      hex: "27104e5d553aaee6950346fe2f154f8e62a59abb4ae9a2d549fcbc585683c7b3",
    },
  ];

  for (const vector of vectors) {
    it(`packs to the pinned field elements: ${vector.name}`, () => {
      const { lo, hi } = packBits(vector.bits);
      expect(lo).toBe(vector.lo);
      expect(hi).toBe(vector.hi);
    });

    it(`commits and serialises to the pinned values: ${vector.name}`, async () => {
      const commitment = await computeCommitment(vector.bits, vector.salt);
      expect(commitment).toBe(vector.commitment);
      // `bytesToHex` is the same helper that produces `commitment_new_hex` on
      // the wire to the validator, so this pins the transmitted form too.
      expect(bytesToHex(bigintToBytes32(commitment))).toBe(vector.hex);
    });
  }

  it("every pinned commitment lies inside the BN254 scalar field", () => {
    for (const vector of vectors) {
      expect(vector.commitment).toBeGreaterThan(BigInt(0));
      expect(vector.commitment).toBeLessThan(
        BigInt(
          "21888242871839275222246405745257275088548364400416034343698204186575808495617"
        )
      );
    }
  });
});
