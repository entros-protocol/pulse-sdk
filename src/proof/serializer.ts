import {
  BN254_BASE_FIELD,
  PROOF_A_SIZE,
  PROOF_B_SIZE,
  PROOF_C_SIZE,
  TOTAL_PROOF_SIZE,
  NUM_PUBLIC_INPUTS,
} from "../config";
import type { RawProof, SolanaProof } from "./types";
import { SCALAR_MODULUS } from "./request";

/**
 * Convert a decimal string to a 32-byte big-endian Uint8Array.
 */
export function toBigEndian32(decStr: string): Uint8Array {
  if (typeof decStr !== "string" || !/^(0|[1-9][0-9]*)$/.test(decStr))
    throw new Error("Expected a canonical unsigned decimal integer");
  let n = BigInt(decStr);
  if (n >= 1n << 256n) throw new Error("Integer exceeds 32 bytes");
  const bytes = new Uint8Array(32);
  for (let i = 31; i >= 0; i--) {
    bytes[i] = Number(n & BigInt(0xff));
    n >>= BigInt(8);
  }
  return bytes;
}

/**
 * Negate a G1 y-coordinate for groth16-solana proof_a format.
 */
function negateG1Y(yDecStr: string): Uint8Array {
  const y = BigInt(yDecStr);
  const yNeg = (BN254_BASE_FIELD - y) % BN254_BASE_FIELD;
  return toBigEndian32(yNeg.toString());
}

/**
 * Serialize an snarkjs proof into the 256-byte format groth16-solana expects.
 *
 * proof_a: 64 bytes (x + negated y)
 * proof_b: 128 bytes (G2 with reversed coordinate ordering: c1 before c0)
 * proof_c: 64 bytes (x + y)
 */
export function serializeProof(
  proof: RawProof,
  publicSignals: string[],
  generation: "legacy" | "request-bound-v1" = "legacy",
): SolanaProof {
  if (proof.protocol !== "groth16" || proof.curve !== "bn128") {
    throw new Error("Unsupported proof protocol or curve");
  }
  for (const point of [proof.pi_a, proof.pi_c]) {
    if (
      !Array.isArray(point) ||
      (point.length !== 2 && point.length !== 3) ||
      (point.length === 3 && point[2] !== "1")
    ) {
      throw new Error("Expected an affine G1 proof point");
    }
  }
  if (
    !Array.isArray(proof.pi_b) ||
    (proof.pi_b.length !== 2 && proof.pi_b.length !== 3) ||
    proof.pi_b.some(
      (coordinate) => !Array.isArray(coordinate) || coordinate.length !== 2,
    ) ||
    (proof.pi_b.length === 3 &&
      (proof.pi_b[2]![0] !== "1" || proof.pi_b[2]![1] !== "0"))
  ) {
    throw new Error("Expected an affine G2 proof point");
  }
  const count =
    generation === "request-bound-v1"
      ? 6
      : generation === "legacy"
        ? NUM_PUBLIC_INPUTS
        : 0;
  if (!count || publicSignals.length !== count) {
    throw new Error(
      `Expected ${count} public signals, got ${publicSignals.length}`,
    );
  }

  for (const coordinate of [
    proof.pi_a[0],
    proof.pi_a[1],
    ...proof.pi_b.slice(0, 2).flat().slice(0, 4),
    proof.pi_c[0],
    proof.pi_c[1],
  ]) {
    toBigEndian32(coordinate!);
    if (BigInt(coordinate!) >= BN254_BASE_FIELD)
      throw new Error("Noncanonical proof coordinate");
  }
  for (const [index, signal] of publicSignals.entries()) {
    toBigEndian32(signal);
    if (
      BigInt(signal) >= SCALAR_MODULUS ||
      (index >= 4 && BigInt(signal) >= 1n << 128n)
    )
      throw new Error("Noncanonical public input");
  }

  // proof_a: x (32 bytes) + negated y (32 bytes)
  const a0 = toBigEndian32(proof.pi_a[0]!);
  const a1 = negateG1Y(proof.pi_a[1]!);
  const proofA = new Uint8Array(PROOF_A_SIZE);
  proofA.set(a0, 0);
  proofA.set(a1, 32);

  // proof_b: G2 reversed coordinate ordering
  const b00 = toBigEndian32(proof.pi_b[0]![1]!); // c1 first
  const b01 = toBigEndian32(proof.pi_b[0]![0]!); // c0 second
  const b10 = toBigEndian32(proof.pi_b[1]![1]!);
  const b11 = toBigEndian32(proof.pi_b[1]![0]!);
  const proofB = new Uint8Array(PROOF_B_SIZE);
  proofB.set(b00, 0);
  proofB.set(b01, 32);
  proofB.set(b10, 64);
  proofB.set(b11, 96);

  // proof_c: x + y (no negation)
  const c0 = toBigEndian32(proof.pi_c[0]!);
  const c1 = toBigEndian32(proof.pi_c[1]!);
  const proofC = new Uint8Array(PROOF_C_SIZE);
  proofC.set(c0, 0);
  proofC.set(c1, 32);

  // Combine into single 256-byte blob
  const proofBytes = new Uint8Array(TOTAL_PROOF_SIZE);
  proofBytes.set(proofA, 0);
  proofBytes.set(proofB, PROOF_A_SIZE);
  proofBytes.set(proofC, PROOF_A_SIZE + PROOF_B_SIZE);

  // Public inputs as 32-byte big-endian arrays
  const publicInputs = publicSignals.map((s) => toBigEndian32(s));

  return { proofBytes, publicInputs };
}
