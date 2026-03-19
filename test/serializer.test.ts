import { describe, it, expect } from "vitest";
import { serializeProof, toBigEndian32 } from "../src/proof/serializer";
import { TOTAL_PROOF_SIZE, NUM_PUBLIC_INPUTS, BN254_BASE_FIELD } from "../src/config";

// Mock proof matching snarkjs output format
const mockProof = {
  pi_a: [
    "12345678901234567890123456789012345678901234567890",
    "98765432109876543210987654321098765432109876543210",
    "1",
  ],
  pi_b: [
    [
      "11111111111111111111111111111111111111111111111111",
      "22222222222222222222222222222222222222222222222222",
    ],
    [
      "33333333333333333333333333333333333333333333333333",
      "44444444444444444444444444444444444444444444444444",
    ],
  ],
  pi_c: [
    "55555555555555555555555555555555555555555555555555",
    "66666666666666666666666666666666666666666666666666",
  ],
  protocol: "groth16",
  curve: "bn128",
};

const mockPublicSignals = [
  "111111111111111111111",
  "222222222222222222222",
  "30",
];

describe("serializer", () => {
  it("toBigEndian32 converts decimal string to 32 bytes", () => {
    const bytes = toBigEndian32("256");
    expect(bytes.length).toBe(32);
    expect(bytes[30]).toBe(1);
    expect(bytes[31]).toBe(0);
  });

  it("toBigEndian32 handles zero", () => {
    const bytes = toBigEndian32("0");
    expect(bytes.every((b) => b === 0)).toBe(true);
  });

  it("produces 256-byte proof output", () => {
    const { proofBytes } = serializeProof(mockProof as any, mockPublicSignals);
    expect(proofBytes.length).toBe(TOTAL_PROOF_SIZE);
  });

  it("produces correct number of public inputs", () => {
    const { publicInputs } = serializeProof(mockProof as any, mockPublicSignals);
    expect(publicInputs.length).toBe(NUM_PUBLIC_INPUTS);
    for (const input of publicInputs) {
      expect(input.length).toBe(32);
    }
  });

  it("negates proof_a y-coordinate", () => {
    const { proofBytes } = serializeProof(mockProof as any, mockPublicSignals);

    // Extract the y-coordinate from proof_a (bytes 32-63)
    let yFromProof = BigInt(0);
    for (let i = 32; i < 64; i++) {
      yFromProof = (yFromProof << BigInt(8)) + BigInt(proofBytes[i]!);
    }

    // The original y
    const yOriginal = BigInt(mockProof.pi_a[1]!);
    const yExpected = (BN254_BASE_FIELD - yOriginal) % BN254_BASE_FIELD;

    expect(yFromProof).toBe(yExpected);
  });

  it("reverses G2 coordinate ordering in proof_b", () => {
    const { proofBytes } = serializeProof(mockProof as any, mockPublicSignals);

    // First 32 bytes of proof_b should be pi_b[0][1] (c1), not pi_b[0][0] (c0)
    const expectedFirst = toBigEndian32(mockProof.pi_b[0]![1]!);
    const actualFirst = proofBytes.slice(64, 96);
    expect(Array.from(actualFirst)).toEqual(Array.from(expectedFirst));
  });
});
