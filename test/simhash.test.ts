import { describe, it, expect } from "vitest";
import { simhash, hammingDistance } from "../src/hashing/simhash";
import { FINGERPRINT_BITS } from "../src/config";

describe("simhash", () => {
  const featureA = Array.from({ length: 100 }, (_, i) => Math.sin(i * 0.1));

  it("produces a 256-bit binary fingerprint", () => {
    const fp = simhash(featureA);
    expect(fp.length).toBe(FINGERPRINT_BITS);
    for (const bit of fp) {
      expect(bit === 0 || bit === 1).toBe(true);
    }
  });

  it("is deterministic", () => {
    const fp1 = simhash(featureA);
    const fp2 = simhash(featureA);
    expect(fp1).toEqual(fp2);
  });

  it("similar vectors produce low Hamming distance", () => {
    // Slightly perturbed version of featureA
    const featureB = featureA.map((v) => v + (Math.random() - 0.5) * 0.01);
    const fpA = simhash(featureA);
    const fpB = simhash(featureB);
    const dist = hammingDistance(fpA, fpB);
    // Small perturbation should produce distance well below 128 (random chance)
    expect(dist).toBeLessThan(64);
  });

  it("dissimilar vectors produce high Hamming distance", () => {
    const featureC = Array.from({ length: 100 }, (_, i) => -Math.cos(i * 3.7));
    const fpA = simhash(featureA);
    const fpC = simhash(featureC);
    const dist = hammingDistance(fpA, fpC);
    // Different vectors should have distance closer to 128 (random)
    expect(dist).toBeGreaterThan(50);
  });

  it("empty feature vector returns all zeros", () => {
    const fp = simhash([]);
    expect(fp.length).toBe(FINGERPRINT_BITS);
    expect(fp.every((b) => b === 0)).toBe(true);
  });

  it("hamming distance is symmetric", () => {
    const fpA = simhash(featureA);
    const fpB = simhash(featureA.map((v) => v + 0.5));
    expect(hammingDistance(fpA, fpB)).toBe(hammingDistance(fpB, fpA));
  });

  it("hamming distance of identical fingerprints is zero", () => {
    const fp = simhash(featureA);
    expect(hammingDistance(fp, fp)).toBe(0);
  });
});
