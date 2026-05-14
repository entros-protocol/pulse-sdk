import { describe, it, expect } from "vitest";
import { mean, variance, skewness, kurtosis, condense } from "../src/extraction/statistics";

describe("statistics", () => {
  it("computes mean correctly", () => {
    expect(mean([1, 2, 3, 4, 5])).toBe(3);
    expect(mean([10])).toBe(10);
    expect(mean([])).toBe(0);
  });

  it("computes variance correctly", () => {
    // Sample variance of [2, 4, 4, 4, 5, 5, 7, 9]
    const v = variance([2, 4, 4, 4, 5, 5, 7, 9]);
    expect(v).toBeCloseTo(4.571, 2);
  });

  it("returns zero variance for constant array", () => {
    expect(variance([5, 5, 5, 5])).toBe(0);
  });

  it("computes skewness correctly", () => {
    // Symmetric distribution has ~0 skewness
    const s = skewness([1, 2, 3, 4, 5]);
    expect(Math.abs(s)).toBeLessThan(0.01);
  });

  it("computes positive skewness for right-skewed data", () => {
    const s = skewness([1, 1, 1, 1, 1, 1, 1, 10]);
    expect(s).toBeGreaterThan(0);
  });

  it("computes kurtosis for normal-like data", () => {
    // Excess kurtosis near 0 for normal-like data
    const k = kurtosis([2, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8]);
    expect(Math.abs(k)).toBeLessThan(2);
  });

  it("condense returns all four stats", () => {
    const result = condense([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(result.mean).toBeCloseTo(5.5);
    expect(result.variance).toBeGreaterThan(0);
    expect(typeof result.skewness).toBe("number");
    expect(typeof result.kurtosis).toBe("number");
  });

  it("handles edge cases", () => {
    expect(variance([1])).toBe(0);
    expect(skewness([1, 2])).toBe(0);
    expect(kurtosis([1, 2, 3])).toBe(0);
  });

  // Pins the cross-person fingerprint-collapse fix: standardized moments
  // (skewness, kurtosis) must be bounded so a single near-silent or low-
  // variance frame cannot inject a 100,000+ value into the audio block,
  // collapse the validator's per-modality z-score, and force ~half the
  // audio bits to a deterministic value across all users.
  describe("bounded standardized moments (cross-person collapse defense)", () => {
    it("kurtosis on an outlier-dominated low-variance series clips at 50", () => {
      // 31 zeros + one 1000 → raw excess kurtosis would be in the 30+ range
      // for n=32; the bound caps it. (Production saw LTAS kurtosis 153,881
      // on near-silent frames, with the same shape — many small values
      // plus one extreme.)
      const series = [...new Array(31).fill(0), 1000];
      expect(kurtosis(series)).toBeLessThanOrEqual(50);
    });

    it("skewness on an outlier-dominated series clips at ±20", () => {
      const series = [...new Array(31).fill(0), 1000];
      const s = skewness(series);
      expect(Math.abs(s)).toBeLessThanOrEqual(20);
    });

    it("normal-range data passes through unaltered (regression guard against over-clipping)", () => {
      // Real human behavioral kurtosis sits in [0, 15]; this fixture
      // returns ~-1.2 (platykurtic), which the bound must NOT clip to 0
      // (kurtosis floor is 0 only because excess-kurtosis already
      // subtracts 3, and clipping in [0, 50] preserves the natural
      // post-correction range).
      const k = kurtosis([2, 3, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8]);
      expect(k).toBeGreaterThanOrEqual(0);
      expect(k).toBeLessThan(2);

      const s = skewness([1, 2, 3, 4, 5]);
      expect(Math.abs(s)).toBeLessThan(0.01);
    });
  });
});
