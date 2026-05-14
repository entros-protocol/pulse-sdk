import { describe, it, expect } from "vitest";
import {
  dctII,
  pitchContourShape,
  PITCH_CONTOUR_SHAPE_FEATURE_COUNT,
} from "../src/extraction/dct";

describe("dctII", () => {
  it("returns a zero-padded array when numCoefficients is zero", () => {
    expect(dctII([1, 2, 3], 0)).toEqual([]);
  });

  it("returns zeros when input is empty", () => {
    expect(dctII([], 5)).toEqual([0, 0, 0, 0, 0]);
  });

  it("constant input concentrates energy in DC coefficient", () => {
    // x_n = c for all n → X_0 = c × N, X_k = 0 for k > 0
    const N = 16;
    const c = 2.5;
    const input = new Array(N).fill(c);
    const out = dctII(input, 5);
    expect(out[0]).toBeCloseTo(c * N, 10);
    for (let k = 1; k < 5; k++) {
      expect(Math.abs(out[k]!)).toBeLessThan(1e-10);
    }
  });

  it("pure cosine input concentrates energy at the matching DCT bin", () => {
    // x_n = cos((π/N)(n + 0.5) × k0) → only X_{k0} should be non-zero
    const N = 32;
    const k0 = 3;
    const input: number[] = [];
    for (let n = 0; n < N; n++) {
      input.push(Math.cos((Math.PI / N) * (n + 0.5) * k0));
    }
    const out = dctII(input, 8);
    // X_{k0} ≈ N/2 (the orthogonality factor), other bins ≈ 0.
    for (let k = 0; k < 8; k++) {
      if (k === k0) {
        expect(out[k]!).toBeCloseTo(N / 2, 8);
      } else {
        expect(Math.abs(out[k]!)).toBeLessThan(1e-9);
      }
    }
  });

  it("pads with zeros when numCoefficients exceeds input length", () => {
    const out = dctII([1, 2, 3], 6);
    expect(out).toHaveLength(6);
    // Trailing positions beyond N are explicitly zero.
    expect(out[3]).toBe(0);
    expect(out[4]).toBe(0);
    expect(out[5]).toBe(0);
  });

  it("returns deterministic output for identical input", () => {
    const input = [1, -2, 3, -4, 5, -6, 7];
    const a = dctII(input, 4);
    const b = dctII(input, 4);
    expect(a).toEqual(b);
  });

  it("is linear (DCT(a×x + b×y) = a×DCT(x) + b×DCT(y))", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [5, 4, 3, 2, 1];
    const a = 2;
    const b = -1;
    const combined = x.map((xi, i) => a * xi + b * y[i]!);
    const dctCombined = dctII(combined, 3);
    const dctX = dctII(x, 3);
    const dctY = dctII(y, 3);
    for (let k = 0; k < 3; k++) {
      expect(dctCombined[k]!).toBeCloseTo(a * dctX[k]! + b * dctY[k]!, 10);
    }
  });
});

describe("pitchContourShape", () => {
  it("returns the documented feature count", () => {
    // Build a contour long enough to satisfy the voiced-frame floor.
    const contour = Array.from({ length: 100 }, (_, i) =>
      120 + 10 * Math.sin(i * 0.1),
    );
    const shape = pitchContourShape(contour);
    expect(shape).toHaveLength(PITCH_CONTOUR_SHAPE_FEATURE_COUNT);
    expect(shape).toHaveLength(5);
  });

  it("returns zeros for empty contour", () => {
    const shape = pitchContourShape([]);
    expect(shape).toHaveLength(5);
    expect(shape.every((v) => v === 0)).toBe(true);
  });

  it("returns zeros when fewer than 2 × numCoefficients voiced frames", () => {
    // 9 voiced frames < 2 × 5 = 10 floor.
    const contour = [0, 0, 100, 110, 120, 130, 0, 0, 105, 115, 125, 135, 100];
    const shape = pitchContourShape(contour);
    expect(shape.every((v) => v === 0)).toBe(true);
  });

  it("ignores unvoiced (F0 = 0) frames", () => {
    // Two contours: one with explicit zeros, one without — same voiced
    // values should produce identical shapes.
    const voiced = [120, 125, 122, 119, 121, 130, 128, 124, 126, 122, 118, 121];
    const sparse = [0, 0, ...voiced, 0, 0];
    const dense = [...voiced];
    const a = pitchContourShape(sparse);
    const b = pitchContourShape(dense);
    for (let k = 0; k < 5; k++) {
      expect(a[k]!).toBeCloseTo(b[k]!, 10);
    }
  });

  it("DC coefficient is near zero after mean-centering", () => {
    // Mean-centering puts the DC component at zero by construction.
    const contour = Array.from({ length: 50 }, () => 100 + Math.random() * 20);
    const shape = pitchContourShape(contour);
    expect(Math.abs(shape[0]!)).toBeLessThan(1e-10);
  });

  it("constant pitch contour produces zero shape vector", () => {
    // A constant pitch (no prosodic shape) has zero variance after centering,
    // so all DCT coefficients are zero.
    const contour = new Array(50).fill(120);
    const shape = pitchContourShape(contour);
    for (const v of shape) expect(Math.abs(v)).toBeLessThan(1e-10);
  });

  it("rising pitch produces distinct coefficients from falling pitch", () => {
    const N = 50;
    const rising = Array.from({ length: N }, (_, i) => 100 + i);
    const falling = Array.from({ length: N }, (_, i) => 100 + (N - 1 - i));
    const a = pitchContourShape(rising);
    const b = pitchContourShape(falling);
    // After mean-centering, rising and falling are negatives of each other,
    // so DCT coefficients flip sign.
    for (let k = 1; k < 5; k++) {
      // Each non-DC coefficient should differ from its mirror by ~ 2× magnitude.
      expect(Math.sign(a[k]!) === Math.sign(b[k]!)).toBe(false);
    }
  });

  it("returns deterministic output for identical input", () => {
    const contour = [120, 125, 130, 128, 124, 126, 122, 118, 121, 119];
    const a = pitchContourShape(contour);
    const b = pitchContourShape(contour);
    expect(a).toEqual(b);
  });

  it("handles non-finite F0 values by treating as unvoiced", () => {
    const contour = [120, NaN, 125, Infinity, 130, -1, 128, 124, 126, 122, 118, 121];
    const shape = pitchContourShape(contour);
    // Same voiced subset as filtering NaN/Inf/-1/0 manually.
    const cleanContour = [120, 125, 130, 128, 124, 126, 122, 118, 121];
    const cleanShape = pitchContourShape(cleanContour);
    // cleanContour has 9 voiced frames, below the 10 floor → both should
    // return zeros.
    expect(shape.every((v) => v === 0)).toBe(true);
    expect(cleanShape.every((v) => v === 0)).toBe(true);
  });
});
