import { describe, expect, it } from "vitest";
import { analyzeAcousticRealism } from "../src/sensor/audio";

/** Simple deterministic pseudorandom generator (LCG) for reproducible unit tests */
function createSeededRandom(seed = 12345) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

describe("analyzeAcousticRealism", () => {
  it("returns zero metrics for empty samples", () => {
    const result = analyzeAcousticRealism(new Float32Array(0), 16000);
    expect(result.flatness).toBe(0);
    expect(result.centroid).toBe(0);
  });

  it("returns zero metrics for insufficient frame samples (<1024)", () => {
    const result = analyzeAcousticRealism(new Float32Array(512), 16000);
    expect(result.flatness).toBe(0);
    expect(result.centroid).toBe(0);
  });

  it("handles non-multiple frame sample lengths (e.g. 4095 samples) cleanly", () => {
    const rng = createSeededRandom(42);
    const samples = new Float32Array(4095);
    for (let i = 0; i < samples.length; i++) {
      samples[i] = (rng() - 0.5) * 2;
    }
    const result = analyzeAcousticRealism(samples, 16000);
    expect(Number.isFinite(result.flatness)).toBe(true);
    expect(Number.isFinite(result.centroid)).toBe(true);
    expect(result.flatness).toBeGreaterThan(0);
    expect(result.centroid).toBeGreaterThan(0);
  });

  it("returns low spectral flatness for a pure sine wave (synthetic single tone)", () => {
    const sampleRate = 16000;
    const length = 4096;
    const samples = new Float32Array(length);
    const freq = 440; // 440 Hz sine wave
    for (let i = 0; i < length; i++) {
      samples[i] = Math.sin((2 * Math.PI * freq * i) / sampleRate);
    }

    const result = analyzeAcousticRealism(samples, sampleRate);
    // Pure sine wave spectral energy is concentrated in 1 bin -> Wiener entropy is very low (< 0.015)
    expect(result.flatness).toBeLessThan(0.015);
    expect(result.centroid).toBeGreaterThan(400);
    expect(result.centroid).toBeLessThan(1500);
  });

  it("returns high spectral flatness for deterministic pseudo-random white noise", () => {
    const rng = createSeededRandom(99);
    const sampleRate = 16000;
    const length = 4096;
    const samples = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      samples[i] = (rng() - 0.5) * 2;
    }

    const result = analyzeAcousticRealism(samples, sampleRate);
    // White noise has flat energy spectrum across all bins -> Wiener entropy is high (> 0.50)
    expect(result.flatness).toBeGreaterThan(0.50);
    expect(result.centroid).toBeGreaterThan(3000);
    expect(result.centroid).toBeLessThan(5000);
  });

  it("returns valid physical acoustic bounds for simulated speech with deterministic room background noise", () => {
    const rng = createSeededRandom(777);
    const sampleRate = 16000;
    const length = 4096;
    const samples = new Float32Array(length);
    // Speech harmonic stack + deterministic room acoustic noise
    for (let i = 0; i < length; i++) {
      const f0 = 180;
      const t = i / sampleRate;
      samples[i] =
        0.4 * Math.sin(2 * Math.PI * f0 * t) +
        0.2 * Math.sin(2 * Math.PI * f0 * 2 * t) +
        0.1 * Math.sin(2 * Math.PI * f0 * 3 * t) +
        0.3 * (rng() - 0.5);
    }

    const result = analyzeAcousticRealism(samples, sampleRate);
    // Physical speech audio has moderate Wiener entropy (0.015 <= flatness <= 0.85)
    expect(result.flatness).toBeGreaterThanOrEqual(0.015);
    expect(result.flatness).toBeLessThanOrEqual(0.85);
    // Physical speech centroid sits between 100 Hz and 6000 Hz
    expect(result.centroid).toBeGreaterThanOrEqual(100);
    expect(result.centroid).toBeLessThanOrEqual(6000);
  });
});
