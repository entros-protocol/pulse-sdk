import { describe, it, expect } from "vitest";
import { realFFT, bandEnergy, peakInBand, nextPow2 } from "../src/extraction/fft";

const SAMPLE_RATE = 64; // Hz — power-of-two for clean bin alignment in tests.
const FFT_SIZE = 256;

function sineSignal(
  length: number,
  freqHz: number,
  sampleRate: number,
  amplitude = 1,
): number[] {
  const out = new Array<number>(length);
  for (let i = 0; i < length; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * freqHz * i) / sampleRate);
  }
  return out;
}

describe("nextPow2", () => {
  it("returns 2 for inputs ≤ 2", () => {
    expect(nextPow2(0)).toBe(2);
    expect(nextPow2(1)).toBe(2);
    expect(nextPow2(2)).toBe(2);
  });

  it("returns the same value when input is already a power of two", () => {
    expect(nextPow2(16)).toBe(16);
    expect(nextPow2(1024)).toBe(1024);
  });

  it("rounds up to the next power of two", () => {
    expect(nextPow2(3)).toBe(4);
    expect(nextPow2(700)).toBe(1024);
    expect(nextPow2(1025)).toBe(2048);
  });
});

describe("realFFT", () => {
  it("rejects non-power-of-two sizes", () => {
    expect(() => realFFT([1, 2, 3], 6)).toThrow();
    expect(() => realFFT([1, 2, 3], 0)).toThrow();
  });

  it("returns DC bin equal to the signal sum for constant input", () => {
    const N = 8;
    const input = new Array<number>(N).fill(0.5);
    const { real, imag } = realFFT(input, N);
    expect(real[0]).toBeCloseTo(N * 0.5, 6);
    // All other bins should be zero (within float tolerance).
    for (let k = 1; k < N; k++) {
      expect(Math.abs(real[k]!)).toBeLessThan(1e-9);
      expect(Math.abs(imag[k]!)).toBeLessThan(1e-9);
    }
  });

  it("places sine energy at the expected bin", () => {
    // 4 cycles in 256 samples at 64 Hz sampleRate → frequency = 1 Hz.
    // Bin k for f Hz at N samples / sampleRate = f * N / sampleRate.
    // freq = 8 Hz, N = 256, sampleRate = 64 → expected bin = 32.
    const freq = 8;
    const input = sineSignal(FFT_SIZE, freq, SAMPLE_RATE, 1);
    const { real, imag } = realFFT(input, FFT_SIZE);

    const expectedBin = (freq * FFT_SIZE) / SAMPLE_RATE;
    const power = real.map((re, k) => re * re + imag[k]! * imag[k]!);
    let bestBin = 0;
    let bestPower = -Infinity;
    for (let k = 0; k < FFT_SIZE / 2; k++) {
      if (power[k]! > bestPower) {
        bestPower = power[k]!;
        bestBin = k;
      }
    }
    expect(bestBin).toBe(expectedBin);
  });

  it("zero-pads input shorter than size", () => {
    const input = [1, 2, 3];
    const { real, imag } = realFFT(input, 4);
    // DC bin = sum = 1 + 2 + 3 + 0 = 6
    expect(real[0]).toBeCloseTo(6, 6);
    expect(imag[0]).toBeCloseTo(0, 6);
  });

  it("truncates input longer than size", () => {
    const input = [1, 2, 3, 4, 5, 6];
    const { real } = realFFT(input, 4);
    // DC bin = sum of first 4 = 1 + 2 + 3 + 4 = 10
    expect(real[0]).toBeCloseTo(10, 6);
  });
});

describe("bandEnergy", () => {
  it("returns zero on empty input", () => {
    expect(bandEnergy([], [], 100, 0, 10)).toBe(0);
  });

  it("returns zero for invalid sample rate", () => {
    const { real, imag } = realFFT([1, 0, -1, 0], 4);
    expect(bandEnergy(real, imag, 0, 0, 10)).toBe(0);
    expect(bandEnergy(real, imag, -100, 0, 10)).toBe(0);
  });

  it("returns zero for an inverted band", () => {
    const { real, imag } = realFFT([1, 0, -1, 0], 4);
    expect(bandEnergy(real, imag, 100, 10, 5)).toBe(0);
  });

  it("captures all sine energy when band brackets the frequency", () => {
    const freq = 8;
    const input = sineSignal(FFT_SIZE, freq, SAMPLE_RATE, 1);
    const { real, imag } = realFFT(input, FFT_SIZE);
    const inBand = bandEnergy(real, imag, SAMPLE_RATE, 7, 9);
    const fullSpectrum = bandEnergy(real, imag, SAMPLE_RATE, 0, SAMPLE_RATE);
    // The single-frequency sine should put effectively all energy in the
    // 7-9 Hz band; allow 1% slack for windowing leakage.
    expect(inBand).toBeGreaterThan(0.99 * fullSpectrum);
  });

  it("returns zero when band excludes the signal frequency", () => {
    const freq = 8;
    const input = sineSignal(FFT_SIZE, freq, SAMPLE_RATE, 1);
    const { real, imag } = realFFT(input, FFT_SIZE);
    const offBand = bandEnergy(real, imag, SAMPLE_RATE, 20, 30);
    const inBand = bandEnergy(real, imag, SAMPLE_RATE, 7, 9);
    expect(offBand).toBeLessThan(0.01 * inBand);
  });
});

describe("realFFT determinism", () => {
  it("returns byte-identical output across repeated calls on identical input", () => {
    const input = sineSignal(FFT_SIZE, 6, SAMPLE_RATE, 1);
    const a = realFFT(input, FFT_SIZE);
    const b = realFFT(input, FFT_SIZE);
    expect(a.real).toEqual(b.real);
    expect(a.imag).toEqual(b.imag);
  });
});

describe("peakInBand", () => {
  it("returns zero amplitude on empty input", () => {
    const result = peakInBand([], [], 100, 0, 10);
    expect(result).toEqual({ freq: 0, amplitude: 0 });
  });

  it("locates the dominant frequency within the requested band", () => {
    // Two superposed sines: 5 Hz (in band) + 20 Hz (out of band).
    const length = FFT_SIZE;
    const input = new Array<number>(length);
    for (let i = 0; i < length; i++) {
      const a = Math.sin((2 * Math.PI * 5 * i) / SAMPLE_RATE);
      const b = 0.5 * Math.sin((2 * Math.PI * 20 * i) / SAMPLE_RATE);
      input[i] = a + b;
    }
    const { real, imag } = realFFT(input, FFT_SIZE);
    const { freq } = peakInBand(real, imag, SAMPLE_RATE, 4, 12);
    expect(freq).toBeCloseTo(5, 1);
  });

  it("returns zero when no bin falls inside the band", () => {
    const input = sineSignal(8, 1, 4, 1);
    const { real, imag } = realFFT(input, 8);
    // Bin spacing = sampleRate / N = 4 / 8 = 0.5 Hz; band [10, 11] is past Nyquist.
    const result = peakInBand(real, imag, 4, 10, 11);
    expect(result).toEqual({ freq: 0, amplitude: 0 });
  });
});
