import { describe, it, expect } from "vitest";
import {
  extractFormantRatios,
  extractLpcAnalysis,
  type LpcAnalysis,
} from "../src/extraction/lpc";

const SAMPLE_RATE = 16000;
const FRAME_SIZE = 2048;
const HOP_SIZE = 160;
const SESSION_LENGTH = SAMPLE_RATE * 12;

function sineSamples(length: number, freqHz: number, amplitude = 0.3): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * freqHz * i) / SAMPLE_RATE);
  }
  return out;
}

function multiToneSamples(length: number, freqs: number[], amplitude = 0.3): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    let sum = 0;
    for (const f of freqs) {
      sum += Math.sin((2 * Math.PI * f * i) / SAMPLE_RATE);
    }
    out[i] = (amplitude / freqs.length) * sum;
  }
  return out;
}

describe("extractLpcAnalysis", () => {
  it("returns the documented shape with LPC coefficient time series", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    expect(analysis.lpcCoefficients).toHaveLength(12); // default lpcOrder
    expect(analysis.numFramesAnalyzed).toBeGreaterThan(0);
    // Each per-coefficient track has one entry per analyzed frame.
    for (const coefTrack of analysis.lpcCoefficients) {
      expect(coefTrack.length).toBeLessThanOrEqual(analysis.numFramesAnalyzed);
    }
  });

  it("preserves backward compat with extractFormantRatios", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const legacy = extractFormantRatios(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    const fresh = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    // Same window, same frame iteration, same formant filter — should produce
    // identical ratio time series.
    expect(fresh.f1f2).toEqual(legacy.f1f2);
    expect(fresh.f2f3).toEqual(legacy.f2f3);
  });

  it("aligns formant absolute and bandwidth time series by index", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    expect(analysis.f1).toHaveLength(analysis.f2.length);
    expect(analysis.f2).toHaveLength(analysis.f3.length);
    expect(analysis.b1).toHaveLength(analysis.b2.length);
    expect(analysis.b2).toHaveLength(analysis.b3.length);
    expect(analysis.f1).toHaveLength(analysis.b1.length);
  });

  it("formants are sorted F1 < F2 < F3 within each frame", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    for (let t = 0; t < analysis.f1.length; t++) {
      expect(analysis.f1[t]!).toBeLessThan(analysis.f2[t]!);
      expect(analysis.f2[t]!).toBeLessThan(analysis.f3[t]!);
    }
  });

  it("bandwidths are positive and bounded", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    for (const arr of [analysis.b1, analysis.b2, analysis.b3]) {
      for (const bw of arr) {
        expect(bw).toBeGreaterThanOrEqual(0);
        expect(bw).toBeLessThan(500); // matches the formant filter
      }
    }
  });

  it("LPC coefficients are finite for non-degenerate input", () => {
    const samples = sineSamples(SESSION_LENGTH, 220);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    for (const coefTrack of analysis.lpcCoefficients) {
      for (const v of coefTrack) {
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  it("returns empty arrays on too-few-frames input", () => {
    const samples = sineSamples(1000, 220);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    expect(analysis.numFramesAnalyzed).toBe(0);
    expect(analysis.f1).toHaveLength(0);
    for (const coefTrack of analysis.lpcCoefficients) {
      expect(coefTrack).toHaveLength(0);
    }
  });

  it("produces deterministic output", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const a = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    const b = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    expect(a).toEqual(b);
  });

  it("type export is consumable", () => {
    const dummy: LpcAnalysis = {
      lpcCoefficients: [[]],
      f1: [],
      f2: [],
      f3: [],
      b1: [],
      b2: [],
      b3: [],
      f1f2: [],
      f2f3: [],
      numFramesAnalyzed: 0,
    };
    expect(dummy.numFramesAnalyzed).toBe(0);
  });
});
