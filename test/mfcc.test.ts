import { describe, it, expect } from "vitest";
import { extractMfccFeatures, MFCC_FEATURE_COUNT } from "../src/extraction/mfcc";

// --- Helpers ---

function silentSamples(length: number): Float32Array {
  return new Float32Array(length);
}

function sineSamples(
  length: number,
  freqHz: number,
  sampleRate: number,
  amplitude = 0.3,
): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * freqHz * i) / sampleRate);
  }
  return out;
}

function noiseSamples(length: number, seed = 1, amplitude = 0.2): Float32Array {
  // Deterministic LCG so test results are reproducible across runs.
  let s = seed >>> 0;
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = amplitude * (((s & 0xffff) / 0xffff) * 2 - 1);
  }
  return out;
}

const SAMPLE_RATE = 16000;
// 12 seconds of audio at 16kHz, the actual session length used in production.
const SESSION_LENGTH = SAMPLE_RATE * 12;
// Frame size derived to match speaker.ts::getFrameSize logic for 16kHz.
const FRAME_SIZE = 2048;
const HOP_SIZE = 160; // 10ms at 16kHz, matches speaker.ts::getHopSize

// --- Tests ---

describe("extractMfccFeatures", () => {
  it("returns the documented feature count", async () => {
    const samples = sineSamples(SESSION_LENGTH, 220, SAMPLE_RATE);
    const features = await extractMfccFeatures(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    expect(features).toHaveLength(MFCC_FEATURE_COUNT);
    expect(MFCC_FEATURE_COUNT).toBe(78); // 13×4 + 13×2
  });

  it("produces all-finite values on real input", async () => {
    const samples = sineSamples(SESSION_LENGTH, 220, SAMPLE_RATE);
    const features = await extractMfccFeatures(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    for (const v of features) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("produces deterministic output for identical input", async () => {
    const samples = sineSamples(SESSION_LENGTH, 440, SAMPLE_RATE);
    const a = await extractMfccFeatures(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    const b = await extractMfccFeatures(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    expect(a).toEqual(b);
  });

  it("produces zero vector on empty input", async () => {
    const features = await extractMfccFeatures(
      new Float32Array(0),
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
    );
    expect(features).toHaveLength(MFCC_FEATURE_COUNT);
    expect(features.every((v) => v === 0)).toBe(true);
  });

  it("produces zero vector on invalid sample rate", async () => {
    const samples = sineSamples(SESSION_LENGTH, 220, SAMPLE_RATE);
    const features = await extractMfccFeatures(samples, 0, FRAME_SIZE, HOP_SIZE);
    expect(features.every((v) => v === 0)).toBe(true);
  });

  it("produces zero vector on too-few-frames input", async () => {
    // 5 frames at hop 160 = 800 samples + frameSize 2048 = 2848 minimum.
    // Provide less to trigger the short-input early return.
    const samples = sineSamples(2000, 220, SAMPLE_RATE);
    const features = await extractMfccFeatures(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    expect(features.every((v) => v === 0)).toBe(true);
  });

  it("delta-MFCC mean is small for stationary input", async () => {
    // A constant-frequency sine wave has stationary MFCCs across frames,
    // so delta-MFCCs should average near zero.
    const samples = sineSamples(SESSION_LENGTH, 440, SAMPLE_RATE);
    const features = await extractMfccFeatures(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    // Layout: indices 0..51 are MFCC moments, 52..77 are delta moments
    // (alternating mean, var per coefficient). Pull the 13 delta means.
    const deltaMeans: number[] = [];
    for (let c = 0; c < 13; c++) {
      deltaMeans.push(features[52 + c * 2]!);
    }
    const meanOfMeans = deltaMeans.reduce((a, b) => a + b, 0) / deltaMeans.length;
    // For a perfectly stationary signal we'd expect 0; allow generous slack
    // since Meyda's MFCC has mild frame-to-frame numerical drift.
    expect(Math.abs(meanOfMeans)).toBeLessThan(0.1);
  });

  it("produces different output for different audio content", async () => {
    const a = await extractMfccFeatures(
      sineSamples(SESSION_LENGTH, 220, SAMPLE_RATE),
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
    );
    const b = await extractMfccFeatures(
      sineSamples(SESSION_LENGTH, 880, SAMPLE_RATE),
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
    );
    // At least one moment must differ measurably.
    let differs = false;
    for (let i = 0; i < a.length; i++) {
      if (Math.abs(a[i]! - b[i]!) > 1e-6) {
        differs = true;
        break;
      }
    }
    expect(differs).toBe(true);
  });

  it("noise input produces finite, non-degenerate features", async () => {
    const samples = noiseSamples(SESSION_LENGTH, 42);
    const features = await extractMfccFeatures(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    expect(features.every(Number.isFinite)).toBe(true);
    // Variance terms should be positive (not all zero) for non-trivial input.
    let positiveVarCount = 0;
    for (let c = 0; c < 13; c++) {
      if (features[c * 4 + 1]! > 0) positiveVarCount++;
    }
    expect(positiveVarCount).toBeGreaterThan(0);
  });
});
