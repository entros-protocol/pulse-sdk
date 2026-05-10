import { describe, it, expect } from "vitest";
import {
  extractVoiceQualityFeatures,
  VOICE_QUALITY_FEATURE_COUNT,
} from "../src/extraction/voice-quality";

const SAMPLE_RATE = 16000;
const FRAME_SIZE = 2048;
const HOP_SIZE = 160;
// 2 seconds is enough to produce ~190 analysis frames at hop 160 — plenty
// to verify feature correctness, while keeping unit-test runtime under 5s
// for the whole suite (the production verification path uses 12-second
// sessions; that's a separate integration-test concern).
const SESSION_LENGTH = SAMPLE_RATE * 2;

function silentSamples(length: number): Float32Array {
  return new Float32Array(length);
}

function sineSamples(
  length: number,
  freqHz: number,
  amplitude = 0.3,
): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * freqHz * i) / SAMPLE_RATE);
  }
  return out;
}

function multiToneSamples(
  length: number,
  freqs: number[],
  amplitude = 0.3,
): Float32Array {
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

// Compute number of frames the extractor will iterate over for given input.
function numFramesOf(length: number): number {
  return Math.max(0, Math.floor((length - FRAME_SIZE) / HOP_SIZE) + 1);
}

describe("extractVoiceQualityFeatures", () => {
  it("returns the documented feature count", async () => {
    const samples = sineSamples(SESSION_LENGTH, 220);
    const f0PerFrame = new Array(numFramesOf(samples.length)).fill(220);
    const features = await extractVoiceQualityFeatures(
      samples,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0PerFrame,
    );
    expect(features).toHaveLength(VOICE_QUALITY_FEATURE_COUNT);
    expect(VOICE_QUALITY_FEATURE_COUNT).toBe(9);
  });

  it("returns all-finite values on valid input", async () => {
    const samples = sineSamples(SESSION_LENGTH, 220);
    const f0PerFrame = new Array(numFramesOf(samples.length)).fill(220);
    const features = await extractVoiceQualityFeatures(
      samples,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0PerFrame,
    );
    for (const v of features) expect(Number.isFinite(v)).toBe(true);
  });

  it("returns deterministic output for identical input", async () => {
    const samples = multiToneSamples(SESSION_LENGTH, [220, 440, 660]);
    const f0 = new Array(numFramesOf(samples.length)).fill(220);
    const a = await extractVoiceQualityFeatures(
      samples,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0,
    );
    const b = await extractVoiceQualityFeatures(
      samples,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0,
    );
    expect(a).toEqual(b);
  });

  it("returns zero vector on empty input", async () => {
    const features = await extractVoiceQualityFeatures(
      new Float32Array(0),
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      [],
    );
    expect(features).toHaveLength(VOICE_QUALITY_FEATURE_COUNT);
    expect(features.every((v) => v === 0)).toBe(true);
  });

  it("returns zero vector on invalid sample rate", async () => {
    const samples = sineSamples(SESSION_LENGTH, 220);
    const f0 = new Array(numFramesOf(samples.length)).fill(220);
    const features = await extractVoiceQualityFeatures(
      samples,
      0,
      FRAME_SIZE,
      HOP_SIZE,
      f0,
    );
    expect(features.every((v) => v === 0)).toBe(true);
  });

  it("subband ratios sum to ≤ 1 (energy can extend above 8 kHz)", async () => {
    const samples = multiToneSamples(SESSION_LENGTH, [200, 1500, 4000, 7000]);
    const f0 = new Array(numFramesOf(samples.length)).fill(200);
    const features = await extractVoiceQualityFeatures(
      samples,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0,
    );
    const [low, mid, high] = features.slice(6, 9);
    expect(low).toBeGreaterThanOrEqual(0);
    expect(mid).toBeGreaterThanOrEqual(0);
    expect(high).toBeGreaterThanOrEqual(0);
    expect(low! + mid! + high!).toBeLessThanOrEqual(1.01);
  });

  it("low-frequency tone produces low-band-dominant ratios", async () => {
    // Energy concentrated at 300 Hz (well within low band <1kHz).
    const samples = sineSamples(SESSION_LENGTH, 300);
    const f0 = new Array(numFramesOf(samples.length)).fill(300);
    const features = await extractVoiceQualityFeatures(
      samples,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0,
    );
    const [low, mid, high] = features.slice(6, 9);
    expect(low!).toBeGreaterThan(mid!);
    expect(low!).toBeGreaterThan(high!);
  });

  it("high-frequency tone produces high-band-dominant ratios", async () => {
    // Energy concentrated at 5500 Hz (well within high band 3-8kHz).
    const samples = sineSamples(SESSION_LENGTH, 5500);
    const f0 = new Array(numFramesOf(samples.length)).fill(5500);
    const features = await extractVoiceQualityFeatures(
      samples,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0,
    );
    const [low, mid, high] = features.slice(6, 9);
    expect(high!).toBeGreaterThan(low!);
    expect(high!).toBeGreaterThan(mid!);
  });

  it("produces different output for different audio content", async () => {
    const a = await extractVoiceQualityFeatures(
      sineSamples(SESSION_LENGTH, 200),
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      new Array(numFramesOf(SESSION_LENGTH)).fill(200),
    );
    const b = await extractVoiceQualityFeatures(
      sineSamples(SESSION_LENGTH, 800),
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      new Array(numFramesOf(SESSION_LENGTH)).fill(800),
    );
    let differs = false;
    for (let i = 0; i < a.length; i++) {
      if (Math.abs(a[i]! - b[i]!) > 1e-6) {
        differs = true;
        break;
      }
    }
    expect(differs).toBe(true);
  });

  it("silent input produces zero subband ratios and finite scalar metrics", async () => {
    const samples = silentSamples(SESSION_LENGTH);
    const f0 = new Array(numFramesOf(samples.length)).fill(0);
    const features = await extractVoiceQualityFeatures(
      samples,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0,
    );
    expect(features).toHaveLength(VOICE_QUALITY_FEATURE_COUNT);
    for (const v of features) expect(Number.isFinite(v)).toBe(true);
    // Subband ratios specifically: total energy is zero on silent input,
    // so each ratio should be 0.
    expect(features[6]).toBe(0);
    expect(features[7]).toBe(0);
    expect(features[8]).toBe(0);
  });

  it("H1-H2 is skipped on unvoiced frames (f0 ≤ 0)", async () => {
    // Provide all-unvoiced f0 → h1h2 array stays empty → mean/var stay 0.
    const samples = sineSamples(SESSION_LENGTH, 220);
    const f0AllUnvoiced = new Array(numFramesOf(samples.length)).fill(0);
    const features = await extractVoiceQualityFeatures(
      samples,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0AllUnvoiced,
    );
    // h1h2Mean (index 4) and h1h2Var (index 5) should both be 0 since
    // mean()/variance() of empty arrays return 0.
    expect(features[4]).toBe(0);
    expect(features[5]).toBe(0);
  });

  it("CPP detects periodic structure: harmonic stack produces higher CPP than white noise", async () => {
    // CPP measures cepstral peak prominence in the F0 quefrency range. A
    // synthetic harmonic stack (fundamental + harmonics) has strong
    // periodic structure → high CPP. White noise has no periodic
    // structure → low / near-zero CPP. This is the discriminative behavior
    // CPP exists for; the assertion locks the math against future
    // regressions in the cepstrum + baseline-regression implementation.
    const f0Hz = 200;
    const harmonicStack = multiToneSamples(
      SESSION_LENGTH,
      [f0Hz, 2 * f0Hz, 3 * f0Hz, 4 * f0Hz, 5 * f0Hz],
    );
    const f0Frames = new Array(numFramesOf(SESSION_LENGTH)).fill(f0Hz);
    const harmonicFeatures = await extractVoiceQualityFeatures(
      harmonicStack,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0Frames,
    );

    const noiseSampleArr = noiseSamples(SESSION_LENGTH, 99, 0.1);
    const noiseFeatures = await extractVoiceQualityFeatures(
      noiseSampleArr,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0Frames,
    );

    // cppMean is at index 0 of the feature vector.
    const harmonicCpp = harmonicFeatures[0]!;
    const noiseCpp = noiseFeatures[0]!;
    expect(harmonicCpp).toBeGreaterThan(noiseCpp);
  });
});

function noiseSamples(length: number, seed: number, amplitude: number): Float32Array {
  let s = seed >>> 0;
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    out[i] = amplitude * (((s & 0xffff) / 0xffff) * 2 - 1);
  }
  return out;
}
