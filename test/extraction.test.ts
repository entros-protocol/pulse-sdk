import { describe, it, expect } from "vitest";
import {
  extractSpeakerFeatures,
  SPEAKER_FEATURE_COUNT,
  extractMotionFeatures,
  extractTouchFeatures,
  extractMouseDynamics,
  fuseFeatures,
} from "../src/index";
import type { AudioCapture, MotionSample, TouchSample } from "../src/index";

// --- Helpers ---

function makeAudio(opts: {
  samples?: Float32Array;
  sampleRate?: number;
  duration?: number;
  length?: number;
}): AudioCapture {
  const length = opts.length ?? 32000;
  const sampleRate = opts.sampleRate ?? 16000;
  const samples = opts.samples ?? randomAudio(length);
  return { samples, sampleRate, duration: samples.length / sampleRate };
}

function randomAudio(length: number, amplitude = 0.3): Float32Array {
  const samples = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    // Simulate speech: sine wave + noise
    samples[i] = amplitude * Math.sin(2 * Math.PI * 200 * i / 16000) +
      (Math.random() - 0.5) * amplitude * 0.3;
  }
  return samples;
}

function silentAudio(length: number): Float32Array {
  return new Float32Array(length); // all zeros
}

function makeMotionSamples(count: number): MotionSample[] {
  return Array.from({ length: count }, (_, i) => ({
    timestamp: i * 16.67, // ~60Hz
    ax: Math.random() * 0.5,
    ay: Math.random() * 0.5,
    az: -9.8 + Math.random() * 0.1,
    gx: Math.random() * 0.01,
    gy: Math.random() * 0.01,
    gz: Math.random() * 0.01,
  }));
}

function makeTouchSamples(count: number): TouchSample[] {
  return Array.from({ length: count }, (_, i) => ({
    timestamp: i * 16.67,
    x: 200 + Math.cos(i * 0.05) * 100,
    y: 300 + Math.sin(i * 0.05) * 100,
    pressure: 0.5 + Math.random() * 0.3,
    width: 10 + Math.random() * 5,
    height: 10 + Math.random() * 5,
  }));
}

// --- Speaker Feature Tests ---

describe("speaker feature extraction", () => {
  it("produces 44 features from normal speech audio", async () => {
    const audio = makeAudio({ length: 32000, sampleRate: 16000 });
    const features = await extractSpeakerFeatures(audio);
    expect(features).toHaveLength(SPEAKER_FEATURE_COUNT);
    expect(features).toHaveLength(44);
  });

  it("produces no NaN values from normal audio", async () => {
    const audio = makeAudio({ length: 32000, sampleRate: 16000 });
    const features = await extractSpeakerFeatures(audio);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i]), `feature[${i}] is ${features[i]}`).toBe(true);
    }
  });

  it("produces no NaN values from silent audio", async () => {
    const audio = makeAudio({ samples: silentAudio(32000), sampleRate: 16000 });
    const features = await extractSpeakerFeatures(audio);
    expect(features).toHaveLength(44);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i]), `feature[${i}] is ${features[i]}`).toBe(true);
    }
  });

  it("produces no NaN values at 48kHz (iOS native rate)", async () => {
    // iOS ignores requested 16kHz and runs AudioContext at 48kHz
    const audio = makeAudio({ length: 48000 * 2, sampleRate: 48000 });
    const features = await extractSpeakerFeatures(audio);
    expect(features).toHaveLength(44);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i]), `feature[${i}] is ${features[i]}`).toBe(true);
    }
  });

  it("produces no NaN values from silent 48kHz audio", async () => {
    const audio = makeAudio({ samples: silentAudio(96000), sampleRate: 48000 });
    const features = await extractSpeakerFeatures(audio);
    expect(features).toHaveLength(44);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i]), `feature[${i}] is ${features[i]}`).toBe(true);
    }
  });

  it("returns zeros for too-short audio", async () => {
    const audio = makeAudio({ length: 500, sampleRate: 16000 });
    const features = await extractSpeakerFeatures(audio);
    expect(features).toHaveLength(44);
    expect(features.every((v) => v === 0)).toBe(true);
  });

  it("produces different features for different audio", async () => {
    const audio1 = makeAudio({ samples: randomAudio(32000, 0.3), sampleRate: 16000 });
    const audio2 = makeAudio({ samples: randomAudio(32000, 0.6), sampleRate: 16000 });
    const f1 = await extractSpeakerFeatures(audio1);
    const f2 = await extractSpeakerFeatures(audio2);
    const identical = f1.every((v, i) => v === f2[i]);
    expect(identical).toBe(false);
  });
});

// --- Kinematic Feature Tests ---

describe("motion feature extraction", () => {
  it("produces 54 features from IMU data", () => {
    const samples = makeMotionSamples(100);
    const features = extractMotionFeatures(samples);
    expect(features).toHaveLength(54);
  });

  it("produces no NaN values", () => {
    const samples = makeMotionSamples(100);
    const features = extractMotionFeatures(samples);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i]), `feature[${i}] is ${features[i]}`).toBe(true);
    }
  });

  it("returns zeros for insufficient samples", () => {
    const features = extractMotionFeatures([]);
    expect(features).toHaveLength(54);
    expect(features.every((v) => v === 0)).toBe(true);
  });

  it("handles constant acceleration (stationary phone)", () => {
    // Phone sitting still: constant gravity on z-axis
    const samples: MotionSample[] = Array.from({ length: 100 }, (_, i) => ({
      timestamp: i * 16.67,
      ax: 0, ay: 0, az: -9.8, gx: 0, gy: 0, gz: 0,
    }));
    const features = extractMotionFeatures(samples);
    expect(features).toHaveLength(54);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i]), `feature[${i}] is ${features[i]}`).toBe(true);
    }
  });
});

describe("touch feature extraction", () => {
  it("produces 36 features from touch data", () => {
    const samples = makeTouchSamples(100);
    const features = extractTouchFeatures(samples);
    expect(features).toHaveLength(36);
  });

  it("produces no NaN values", () => {
    const samples = makeTouchSamples(100);
    const features = extractTouchFeatures(samples);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i]), `feature[${i}] is ${features[i]}`).toBe(true);
    }
  });

  it("returns zeros for insufficient samples", () => {
    const features = extractTouchFeatures([]);
    expect(features).toHaveLength(36);
    expect(features.every((v) => v === 0)).toBe(true);
  });
});

describe("mouse dynamics extraction", () => {
  it("produces 54 features from pointer data", () => {
    const samples = makeTouchSamples(100);
    const features = extractMouseDynamics(samples);
    expect(features).toHaveLength(54);
  });

  it("produces no NaN values", () => {
    const samples = makeTouchSamples(100);
    const features = extractMouseDynamics(samples);
    for (let i = 0; i < features.length; i++) {
      expect(Number.isFinite(features[i]), `feature[${i}] is ${features[i]}`).toBe(true);
    }
  });

  it("returns zeros for insufficient samples", () => {
    const features = extractMouseDynamics([]);
    expect(features).toHaveLength(54);
    expect(features.every((v) => v === 0)).toBe(true);
  });
});

// --- Fusion Tests ---

describe("feature fusion (normalizeGroup + fuseFeatures)", () => {
  it("produces 134 features from all three modalities", () => {
    const audio = new Array(44).fill(0).map(() => Math.random() * 100);
    const motion = new Array(54).fill(0).map(() => Math.random());
    const touch = new Array(36).fill(0).map(() => Math.random() * 10);
    const fused = fuseFeatures(audio, motion, touch);
    expect(fused).toHaveLength(134);
  });

  it("produces no NaN values even with NaN inputs", () => {
    const audio = [NaN, 1, 2, ...new Array(41).fill(0)];
    const motion = new Array(54).fill(0.5);
    const touch = new Array(36).fill(1);
    const fused = fuseFeatures(audio, motion, touch);
    for (let i = 0; i < fused.length; i++) {
      expect(Number.isFinite(fused[i]), `fused[${i}] is ${fused[i]}`).toBe(true);
    }
  });

  it("normalizes each group to zero mean", () => {
    const audio = [100, 200, 300, ...new Array(41).fill(150)];
    const motion = new Array(54).fill(0.5);
    const touch = new Array(36).fill(10);
    const fused = fuseFeatures(audio, motion, touch);

    // Audio group (indices 0-43) should have mean ≈ 0
    const audioSlice = fused.slice(0, 44);
    const audioMean = audioSlice.reduce((a, b) => a + b, 0) / audioSlice.length;
    expect(Math.abs(audioMean)).toBeLessThan(0.001);
  });

  it("handles all-zero modality", () => {
    const audio = new Array(44).fill(0);
    const motion = new Array(54).fill(0.5);
    const touch = new Array(36).fill(1);
    const fused = fuseFeatures(audio, motion, touch);
    expect(fused).toHaveLength(134);
    // All-zero group normalizes to all zeros
    expect(fused.slice(0, 44).every((v) => v === 0)).toBe(true);
  });
});
