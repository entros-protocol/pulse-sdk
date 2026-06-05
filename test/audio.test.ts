import { describe, it, expect } from "vitest";
import { captureAudio, normalizeCaptureRMS } from "../src/sensor/audio";

// Helper: compute RMS over a Float32Array. Used by the assertions to
// confirm the helper achieves the documented target without re-deriving
// it inside every test.
function rms(samples: Float32Array): number {
  if (samples.length === 0) return 0;
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) sumSq += samples[i]! * samples[i]!;
  return Math.sqrt(sumSq / samples.length);
}

describe("normalizeCaptureRMS", () => {
  it("returns empty buffer unchanged", () => {
    const out = normalizeCaptureRMS(new Float32Array(0));
    expect(out).toHaveLength(0);
  });

  it("returns pure-silence buffer unchanged (no noise-floor amplification)", () => {
    const samples = new Float32Array(1024);
    const out = normalizeCaptureRMS(samples);
    expect(out).toBe(samples); // same reference, no allocation
    expect(rms(out)).toBe(0);
  });

  it("amplifies a quiet capture (RMS 0.005) up to the 0.05 target", () => {
    const samples = new Float32Array(1024);
    samples.fill(0.005); // constant DC at 0.005 → RMS = 0.005
    const out = normalizeCaptureRMS(samples);
    expect(rms(out)).toBeCloseTo(0.05, 4);
  });

  it("attenuates a loud capture (RMS 0.5) down to the 0.05 target", () => {
    const samples = new Float32Array(1024);
    samples.fill(0.5);
    const out = normalizeCaptureRMS(samples);
    expect(rms(out)).toBeCloseTo(0.05, 4);
  });

  it("caps gain at 50× for a near-silent capture (RMS 0.0005 stays under target)", () => {
    // RMS 0.0005 → ideal gain 100×. Cap at 50× → output RMS ~0.025, not 0.05.
    const samples = new Float32Array(1024);
    samples.fill(5e-4);
    const out = normalizeCaptureRMS(samples);
    expect(rms(out)).toBeCloseTo(0.025, 4);
    expect(rms(out)).toBeLessThan(0.05); // explicitly under target
  });

  it("clamps transients to [-1, 1] and keeps surrounding samples finite + amplified", () => {
    // Mostly DC at 0.005 with one transient at 0.5. The transient pushes
    // overall RMS up enough that the surrounding amplification factor is
    // smaller than the no-transient case (RMS includes the transient by
    // definition), but the clamp still has to absorb whatever gain × 0.5
    // produces. The test pins the clamp invariant; it doesn't pin the
    // surrounding gain (which is a function of the transient's RMS
    // contribution and isn't load-bearing for the cross-person fix).
    const samples = new Float32Array(1024);
    samples.fill(0.005);
    samples[100] = 0.5;
    const out = normalizeCaptureRMS(samples);
    // Clamp invariant: every sample bounded by [-1, 1].
    for (const v of out) {
      expect(v).toBeLessThanOrEqual(1.0);
      expect(v).toBeGreaterThanOrEqual(-1.0);
      expect(Number.isFinite(v)).toBe(true);
    }
    // Surrounding samples got SOME amplification (positive, larger than
    // input), even if the transient skewed RMS down vs the no-transient
    // case.
    expect(out[0]).toBeGreaterThan(samples[0]!);
  });

  it("never returns NaN or Infinity even on adversarial inputs", () => {
    const adversarial = new Float32Array([0, 0, 1e-10, 0, -1e-10, 0]);
    const out = normalizeCaptureRMS(adversarial);
    for (const v of out) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

// --- Capture-ready gate (first-attempt cold-start fix) ---
//
// Minimal Web Audio mock so captureAudio's onReady contract can be pinned
// without a real microphone / AudioContext. Only the surface captureAudio
// actually touches is implemented.

class MockScriptProcessor {
  onaudioprocess:
    | ((e: { inputBuffer: { getChannelData: (ch: number) => Float32Array } }) => void)
    | null = null;
  connect() {}
  disconnect() {}
}

class MockAudioContext {
  static lastProcessor: MockScriptProcessor | null = null;
  readonly sampleRate = 16000;
  readonly destination = {};
  constructor(_options?: unknown) {}
  async resume() {}
  createMediaStreamSource() {
    return { connect() {}, disconnect() {} };
  }
  createScriptProcessor() {
    const p = new MockScriptProcessor();
    MockAudioContext.lastProcessor = p;
    return p;
  }
  close() {
    return Promise.resolve();
  }
}

function fireFrame(p: MockScriptProcessor, value: number) {
  const buf = new Float32Array(4096).fill(value);
  p.onaudioprocess?.({ inputBuffer: { getChannelData: () => buf } });
}

describe("captureAudio onReady gate", () => {
  it("fires onReady exactly once, on the first delivered audio frame", async () => {
    const g = globalThis as { AudioContext?: unknown };
    const original = g.AudioContext;
    g.AudioContext = MockAudioContext as unknown as typeof AudioContext;
    try {
      const stream = { getTracks: () => [{ stop() {} }] } as unknown as MediaStream;
      const controller = new AbortController();
      let readyCount = 0;
      const capture = captureAudio({
        stream,
        onReady: () => {
          readyCount += 1;
        },
        signal: controller.signal,
        minDurationMs: 0,
        maxDurationMs: 1000,
      });

      // Let the async setup (new AudioContext + await resume + connect) run.
      await new Promise((r) => setTimeout(r, 0));
      const proc = MockAudioContext.lastProcessor;
      expect(proc).toBeTruthy();

      // The whole point: not "ready" until a real frame is delivered, so the
      // speak prompt can't start before audio is flowing.
      expect(readyCount).toBe(0);

      fireFrame(proc!, 0.1);
      expect(readyCount).toBe(1);

      // Subsequent frames must not re-fire it.
      fireFrame(proc!, 0.1);
      fireFrame(proc!, 0.1);
      expect(readyCount).toBe(1);

      controller.abort();
      const result = await capture;
      expect(result.samples.length).toBeGreaterThan(0);
    } finally {
      g.AudioContext = original;
    }
  });
});
