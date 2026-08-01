import { describe, it, expect } from "vitest";
import { MAX_TRANSMITTED_CAPTURE_MS } from "../src/config";
import { captureAudio, describeInputLevel, normalizeCaptureRMS } from "../src/sensor/audio";

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
  readonly sampleRate: number = 16000;
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

  it("detects virtual device when track label matches loopback keyword", async () => {
    const g = globalThis as { AudioContext?: unknown };
    const original = g.AudioContext;
    g.AudioContext = MockAudioContext as unknown as typeof AudioContext;
    try {
      const stream = {
        getTracks: () => [{ stop() {} }],
        getAudioTracks: () => [{ label: "VB-Audio Cable" }],
      } as unknown as MediaStream;
      const controller = new AbortController();
      const capture = captureAudio({
        stream,
        signal: controller.signal,
        minDurationMs: 0,
        maxDurationMs: 1000,
      });

      await new Promise((r) => setTimeout(r, 0));
      const proc = MockAudioContext.lastProcessor;
      expect(proc).toBeTruthy();
      fireFrame(proc!, 0.1);

      controller.abort();
      const result = await capture;
      expect(result.virtualDevice).toBe(true);
    } finally {
      g.AudioContext = original;
    }
  });
});

/**
 * The branch the canonical-rate change exists for, and the one nothing
 * exercised: a browser that ignores the 16 kHz `AudioContext` request.
 *
 * `test/resample.test.ts` covers the transform in isolation, but nothing
 * previously drove `stopCapture` at a non-16 kHz rate, so the wiring itself
 * (resample before RMS normalization, the reported rate, the reported
 * duration) was untested.
 */
class MockAudioContext48k extends MockAudioContext {
  override readonly sampleRate = 48000;
}

describe("captureAudio canonicalizes the sample rate", () => {
  async function capture(ctx: unknown, frames: number) {
    const g = globalThis as { AudioContext?: unknown };
    const original = g.AudioContext;
    g.AudioContext = ctx as typeof AudioContext;
    try {
      const stream = {
        getTracks: () => [{ stop() {} }],
        getAudioTracks: () => [{ label: "Built-in Microphone" }],
      } as unknown as MediaStream;
      const controller = new AbortController();
      const pending = captureAudio({
        stream,
        signal: controller.signal,
        minDurationMs: 0,
        maxDurationMs: 5000,
      });
      await new Promise((r) => setTimeout(r, 0));
      const proc = MockAudioContext.lastProcessor!;
      for (let i = 0; i < frames; i++) fireFrame(proc, 0.1);
      controller.abort();
      return await pending;
    } finally {
      g.AudioContext = original;
    }
  }

  it("reports 16 kHz and a decimated buffer when the browser delivers 48 kHz", async () => {
    const frames = 60; // 60 x 4096 = 245,760 source samples, 5.12 s at 48 kHz
    const result = await capture(MockAudioContext48k, frames);

    expect(result.sampleRate).toBe(16000);
    expect(result.samples.length).toBe(Math.round((frames * 4096) / 3));
    // Duration is the length of the buffer handed over, which here equals
    // wall-clock because nothing was trimmed or capped.
    expect(result.duration).toBeCloseTo((frames * 4096) / 48000, 3);
  });

  it("keeps a 16 kHz capture at its original length while still filtering it", async () => {
    const frames = 60;
    const result = await capture(MockAudioContext, frames);

    expect(result.sampleRate).toBe(16000);
    expect(result.samples.length).toBe(frames * 4096);
    expect(result.duration).toBeCloseTo((frames * 4096) / 16000, 3);
  });
});

/**
 * The capture window mark. `startAudio` resolves as soon as audio is genuinely
 * flowing, so the speak prompt never appears during the microphone's cold
 * start, which means recording begins before the prompt does, with the
 * challenge fetch and the countdown inside that gap. The host marks when the
 * window actually opens and everything before it is discarded.
 */
describe("captureAudio capture-window mark", () => {
  async function captureWithMark(
    framesBefore: number,
    framesAfter: number,
    opts: { markTimes?: number; markAfterStop?: boolean } = {}
  ) {
    const g = globalThis as { AudioContext?: unknown };
    const original = g.AudioContext;
    g.AudioContext = MockAudioContext as unknown as typeof AudioContext;
    try {
      const stream = {
        getTracks: () => [{ stop() {} }],
        getAudioTracks: () => [{ label: "Built-in Microphone" }],
      } as unknown as MediaStream;
      const controller = new AbortController();
      const windowController = new AbortController();
      const mark = () => windowController.abort();
      const pending = captureAudio({
        stream,
        signal: controller.signal,
        minDurationMs: 0,
        maxDurationMs: 5000,
        captureWindowSignal: windowController.signal,
      });
      await new Promise((r) => setTimeout(r, 0));
      const proc = MockAudioContext.lastProcessor!;

      for (let i = 0; i < framesBefore; i++) fireFrame(proc, 0.02);
      if (!opts.markAfterStop) {
        for (let i = 0; i < (opts.markTimes ?? 1); i++) mark();
      }
      for (let i = 0; i < framesAfter; i++) fireFrame(proc, 0.2);

      controller.abort();
      const result = await pending;
      if (opts.markAfterStop) mark();
      return result;
    } finally {
      g.AudioContext = original;
    }
  }

  it("keeps only what was recorded after the window opened", async () => {
    const result = await captureWithMark(12, 30);
    expect(result.samples.length).toBe(30 * 4096);
  });

  /**
   * The returned buffer must own exactly its own bytes. A `subarray` view
   * would hand callers a `Float32Array` whose `.buffer` is larger than its
   * contents and keep the whole pre-trim recording alive behind it.
   */
  it("returns a buffer that owns exactly its own bytes", async () => {
    const result = await captureWithMark(12, 30);
    expect(result.samples.buffer.byteLength).toBe(result.samples.byteLength);
  });

  it("cannot be moved by a second signal, so a double-invoked effect is safe", async () => {
    const result = await captureWithMark(12, 30, { markTimes: 3 });
    expect(result.samples.length).toBe(30 * 4096);
  });

  it("keeps the whole recording when the host never marks", async () => {
    const g = globalThis as { AudioContext?: unknown };
    const original = g.AudioContext;
    g.AudioContext = MockAudioContext as unknown as typeof AudioContext;
    try {
      const stream = {
        getTracks: () => [{ stop() {} }],
        getAudioTracks: () => [{ label: "Built-in Microphone" }],
      } as unknown as MediaStream;
      const controller = new AbortController();
      const pending = captureAudio({
        stream,
        signal: controller.signal,
        minDurationMs: 0,
        maxDurationMs: 5000,
      });
      await new Promise((r) => setTimeout(r, 0));
      for (let i = 0; i < 20; i++) fireFrame(MockAudioContext.lastProcessor!, 0.1);
      controller.abort();
      const result = await pending;
      expect(result.samples.length).toBe(20 * 4096);
    } finally {
      g.AudioContext = original;
    }
  });

  it("ignores a mark that arrives after the capture has already closed", async () => {
    const result = await captureWithMark(12, 30, { markAfterStop: true });
    expect(result.samples.length).toBe(42 * 4096);
  });
});

/**
 * The transmitted-length cap. It mirrors the validator's own
 * `MAX_AUDIO_SAMPLES`, past which phrase binding is skipped and the
 * verification silently passes, so the cap exists to make that unreachable.
 *
 * Which end survives depends on where the phrase is. With a mark, index 0 is
 * the prompt and speech is at the front. Without one, every integrator who
 * has not adopted `markCaptureStart`, index 0 is recorder start and speech is
 * at the end, so keeping the head would delete it and leave the validator
 * transcribing silence.
 */
describe("captureAudio transmitted-length cap", () => {
  const CAP_SAMPLES = (MAX_TRANSMITTED_CAPTURE_MS / 1000) * 16000;
  const FRAMES_PAST_CAP = Math.ceil(CAP_SAMPLES / 4096) + 1;

  async function overrun(mark: boolean) {
    const g = globalThis as { AudioContext?: unknown };
    const original = g.AudioContext;
    g.AudioContext = MockAudioContext as unknown as typeof AudioContext;
    try {
      const stream = {
        getTracks: () => [{ stop() {} }],
        getAudioTracks: () => [{ label: "Built-in Microphone" }],
      } as unknown as MediaStream;
      const controller = new AbortController();
      const windowController = new AbortController();
      const pending = captureAudio({
        stream,
        signal: controller.signal,
        minDurationMs: 0,
        maxDurationMs: 120_000,
        captureWindowSignal: windowController.signal,
      });
      await new Promise((r) => setTimeout(r, 0));
      const proc = MockAudioContext.lastProcessor!;

      if (mark) windowController.abort();
      // Sign, not magnitude: `normalizeCaptureRMS` rescales every capture to
      // the same RMS, so only the sign survives to identify which end was kept.
      for (let i = 0; i < FRAMES_PAST_CAP; i++) fireFrame(proc, 0.5);
      for (let i = 0; i < 20; i++) fireFrame(proc, -0.5);

      controller.abort();
      return await pending;
    } finally {
      g.AudioContext = original;
    }
  }

  /**
   * Pinned against the server's number, not against our own constant, or the
   * assertion moves with whatever it is meant to be checking.
   *
   * `entros-validation::phrase_binding::MAX_AUDIO_SAMPLES` is 320_000, and the
   * comparison there is `projected > MAX`, so 320_000 exactly is the largest
   * capture that still gets phrase binding. Margin is zero by design: one
   * sample more and the validator skips the check and passes silently.
   */
  it("caps at exactly the validator's MAX_AUDIO_SAMPLES", async () => {
    expect(CAP_SAMPLES).toBe(320_000);
    const result = await overrun(true);
    expect(result.samples.length).toBe(320_000);
  });

  it("keeps the head when the host marked, because speech starts at the mark", async () => {
    const result = await overrun(true);
    expect(result.samples[1000]).toBeGreaterThan(0);
  });

  it("keeps the tail when the host did not mark, because speech ends the capture", async () => {
    const result = await overrun(false);
    expect(result.samples.length).toBe(CAP_SAMPLES);
    expect(result.samples[result.samples.length - 1000]!).toBeLessThan(0);
  });
});

/**
 * Where the transmitted buffer sits on the wall clock.
 *
 * Every other modality is aligned to this. `accel_magnitude` is resampled onto
 * it, then correlated against the F0 contour server-side, so if the window is
 * wrong the coupling check is measuring two different moments. That is exactly
 * what 4.0.0 shipped: the lead-in trim moved the audio and nothing told motion.
 *
 * Derived from `markedAtSample`, which is exact, rather than from the instant
 * the mark fired, which is only accurate to one 4096-sample buffer. At 16 kHz
 * that is 256 ms, five times the validator's whole lag search.
 */
describe("captureAudio reports the transmitted window", () => {
  const RATE = 16_000;
  const FRAME_MS = (4096 / RATE) * 1000; // 256ms per buffer

  async function run(opts: { frames: number; markAfter?: number }) {
    const g = globalThis as { AudioContext?: unknown };
    const original = g.AudioContext;
    g.AudioContext = MockAudioContext as unknown as typeof AudioContext;
    try {
      const stream = { getTracks: () => [{ stop() {} }] } as unknown as MediaStream;
      const controller = new AbortController();
      const windowController = new AbortController();
      const pending = captureAudio({
        stream,
        signal: controller.signal,
        captureWindowSignal: windowController.signal,
        minDurationMs: 0,
        maxDurationMs: 5000,
      });
      await new Promise((r) => setTimeout(r, 0));
      const proc = MockAudioContext.lastProcessor!;

      const beforeFirstFrame = performance.now();
      for (let i = 0; i < opts.frames; i++) {
        if (i === opts.markAfter) windowController.abort();
        fireFrame(proc, 0.1);
      }
      controller.abort();
      return { result: await pending, beforeFirstFrame };
    } finally {
      g.AudioContext = original;
    }
  }

  it("anchors the window at the mark, one buffer behind the first frame", async () => {
    // Mark after 3 buffers, so the trim drops 12,288 samples = 768ms, and the
    // window opens 768ms after the audio itself began.
    const { result, beforeFirstFrame } = await run({ frames: 10, markAfter: 3 });

    const epoch = result.windowStartMs - 3 * FRAME_MS;
    // The epoch is one buffer behind the first `onaudioprocess`, because a
    // buffer is delivered only once it is full.
    expect(epoch).toBeGreaterThanOrEqual(beforeFirstFrame - FRAME_MS - 50);
    expect(epoch).toBeLessThanOrEqual(beforeFirstFrame - FRAME_MS + 50);

    // 7 buffers survive the trim.
    expect(result.duration).toBeCloseTo((7 * 4096) / RATE, 3);
    expect(result.windowEndMs - result.windowStartMs).toBeCloseTo(result.duration * 1000, 3);
  });

  it("anchors on the far edge when the host never marked", async () => {
    // No mark means nothing is trimmed, so the window covers the whole buffer
    // and opens at the audio's own start rather than partway through it.
    const { result, beforeFirstFrame } = await run({ frames: 10 });

    expect(result.duration).toBeCloseTo((10 * 4096) / RATE, 3);
    expect(result.windowStartMs).toBeGreaterThanOrEqual(beforeFirstFrame - FRAME_MS - 50);
    expect(result.windowStartMs).toBeLessThanOrEqual(beforeFirstFrame - FRAME_MS + 50);
  });

  it("reports an empty window for a capture that produced nothing", async () => {
    // No frames at all. Equal bounds make `extractAccelerationMagnitude`
    // return no contour, so the coupling check skips rather than correlating
    // against a buffer that never existed.
    const { result } = await run({ frames: 0 });
    expect(result.samples.length).toBe(0);
    expect(result.windowEndMs - result.windowStartMs).toBe(0);
  });
});

/**
 * What the microphone delivered, as distinct from what was transmitted.
 *
 * `normalizeCaptureRMS` rescales the buffer toward `TARGET_CAPTURE_RMS` before
 * it leaves the device, so the level of the transmitted audio is a property of
 * that target and says nothing about the capture. A validator can therefore see
 * healthy audio while the user was barely audible, which is exactly the state
 * that made the "microphone too quiet" warning impossible to adjudicate from
 * server logs on 2026-08-01.
 *
 * The pair that resolves it is `gainClipped` against `voicedFrameRatio`.
 * Normalisation recovers input down to 0.05 / 50 = 0.001 RMS, while hosts warn
 * at 0.008, which is eight times stricter. Input landing between the two
 * produces a good capture and a warning at once, and only these fields say so.
 */
describe("describeInputLevel", () => {
  const RATE = 16_000;

  /** A tone at a known RMS. Amplitude a gives RMS a/√2. */
  function tone(rms: number, seconds = 1): Float32Array {
    const amp = rms * Math.SQRT2;
    const n = Math.round(RATE * seconds);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * 220 * i) / RATE);
    return out;
  }

  it("measures the input, not the normalized output", () => {
    // The whole contract. A quiet capture stays quiet in this reading even
    // though the buffer that ships is loud.
    const quiet = tone(0.004);
    const level = describeInputLevel(quiet);
    expect(level.rms).toBeCloseTo(0.004, 4);

    const shipped = normalizeCaptureRMS(quiet);
    let sumSq = 0;
    for (let i = 0; i < shipped.length; i++) sumSq += shipped[i]! * shipped[i]!;
    const shippedRms = Math.sqrt(sumSq / shipped.length);
    expect(shippedRms, "the transmitted buffer should have been brought up").toBeCloseTo(0.05, 3);
    expect(
      level.rms,
      "measuring the transmitted buffer would have reported the target, not the microphone",
    ).toBeLessThan(shippedRms / 10);
  });

  it("reports the gain that was actually applied", () => {
    // Mirrors `normalizeCaptureRMS` rather than re-deriving, so the two cannot
    // drift into disagreeing about the same capture.
    const level = describeInputLevel(tone(0.01));
    expect(level.gain).toBeCloseTo(0.05 / 0.01, 2);
    expect(level.gainClipped).toBe(false);
  });

  it("flags input below what the gain ceiling can recover", () => {
    // 0.05 / 50 = 0.001 is the floor. Below it the transmitted audio is still
    // quiet, and the microphone genuinely was the problem.
    const level = describeInputLevel(tone(0.0005));
    expect(level.gainClipped).toBe(true);
    expect(level.gain).toBe(50);

    // Just above the floor is recoverable, so it must not be flagged.
    expect(describeInputLevel(tone(0.002)).gainClipped).toBe(false);
  });

  it("separates the recoverable-but-warned case from the genuinely quiet one", () => {
    // The case that explains the reported UI regression: below the host's 0.008
    // warning bar, above the 0.001 floor. Capture is fine, warning still fires.
    const level = describeInputLevel(tone(0.004));
    expect(level.gainClipped, "this capture is recoverable").toBe(false);
    expect(level.voicedFrameRatio, "yet nothing clears the host's threshold").toBe(0);
  });

  it("counts voiced frames against the host's own threshold", () => {
    // Half the buffer above 0.008, half below.
    const n = RATE;
    const buf = new Float32Array(n);
    const loud = 0.05 * Math.SQRT2;
    for (let i = 0; i < n / 2; i++) buf[i] = loud * Math.sin((2 * Math.PI * 220 * i) / RATE);
    const level = describeInputLevel(buf);
    expect(level.voicedFrameRatio).toBeGreaterThan(0.4);
    expect(level.voicedFrameRatio).toBeLessThan(0.6);
    expect(level.peak).toBeCloseTo(loud, 2);
  });

  it("returns a usable answer for an empty capture", () => {
    // A failed capture must not divide by zero on the way to reporting itself.
    expect(describeInputLevel(new Float32Array(0))).toEqual({
      rms: 0,
      peak: 0,
      gain: 1,
      gainClipped: false,
      voicedFrameRatio: 0,
    });
  });
});
