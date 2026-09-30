import { beforeEach, afterEach, describe, expect, it, vi } from "vitest";

/**
 * A timing regression the mocked-recorder session tests cannot catch: the real
 * recorder's clock origin is only an arrival bound, so it can refine after the
 * round has begun. A late bound must not lose speech the person produced
 * after the reveal, and the deferred commit copy must resolve the refined
 * window instead of the boundaries the round started with. This mirrors the
 * native recorder/controller regression in
 * entros-mobile/src/flows/__tests__/pairedRecordingClock.test.ts.
 */

vi.mock("../src/sensor/audio", () => ({
  audioCaptureConstraints: () => ({}),
  detectVirtualAudioInput: async () => false,
  readVoiceIsolationApplied: () => null,
  describeInputLevel: () => "typical",
  normalizeCaptureRMS: (samples: Float32Array) => samples,
}));

vi.mock("../src/sensor/motion", () => ({
  requestMotionPermission: async () => false,
  captureMotion: async () => [],
}));

const server = vi.hoisted(() => ({
  commits: [] as Record<string, unknown>[],
  cues: [] as Record<string, unknown>[],
}));

vi.mock("../src/transport/post-json", async () => {
  const transcript = await import("../src/paired/transcript");
  const sessionNonce = new Uint8Array(32).fill(4);
  const attempt = new Uint8Array(32).fill(5);
  const words = ["balance", "garden", "silver"];
  const target = transcript.encodePathTarget([
    { x: 200, y: 200 },
    { x: 500, y: 700 },
    { x: 800, y: 300 },
  ]);
  const reveal = (index: number) => {
    const nonce = new Uint8Array(32).fill(10 + index);
    const cue = transcript.cueCommitment(sessionNonce, index, nonce, new Uint8Array(32).fill(100 + index), { x: 200, y: 800 });
    return {
      round_index: index,
      round_nonce: transcript.toHex(nonce),
      word: words[index - 1],
      path_target_hex: transcript.toHex(target),
      cue_commitment: transcript.toHex(cue),
      challenge_digest: transcript.toHex(
        transcript.challengeDigestV2(sessionNonce, index, nonce, words[index - 1]!, target, cue),
      ),
      expires_in_ms: 12_000,
    };
  };
  const reply = (status: number, body: Record<string, unknown>) => ({
    status,
    body,
    header: () => null as string | null,
  });
  return {
    TransportError: class extends Error {},
    postJson: async (url: string, payload: Record<string, unknown>) => {
      if (url.endsWith("/challenge/paired")) {
        return reply(200, {
          protocol: "paired",
          protocol_version: 2,
          session_id: "00112233445566778899aabbccddeeff",
          session_nonce: transcript.toHex(sessionNonce),
          attempt_binding: transcript.toHex(attempt),
          rounds: 3,
          tier: "trace",
          session_expiry_unix_ms: 1_790_000_600_000,
          expires_in_ms: 600_000,
          audio_format: "pcm_s16le_16000_mono",
          bounds: {
            max_round_samples: 192_000,
            max_session_samples: 576_000,
            min_path_points: 8,
            max_path_points: 64,
          },
          reveal: reveal(1),
        });
      }
      if (url.endsWith("/paired/cue")) {
        server.cues.push(payload);
        return reply(200, {
          session_id: payload.session_id,
          round_index: payload.round_index,
          round_nonce: payload.round_nonce,
          challenge_digest: payload.challenge_digest,
          point: { x: 200, y: 800 },
          salt: transcript.toHex(new Uint8Array(32).fill(100 + Number(payload.round_index))),
          expires_in_ms: 6_000,
        });
      }
      if (url.endsWith("/paired/commit")) {
        server.commits.push(payload);
        const index = payload.round_index as number;
        return reply(200, {
          state: index < 3 ? "awaiting_commit" : "ready_to_finalize",
          accepted_round: index,
          commitment: payload.commitment,
          session_expires_in_ms: index < 3 ? 500_000 : 120_000,
          ...(index < 3 ? { reveal: reveal(index + 1) } : {}),
        });
      }
      throw new Error(`unexpected url ${url}`);
    },
  };
});

import {
  AUDIO_FORMAT,
  audioDigest,
  challengeDigestV2,
  cueCommitment,
  encodePathTarget,
  toHex,
} from "../src/paired/transcript";
import { PairedSession, type PairedPhase, type PairedRoundView } from "../src/paired/session";
import { encodePcm16 } from "../src/sensor/encode";
import { createStreamingCanonicalizer } from "../src/sensor/resample";

const WALLET = "11111111111111111111111111111111";
const SESSION_NONCE = new Uint8Array(32).fill(4);
const ROUND_NONCE = new Uint8Array(32).fill(11);
const CUE_SALT = new Uint8Array(32).fill(101);
const WORD = "balance";
const TARGET = encodePathTarget([
  { x: 200, y: 200 },
  { x: 500, y: 700 },
  { x: 800, y: 300 },
]);
const CHALLENGE = challengeDigestV2(
  SESSION_NONCE,
  1,
  ROUND_NONCE,
  WORD,
  TARGET,
  cueCommitment(SESSION_NONCE, 1, ROUND_NONCE, CUE_SALT, { x: 200, y: 800 }),
);

/** The wall clock the recorder and the session share, in ms. */
let nowMs = 10_296;

function device() {
  const state = { trackStops: 0, closed: 0 };
  const processor: {
    onaudioprocess: ((event: unknown) => void) | null;
    connect(): void;
    disconnect(): void;
  } = { onaudioprocess: null, connect() {}, disconnect() {} };
  vi.stubGlobal("navigator", {
    mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop: () => state.trackStops++ }] }) },
  });
  vi.stubGlobal("AudioContext", class {
    sampleRate = 16_000;
    destination = {};
    async resume() {}
    async close() { state.closed++; }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createScriptProcessor() { return processor; }
  });
  return {
    state,
    emit(samples: Float32Array) {
      processor.onaudioprocess?.({ inputBuffer: { getChannelData: () => samples } });
    },
    connected() {
      return processor.onaudioprocess !== null;
    },
  };
}

function surface() {
  const listeners = new Map<string, ((event: unknown) => void)[]>();
  return {
    element: {
      addEventListener: (type: string, handler: (event: unknown) => void) => {
        listeners.set(type, [...(listeners.get(type) ?? []), handler]);
      },
      removeEventListener: (type: string, handler: (event: unknown) => void) => {
        listeners.set(
          type,
          (listeners.get(type) ?? []).filter((candidate) => candidate !== handler),
        );
      },
      getBoundingClientRect: () => ({ left: 0, top: 0, width: 1000, height: 1000 }),
    } as unknown as HTMLElement,
    press(type: "pointerdown" | "pointermove", x: number, y: number, buttons = 1) {
      for (const handler of listeners.get(type) ?? []) {
        handler({ clientX: x, clientY: y, buttons, pointerType: "mouse", pressure: 0.5, width: 1, height: 1 });
      }
    },
  };
}

function pipeline() {
  return {
    readProjectionPolicy: async () => ({ current: 1, minimum: 0 }),
    process: async () => ({ success: true, commitment: new Uint8Array(32), isFirstVerification: true }),
    processReset: async () => ({ success: true, commitment: new Uint8Array(32), isFirstVerification: false }),
    relayerUrl: "https://executor.example",
    relayerApiKey: "key",
  };
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 5));

/** One 4,096-sample buffer: a quarter second at the canonical rate. */
function buffer(fill: { from: number; to: number; level: number } | null): Float32Array {
  const samples = new Float32Array(4096);
  if (fill) samples.fill(fill.level, fill.from, fill.to);
  return samples;
}

beforeEach(() => {
  vi.spyOn(performance, "now").mockImplementation(() => nowMs);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  server.commits = [];
  server.cues = [];
});

describe("real recorder and session controller timing", () => {
  it("reconciles a delayed recorder clock at commit without losing early quiet speech", async () => {
    const mic = device();
    const phases: PairedPhase[] = [];
    const cueRounds: number[] = [];
    const reveals: number[] = [];
    const session = new PairedSession(pipeline(), {
      onPhase: (phase) => phases.push(phase),
      onCue: (cue) => cueRounds.push(cue.roundIndex),
      onReveal: (round: PairedRoundView) => reveals.push(round.roundIndex),
    });
    const trace = surface();
    nowMs = 10_296;
    const started = session.start(WALLET, trace.element);
    while (!mic.connected()) await Promise.resolve();
    // First buffer: its arrival places the clock origin 256 ms before now.
    mic.emit(buffer(null));
    // The round opens at the next clock reading.
    nowMs = 10_300;
    await started;
    expect(reveals).toEqual([1]);

    // A quieter buffer arrives with less delay, so the origin moves back 30 ms
    // after the round began. The pulse is real speech evidence from after the
    // reveal, though below the tracker's speech floor.
    nowMs = 10_522;
    mic.emit(buffer({ from: 704, to: 2304, level: 0.005 }));
    for (const [x, y] of [
      [200, 200],
      [350, 450],
      [500, 700],
      [650, 500],
      [800, 300],
    ]) {
      trace.press("pointermove", x!, y!);
    }
    expect(session.continueRound()).toBe(true);
    await settle();
    expect(cueRounds).toEqual([1]);
    expect(session.currentPhase).toBe("cue");

    nowMs = 10_523;
    trace.press("pointermove", 200, 800);
    nowMs = 10_778;
    // The cue point is reached on the first frame past it, which finishes the
    // round and defers the commit copy out of the audio callback.
    mic.emit(buffer(null));
    expect(session.currentPhase).toBe("committing");
    // A further buffer arrives whose bound moves the origin back 5 ms before
    // the deferred copy runs. The commit must resolve the refined window.
    nowMs = 11_029;
    mic.emit(buffer(null));
    await settle();

    expect(server.commits).toHaveLength(1);
    const commit = server.commits[0]!;
    expect(commit.round_index).toBe(1);
    expect(commit.audio_byte_length).toBe(8_320);

    // The exact bytes: the same buffers through the same chunked
    // canonicalizer, from sample 4,720 to 8,880.
    const canonicalizer = createStreamingCanonicalizer(16_000);
    const expected = [
      canonicalizer.push(buffer(null)),
      canonicalizer.push(buffer({ from: 704, to: 2304, level: 0.005 })),
      canonicalizer.push(buffer(null)),
      canonicalizer.push(buffer(null)),
    ];
    const length = expected.reduce((total, part) => total + part.length, 0);
    const canonical = new Float32Array(length);
    let offset = 0;
    for (const part of expected) {
      canonical.set(part, offset);
      offset += part.length;
    }
    const segment = encodePcm16(canonical.subarray(4_720, 8_880));
    expect(segment.some((value) => value !== 0)).toBe(true);
    expect(commit.audio_digest).toBe(toHex(audioDigest(SESSION_NONCE, 1, CHALLENGE, AUDIO_FORMAT, segment)));

    // The commit was accepted, so the next round opened.
    expect(reveals).toEqual([1, 2]);
    expect(phases).toContain("committing");

    session.abort();
    await settle();
    expect(session.currentPhase).toBe("failed");
    expect(mic.state.trackStops).toBe(1);
    expect(mic.state.closed).toBe(1);
  });

  it("keeps speech readiness when a delayed clock bound refines across the spoken word", async () => {
    const mic = device();
    const cueRounds: number[] = [];
    const session = new PairedSession(pipeline(), {
      onCue: (cue) => cueRounds.push(cue.roundIndex),
    });
    const trace = surface();
    nowMs = 10_296;
    const started = session.start(WALLET, trace.element);
    while (!mic.connected()) await Promise.resolve();
    mic.emit(buffer(null));
    nowMs = 10_300;
    await started;

    // The word, spoken right at the reveal: four full frames above the floor.
    nowMs = 10_522;
    mic.emit(buffer({ from: 704, to: 3_904, level: 0.05 }));
    // Quiet buffers while speech plus its quiet interval latches readiness.
    nowMs = 10_778;
    mic.emit(buffer(null));
    nowMs = 11_034;
    mic.emit(buffer(null));
    // A stalled pipeline now catches up: each buffer still carries 256 ms of
    // audio, but the clock advances far less, so every arrival bound moves the
    // recording origin further back — across the spoken word.
    nowMs = 11_090;
    mic.emit(buffer(null));
    nowMs = 11_150;
    mic.emit(buffer(null));
    nowMs = 11_210;
    mic.emit(buffer(null));

    // The round still completes on its own: the refined boundary stopped at
    // the word, so readiness survived and the cue request fires here.
    nowMs = 11_300;
    for (const [x, y] of [
      [200, 200],
      [350, 450],
      [500, 700],
      [650, 500],
      [800, 300],
    ]) {
      trace.press("pointermove", x!, y!);
    }
    mic.emit(buffer(null));
    await settle();
    expect(server.cues).toHaveLength(1);
    expect(cueRounds).toEqual([1]);
    expect(session.currentPhase).toBe("cue");

    nowMs = 11_340;
    trace.press("pointermove", 200, 800);
    nowMs = 11_360;
    mic.emit(buffer(null));
    await settle();

    expect(server.commits).toHaveLength(1);
    const commit = server.commits[0]!;
    expect(commit.round_index).toBe(1);
    // The committed segment starts at the clamped boundary and keeps the word:
    // canonical samples 4,800 through 36,800.
    const canonicalizer = createStreamingCanonicalizer(16_000);
    const parts = [
      canonicalizer.push(buffer(null)),
      canonicalizer.push(buffer({ from: 704, to: 3_904, level: 0.05 })),
      ...Array.from({ length: 7 }, () => canonicalizer.push(buffer(null))),
    ];
    const length = parts.reduce((total, part) => total + part.length, 0);
    const canonical = new Float32Array(length);
    let offset = 0;
    for (const part of parts) {
      canonical.set(part, offset);
      offset += part.length;
    }
    const segment = encodePcm16(canonical.subarray(4_800, 36_800));
    expect(segment.subarray(0, 6_400).some((value) => value !== 0)).toBe(true);
    expect(commit.audio_byte_length).toBe(segment.length);
    expect(commit.audio_digest).toBe(toHex(audioDigest(SESSION_NONCE, 1, CHALLENGE, AUDIO_FORMAT, segment)));

    session.abort();
    await settle();
    expect(session.currentPhase).toBe("failed");
  });
});
