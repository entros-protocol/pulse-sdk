import { PublicKey } from "@solana/web3.js";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PROGRAM_IDS } from "../src/config";
import type { PostJsonResponse } from "../src/transport/post-json";

const recorderState = vi.hoisted(() => ({
  frame: null as null | ((level: number, endSample: number) => void),
  framed: 0,
  released: 0,
  stopped: 0,
  gate: null as null | Promise<void>,
  refuse: null as null | Error,
}));

vi.mock("../src/paired/recorder", () => ({
  startPairedRecorder: async (onFrame: (level: number, endSample: number) => void) => {
    if (recorderState.gate) await recorderState.gate;
    if (recorderState.refuse) throw recorderState.refuse;
    recorderState.frame = onFrame;
    recorderState.framed = 0;
    return {
      virtualDevice: false,
      voiceIsolationApplied: null,
      framedSamples: () => recorderState.framed,
      sampleIndexAt: () => recorderState.framed,
      timeAt: (sample: number) => sample / 16,
      slice: (start: number, end: number) => {
        const out = new Float32Array(end - start);
        for (let index = 0; index < out.length; index++) {
          out[index] = 0.1 * Math.sin((2 * Math.PI * 200 * (start + index)) / 16_000);
        }
        return out;
      },
      releaseBefore: (sample: number) => {
        recorderState.released = sample;
      },
      stop: async () => {
        recorderState.stopped++;
      },
    };
  },
}));

vi.mock("../src/sensor/motion", () => ({
  requestMotionPermission: async () => false,
  captureMotion: async () => [],
}));

const server = vi.hoisted(() => ({
  commits: [] as Record<string, unknown>[],
  refuseRound: 0,
  holdRound: 0,
  release: null as null | (() => void),
  opens: 0,
  openFailures: [] as { status: number; body: Record<string, unknown> }[],
  revealExpiresMs: 120_000,
  readyExpiresMs: 120_000,
  finalizes: [] as Record<string, unknown>[],
  finalizeReplies: [] as { status: number; body: Record<string, unknown> }[],
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
    return {
      round_index: index,
      round_nonce: transcript.toHex(nonce),
      word: words[index - 1],
      path_target_hex: transcript.toHex(target),
      challenge_digest: transcript.toHex(
        transcript.challengeDigest(sessionNonce, index, nonce, words[index - 1]!, target),
      ),
      expires_in_ms: server.revealExpiresMs,
    };
  };
  const reply = (status: number, body: Record<string, unknown>): PostJsonResponse => ({
    status,
    body,
    header: () => null,
  });
  return {
    TransportError: class extends Error {},
    postJson: async (url: string, payload: Record<string, unknown>) => {
      if (url.endsWith("/challenge/paired")) {
        server.opens++;
        const failure = server.openFailures.shift();
        if (failure) return reply(failure.status, failure.body);
        return reply(200, {
          protocol: "paired",
          protocol_version: 1,
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
      if (url.endsWith("/paired/commit")) {
        server.commits.push(payload);
        const index = payload.round_index as number;
        if (index === server.refuseRound) {
          return reply(409, { error: "refused", reason: "session_superseded" });
        }
        if (index === server.holdRound) {
          await new Promise<void>((resolve) => (server.release = resolve));
        }
        return reply(200, {
          state: index < 3 ? "awaiting_commit" : "ready_to_finalize",
          accepted_round: index,
          commitment: payload.commitment,
          session_expires_in_ms: index < 3 ? 500_000 : server.readyExpiresMs,
          ...(index < 3 ? { reveal: reveal(index + 1) } : {}),
        });
      }
      if (url.endsWith("/validate-session")) {
        server.finalizes.push(payload);
        const next = server.finalizeReplies.shift() ?? { status: 400, body: { reason: "phrase_content_mismatch" } };
        return reply(next.status, next.body);
      }
      throw new Error(`unexpected url ${url}`);
    },
  };
});

import { PairedSession, type PairedRoundView } from "../src/paired/session";
import { PulseSDK, type ValidationTarget } from "../src/pulse";

const WALLET = "11111111111111111111111111111111";

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

function frames(level: number, count: number) {
  for (let index = 0; index < count; index++) {
    recorderState.framed += 800;
    recorderState.frame?.(level, recorderState.framed);
  }
}

const settle = () => new Promise((resolve) => setTimeout(resolve, 5));

/** Waits, within a bound, for a commit in flight to land. */
async function landed(session: PairedSession) {
  const end = Date.now() + 2_000;
  while (session.currentPhase === "committing" && Date.now() < end) await settle();
}

async function playRound(trace: ReturnType<typeof surface>, session: PairedSession) {
  trace.press("pointerdown", 190, 210);
  for (const [x, y] of [
    [200, 200],
    [350, 450],
    [500, 700],
    [650, 500],
    [800, 300],
  ]) {
    trace.press("pointermove", x!, y!);
    frames(0.001, 1);
  }
  frames(0.001, 5);
  frames(0.05, 6);
  frames(0.001, 13);
  await settle();
  await landed(session);
}

function pipeline(capture: { target?: ValidationTarget }) {
  return {
    readProjectionPolicy: async () => ({ current: 1, minimum: 0 }),
    process: async (_data: unknown, _wallet: unknown, _connection: unknown, _progress: unknown, _policy: unknown, target: ValidationTarget) => {
      capture.target = target;
      return { success: true, commitment: new Uint8Array(32), isFirstVerification: true };
    },
    processReset: async () => ({ success: true, commitment: new Uint8Array(32), isFirstVerification: false }),
    relayerUrl: "https://executor.example",
    relayerApiKey: "key",
  };
}

afterEach(() => {
  server.commits = [];
  server.refuseRound = 0;
  server.holdRound = 0;
  server.release = null;
  server.opens = 0;
  server.openFailures = [];
  server.revealExpiresMs = 120_000;
  server.readyExpiresMs = 120_000;
  server.finalizes = [];
  server.finalizeReplies = [];
  recorderState.stopped = 0;
  recorderState.gate = null;
  recorderState.refuse = null;
});

function receiptFor(finalDigestHex: string, tier = 2) {
  const message = Buffer.alloc(136);
  Buffer.from("entros-validator-receipt-v3\0", "ascii").copy(message, 0);
  message[28] = 1;
  message.writeUInt16LE(1, 29);
  Buffer.alloc(32, 7).copy(message, 63);
  message.writeBigInt64LE(1_790_000_000n, 95);
  Buffer.from(finalDigestHex, "hex").copy(message, 103);
  message[135] = tier;
  return {
    validator_pubkey_hex: "8c".repeat(32),
    signature_hex: "ab".repeat(64),
    message_hex: message.toString("hex"),
  };
}

async function playSession(session: PairedSession, trace: ReturnType<typeof surface>) {
  await session.start(WALLET, trace.element);
  await playRound(trace, session);
  await playRound(trace, session);
  await playRound(trace, session);
}

describe("paired session", () => {
  it("runs three rounds in order and commits each before the next reveal", async () => {
    const reveals: PairedRoundView[] = [];
    const capture: { target?: ValidationTarget } = {};
    const session = new PairedSession(pipeline(capture), { onReveal: (round) => reveals.push(round) });
    const trace = surface();
    await session.start(WALLET, trace.element);
    expect(reveals.map((round) => round.word)).toEqual(["balance"]);

    await playRound(trace, session);
    expect(server.commits).toHaveLength(1);
    expect(reveals.map((round) => round.word)).toEqual(["balance", "garden"]);
    await playRound(trace, session);
    await playRound(trace, session);
    expect(server.commits.map((commit) => commit.round_index)).toEqual([1, 2, 3]);
    expect(session.currentPhase).toBe("ready");

    // Each commit chains from the one before it.
    expect(server.commits[1]!.previous_commitment).toBe(server.commits[0]!.commitment);
    expect(server.commits[2]!.previous_commitment).toBe(server.commits[1]!.commitment);
    expect(new Set(server.commits.map((commit) => commit.idempotency_key)).size).toBe(3);

    const result = await session.complete({}, {});
    expect(result.success).toBe(true);
    const body = capture.target!.body({
      walletAddress: WALLET,
      features: [0.5],
      f0Contour: [],
      accelMagnitude: [],
      captureTiming: undefined,
      clientSignals: { v: 1 } as never,
      receiptPurpose: "mint",
    });
    const segments = body.segments as { round_index: number; audio_b64: string }[];
    expect(segments.map((segment) => segment.round_index)).toEqual([1, 2, 3]);
    expect(capture.target!.path).toBe("/validate-session");
    expect(body.capture_protocol).toBe("paired");
    expect(capture.target!.accept!({ valid: true })).toMatch(/no receipt/);

    const receipt = receiptFor(body.final_digest as string);
    const success = {
      valid: true,
      signed_receipt: receipt,
      commitment_hex: "07".repeat(32),
      salt_hex: "05".repeat(32),
      assurance_tier: 0,
    };
    expect(capture.target!.accept!(success)).toBeNull();

    const otherSession = Buffer.from(receipt.message_hex, "hex");
    otherSession[110] ^= 1;
    expect(
      capture.target!.accept!({ ...success, signed_receipt: { ...receipt, message_hex: otherSession.toString("hex") } }),
    ).toMatch(/does not match/);
  });

  it("stops every sensor once the last round is committed", async () => {
    const session = new PairedSession(pipeline({}), {});
    await playSession(session, surface());
    expect(session.currentPhase).toBe("ready");
    expect(recorderState.stopped).toBe(1);
  });

  it("never draws or reaches on hover", async () => {
    const session = new PairedSession(pipeline({}), {});
    const trace = surface();
    await session.start(WALLET, trace.element);
    for (const [x, y] of [
      [200, 200],
      [500, 700],
      [800, 300],
    ]) {
      trace.press("pointermove", x!, y!, 0);
    }
    frames(0.001, 5);
    frames(0.05, 6);
    frames(0.001, 20);
    await settle();
    expect(server.commits).toHaveLength(0);
    expect(session.currentPhase).toBe("round");
  });

  it("ends the session on a protocol refusal and reports why", async () => {
    server.refuseRound = 1;
    const failures: string[] = [];
    const session = new PairedSession(pipeline({}), {
      onFailure: (error) => failures.push(error.reason),
    });
    const trace = surface();
    await session.start(WALLET, trace.element);
    await playRound(trace, session);
    expect(failures).toEqual(["session_superseded"]);
    expect(session.currentPhase).toBe("failed");
  });

  it("releases the microphone when aborted while it starts", async () => {
    let open!: () => void;
    recorderState.gate = new Promise((resolve) => (open = resolve));
    const failures: string[] = [];
    const session = new PairedSession(pipeline({}), { onFailure: (error) => failures.push(error.reason) });
    const started = session.start(WALLET, surface().element);
    await settle();
    session.abort();
    open();
    await expect(started).resolves.toBeUndefined();
    expect(recorderState.stopped).toBeGreaterThan(0);
    expect(server.opens).toBe(0);
    expect(failures).toEqual([]);
    expect(session.currentPhase).toBe("failed");
  });

  it("reports a failure to start once, by rejecting", async () => {
    server.openFailures = [{ status: 429, body: { reason: "session_budget_exhausted", retry_after: 30 } }];
    const failures: string[] = [];
    const session = new PairedSession(pipeline({}), { onFailure: (error) => failures.push(error.reason) });
    await expect(session.start(WALLET, surface().element)).rejects.toMatchObject({
      reason: "session_budget_exhausted",
      status: 429,
      retryAfterSecs: 30,
    });
    expect(failures).toEqual([]);
    expect(recorderState.stopped).toBeGreaterThan(0);
  });

  it("retries a briefly rate-limited open and gives up on a long wait", async () => {
    server.openFailures = [{ status: 429, body: { reason: "ip_rate_limited", retry_after: 1 } }];
    const session = new PairedSession(pipeline({}), {});
    await session.start(WALLET, surface().element);
    expect(server.opens).toBe(2);
    expect(session.currentPhase).toBe("round");
    session.abort();

    // A wait longer than the open's retry window ends the start with the server's refusal.
    server.opens = 0;
    server.openFailures = [{ status: 429, body: { reason: "session_budget_exhausted", retry_after: 600 } }];
    const refused = new PairedSession(pipeline({}), {});
    await expect(refused.start(WALLET, surface().element)).rejects.toMatchObject({
      reason: "session_budget_exhausted",
      retryAfterSecs: 600,
    });
    expect(server.opens).toBe(1);
  });

  it("retries an open through a validator outage", async () => {
    server.openFailures = [{ status: 503, body: { reason: "validation_unavailable" } }];
    const reveals: string[] = [];
    const session = new PairedSession(pipeline({}), { onReveal: (round) => reveals.push(round.word) });
    await session.start(WALLET, surface().element);
    expect(server.opens).toBe(2);
    expect(reveals).toEqual(["balance"]);
  });

  it("waits for a trace that forms a path before it ends a round", async () => {
    const stalls: number[] = [];
    const session = new PairedSession(pipeline({}), { onStall: (index) => stalls.push(index) });
    const trace = surface();
    await session.start(WALLET, trace.element);
    // Pressed at one spot only: the trace has no length.
    trace.press("pointerdown", 200, 200);
    trace.press("pointermove", 200, 200);
    frames(0.001, 5);
    frames(0.05, 6);
    frames(0.001, 300);
    await settle();
    expect(server.commits).toHaveLength(0);
    expect(session.currentPhase).toBe("round");
    expect(stalls).toEqual([1]);
    expect(session.continueRound()).toBe(false);

    for (const [x, y] of [
      [500, 700],
      [800, 300],
    ]) {
      trace.press("pointermove", x!, y!);
    }
    expect(session.continueRound()).toBe(true);
    await settle();
    expect(server.commits).toHaveLength(1);
  });

  it("keeps a round whose trace paused at the first dot", async () => {
    const session = new PairedSession(pipeline({}), {});
    const trace = surface();
    await session.start(WALLET, trace.element);
    const start = performance.now();
    const at = vi.spyOn(performance, "now");
    trace.press("pointerdown", 200, 200);
    // Two seconds at the first dot, then the rest of the path in a tenth of that.
    for (let step = 0; step < 200; step++) {
      at.mockReturnValue(start + step * 10);
      trace.press("pointermove", 200, 200);
    }
    [
      [350, 450],
      [500, 700],
      [650, 500],
      [800, 300],
    ].forEach(([x, y], index) => {
      at.mockReturnValue(start + 2_000 + index * 50);
      trace.press("pointermove", x!, y!);
    });
    at.mockRestore();
    frames(0.001, 5);
    frames(0.05, 6);
    frames(0.001, 13);
    await settle();
    await landed(session);
    expect(server.commits).toHaveLength(1);
  });

  it("counts waypoints only in the issued order", async () => {
    const stalls: number[] = [];
    const session = new PairedSession(pipeline({}), { onStall: (index) => stalls.push(index) });
    const trace = surface();
    await session.start(WALLET, trace.element);
    trace.press("pointerdown", 800, 300);
    for (const [x, y] of [
      [650, 500],
      [500, 700],
      [350, 450],
      [200, 200],
    ]) {
      trace.press("pointermove", x!, y!);
      frames(0.001, 1);
    }
    frames(0.05, 6);
    frames(0.001, 300);
    await settle();
    expect(server.commits).toHaveLength(0);
    expect(stalls).toEqual([1]);
    // Speech the tracker heard does not stand in for a trace in the wrong order.
    expect(session.continueRound()).toBe(false);
  });

  it("ends a round that outlives its reveal", async () => {
    server.revealExpiresMs = 30;
    const failures: string[] = [];
    const session = new PairedSession(pipeline({}), { onFailure: (error) => failures.push(error.reason) });
    await session.start(WALLET, surface().element);
    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(failures).toEqual(["round_expired"]);
    expect(session.currentPhase).toBe("failed");
    expect(recorderState.stopped).toBe(1);
  });

  it("ends a ready session that is never finalized in its window", async () => {
    server.readyExpiresMs = 400;
    const failures: string[] = [];
    const session = new PairedSession(pipeline({}), { onFailure: (error) => failures.push(error.reason) });
    await playSession(session, surface());
    expect(session.currentPhase).toBe("ready");
    await new Promise((resolve) => setTimeout(resolve, 800));
    expect(failures).toEqual(["session_expired"]);
  });

  it("refuses a finalize locally once the session has expired", async () => {
    server.readyExpiresMs = 1_000;
    const capture: { target?: ValidationTarget } = {};
    const session = new PairedSession(pipeline(capture), {});
    await playSession(session, surface());
    // The clock passes the finalize window before the host calls `complete`.
    vi.spyOn(performance, "now").mockReturnValue(performance.now() + 2_000);
    const result = await session.complete({}, {});
    vi.restoreAllMocks();
    expect(result).toMatchObject({ success: false, reason: "session_expired" });
    expect(capture.target).toBeUndefined();
  });

  it("changes nothing after an abort, even when a commit lands later", async () => {
    server.holdRound = 1;
    const phases: string[] = [];
    const reveals: number[] = [];
    const session = new PairedSession(pipeline({}), {
      onPhase: (phase) => phases.push(phase),
      onReveal: (round) => reveals.push(round.roundIndex),
    });
    const trace = surface();
    await session.start(WALLET, trace.element);
    await playRound(trace, session);
    expect(session.currentPhase).toBe("committing");
    session.abort();
    server.release?.();
    await settle();
    expect(session.currentPhase).toBe("failed");
    expect(phases[phases.length - 1]).toBe("failed");
    expect(reveals).toEqual([1]);
  });

  it("refuses the validator's answer once aborted during finalize", async () => {
    const capture: { target?: ValidationTarget } = {};
    let proceed!: () => void;
    const session = new PairedSession(
      {
        ...pipeline(capture),
        process: async (...args: unknown[]) => {
          capture.target = args[5] as ValidationTarget;
          await new Promise<void>((resolve) => (proceed = resolve));
          return { success: true, commitment: new Uint8Array(32), isFirstVerification: true };
        },
      },
      {},
    );
    await playSession(session, surface());
    const completing = session.complete({}, {});
    await settle();
    session.abort();
    expect(capture.target!.accept!({ valid: true })).toMatch(/cancelled/);
    proceed();
    await completing;
    expect(session.currentPhase).toBe("failed");
  });

  it("passes a refused microphone through as the browser's error", async () => {
    recorderState.refuse = new DOMException("Permission denied", "NotAllowedError");
    const session = new PairedSession(pipeline({}), {});
    await expect(session.start(WALLET, surface().element)).rejects.toMatchObject({ name: "NotAllowedError" });
    expect(session.currentPhase).toBe("failed");
  });

  it("refuses to complete before the last round is committed", async () => {
    const session = new PairedSession(pipeline({}), {});
    await session.start(WALLET, surface().element);
    await expect(session.complete({}, {})).rejects.toThrow(/last round/);
  });
});

describe("paired session through the verification pipeline", () => {
  const registryProgramId = new PublicKey(PROGRAM_IDS.entrosRegistry);
  const [protocolConfigPda] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("protocol_config")],
    registryProgramId,
  );
  const connection = {
    getAccountInfo: async (address: PublicKey) => {
      if (!address.equals(protocolConfigPda)) return null;
      const data = Buffer.alloc(113);
      data.writeUInt16LE(1, 109);
      return { data, owner: registryProgramId };
    },
  };

  it("never asks the wallet to sign when the validator's receipt is refused", async () => {
    const wallet = {
      publicKey: new PublicKey(WALLET),
      calls: 0,
      async signTransaction<T>(transaction: T): Promise<T> {
        this.calls++;
        return transaction;
      },
      async sendTransaction(): Promise<string> {
        this.calls++;
        return "signature";
      },
    };
    const sdk = new PulseSDK({ cluster: "devnet", relayerUrl: "https://executor.example", relayerApiKey: "key" });
    const session = sdk.createPairedSession();
    await playSession(session, surface());
    // A receipt for another session, which the SDK must refuse before any signature.
    server.finalizeReplies = [
      {
        status: 200,
        body: {
          valid: true,
          signed_receipt: receiptFor("00".repeat(32)),
          commitment_hex: "07".repeat(32),
          salt_hex: "05".repeat(32),
          assurance_tier: 2,
        },
      },
    ];
    const result = await session.complete(wallet, connection);
    expect(server.finalizes).toHaveLength(1);
    expect(server.finalizes[0]!.capture_protocol).toBe("paired");
    expect(result.success).toBe(false);
    expect(result.error).toMatch(/does not match/);
    expect(wallet.calls).toBe(0);
  });

  it("resends a finalize the relayer was too busy to take", async () => {
    const sdk = new PulseSDK({ cluster: "devnet", relayerUrl: "https://executor.example", relayerApiKey: "key" });
    const session = sdk.createPairedSession();
    await playSession(session, surface());
    server.finalizeReplies = [
      { status: 503, body: { reason: "session_busy", retry_after: 0 } },
      { status: 400, body: { reason: "trace_incomplete", biometric_risk: 0 } },
    ];
    const result = await session.complete({ publicKey: new PublicKey(WALLET) }, connection);
    expect(server.finalizes).toHaveLength(2);
    expect(server.finalizes[0]).toEqual(server.finalizes[1]);
    expect(result).toMatchObject({ success: false, reason: "trace_incomplete" });
  });
});
