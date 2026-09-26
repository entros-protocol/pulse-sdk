import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  buildCommit,
  buildFinalizeBody,
  checkFinalizeSuccess,
  commitWithRetry,
  finalizeRetryAfterMs,
  initialCommitment,
  PairedProtocolError,
  parseCommitResponse,
  parseOpenResponse,
  parseReveal,
  refusalOf,
  sessionFinalDigest,
} from "../src/paired/client";
import { toHex } from "../src/paired/transcript";
import type { PostJsonResponse } from "../src/transport/post-json";

const vectors = JSON.parse(
  readFileSync(resolve(__dirname, "fixtures/paired-round-vectors.json"), "utf8"),
);
const session = vectors.sessions[0];
const rounds = session.roundEntries;

function reveal(index: number, overrides: Record<string, unknown> = {}) {
  const round = rounds[index - 1];
  return {
    round_index: round.index,
    round_nonce: round.roundNonceHex,
    word: round.word,
    path_target_hex: round.pathTargetHex,
    challenge_digest: round.challengeDigestHex,
    expires_in_ms: 120_000,
    ...overrides,
  };
}

function openBody(overrides: Record<string, unknown> = {}) {
  return {
    protocol: "paired",
    protocol_version: 1,
    session_id: "00112233445566778899aabbccddeeff",
    session_nonce: session.sessionNonceHex,
    attempt_binding: session.attemptBindingDigestHex,
    rounds: 3,
    tier: "trace",
    session_expiry_unix_ms: session.sessionExpiryUnixMs,
    expires_in_ms: 600_000,
    audio_format: "pcm_s16le_16000_mono",
    bounds: {
      max_round_samples: 192_000,
      max_session_samples: 576_000,
      min_path_points: 8,
      max_path_points: 64,
    },
    reveal: reveal(1),
    ...overrides,
  };
}

const response = (
  status: number,
  body: Record<string, unknown> = {},
  retryAfter: string | null = null,
): PostJsonResponse => ({
  status,
  body,
  header: (name: string) => (name === "retry-after" ? retryAfter : null),
});

function clock(start = 0) {
  let now = start;
  const waits: number[] = [];
  return {
    waits,
    now: () => now,
    sleep: async (ms: number) => {
      waits.push(ms);
      now += ms;
    },
  };
}

describe("paired open response", () => {
  it("reads a well-formed session and chains from its C_0", () => {
    const open = parseOpenResponse(openBody(), 1_000);
    expect(open.reveal.word).toBe(rounds[0].word);
    expect(open.reveal.waypoints).toHaveLength(4);
    expect(open.expiresAtMs).toBe(601_000);
    expect(toHex(initialCommitment(open))).toBe(session.sessionCommitmentHex);
  });

  it("refuses a reveal whose digest does not recompute", () => {
    const tampered = openBody({ reveal: reveal(1, { word: "garden" }) });
    expect(() => parseOpenResponse(tampered, 0)).toThrow("challenge_mismatch");
  });

  it("refuses another protocol, tier, format or bound", () => {
    for (const change of [
      { protocol: "single" },
      { protocol_version: 2 },
      { tier: "speech_only" },
      { audio_format: "pcm_s16le_48000_mono" },
      { rounds: 4 },
      { bounds: { max_round_samples: 320_000, max_session_samples: 576_000, min_path_points: 8, max_path_points: 64 } },
    ]) {
      expect(() => parseOpenResponse(openBody(change), 0)).toThrow(PairedProtocolError);
    }
  });

  it("refuses a malformed session id or nonce", () => {
    expect(() => parseOpenResponse(openBody({ session_id: "ABC" }), 0)).toThrow("malformed_response");
    expect(() => parseOpenResponse(openBody({ session_nonce: "00" }), 0)).toThrow("malformed_response");
  });
});

describe("paired commit", () => {
  const open = parseOpenResponse(openBody(), 0);
  const round = rounds[0];
  const built = buildCommit({
    open,
    reveal: open.reveal,
    walletId: "wallet",
    previousCommitment: initialCommitment(open),
    segment: Uint8Array.from(Buffer.from(round.audioSegmentHex, "hex")),
    coarsePath: Uint8Array.from(Buffer.from(round.coarsePathHex, "hex")),
    pointCount: round.pathPointCount,
    idempotencyKey: new Uint8Array(16).fill(1),
  });

  it("builds the commitment the vectors expect", () => {
    expect(built.body.commitment).toBe(round.commitmentHex);
    expect(built.body.audio_digest).toBe(round.audioDigestHex);
    expect(built.body.audio_format).toBe("pcm_s16le_16000_mono");
    expect(built.body.idempotency_key).toBe("01".repeat(16));
    expect(built.body).not.toHaveProperty("request_digest");
  });

  const accepted = (overrides: Record<string, unknown> = {}) => ({
    state: "awaiting_commit",
    accepted_round: 1,
    commitment: round.commitmentHex,
    reveal: reveal(2),
    session_expires_in_ms: 500_000,
    ...overrides,
  });

  it("accepts a response for this round with the next reveal", () => {
    const parsed = parseCommitResponse(accepted(), built.round, open.sessionNonce, 1_000);
    expect(parsed.next?.roundIndex).toBe(2);
    expect(parsed.sessionEndsAtMs).toBe(501_000);
  });

  it("never lets the next round outlast the session", () => {
    const parsed = parseCommitResponse(
      accepted({ session_expires_in_ms: 60_000 }),
      built.round,
      open.sessionNonce,
      0,
    );
    expect(parsed.next?.expiresAtMs).toBe(60_000);
  });

  it("refuses a state that does not follow this round", () => {
    expect(() =>
      parseCommitResponse(accepted({ state: "ready_to_finalize" }), built.round, open.sessionNonce, 0),
    ).toThrow("malformed_response");
    expect(() =>
      parseCommitResponse(
        accepted({ state: "awaiting_commit" }),
        { ...built.round, roundIndex: 3 },
        open.sessionNonce,
        0,
      ),
    ).toThrow("malformed_response");
  });

  it("refuses a response that names another commitment", () => {
    expect(() =>
      parseCommitResponse(accepted({ commitment: "00".repeat(32) }), built.round, open.sessionNonce, 0),
    ).toThrow("commitment_mismatch");
  });

  it("retries a lost response with the identical body until it lands", async () => {
    const sent: string[] = [];
    const replies = [
      () => Promise.reject(new Error("offline")),
      () => Promise.resolve(response(503)),
      () => Promise.resolve(response(200, { state: "awaiting_commit" })),
    ];
    const time = clock();
    const body = await commitWithRetry(
      (payload) => {
        sent.push(JSON.stringify(payload));
        return replies.shift()!();
      },
      built.body,
      60_000,
      time,
    );
    expect(body.state).toBe("awaiting_commit");
    expect(new Set(sent).size).toBe(1);
    expect(sent).toHaveLength(3);
    expect(time.waits).toEqual([250, 500]);
  });

  it("honours retry_after and stops at the round deadline", async () => {
    const time = clock();
    await expect(
      commitWithRetry(
        () => Promise.resolve(response(429, { reason: "rate_limited", retry_after: 3 })),
        built.body,
        7_000,
        time,
      ),
    ).rejects.toThrow("validation_unavailable");
    expect(time.waits).toEqual([3_000, 3_000]);
  });

  it("ends at once on a protocol refusal", async () => {
    const time = clock();
    await expect(
      commitWithRetry(
        () => Promise.resolve(response(409, { reason: "session_superseded" })),
        built.body,
        60_000,
        time,
      ),
    ).rejects.toMatchObject({ reason: "session_superseded", status: 409 });
    expect(time.waits).toEqual([]);
  });
});

describe("paired finalize body", () => {
  it("orders the rounds and binds the final digest", () => {
    const open = parseOpenResponse(openBody(), 0);
    let previous = initialCommitment(open);
    const committed = [];
    let current = open.reveal;
    for (const [index, round] of rounds.entries()) {
      const { round: done } = buildCommit({
        open,
        reveal: current,
        walletId: "wallet",
        previousCommitment: previous,
        segment: Uint8Array.from(Buffer.from(round.audioSegmentHex, "hex")),
        coarsePath: Uint8Array.from(Buffer.from(round.coarsePathHex, "hex")),
        pointCount: round.pathPointCount,
        idempotencyKey: new Uint8Array(16).fill(index + 1),
      });
      committed.push(done);
      previous = done.commitment;
      if (index + 1 < rounds.length) {
        current = parseReveal(reveal(index + 2), open.sessionNonce, 0, open.expiresAtMs);
      }
    }
    const body = buildFinalizeBody({
      open,
      walletId: "wallet",
      rounds: [committed[2]!, committed[0]!, committed[1]!],
      features: [0.5],
      baselineReset: false,
    });
    expect(body.final_digest).toBe(session.finalDigestHex);
    expect((body.segments as { audio_b64: string }[])[0]!.audio_b64).toBe(
      Buffer.from(rounds[0].audioSegmentHex, "hex").toString("base64"),
    );
    expect((body.segments as { round_index: number }[]).map((segment) => segment.round_index)).toEqual([
      1, 2, 3,
    ]);
    expect(body).not.toHaveProperty("attestation");
    expect(body.projection_version).toBe(1);
  });
});

describe("paired refusals and retries", () => {
  it("names a refusal by its reason, or by its status when it has none", () => {
    expect(refusalOf(response(409, { reason: "session_active", retry_after: 42 }))).toMatchObject({
      reason: "session_active",
      status: 409,
      retryAfterSecs: 42,
    });
    expect(refusalOf(response(429, {}, "7")).retryAfterSecs).toBe(7);
    expect(refusalOf(response(404)).reason).toBe("unsupported_session");
    expect(refusalOf(response(502)).reason).toBe("validation_unavailable");
    expect(refusalOf(response(401)).reason).toBe("malformed_response");
  });

  it("resends a finalize only while nothing was judged", () => {
    expect(finalizeRetryAfterMs(null, 0)).toBe(250);
    expect(finalizeRetryAfterMs(response(408), 1)).toBe(500);
    expect(finalizeRetryAfterMs(response(503, { reason: "session_busy", retry_after: 1 }), 0)).toBe(1_000);
    expect(finalizeRetryAfterMs(response(503, { reason: "validation_unavailable" }), 6)).toBe(4_000);
    expect(finalizeRetryAfterMs(response(503, { reason: "technical_failure" }), 0)).toBeNull();
    expect(finalizeRetryAfterMs(response(400, { reason: "trace_incomplete" }), 0)).toBeNull();
    expect(finalizeRetryAfterMs(response(409, { reason: "session_consumed" }), 0)).toBeNull();
    expect(finalizeRetryAfterMs(response(200, { valid: true }), 0)).toBeNull();
  });
});

describe("paired finalize success", () => {
  const open = parseOpenResponse(openBody(), 0);
  const committed = rounds.map((round: Record<string, string>, index: number) => ({
    roundIndex: index + 1,
    segment: new Uint8Array(0),
    coarsePath: new Uint8Array(0),
    pointCount: 0,
    audioDigest: Uint8Array.from(Buffer.from(round.audioDigestHex!, "hex")),
    pathDigest: Uint8Array.from(Buffer.from(round.pathDigestHex!, "hex")),
    commitment: Uint8Array.from(Buffer.from(round.commitmentHex!, "hex")),
  }));
  const finalDigest = sessionFinalDigest(open, committed);
  const wallet = new Uint8Array(32).fill(7);

  function receipt(tier = 2, digest = finalDigest) {
    const message = Buffer.alloc(136);
    Buffer.from("entros-validator-receipt-v3\0", "ascii").copy(message, 0);
    message[28] = 1;
    message.writeUInt16LE(1, 29);
    Buffer.from(wallet).copy(message, 31);
    Buffer.alloc(32, 9).copy(message, 63);
    message.writeBigInt64LE(1_790_000_000n, 95);
    Buffer.from(digest).copy(message, 103);
    message[135] = tier;
    return {
      validator_pubkey_hex: "8c".repeat(32),
      signature_hex: "ab".repeat(64),
      message_hex: message.toString("hex"),
    };
  }
  const success = (overrides: Record<string, unknown> = {}) => ({
    valid: true,
    signed_receipt: receipt(),
    commitment_hex: "09".repeat(32),
    salt_hex: "05".repeat(32),
    assurance_tier: 0,
    ...overrides,
  });
  const binding = { purpose: "mint" as const, wallet, finalDigest };

  it("takes the tier from the signed receipt, not the body", () => {
    expect(checkFinalizeSuccess(success(), binding)).toEqual({ assuranceTier: 2 });
  });

  it("refuses a receipt that does not bind this session or is malformed", () => {
    const otherSession = new Uint8Array(finalDigest);
    otherSession[0] ^= 1;
    for (const body of [
      success({ signed_receipt: receipt(2, otherSession) }),
      success({ signed_receipt: { message_hex: 1 } }),
      success({ signed_receipt: "receipt" }),
      success({ salt_hex: undefined }),
      success({ commitment_hex: "09" }),
    ]) {
      expect(checkFinalizeSuccess(body, binding)).toMatch(/does not match/);
    }
    expect(checkFinalizeSuccess(success(), { ...binding, purpose: "reset" })).toMatch(/does not match/);
  });

  it("requires a receipt for a transition and none for an update", () => {
    expect(checkFinalizeSuccess({ valid: true }, binding)).toMatch(/no receipt/);
    expect(checkFinalizeSuccess({ valid: true }, { wallet, finalDigest })).toEqual({});
  });
});
