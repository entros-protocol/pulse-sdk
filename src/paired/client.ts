/**
 * The paired-session wire contract: open and commit responses, commit and
 * finalize bodies, the finalize receipt check, and the retry rules.
 *
 * Every response is checked before use. A reveal whose digest does not
 * recompute is refused, because the next commitment would bind a challenge the
 * server never issued.
 */

import { bytesToBase64 } from "../sensor/encode";
import { decodeSignedReceipt, receiptMatchesBinding } from "../submit/receipt";
import type { SignedReceiptDto } from "../submit/types";
import type { PostJsonResponse } from "../transport/post-json";
import {
  AUDIO_FORMAT,
  audioDigest,
  challengeDigest,
  decodePathTarget,
  type Digest,
  equalBytes,
  evidenceManifest,
  finalDigest,
  fromHex,
  type GridPoint,
  MAX_PATH_POINTS,
  MIN_PATH_POINTS,
  pathDigest,
  roundCommitment,
  sessionCommitment,
  toHex,
} from "./transcript";
import { MAX_ROUND_SAMPLES } from "./segment";

export const PAIRED_PROTOCOL_VERSION = 1;
export const PAIRED_ROUNDS = 3;
export const MAX_SESSION_SAMPLES = 576_000;

/** A protocol rule or a malformed server response ended the session. */
export class PairedProtocolError extends Error {
  constructor(
    readonly reason: string,
    readonly status?: number,
    readonly retryAfterSecs?: number,
  ) {
    super(reason);
    this.name = "PairedProtocolError";
  }
}

export interface PairedReveal {
  roundIndex: number;
  roundNonce: Uint8Array;
  word: string;
  pathTarget: Uint8Array;
  waypoints: GridPoint[];
  challengeDigest: Digest;
  /** When the round expires, on the local clock. */
  expiresAtMs: number;
}

export interface PairedSessionOpen {
  sessionId: string;
  sessionNonce: Uint8Array;
  attemptBinding: Digest;
  sessionExpiryUnixMs: number;
  /** When the session expires, on the local clock. */
  expiresAtMs: number;
  reveal: PairedReveal;
}

type Json = Record<string, unknown>;

const isRecord = (value: unknown): value is Json =>
  value !== null && typeof value === "object" && !Array.isArray(value);

function field<T>(record: Json, key: string, check: (value: unknown) => value is T): T {
  const value = record[key];
  if (!check(value)) throw new PairedProtocolError("malformed_response");
  return value;
}

const isString = (value: unknown): value is string => typeof value === "string";
const isCount = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= 0;

function hexField(record: Json, key: string, bytes: number): Uint8Array {
  const value = fromHex(field(record, key, isString), bytes);
  if (!value) throw new PairedProtocolError("malformed_response");
  return value;
}

/**
 * The refusal a non-success response carries. A response without a reason is
 * named by its status: a relayer without paired sessions, a transient fault,
 * or a response this client cannot read.
 */
export function refusalOf(response: PostJsonResponse): PairedProtocolError {
  const body = response.body;
  const header = Number(response.header("retry-after"));
  const retryAfter =
    typeof body.retry_after === "number" && body.retry_after > 0
      ? body.retry_after
      : Number.isFinite(header) && header > 0
        ? header
        : undefined;
  const reason =
    typeof body.reason === "string"
      ? body.reason
      : response.status === 404
        ? "unsupported_session"
        : transient(response.status)
          ? "validation_unavailable"
          : "malformed_response";
  return new PairedProtocolError(reason, response.status, retryAfter);
}

/** A status worth sending the same request again for. */
const transient = (status: number) => status === 408 || status === 429 || status >= 500;

/**
 * Reads a reveal and checks its digest against the session nonce. Its expiry
 * never runs past the session's end.
 */
export function parseReveal(
  value: unknown,
  sessionNonce: Uint8Array,
  nowMs: number,
  sessionEndsAtMs: number,
): PairedReveal {
  if (!isRecord(value)) throw new PairedProtocolError("malformed_response");
  const roundIndex = field(value, "round_index", isCount);
  const roundNonce = hexField(value, "round_nonce", 32);
  const word = field(value, "word", isString);
  if (!/^[a-z]{1,32}$/.test(word)) throw new PairedProtocolError("malformed_response");
  const pathTarget = fromHex(field(value, "path_target_hex", isString));
  if (!pathTarget) throw new PairedProtocolError("malformed_response");
  let waypoints: GridPoint[];
  try {
    waypoints = decodePathTarget(pathTarget);
  } catch {
    throw new PairedProtocolError("malformed_response");
  }
  const digest = hexField(value, "challenge_digest", 32);
  if (!equalBytes(digest, challengeDigest(sessionNonce, roundIndex, roundNonce, word, pathTarget))) {
    throw new PairedProtocolError("challenge_mismatch");
  }
  return {
    roundIndex,
    roundNonce,
    word,
    pathTarget,
    waypoints,
    challengeDigest: digest,
    expiresAtMs: Math.min(nowMs + field(value, "expires_in_ms", isCount), sessionEndsAtMs),
  };
}

/** Reads an open response and refuses anything outside protocol version 1. */
export function parseOpenResponse(value: unknown, nowMs: number): PairedSessionOpen {
  if (!isRecord(value)) throw new PairedProtocolError("malformed_response");
  if (
    value.protocol !== "paired" ||
    value.protocol_version !== PAIRED_PROTOCOL_VERSION ||
    value.rounds !== PAIRED_ROUNDS ||
    value.tier !== "trace" ||
    value.audio_format !== AUDIO_FORMAT
  ) {
    throw new PairedProtocolError("unsupported_session");
  }
  const bounds = value.bounds;
  if (
    !isRecord(bounds) ||
    bounds.max_round_samples !== MAX_ROUND_SAMPLES ||
    bounds.max_session_samples !== MAX_SESSION_SAMPLES ||
    bounds.min_path_points !== MIN_PATH_POINTS ||
    bounds.max_path_points !== MAX_PATH_POINTS
  ) {
    throw new PairedProtocolError("unsupported_session");
  }
  const sessionId = field(value, "session_id", isString);
  if (!/^[0-9a-f]{32}$/.test(sessionId)) throw new PairedProtocolError("malformed_response");
  const sessionNonce = hexField(value, "session_nonce", 32);
  const expiresAtMs = nowMs + field(value, "expires_in_ms", isCount);
  const reveal = parseReveal(value.reveal, sessionNonce, nowMs, expiresAtMs);
  if (reveal.roundIndex !== 1) throw new PairedProtocolError("malformed_response");
  return {
    sessionId,
    sessionNonce,
    attemptBinding: hexField(value, "attempt_binding", 32),
    sessionExpiryUnixMs: field(value, "session_expiry_unix_ms", isCount),
    expiresAtMs,
    reveal,
  };
}

/** `C_0`, the commitment round 1 chains from. */
export function initialCommitment(open: PairedSessionOpen): Digest {
  return sessionCommitment(
    open.sessionNonce,
    open.attemptBinding,
    PAIRED_ROUNDS,
    open.sessionExpiryUnixMs,
  );
}

/** One round's committed evidence, kept for the finalize request. */
export interface CommittedRound {
  roundIndex: number;
  segment: Uint8Array;
  coarsePath: Uint8Array;
  pointCount: number;
  audioDigest: Digest;
  pathDigest: Digest;
  commitment: Digest;
}

export interface CommitInput {
  open: PairedSessionOpen;
  reveal: PairedReveal;
  walletId: string;
  previousCommitment: Digest;
  segment: Uint8Array;
  coarsePath: Uint8Array;
  pointCount: number;
  idempotencyKey: Uint8Array;
}

export function buildCommit(input: CommitInput): { body: Json; round: CommittedRound } {
  const { open, reveal } = input;
  const audio = audioDigest(
    open.sessionNonce,
    reveal.roundIndex,
    reveal.challengeDigest,
    AUDIO_FORMAT,
    input.segment,
  );
  const path = pathDigest(open.sessionNonce, reveal.roundIndex, reveal.challengeDigest, input.coarsePath);
  const commitment = roundCommitment({
    sessionNonce: open.sessionNonce,
    roundIndex: reveal.roundIndex,
    roundNonce: reveal.roundNonce,
    challenge: reveal.challengeDigest,
    previous: input.previousCommitment,
    audioFormat: AUDIO_FORMAT,
    audioByteLength: input.segment.length,
    audio,
    pathPointCount: input.pointCount,
    path,
  });
  return {
    body: {
      wallet_id: input.walletId,
      session_id: open.sessionId,
      round_index: reveal.roundIndex,
      round_nonce: toHex(reveal.roundNonce),
      challenge_digest: toHex(reveal.challengeDigest),
      previous_commitment: toHex(input.previousCommitment),
      audio_format: AUDIO_FORMAT,
      audio_byte_length: input.segment.length,
      audio_digest: toHex(audio),
      path_point_count: input.pointCount,
      path_digest: toHex(path),
      commitment: toHex(commitment),
      idempotency_key: toHex(input.idempotencyKey),
    },
    round: {
      roundIndex: reveal.roundIndex,
      segment: input.segment,
      coarsePath: input.coarsePath,
      pointCount: input.pointCount,
      audioDigest: audio,
      pathDigest: path,
      commitment,
    },
  };
}

export interface CommitAccepted {
  /** The next round, or null once the last round is committed. */
  next: PairedReveal | null;
  /** When the session ends unless the client acts, on the local clock. */
  sessionEndsAtMs: number;
}

/**
 * Reads a commit response and checks it names this round, this commitment and
 * the state that follows this round: another reveal until the last round, then
 * none.
 */
export function parseCommitResponse(
  value: unknown,
  round: CommittedRound,
  sessionNonce: Uint8Array,
  nowMs: number,
): CommitAccepted {
  if (!isRecord(value)) throw new PairedProtocolError("malformed_response");
  const last = round.roundIndex === PAIRED_ROUNDS;
  if (value.state !== (last ? "ready_to_finalize" : "awaiting_commit")) {
    throw new PairedProtocolError("malformed_response");
  }
  if (field(value, "accepted_round", isCount) !== round.roundIndex) {
    throw new PairedProtocolError("malformed_response");
  }
  if (!equalBytes(hexField(value, "commitment", 32), round.commitment)) {
    throw new PairedProtocolError("commitment_mismatch");
  }
  const sessionEndsAtMs = nowMs + field(value, "session_expires_in_ms", isCount);
  const next = last ? null : parseReveal(value.reveal, sessionNonce, nowMs, sessionEndsAtMs);
  if (next && next.roundIndex !== round.roundIndex + 1) {
    throw new PairedProtocolError("malformed_response");
  }
  return { next, sessionEndsAtMs };
}

export interface RetryClock {
  now(): number;
  sleep(ms: number): Promise<void>;
}

/** The wait before the next attempt: 250 ms, doubling to 4 s. */
export function backoffMs(attempt: number): number {
  return Math.min(250 * 2 ** attempt, 4_000);
}

/** What one attempt produced: a value, or the refusal to report if no retry lands. */
export type Attempt<T> = { value: T } | { retry: PairedProtocolError; waitMs?: number };

/**
 * Runs `attempt` until it produces a value or no retry fits before
 * `deadlineMs`. An attempt that throws ends the run with that error.
 */
export async function retryUntil<T>(
  attempt: () => Promise<Attempt<T>>,
  deadlineMs: number,
  clock: RetryClock,
): Promise<T> {
  for (let count = 0; ; count++) {
    const result = await attempt();
    if ("value" in result) return result.value;
    const wait = Math.max(backoffMs(count), result.waitMs ?? 0);
    if (clock.now() + wait >= deadlineMs) throw result.retry;
    await clock.sleep(wait);
  }
}

/**
 * Sends one commit until it lands or its round can no longer succeed. Every
 * attempt carries the same body and the same idempotency key, so a retry after
 * a lost response returns the stored result instead of a second commit.
 *
 * A network error, 408, 429 or any 5xx retries with backoff. Any other
 * rejection ends the round with its reason. A round whose retries run out
 * reports the service as unavailable, since no refusal ever arrived.
 */
export function commitWithRetry(
  post: (payload: Json) => Promise<PostJsonResponse>,
  payload: Json,
  deadlineMs: number,
  clock: RetryClock,
): Promise<Json> {
  return retryUntil<Json>(
    async () => {
      let response: PostJsonResponse;
      try {
        response = await post(payload);
      } catch (error) {
        // A refusal raised before the request went out ends the round. A failed request is retried.
        if (error instanceof PairedProtocolError) throw error;
        return { retry: new PairedProtocolError("validation_unavailable") };
      }
      if (response.status >= 200 && response.status < 300) return { value: response.body };
      const refusal = refusalOf(response);
      if (!transient(response.status)) throw refusal;
      return {
        retry: new PairedProtocolError("validation_unavailable", response.status),
        waitMs: (refusal.retryAfterSecs ?? 0) * 1_000,
      };
    },
    deadlineMs,
    clock,
  );
}

/**
 * When to send a finalize again, or null to stop. A resend is safe: the
 * server consumes a session once, so a finalize that reached it is refused on
 * resend and never judged twice. A busy relayer, a stalled upload and an
 * unreachable validator are resent. A verdict, a refusal and a fault after
 * consumption are not. `response` is null when no response arrived.
 */
export function finalizeRetryAfterMs(response: PostJsonResponse | null, attempt: number): number | null {
  if (response === null || response.status === 408) return backoffMs(attempt);
  if (response.status < 500 || response.body.reason === "technical_failure") return null;
  const refusal = refusalOf(response);
  return Math.max(backoffMs(attempt), (refusal.retryAfterSecs ?? 0) * 1_000);
}

/** The final digest over every committed round, in order. */
export function sessionFinalDigest(open: PairedSessionOpen, rounds: readonly CommittedRound[]): Digest {
  const last = rounds[rounds.length - 1];
  if (!last || rounds.length !== PAIRED_ROUNDS) throw new PairedProtocolError("session_not_ready");
  return finalDigest(
    open.sessionNonce,
    last.commitment,
    PAIRED_ROUNDS,
    evidenceManifest(rounds.map((round) => ({ audio: round.audioDigest, path: round.pathDigest }))),
  );
}

export interface FinalizeInput {
  open: PairedSessionOpen;
  walletId: string;
  rounds: readonly CommittedRound[];
  features: number[];
  f0Contour?: number[];
  accelMagnitude?: number[];
  captureTiming?: unknown;
  clientSignals?: unknown;
  baselineReset: boolean;
}

export function buildFinalizeBody(input: FinalizeInput): Json {
  const rounds = [...input.rounds].sort((left, right) => left.roundIndex - right.roundIndex);
  const body: Json = {
    capture_protocol: "paired",
    wallet_id: input.walletId,
    projection_version: 1,
    session_id: input.open.sessionId,
    final_digest: toHex(sessionFinalDigest(input.open, rounds)),
    segments: rounds.map((round) => ({
      round_index: round.roundIndex,
      audio_b64: bytesToBase64(round.segment),
      coarse_path_hex: toHex(round.coarsePath),
    })),
    features: input.features,
    baseline_reset: input.baselineReset,
  };
  if (input.f0Contour) body.f0_contour = input.f0Contour;
  if (input.accelMagnitude) body.accel_magnitude = input.accelMagnitude;
  if (input.captureTiming !== undefined) body.capture_timing = input.captureTiming;
  if (input.clientSignals !== undefined) body.client_signals = input.clientSignals;
  return body;
}

export interface FinalizeBinding {
  /** The transition the receipt must authorise. Absent for an update, which needs none. */
  purpose?: "mint" | "rebaseline" | "reset";
  wallet: Uint8Array;
  finalDigest: Digest;
}

const RECEIPT_PURPOSES = { mint: 1, rebaseline: 2, reset: 3 } as const;
const RECEIPT_MISMATCH = "The validator's receipt does not match this session.";

/**
 * Checks a finalize success body before anything asks the wallet to sign.
 * Returns the tier the receipt signs, or a message refusing the body. A
 * transition needs a version 3 receipt that binds this session, this wallet,
 * the transition and the commitment, and the commitment's salt.
 */
export function checkFinalizeSuccess(
  body: Json,
  binding: FinalizeBinding,
): { assuranceTier?: number } | string {
  const receipt = body.signed_receipt;
  if (receipt === undefined) {
    return binding.purpose ? "The validator returned no receipt for this session." : {};
  }
  if (
    !binding.purpose ||
    !isRecord(receipt) ||
    !isString(receipt.validator_pubkey_hex) ||
    !isString(receipt.message_hex) ||
    !isString(receipt.signature_hex)
  ) {
    return RECEIPT_MISMATCH;
  }
  const dto = receipt as unknown as SignedReceiptDto;
  const decoded = decodeSignedReceipt(dto);
  const commitment = isString(body.commitment_hex) ? fromHex(body.commitment_hex, 32) : null;
  const salt = isString(body.salt_hex) ? fromHex(body.salt_hex, 32) : null;
  if (
    !decoded ||
    decoded.version !== 3 ||
    decoded.assuranceTier === null ||
    !commitment ||
    !salt ||
    !receiptMatchesBinding(dto, {
      purpose: RECEIPT_PURPOSES[binding.purpose],
      projectionVersion: 1,
      wallet: binding.wallet,
      commitment,
      finalDigest: binding.finalDigest,
    })
  ) {
    return RECEIPT_MISMATCH;
  }
  return { assuranceTier: decoded.assuranceTier };
}
