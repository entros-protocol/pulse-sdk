/**
 * A paired verification: three rounds of one word and one short path, then one
 * finalize.
 *
 * The server reveals round k + 1 only after it accepts the commitment for
 * round k. That shows the client fixed its round evidence before it saw the
 * next challenge. It does not prove capture time, sensor origin, human
 * presence or physiological coupling.
 *
 * Recording runs without a break until the last round is committed. A round
 * ends when the tracker hears speech and the trace reaches every waypoint in
 * order, and its segment is final at that moment: the window is cut from the
 * canonical stream, encoded, hashed and committed. Features are extracted from
 * the committed segments joined and levelled once.
 */

import { canonicalKeyBytes } from "../agent/base58";
import type { ProjectionPolicy } from "../identity/anchor";
import { describeInputLevel, normalizeCaptureRMS } from "../sensor/audio";
import { encodePcm16 } from "../sensor/encode";
import { captureMotion, requestMotionPermission } from "../sensor/motion";
import type { MotionSample, SensorData, TouchSample } from "../sensor/types";
import type { VerificationResult } from "../submit/types";
import { postJson, type PostJsonResponse } from "../transport/post-json";
import type { ValidationTarget } from "../pulse";
import {
  buildCommit,
  buildFinalizeBody,
  checkFinalizeSuccess,
  commitWithRetry,
  type CommittedRound,
  finalizeRetryAfterMs,
  initialCommitment,
  PAIRED_ROUNDS,
  PairedProtocolError,
  type PairedReveal,
  type PairedSessionOpen,
  parseCommitResponse,
  parseOpenResponse,
  refusalOf,
  type RetryClock,
  retryUntil,
  sessionFinalDigest,
} from "./client";
import { CoarsePathError, scorePath, toCoarsePath, toGridPoint, type TracePoint } from "./coarse-path";
import { type PairedRecorder, startPairedRecorder } from "./recorder";
import { joinSegments, roundWindow, runsToSamples, type SampleRange } from "./segment";
import { createRoundTracker } from "./tracker";
import { encodeCoarsePath, type GridPoint } from "./transcript";

/** A session lives ten minutes. Motion capture never needs to outlast it. */
const SESSION_MOTION_BOUND_MS = 600_000;
/** Opening a session keeps retrying a validator outage for this long. */
const OPEN_RETRY_MS = 15_000;
/** One open request may run this long. */
const OPEN_REQUEST_MS = 15_000;
/** One commit request may run this long. */
const COMMIT_REQUEST_MS = 10_000;

/** What the interface shows for one round. */
export interface PairedRoundView {
  roundIndex: number;
  rounds: number;
  word: string;
  /** Waypoints on the 0 to 1000 grid of the trace surface, to be traced in order. */
  waypoints: GridPoint[];
}

export type PairedPhase =
  | "idle"
  | "opening"
  | "round"
  | "committing"
  | "ready"
  | "finalizing"
  | "done"
  | "failed";

export interface PairedSessionOptions {
  /**
   * Opens the session and returns the executor's JSON for an accepted open.
   * The default posts to the relayer's `/challenge/paired`.
   *
   * Report a refusal by throwing a `PairedProtocolError` with the server's
   * reason, status and `retry_after`. A 408, 429 or 5xx status, or any error
   * that is not a `PairedProtocolError`, is retried for up to 15 s.
   */
  openSession?: (wallet: string) => Promise<unknown>;
  /** A round was revealed. Show its word and path. */
  onReveal?: (round: PairedRoundView) => void;
  /** The session moved to another phase. */
  onPhase?: (phase: PairedPhase) => void;
  /** The round has waited long enough that the interface may offer to continue. */
  onStall?: (roundIndex: number) => void;
  /** One level for each 50 ms frame, for a live meter. */
  onLevel?: (rms: number) => void;
  /**
   * The session cannot continue after `start` resolved. The host starts a new
   * one. A failure during `start` rejects `start` instead, so each failure is
   * reported once.
   */
  onFailure?: (error: PairedProtocolError) => void;
}

export interface PairedPipeline {
  /** The projection policy on chain. Paired sessions run under projection 1 only. */
  readProjectionPolicy(connection?: unknown): Promise<ProjectionPolicy>;
  process(
    sensorData: SensorData,
    wallet: unknown,
    connection: unknown,
    onProgress: ((stage: string) => void) | undefined,
    policy: ProjectionPolicy,
    target: ValidationTarget,
  ): Promise<VerificationResult>;
  processReset(
    sensorData: SensorData,
    wallet: unknown,
    connection: unknown,
    onProgress: ((stage: string) => void) | undefined,
    policy: ProjectionPolicy,
    target: ValidationTarget,
  ): Promise<VerificationResult>;
  relayerUrl?: string;
  relayerApiKey?: string;
}

/** The paired result carries the tier the validator signed into the receipt. */
export type PairedVerificationResult = VerificationResult & { assuranceTier?: number };

const clock: RetryClock = {
  now: () => performance.now(),
  sleep: (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
};

function randomBytes(length: number): Uint8Array {
  const out = new Uint8Array(length);
  crypto.getRandomValues(out);
  return out;
}

export class PairedSession {
  private phase: PairedPhase = "idle";
  private recorder: PairedRecorder | null = null;
  private readonly tracker = createRoundTracker();
  private open: PairedSessionOpen | null = null;
  private reveal: PairedReveal | null = null;
  private previous: Uint8Array | null = null;
  private readonly committed: CommittedRound[] = [];
  /** When the session ends unless the client acts, on the local clock, as the server last said. */
  private sessionEndsAtMs = 0;
  private roundStart = 0;
  private trackerStart = 0;
  private roundTrace: TracePoint[] = [];
  private pendingReaches: { sample: number; point: GridPoint }[] = [];
  private stalled = false;
  private walletId = "";
  private wallet: Uint8Array | null = null;
  private surface: HTMLElement | null = null;
  private detachPointer: (() => void) | null = null;
  private readonly touch: TouchSample[] = [];
  private motionController: AbortController | null = null;
  private motionPromise: Promise<MotionSample[]> | null = null;
  private released: Promise<MotionSample[]> | null = null;
  private windowStartMs = 0;
  private windowEndMs = 0;
  private deadline: ReturnType<typeof setTimeout> | null = null;
  /** Cancels every open and commit request once the session ends. */
  private readonly requests = new AbortController();

  /** @internal Hosts create sessions through `PulseSDK.createPairedSession`. */
  constructor(
    private readonly pipeline: PairedPipeline,
    private readonly options: PairedSessionOptions = {},
  ) {}

  get currentPhase(): PairedPhase {
    return this.phase;
  }

  private get closed(): boolean {
    return this.phase === "failed" || this.phase === "done";
  }

  private setPhase(phase: PairedPhase): void {
    this.phase = phase;
    this.options.onPhase?.(phase);
  }

  private fail(error: unknown, notify = true): PairedProtocolError {
    const failure =
      error instanceof PairedProtocolError ? error : new PairedProtocolError("technical_failure");
    if (this.closed) return failure;
    this.setPhase("failed");
    this.disarm();
    this.requests.abort();
    void this.release();
    if (notify) this.options.onFailure?.(failure);
    return failure;
  }

  /** Ends the session with `reason` at `atMs` on the local clock, unless something ends it first. */
  private arm(atMs: number, reason: "round_expired" | "session_expired"): void {
    this.disarm();
    this.deadline = setTimeout(
      () => this.fail(new PairedProtocolError(reason)),
      Math.max(0, atMs - performance.now()),
    );
  }

  private disarm(): void {
    if (this.deadline !== null) clearTimeout(this.deadline);
    this.deadline = null;
  }

  /**
   * Starts the microphone, motion and the trace surface, opens the session and
   * reveals round 1. Call it from the user gesture that begins verification,
   * because motion permission needs one.
   *
   * Rejects with a `PairedProtocolError` when the session cannot start, or
   * with the browser's `DOMException` when the microphone is refused or
   * missing. After `abort()`, it releases every sensor it had started and
   * resolves without revealing a round.
   */
  async start(wallet: string, surface: HTMLElement, connection?: unknown): Promise<void> {
    if (this.phase !== "idle") throw new Error("This paired session has already started.");
    this.setPhase("opening");
    this.walletId = wallet;
    this.wallet = canonicalKeyBytes(wallet);
    this.surface = surface;
    try {
      if (!this.wallet) throw new PairedProtocolError("invalid_request");
      // Motion permission and the audio context both need the user's gesture,
      // so they come before anything that waits on the network.
      const motionGranted = await requestMotionPermission();
      if (this.closed) return;
      if (motionGranted) {
        this.motionController = new AbortController();
        this.motionPromise = captureMotion({
          signal: this.motionController.signal,
          permissionGranted: true,
          maxDurationMs: SESSION_MOTION_BOUND_MS,
        }).catch(() => []);
      }
      const recorder = await startPairedRecorder((level, endSample) => this.onFrame(level, endSample));
      this.recorder = recorder;
      // `abort()` may have run while the microphone started, before there was one to stop.
      if (this.closed) return void (await recorder.stop());

      const policy = await this.pipeline.readProjectionPolicy(connection);
      if (this.closed) return;
      if (policy.current !== 1) throw new PairedProtocolError("projection_not_supported");

      const response = await this.openWithRetry(wallet);
      if (this.closed) return;
      const opened = parseOpenResponse(response, performance.now());
      this.open = opened;
      this.previous = initialCommitment(opened);
      this.sessionEndsAtMs = opened.expiresAtMs;
      this.attachPointer(surface);
      this.beginRound(opened.reveal);
    } catch (error) {
      if (this.closed) return;
      const failure = this.fail(error, false);
      // A refused or missing microphone keeps its own error, so a host can explain it.
      throw error instanceof DOMException ? error : failure;
    }
  }

  private openWithRetry(wallet: string): Promise<unknown> {
    return retryUntil<unknown>(
      async () => {
        try {
          return { value: await this.openSession(wallet) };
        } catch (error) {
          if (this.closed) throw error;
          if (!(error instanceof PairedProtocolError)) {
            return { retry: new PairedProtocolError("validation_unavailable") };
          }
          const status = error.status;
          if (status !== undefined && status !== 408 && status !== 429 && status < 500) throw error;
          return { retry: error, waitMs: (error.retryAfterSecs ?? 0) * 1_000 };
        }
      },
      performance.now() + OPEN_RETRY_MS,
      clock,
    );
  }

  private async openSession(wallet: string): Promise<unknown> {
    if (this.options.openSession) return this.options.openSession(wallet);
    const response = await this.post("/challenge/paired", { wallet, tier: "trace" }, OPEN_REQUEST_MS);
    if (response.status < 200 || response.status >= 300) throw refusalOf(response);
    return response.body;
  }

  private post(path: string, payload: Record<string, unknown>, deadlineMs: number): Promise<PostJsonResponse> {
    if (this.closed) return Promise.reject(new PairedProtocolError("technical_failure"));
    if (!this.pipeline.relayerUrl) {
      return Promise.reject(new PairedProtocolError("validation_unavailable"));
    }
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.pipeline.relayerApiKey) headers["X-API-Key"] = this.pipeline.relayerApiKey;
    return postJson(`${new URL(this.pipeline.relayerUrl).origin}${path}`, payload, {
      headers,
      deadlineMs,
      signal: this.requests.signal,
    });
  }

  private beginRound(reveal: PairedReveal): void {
    if (this.closed || !this.recorder) return;
    this.reveal = reveal;
    this.roundTrace = [];
    this.pendingReaches = [];
    this.stalled = false;
    this.trackerStart = this.recorder.framedSamples();
    if (reveal.roundIndex === 1) {
      this.roundStart = this.trackerStart;
      this.windowStartMs = this.recorder.timeAt(this.trackerStart);
    }
    this.tracker.begin(reveal.waypoints, true);
    this.arm(
      reveal.expiresAtMs,
      reveal.expiresAtMs < this.sessionEndsAtMs ? "round_expired" : "session_expired",
    );
    this.setPhase("round");
    this.options.onReveal?.({
      roundIndex: reveal.roundIndex,
      rounds: PAIRED_ROUNDS,
      word: reveal.word,
      waypoints: reveal.waypoints,
    });
  }

  private onFrame(level: number, endSample: number): void {
    this.options.onLevel?.(level);
    if (this.phase !== "round") {
      this.tracker.observe(level);
      return;
    }
    const due = this.pendingReaches.filter((reach) => reach.sample <= endSample);
    this.pendingReaches = this.pendingReaches.filter((reach) => reach.sample > endSample);
    for (const reach of due) this.tracker.reach(reach.point);
    const progress = this.tracker.frame(level);
    // The server scores the committed outline, so a round ends only on an
    // outline that passes. Otherwise it waits like a stalled round.
    const outline = progress === "complete" ? this.completedOutline() : null;
    if (outline) {
      this.finishRound(endSample, outline);
    } else if (progress !== "open" && !this.stalled) {
      this.stalled = true;
      this.options.onStall?.(this.reveal?.roundIndex ?? 0);
    }
  }

  /**
   * Ends a stalled round by hand, for speech the tracker did not pick up.
   * Refuses until the trace has reached every waypoint in order, and returns
   * whether the round ended.
   */
  continueRound(): boolean {
    if (this.phase !== "round" || !this.stalled || !this.recorder) return false;
    const outline = this.completedOutline();
    if (!outline) return false;
    this.finishRound(this.recorder.framedSamples(), outline);
    return true;
  }

  /** The round's outline, when it reaches every waypoint in order by the server's rule. */
  private completedOutline(): GridPoint[] | null {
    if (!this.surface || !this.reveal) return null;
    const rect = this.surface.getBoundingClientRect();
    let outline: GridPoint[];
    try {
      outline = toCoarsePath(this.roundTrace, { width: rect.width, height: rect.height });
    } catch (error) {
      if (error instanceof CoarsePathError) return null;
      throw error;
    }
    return scorePath(this.reveal.waypoints, outline).inOrder ? outline : null;
  }

  private finishRound(mark: number, outline: GridPoint[]): void {
    // Fixed at the mark: frames heard after it belong to the next round.
    const window = roundWindow(
      this.roundStart,
      mark,
      runsToSamples(this.trackerStart, this.tracker.runs()),
    );
    this.setPhase("committing");
    // Leave the audio callback before hashing and posting.
    setTimeout(() => {
      void this.commitRound(mark, window, outline).catch((error: unknown) => this.fail(error));
    }, 0);
  }

  private async commitRound(mark: number, window: SampleRange, outline: GridPoint[]): Promise<void> {
    const { recorder, open, reveal, previous } = this;
    if (this.closed) return;
    if (!recorder || !open || !reveal || !previous) throw new PairedProtocolError("technical_failure");
    const segment = encodePcm16(recorder.slice(window.start, window.end));
    const { body, round } = buildCommit({
      open,
      reveal,
      walletId: this.walletId,
      previousCommitment: previous,
      segment,
      coarsePath: encodeCoarsePath(outline),
      pointCount: outline.length,
      idempotencyKey: randomBytes(16),
    });
    // The next round starts at this mark, so everything before it is released.
    recorder.releaseBefore(mark);
    this.roundStart = mark;
    this.windowEndMs = recorder.timeAt(mark);

    const payload = await commitWithRetry(
      (json) => this.post("/paired/commit", json, COMMIT_REQUEST_MS),
      body,
      reveal.expiresAtMs,
      clock,
    );
    if (this.closed) return;
    const accepted = parseCommitResponse(payload, round, open.sessionNonce, performance.now());
    this.committed.push(round);
    this.previous = round.commitment;
    this.sessionEndsAtMs = accepted.sessionEndsAtMs;
    if (accepted.next) {
      this.beginRound(accepted.next);
      return;
    }
    // Every round is committed. Nothing more is recorded, so every sensor stops now.
    void this.release();
    this.arm(this.sessionEndsAtMs, "session_expired");
    this.setPhase("ready");
  }

  private attachPointer(surface: HTMLElement): void {
    const record = (event: PointerEvent) => {
      const now = performance.now();
      this.touch.push({
        timestamp: now,
        x: event.clientX,
        y: event.clientY,
        pressure: event.pressure,
        width: event.width,
        height: event.height,
      });
      if (this.phase !== "round" || !this.recorder) return;
      const rect = surface.getBoundingClientRect();
      const local = { x: event.clientX - rect.left, y: event.clientY - rect.top, t: now };
      this.roundTrace.push(local);
      this.pendingReaches.push({
        sample: this.recorder.sampleIndexAt(now),
        point: toGridPoint(local, { width: rect.width, height: rect.height }),
      });
    };
    // Pressed points only. The state comes from each event, never from a flag a
    // release clears, because a release lost to an unmount would leave the
    // surface drawing on hover.
    const onDown = (event: PointerEvent) => {
      if (event.pointerType === "mouse" && (event.buttons & 1) !== 1) return;
      record(event);
    };
    const onMove = (event: PointerEvent) => {
      if ((event.buttons & 1) !== 1) return;
      record(event);
    };
    surface.addEventListener("pointerdown", onDown);
    surface.addEventListener("pointermove", onMove);
    this.detachPointer = () => {
      surface.removeEventListener("pointerdown", onDown);
      surface.removeEventListener("pointermove", onMove);
      this.detachPointer = null;
    };
  }

  /** Stops every sensor once, the microphone first, and returns the motion recorded. */
  private release(): Promise<MotionSample[]> {
    this.released ??= (async () => {
      this.detachPointer?.();
      await this.recorder?.stop();
      this.motionController?.abort();
      return (await this.motionPromise) ?? [];
    })();
    return this.released;
  }

  /**
   * Stops every sensor and ends the session without a verdict. Safe at any
   * point, including while `start` is still waiting. Once the session is
   * finalizing, the validator's answer is refused before any wallet prompt.
   */
  abort(): void {
    if (this.closed) return;
    this.setPhase("failed");
    this.disarm();
    this.requests.abort();
    void this.release();
  }

  /** Validates the session and runs the mint, update or rebaseline that follows. */
  async complete(
    wallet: unknown,
    connection: unknown,
    onProgress?: (stage: string) => void,
  ): Promise<PairedVerificationResult> {
    return this.finalize(wallet, connection, onProgress, false);
  }

  /** Validates the session as a baseline reset. */
  async completeReset(
    wallet: unknown,
    connection: unknown,
    onProgress?: (stage: string) => void,
  ): Promise<PairedVerificationResult> {
    return this.finalize(wallet, connection, onProgress, true);
  }

  private async finalize(
    wallet: unknown,
    connection: unknown,
    onProgress: ((stage: string) => void) | undefined,
    reset: boolean,
  ): Promise<PairedVerificationResult> {
    const open = this.open;
    const walletBytes = this.wallet;
    if (this.phase !== "ready" || !open || !walletBytes) {
      throw new Error("A paired session completes only after its last round is committed.");
    }
    if (performance.now() >= this.sessionEndsAtMs) {
      this.fail(new PairedProtocolError("session_expired"), false);
      return failure("session_expired", "The verification session expired. Start a new verification.");
    }
    // Spent before the request goes out. The server consumes the session once,
    // so a second finalize would only be refused.
    this.disarm();
    this.setPhase("finalizing");
    const motion = await this.release();
    const recorder = this.recorder;
    const joined = joinSegments(this.committed.map((round) => round.segment));
    const samples = normalizeCaptureRMS(joined);
    const sensorData: SensorData = {
      audio: {
        samples,
        sampleRate: 16_000,
        duration: samples.length / 16_000,
        windowStartMs: this.windowStartMs,
        windowEndMs: this.windowEndMs,
        inputLevel: describeInputLevel(joined),
        virtualDevice: recorder?.virtualDevice ?? false,
        voiceIsolationApplied: recorder?.voiceIsolationApplied ?? null,
      },
      motion,
      touch: this.touch,
      modalities: { audio: true, motion: motion.length > 0, touch: this.touch.length > 0 },
    };

    let assuranceTier: number | undefined;
    const finalDigest = sessionFinalDigest(open, this.committed);
    let receiptPurpose: "mint" | "rebaseline" | "reset" | undefined;
    const target: ValidationTarget = {
      path: "/validate-session",
      deadlineMs: () => Math.max(1, this.sessionEndsAtMs - performance.now()),
      retryAfterMs: finalizeRetryAfterMs,
      body: (evidence) => {
        receiptPurpose = evidence.receiptPurpose;
        return buildFinalizeBody({
          open,
          walletId: evidence.walletAddress,
          rounds: this.committed,
          features: evidence.features,
          f0Contour: evidence.f0Contour,
          accelMagnitude: evidence.accelMagnitude,
          captureTiming: evidence.captureTiming,
          clientSignals: evidence.clientSignals,
          baselineReset: reset,
        });
      },
      accept: (body) => {
        if (this.phase === "failed") return "The verification was cancelled.";
        const checked = checkFinalizeSuccess(body, {
          purpose: receiptPurpose,
          wallet: walletBytes,
          finalDigest,
        });
        if (typeof checked === "string") return checked;
        assuranceTier = checked.assuranceTier;
        return null;
      },
    };

    const policy = await this.pipeline.readProjectionPolicy(connection);
    if (policy.current !== 1) {
      if (!this.closed) this.setPhase("failed");
      return failure(
        "projection_not_supported",
        "The protocol projection changed during the session. Start a new verification.",
      );
    }
    const result = reset
      ? await this.pipeline.processReset(sensorData, wallet, connection, onProgress, policy, target)
      : await this.pipeline.process(sensorData, wallet, connection, onProgress, policy, target);
    if (!this.closed) this.setPhase(result.success ? "done" : "failed");
    return { ...result, assuranceTier };
  }
}

function failure(reason: string, error: string): PairedVerificationResult {
  return {
    success: false,
    commitment: new Uint8Array(32),
    isFirstVerification: false,
    error,
    reason,
    failedAt: "capture",
  };
}
