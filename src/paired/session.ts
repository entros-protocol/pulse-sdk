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
  buildCueRequest,
  parseCueResponse,
  PAIRED_PROTOCOL_VERSION,
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
import { joinSegments, MAX_ROUND_SAMPLES, type SampleRange } from "./segment";
import { createRoundTracker, FRAME_SAMPLES, WAYPOINT_REACH } from "./tracker";
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
  expiresAtMs: number;
}

export type PairedPhase =
  | "idle"
  | "opening"
  | "round"
  | "cue_loading"
  | "cue"
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
  onCue?: (cue: {roundIndex: number; point: GridPoint; expiresAtMs: number}) => void;
  /** The controller's coarse-outline check permits the manual speech fallback. */
  onContinueAvailable?: (available: boolean) => void;
  /** The session moved to another phase. */
  onPhase?: (phase: PairedPhase) => void;
  /** The round has waited long enough that the interface may offer to continue. */
  onStall?: (roundIndex: number) => void;
  /** Amplitude and tracker speech activity for each 50 ms frame. Activity is false outside a round. */
  onLevel?: (rms: number, speechActive: boolean) => void;
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

function retryClock(signal: AbortSignal): RetryClock {
  return {
    now: () => performance.now(),
    sleep: (ms) => new Promise<void>((resolve, reject) => {
      if (signal.aborted) { reject(new PairedProtocolError("technical_failure")); return; }
      const onAbort = () => { clearTimeout(timer); reject(new PairedProtocolError("technical_failure")); };
      const timer = setTimeout(() => {signal.removeEventListener("abort",onAbort);resolve();},ms);
      signal.addEventListener("abort",onAbort,{once:true});
    }),
  };
}

function randomBytes(length: number): Uint8Array {
  const out = new Uint8Array(length);
  crypto.getRandomValues(out);
  return out;
}

export class PairedSession {
  private phase: PairedPhase = "idle";
  private processingEvidence = false;
  private recorder: PairedRecorder | null = null;
  private readonly tracker = createRoundTracker();
  private open: PairedSessionOpen | null = null;
  private reveal: PairedReveal | null = null;
  private previous: Uint8Array | null = null;
  private readonly committed: CommittedRound[] = [];
  /** When the session ends unless the client acts, on the local clock, as the server last said. */
  private sessionEndsAtMs = 0;
  private roundStart = 0;
  private roundStartedAtMs = 0;
  private trackerStart = 0;
  private roundTrace: TracePoint[] = [];
  private pendingReaches: { timeMs: number; point: GridPoint }[] = [];
  private stalled = false;
  private continueAvailable = false;
  private cuePoint: GridPoint | null = null;
  private cueRevealedAtMs = 0;
  private cueReached = false;
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

  /** Local capture readiness. Reading this does not finish or submit the round. */
  get currentRoundStatus(): { speechReady: boolean; traceReady: boolean } | null {
    if (!["round","cue_loading","cue"].includes(this.phase)) return null;
    return {
      speechReady: this.tracker.speechReady(),
      traceReady: this.completedOutline() !== null,
    };
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
    this.discardAfterRelease();
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
      // `abort()` may have run while the microphone started, before there was one to stop.
      if (this.closed) return void (await recorder.stop());
      this.recorder = recorder;

      await this.awaitRecorder(recorder);
      if (this.closed) return;
      const policy = await this.pipeline.readProjectionPolicy(connection);
      if (this.closed) return;
      if (policy.current !== 1) throw new PairedProtocolError("projection_not_supported");

      const response = await this.openWithRetry(wallet);
      if (this.closed) return;
      const opened = parseOpenResponse(response.body, response.startedAtMs);
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

  private openWithRetry(wallet: string): Promise<{body: unknown; startedAtMs: number}> {
    return retryUntil<{body: unknown; startedAtMs: number}>(
      async () => {
        try {
          const startedAtMs = performance.now();
          return { value: {body: await this.openSession(wallet), startedAtMs} };
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
      retryClock(this.requests.signal),
    );
  }

  private async openSession(wallet: string): Promise<unknown> {
    if (this.options.openSession) return this.options.openSession(wallet);
    const response = await this.post("/challenge/paired", { wallet, tier: "trace", protocol_version: PAIRED_PROTOCOL_VERSION }, OPEN_REQUEST_MS);
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

  private awaitRecorder(recorder: PairedRecorder): Promise<void> {
    return new Promise((resolve,reject) => {
      const signal=this.requests.signal;
      const cleanup=() => {clearTimeout(timer);signal.removeEventListener("abort",cancel);};
      const cancel=() => {cleanup();reject(new PairedProtocolError("technical_failure"));};
      const timer=setTimeout(cancel,OPEN_REQUEST_MS);
      signal.addEventListener("abort",cancel,{once:true});
      if (signal.aborted) {cancel();return;}
      recorder.ready.then(() => {cleanup();resolve();},error => {cleanup();reject(error);});
    });
  }

  private beginRound(reveal: PairedReveal): void {
    if (this.closed || !this.recorder) return;
    if (performance.now() >= reveal.expiresAtMs) {this.fail(new PairedProtocolError("round_expired"));return;}
    this.reveal = reveal;
    this.roundTrace = [];
    this.pendingReaches = [];
    this.stalled = false;
    this.cuePoint = null;
    this.cueReached = false;
    this.continueAvailable = false;
    this.options.onContinueAvailable?.(false);
    this.roundStart = this.recorder.markNow();
    this.roundStartedAtMs = this.recorder.timeAt(this.roundStart);
    this.trackerStart = Math.ceil(this.roundStart / FRAME_SAMPLES) * FRAME_SAMPLES;
    this.recorder.releaseBefore(this.roundStart);
    if (reveal.roundIndex === 1) this.windowStartMs = this.recorder.timeAt(this.roundStart);
    this.tracker.begin(reveal.waypoints,true);
    this.arm(reveal.expiresAtMs,"round_expired");
    this.setPhase("round");
    this.options.onReveal?.({roundIndex:reveal.roundIndex,rounds:PAIRED_ROUNDS,word:reveal.word,waypoints:reveal.waypoints,expiresAtMs:reveal.expiresAtMs});
  }

  private onFrame(level: number, endSample: number): void {
    if (this.closed) return;
    const active = this.phase === "round" || this.phase === "cue_loading" || this.phase === "cue";
    if (active && this.recorder) {
      this.roundStart = this.recorder.sampleIndexAt(this.roundStartedAtMs);
      const start = Math.ceil(this.roundStart / FRAME_SAMPLES) * FRAME_SAMPLES;
      if (this.phase === "round" && start > this.trackerStart) this.tracker.discardPrefix((start - this.trackerStart) / FRAME_SAMPLES);
      this.trackerStart = start;
    }
    if (active && endSample - this.roundStart > MAX_ROUND_SAMPLES) {this.fail(new PairedProtocolError("evidence_bounds_invalid"));return;}
    if (active && this.reveal && performance.now() >= this.reveal.expiresAtMs) {this.fail(new PairedProtocolError("round_expired"));return;}
    if (!active || this.phase === "cue_loading" || endSample <= this.trackerStart) {
      this.tracker.observe(level);
      this.options.onLevel?.(level,false);
      return;
    }
    const due = this.pendingReaches.filter(reach => this.recorder!.sampleIndexAt(reach.timeMs) <= endSample);
    this.pendingReaches = this.pendingReaches.filter(reach => this.recorder!.sampleIndexAt(reach.timeMs) > endSample);
    if (this.phase === "cue") {
      const point = this.cuePoint;
      if (point && due.some(reach => reach.timeMs >= this.cueRevealedAtMs && Math.hypot(reach.point.x-point.x,reach.point.y-point.y) <= WAYPOINT_REACH)) this.cueReached=true;
      this.options.onLevel?.(level,false);
      if (this.closed) return;
      const outline=this.cueReached ? this.completedOutline() : null;
      if (outline) this.finishRound(endSample,outline);
      return;
    }
    for (const reach of due) this.tracker.reach(reach.point);
    const progress=this.tracker.frame(level);
    this.options.onLevel?.(level,this.tracker.speechActive());
    if (this.closed || this.phase !== "round") return;
    const outline=this.completedOutline();
    const available=outline !== null;
    if (available !== this.continueAvailable) {this.continueAvailable=available;this.options.onContinueAvailable?.(available);}
    if (outline && this.tracker.speechReady()) {void this.requestCue().catch(error => this.fail(error));}
    else if (progress === "stalled" && !this.stalled) {this.stalled=true;this.options.onStall?.(this.reveal?.roundIndex ?? 0);}
  }

  /** Requests the cue after the visible outline passes. The person must speak first. */
  continueRound(): boolean {
    if (this.phase !== "round" || !this.recorder || !this.reveal || performance.now() >= this.reveal.expiresAtMs || !this.completedOutline()) return false;
    void this.requestCue().catch(error => this.fail(error));
    return true;
  }

  private async requestCue(): Promise<void> {
    const {open,reveal,recorder}=this;
    if (this.phase !== "round" || !open || !reveal || !recorder) return;
    this.setPhase("cue_loading");
    this.continueAvailable=false;
    this.options.onContinueAvailable?.(false);
    this.pendingReaches=[];
    let startedAtMs=performance.now();
    const payload=await commitWithRetry(json => {
      startedAtMs=performance.now();
      return this.post("/paired/cue",json,Math.min(COMMIT_REQUEST_MS,reveal.expiresAtMs-startedAtMs));
    },buildCueRequest(open,reveal,this.walletId),reveal.expiresAtMs,retryClock(this.requests.signal));
    if (this.closed || this.reveal !== reveal) return;
    const cue=parseCueResponse(payload,open,reveal,startedAtMs);
    if (performance.now() >= cue.expiresAtMs) throw new PairedProtocolError("round_expired");
    this.cuePoint=cue.point;
    this.cueRevealedAtMs=recorder.timeAt(recorder.markNow());
    this.reveal={...reveal,waypoints:[...reveal.waypoints,cue.point],expiresAtMs:cue.expiresAtMs};
    this.arm(cue.expiresAtMs,"round_expired");
    this.setPhase("cue");
    this.options.onCue?.({roundIndex:reveal.roundIndex,point:cue.point,expiresAtMs:cue.expiresAtMs});
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
    if (this.phase !== "cue" || mark <= this.roundStart || mark - this.roundStart > MAX_ROUND_SAMPLES) {
      this.fail(new PairedProtocolError("evidence_bounds_invalid"));return;
    }
    const endedAtMs = this.recorder!.timeAt(mark);
    this.setPhase("committing");
    // Leave the audio callback before hashing and posting.
    setTimeout(() => {
      void this.commitRound(endedAtMs, outline).catch((error: unknown) => this.fail(error));
    }, 0);
  }

  private async commitRound(endedAtMs: number, outline: GridPoint[]): Promise<void> {
    const { recorder, open, reveal, previous } = this;
    if (this.closed) return;
    if (!recorder || !open || !reveal || !previous) throw new PairedProtocolError("technical_failure");
    const mark = recorder.sampleIndexAt(endedAtMs);
    const window: SampleRange = {start:recorder.sampleIndexAt(this.roundStartedAtMs),end:mark};
    if (window.start >= mark || mark - window.start > MAX_ROUND_SAMPLES) throw new PairedProtocolError("evidence_bounds_invalid");
    const segment = encodePcm16(recorder.slice(window.start, window.end));
    if (reveal.roundIndex === 1) this.windowStartMs = recorder.timeAt(window.start);
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
    // Committed audio is copied above. Inter-round waiting never enters the next segment.
    recorder.releaseBefore(mark);
    this.windowEndMs = recorder.timeAt(mark);

    let startedAtMs=performance.now();
    let payload: unknown;
    try {
    payload = await commitWithRetry(
      (json) => {startedAtMs=performance.now();return this.post("/paired/commit", json, Math.min(COMMIT_REQUEST_MS,reveal.expiresAtMs-startedAtMs));},
      body,
      reveal.expiresAtMs,
      retryClock(this.requests.signal),
    );
    } finally {
      if (this.closed) round.segment.fill(0);
    }
    if (this.closed) return;
    const accepted = parseCommitResponse(payload, round, open.sessionNonce, startedAtMs);
    this.committed.push(round);
    this.previous = round.commitment;
    this.sessionEndsAtMs = Math.min(this.sessionEndsAtMs,accepted.sessionEndsAtMs);
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
      if ((this.phase !== "round" && this.phase !== "cue") || !this.recorder) return;
      const rect = surface.getBoundingClientRect();
      const local = { x: event.clientX - rect.left, y: event.clientY - rect.top, t: now };
      this.roundTrace.push(local);
      this.pendingReaches.push({
        timeMs: now,
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

  private discardEvidence(): void {
    for (const round of this.committed) round.segment.fill(0);
    this.committed.length = 0;
    this.touch.length = 0;
    this.roundTrace = [];
    this.pendingReaches = [];
    this.motionPromise = null;
    this.motionController = null;
    this.released = null;
    this.recorder = null;
    this.surface = null;
  }

  private discardAfterRelease(): void {
    if (this.processingEvidence) return;
    void this.release().then(motion => { motion.length = 0; }).finally(() => this.discardEvidence());
  }

  /** Stops every sensor once, the microphone first, and returns the motion recorded. */
  private release(): Promise<MotionSample[]> {
    this.released ??= (async () => {
      this.detachPointer?.();
      await this.recorder?.stop().catch(() => undefined);
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
    this.discardAfterRelease();
  }

  private guardedWallet(wallet: unknown): unknown {
    if (wallet === null || typeof wallet !== "object") return wallet;
    const check = () => {
      if (this.requests.signal.aborted || this.closed) throw new PairedProtocolError("technical_failure");
      if (performance.now() >= this.sessionEndsAtMs) throw new PairedProtocolError("session_expired");
      const key: unknown = Reflect.get(wallet,"publicKey",wallet);
      if (key === null || typeof key !== "object") throw new PairedProtocolError("invalid_request");
      const encode: unknown = Reflect.get(key,"toBase58",key);
      if (typeof encode !== "function" || Reflect.apply(encode,key,[]) !== this.walletId) throw new PairedProtocolError("invalid_request");
    };
    const signing = new Set(["signMessage","signTransaction","signAllTransactions","sendTransaction"]);
    return new Proxy(wallet, {
      get: (target,property) => {
        const value: unknown = Reflect.get(target,property,target);
        if (typeof value !== "function") return value;
        if (!signing.has(String(property))) return value.bind(target);
        return async (...args: unknown[]) => {
          check();
          const result: unknown = await Reflect.apply(value,target,args);
          if (property !== "sendTransaction") check();
          return result;
        };
      },
    });
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
    this.processingEvidence = true;
    let motion: MotionSample[] = [];
    let joined: Float32Array = new Float32Array(0);
    let samples: Float32Array = new Float32Array(0);
    try {
    motion = await this.release();
    if (this.closed) return failure("technical_failure","The verification was cancelled.");
    const recorder = this.recorder;
    joined = joinSegments(this.committed.map((round) => round.segment));
    samples = normalizeCaptureRMS(joined);
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
      signal: this.requests.signal,
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
    if (this.closed) return failure("technical_failure","The verification was cancelled.");
    const protectedWallet = this.guardedWallet(wallet);
    const result = reset
      ? await this.pipeline.processReset(sensorData, protectedWallet, connection, onProgress, policy, target)
      : await this.pipeline.process(sensorData, protectedWallet, connection, onProgress, policy, target);
    if (this.closed) return failure("technical_failure", "The verification was cancelled.");
    this.setPhase(result.success ? "done" : "failed");
    return { ...result, assuranceTier };
    } catch (error) {
      throw this.fail(error,false);
    } finally {
      joined.fill(0);
      samples.fill(0);
      motion.length = 0;
      this.processingEvidence = false;
      this.discardEvidence();
    }
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
