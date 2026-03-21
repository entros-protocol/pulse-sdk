import type { PulseConfig } from "./config";
import { DEFAULT_THRESHOLD, DEFAULT_CAPTURE_MS } from "./config";
import type { SensorData, AudioCapture, MotionSample, TouchSample, StageState } from "./sensor/types";
import type { TBH } from "./hashing/types";
import type { SolanaProof } from "./proof/types";
import type { VerificationResult } from "./submit/types";
import type { StoredVerificationData } from "./identity/types";

import { captureAudio } from "./sensor/audio";
import { captureMotion } from "./sensor/motion";
import { captureTouch } from "./sensor/touch";
import { extractMFCC } from "./extraction/mfcc";
import {
  extractMotionFeatures,
  extractTouchFeatures,
} from "./extraction/kinematic";
import { fuseFeatures } from "./extraction/statistics";
import { simhash } from "./hashing/simhash";
import { generateTBH } from "./hashing/poseidon";
import { prepareCircuitInput, generateProof } from "./proof/prover";
import { serializeProof } from "./proof/serializer";
import { submitViaWallet } from "./submit/wallet";
import { submitViaRelayer } from "./submit/relayer";
import {
  storeVerificationData,
  loadVerificationData,
} from "./identity/anchor";

type ResolvedConfig = Required<Pick<PulseConfig, "cluster" | "threshold">> &
  PulseConfig;

/**
 * Extract features from sensor data and fuse into a single vector.
 */
function extractFeatures(data: SensorData): number[] {
  const audioFeatures = data.audio
    ? extractMFCC(data.audio)
    : new Array(169).fill(0);
  const motionFeatures = extractMotionFeatures(data.motion);
  const touchFeatures = extractTouchFeatures(data.touch);
  return fuseFeatures(audioFeatures, motionFeatures, touchFeatures);
}

/**
 * Shared pipeline: features → simhash → TBH → proof → submit.
 * Used by both PulseSDK.verify() and PulseSession.complete().
 */
// Minimum sample counts for meaningful feature extraction
const MIN_AUDIO_SAMPLES = 16000; // ~1 second at 16kHz
const MIN_MOTION_SAMPLES = 10;
const MIN_TOUCH_SAMPLES = 10;

async function processSensorData(
  sensorData: SensorData,
  config: ResolvedConfig,
  wallet?: any,
  connection?: any
): Promise<VerificationResult> {
  // Data quality gate: reject if insufficient behavioral data captured
  const audioSamples = sensorData.audio?.samples.length ?? 0;
  const motionSamples = sensorData.motion.length;
  const touchSamples = sensorData.touch.length;

  // Need at least audio OR (motion + touch) to produce a meaningful fingerprint
  const hasAudio = audioSamples >= MIN_AUDIO_SAMPLES;
  const hasMotion = motionSamples >= MIN_MOTION_SAMPLES;
  const hasTouch = touchSamples >= MIN_TOUCH_SAMPLES;

  if (!hasAudio && !hasMotion && !hasTouch) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: true,
      error: "Insufficient behavioral data. Please speak the phrase and trace the curve during capture.",
    };
  }

  if (!hasAudio) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: true,
      error: "No voice data detected. Please speak the phrase clearly during capture.",
    };
  }

  // Extract features
  const features = extractFeatures(sensorData);

  // Generate fingerprint via SimHash
  const fingerprint = simhash(features);

  // Generate TBH (Poseidon commitment)
  const tbh = await generateTBH(fingerprint);

  // Check for previous verification data
  const previousData = loadVerificationData();
  const isFirstVerification = !previousData;

  let solanaProof: SolanaProof | null = null;

  if (!isFirstVerification && previousData) {
    const previousTBH: TBH = {
      fingerprint: previousData.fingerprint,
      salt: BigInt(previousData.salt),
      commitment: BigInt(previousData.commitment),
      commitmentBytes: new Uint8Array(32),
    };

    const circuitInput = prepareCircuitInput(
      tbh,
      previousTBH,
      config.threshold
    );

    const wasmPath = config.wasmUrl ?? "";
    const zkeyPath = config.zkeyUrl ?? "";

    const { proof, publicSignals } = await generateProof(
      circuitInput,
      wasmPath,
      zkeyPath
    );

    solanaProof = serializeProof(proof, publicSignals);
  }

  // Submit
  let submission;

  if (wallet && connection) {
    if (isFirstVerification) {
      submission = await submitViaWallet(
        solanaProof ?? { proofBytes: new Uint8Array(0), publicInputs: [] },
        tbh.commitmentBytes,
        { wallet, connection, isFirstVerification: true }
      );
    } else {
      submission = await submitViaWallet(solanaProof!, tbh.commitmentBytes, {
        wallet,
        connection,
        isFirstVerification: false,
      });
    }
  } else if (config.relayerUrl) {
    submission = await submitViaRelayer(
      solanaProof ?? { proofBytes: new Uint8Array(0), publicInputs: [] },
      tbh.commitmentBytes,
      { relayerUrl: config.relayerUrl, isFirstVerification }
    );
  } else {
    return {
      success: false,
      commitment: tbh.commitmentBytes,
      isFirstVerification,
      error: "No wallet or relayer configured",
    };
  }

  // Store verification data locally for next re-verification
  if (submission.success) {
    storeVerificationData({
      fingerprint: tbh.fingerprint,
      salt: tbh.salt.toString(),
      commitment: tbh.commitment.toString(),
      timestamp: Date.now(),
    });
  }

  return {
    success: submission.success,
    commitment: tbh.commitmentBytes,
    txSignature: submission.txSignature,
    isFirstVerification,
    error: submission.error,
  };
}

/**
 * PulseSession — event-driven staged capture session.
 *
 * Gives the caller control over when each sensor stage starts and stops.
 * After all stages complete, call complete() to run the processing pipeline.
 *
 * Usage:
 *   const session = pulse.createSession(touchElement);
 *   await session.startAudio();
 *   // ... user speaks ...
 *   await session.stopAudio();
 *   await session.startMotion();
 *   // ... user holds device ...
 *   await session.stopMotion();
 *   await session.startTouch();
 *   // ... user traces curve ...
 *   await session.stopTouch();
 *   const result = await session.complete(wallet, connection);
 */
export class PulseSession {
  private config: ResolvedConfig;
  private touchElement: HTMLElement | undefined;

  private audioStageState: StageState = "idle";
  private motionStageState: StageState = "idle";
  private touchStageState: StageState = "idle";

  private audioController: AbortController | null = null;
  private motionController: AbortController | null = null;
  private touchController: AbortController | null = null;

  private audioPromise: Promise<AudioCapture | null> | null = null;
  private motionPromise: Promise<MotionSample[]> | null = null;
  private touchPromise: Promise<TouchSample[]> | null = null;

  private audioData: AudioCapture | null = null;
  private motionData: MotionSample[] = [];
  private touchData: TouchSample[] = [];

  constructor(config: ResolvedConfig, touchElement?: HTMLElement) {
    this.config = config;
    this.touchElement = touchElement;
  }

  // --- Audio ---

  async startAudio(onAudioLevel?: (rms: number) => void): Promise<void> {
    if (this.audioStageState !== "idle")
      throw new Error("Audio capture already started");
    this.audioStageState = "capturing";
    this.audioController = new AbortController();
    this.audioPromise = captureAudio({
      signal: this.audioController.signal,
      onAudioLevel,
    }).catch(() => null);
  }

  async stopAudio(): Promise<AudioCapture | null> {
    if (this.audioStageState !== "capturing")
      throw new Error("Audio capture not active");
    this.audioController!.abort();
    this.audioData = await this.audioPromise!;
    this.audioStageState = "captured";
    return this.audioData;
  }

  skipAudio(): void {
    if (this.audioStageState !== "idle")
      throw new Error("Audio capture already started");
    this.audioStageState = "skipped";
  }

  // --- Motion ---

  async startMotion(): Promise<void> {
    if (this.motionStageState !== "idle")
      throw new Error("Motion capture already started");
    this.motionStageState = "capturing";
    this.motionController = new AbortController();
    this.motionPromise = captureMotion({
      signal: this.motionController.signal,
    }).catch(() => []);
  }

  async stopMotion(): Promise<MotionSample[]> {
    if (this.motionStageState !== "capturing")
      throw new Error("Motion capture not active");
    this.motionController!.abort();
    this.motionData = await this.motionPromise!;
    this.motionStageState = "captured";
    return this.motionData;
  }

  skipMotion(): void {
    if (this.motionStageState !== "idle")
      throw new Error("Motion capture already started");
    this.motionStageState = "skipped";
  }

  // --- Touch ---

  async startTouch(): Promise<void> {
    if (this.touchStageState !== "idle")
      throw new Error("Touch capture already started");
    if (!this.touchElement)
      throw new Error("No touch element provided to session");
    this.touchStageState = "capturing";
    this.touchController = new AbortController();
    this.touchPromise = captureTouch(this.touchElement, {
      signal: this.touchController.signal,
    }).catch(() => []);
  }

  async stopTouch(): Promise<TouchSample[]> {
    if (this.touchStageState !== "capturing")
      throw new Error("Touch capture not active");
    this.touchController!.abort();
    this.touchData = await this.touchPromise!;
    this.touchStageState = "captured";
    return this.touchData;
  }

  skipTouch(): void {
    if (this.touchStageState !== "idle")
      throw new Error("Touch capture already started");
    this.touchStageState = "skipped";
  }

  // --- Complete ---

  async complete(wallet?: any, connection?: any): Promise<VerificationResult> {
    const active: string[] = [];
    if (this.audioStageState === "capturing") active.push("audio");
    if (this.motionStageState === "capturing") active.push("motion");
    if (this.touchStageState === "capturing") active.push("touch");
    if (active.length > 0) {
      throw new Error(
        `Cannot complete: stages still capturing: ${active.join(", ")}`
      );
    }

    const sensorData: SensorData = {
      audio: this.audioData,
      motion: this.motionData,
      touch: this.touchData,
      modalities: {
        audio: this.audioData !== null,
        motion: this.motionData.length > 0,
        touch: this.touchData.length > 0,
      },
    };

    return processSensorData(sensorData, this.config, wallet, connection);
  }
}

/**
 * PulseSDK — main entry point for IAM Protocol verification.
 *
 * Two usage modes:
 *   1. Simple (backward-compatible): pulse.verify(touchElement) — captures all sensors
 *      for DEFAULT_CAPTURE_MS in parallel, then processes.
 *   2. Staged (event-driven): pulse.createSession(touchElement) — caller controls
 *      when each sensor stage starts and stops.
 */
export class PulseSDK {
  private config: ResolvedConfig;

  constructor(config: PulseConfig) {
    this.config = {
      threshold: DEFAULT_THRESHOLD,
      ...config,
    };
  }

  /**
   * Create a staged capture session for event-driven control.
   */
  createSession(touchElement?: HTMLElement): PulseSession {
    return new PulseSession(this.config, touchElement);
  }

  /**
   * Run a full verification with automatic timed capture (backward-compatible).
   * Captures all sensors in parallel for DEFAULT_CAPTURE_MS, then processes.
   */
  async verify(
    touchElement?: HTMLElement,
    wallet?: any,
    connection?: any
  ): Promise<VerificationResult> {
    try {
      const session = this.createSession(touchElement);
      const stopPromises: Promise<void>[] = [];

      // Audio
      try {
        await session.startAudio();
        stopPromises.push(
          new Promise<void>((r) => setTimeout(r, DEFAULT_CAPTURE_MS)).then(
            () => {
              session.stopAudio();
            }
          )
        );
      } catch {
        session.skipAudio();
      }

      // Motion
      try {
        await session.startMotion();
        stopPromises.push(
          new Promise<void>((r) => setTimeout(r, DEFAULT_CAPTURE_MS)).then(
            () => {
              session.stopMotion();
            }
          )
        );
      } catch {
        session.skipMotion();
      }

      // Touch
      if (touchElement) {
        try {
          await session.startTouch();
          stopPromises.push(
            new Promise<void>((r) => setTimeout(r, DEFAULT_CAPTURE_MS)).then(
              () => {
                session.stopTouch();
              }
            )
          );
        } catch {
          session.skipTouch();
        }
      } else {
        session.skipTouch();
      }

      await Promise.all(stopPromises);
      return session.complete(wallet, connection);
    } catch (err: any) {
      return {
        success: false,
        commitment: new Uint8Array(32),
        isFirstVerification: true,
        error: err.message ?? String(err),
      };
    }
  }
}
