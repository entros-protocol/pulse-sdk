import type { PulseConfig } from "./config";
import { DEFAULT_THRESHOLD, DEFAULT_CAPTURE_MS, PROGRAM_IDS } from "./config";
import type { SensorData, AudioCapture, MotionSample, TouchSample, StageState } from "./sensor/types";
import type { TBH } from "./hashing/types";
import type { SolanaProof } from "./proof/types";
import type { VerificationResult } from "./submit/types";
import type { StoredVerificationData } from "./identity/types";

import { captureAudio } from "./sensor/audio";
import { captureMotion, requestMotionPermission } from "./sensor/motion";
import { captureTouch } from "./sensor/touch";
import { extractSpeakerFeatures, SPEAKER_FEATURE_COUNT } from "./extraction/speaker";
import {
  extractMotionFeatures,
  extractTouchFeatures,
  extractMouseDynamics,
} from "./extraction/kinematic";
import { fuseFeatures } from "./extraction/statistics";
import { simhash, hammingDistance } from "./hashing/simhash";
import { generateTBH, bigintToBytes32 } from "./hashing/poseidon";
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
async function extractFeatures(data: SensorData): Promise<number[]> {
  if (!data.audio) {
    throw new Error("Audio data required for feature extraction");
  }
  const audioFeatures = await extractSpeakerFeatures(data.audio);

  const hasMotion = data.motion.length >= MIN_MOTION_SAMPLES;
  const hasTouch = data.touch.length >= MIN_TOUCH_SAMPLES;

  // On mobile (both IMU and touch available), use touch/pointer dynamics for
  // kinematic features. Stationary IMU reads constant gravity — the derivatives
  // are near-zero and produce identical features across sessions. Finger tracing
  // has natural inter-session variance because no two paths are identical.
  const motionFeatures =
    hasMotion && hasTouch
      ? extractMouseDynamics(data.touch)
      : hasMotion
        ? extractMotionFeatures(data.motion)
        : extractMouseDynamics(data.touch);

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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Solana types are optional peer deps
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

  // Re-verification requires audio + at least one other modality.
  // Audio-only fingerprints lack inter-session variance from motion/touch,
  // producing identical SimHash results that fail the min_distance constraint.
  let hasPreviousData: boolean;
  if (wallet && connection) {
    const walletPubkey = wallet.adapter?.publicKey ?? wallet.publicKey;
    if (walletPubkey) {
      try {
        const { PublicKey } = await import("@solana/web3.js");
        const programId = new PublicKey(PROGRAM_IDS.iamAnchor);
        const [identityPda] = PublicKey.findProgramAddressSync(
          [new TextEncoder().encode("identity"), walletPubkey.toBuffer()],
          programId
        );
        const accountInfo = await connection.getAccountInfo(identityPda);
        hasPreviousData = !!accountInfo;
      } catch {
        hasPreviousData = (await loadVerificationData()) !== null;
      }
    } else {
      hasPreviousData = (await loadVerificationData()) !== null;
    }
  } else {
    hasPreviousData = (await loadVerificationData()) !== null;
  }
  if (hasPreviousData && !hasMotion && !hasTouch) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: false,
      error: "Insufficient sensor data for re-verification. Please trace the curve and allow motion access.",
    };
  }

  // Extract features
  const features = await extractFeatures(sensorData);

  // Diagnostic: log feature vector composition
  const nonZero = features.filter((v) => v !== 0).length;
  console.log(
    `[IAM SDK] Feature vector: ${features.length} dimensions, ${nonZero} non-zero. ` +
    `Audio[0..43]: ${features.slice(0, 44).filter((v) => v !== 0).length} non-zero. ` +
    `Motion/Mouse[44..97]: ${features.slice(44, 98).filter((v) => v !== 0).length} non-zero. ` +
    `Touch[98..133]: ${features.slice(98, 134).filter((v) => v !== 0).length} non-zero.`
  );

  // Generate fingerprint via SimHash
  const fingerprint = simhash(features);

  // Generate TBH (Poseidon commitment)
  const tbh = await generateTBH(fingerprint);

  // Determine if this is a first verification.
  // Wallet-connected: check on-chain IdentityState PDA (source of truth).
  // Walletless: check localStorage for stored fingerprint.
  let isFirstVerification: boolean;
  const previousData = await loadVerificationData();

  if (wallet && connection) {
    const walletPubkey = wallet.adapter?.publicKey ?? wallet.publicKey;
    if (walletPubkey) {
      // Check if IdentityState PDA exists on-chain (simple existence check, no IDL needed)
      try {
        const { PublicKey } = await import("@solana/web3.js");
        const programId = new PublicKey(PROGRAM_IDS.iamAnchor);
        const [identityPda] = PublicKey.findProgramAddressSync(
          [new TextEncoder().encode("identity"), walletPubkey.toBuffer()],
          programId
        );
        const accountInfo = await connection.getAccountInfo(identityPda);
        isFirstVerification = !accountInfo;
      } catch {
        isFirstVerification = !previousData;
      }
    } else {
      isFirstVerification = !previousData;
    }
  } else {
    isFirstVerification = !previousData;
  }

  // Edge case: on-chain identity exists but local fingerprint is missing
  // (cleared browser data, new device, different browser). Can't generate
  // Hamming distance proof without the previous fingerprint.
  if (!isFirstVerification && !previousData) {
    return {
      success: false,
      commitment: tbh.commitmentBytes,
      isFirstVerification: false,
      error: "Previous behavioral fingerprint not found on this device. Your IAM Anchor exists on-chain but the local baseline data is missing. Please verify from the original device, or contact the integrator for a baseline reset.",
    };
  }

  let solanaProof: SolanaProof | null = null;

  if (!isFirstVerification && previousData) {
    const previousTBH: TBH = {
      fingerprint: previousData.fingerprint,
      salt: BigInt(previousData.salt),
      commitment: BigInt(previousData.commitment),
      commitmentBytes: bigintToBytes32(BigInt(previousData.commitment)),
    };

    const distance = hammingDistance(fingerprint, previousData.fingerprint);
    console.log(
      `[IAM SDK] Re-verification: Hamming distance = ${distance} / 256 bits (threshold = ${config.threshold})`
    );

    const circuitInput = prepareCircuitInput(
      tbh,
      previousTBH,
      config.threshold
    );

    const wasmPath = config.wasmUrl;
    const zkeyPath = config.zkeyUrl;

    if (!wasmPath || !zkeyPath) {
      return {
        success: false,
        commitment: tbh.commitmentBytes,
        isFirstVerification: false,
        error: "wasmUrl and zkeyUrl must be configured for re-verification proof generation",
      };
    }

    try {
      const { proof, publicSignals } = await generateProof(
        circuitInput,
        wasmPath,
        zkeyPath
      );
      solanaProof = serializeProof(proof, publicSignals);
    } catch (proofErr: any) {
      // Include diagnostics in error for mobile debugging (no devtools)
      const audioNZ = features.slice(0, 44).filter((v) => v !== 0).length;
      const motionNZ = features.slice(44, 98).filter((v) => v !== 0).length;
      const touchNZ = features.slice(98, 134).filter((v) => v !== 0).length;
      const rawAudio = sensorData.audio?.samples.length ?? 0;
      const rawMotion = sensorData.motion.length;
      const rawTouch = sensorData.touch.length;
      // First 3 feature values as a fingerprint to detect identical data
      const sig = features.slice(0, 3).map((v) => v.toFixed(4)).join(",");
      return {
        success: false,
        commitment: tbh.commitmentBytes,
        isFirstVerification: false,
        error: `Proof failed (dist=${distance}, feat=${audioNZ}/${motionNZ}/${touchNZ}, raw=${rawAudio}/${rawMotion}/${rawTouch}, sig=${sig}): ${proofErr?.message ?? proofErr}`,
      };
    }
  }

  // Submit
  let submission;

  if (wallet && connection) {
    if (isFirstVerification) {
      submission = await submitViaWallet(
        solanaProof ?? { proofBytes: new Uint8Array(0), publicInputs: [] },
        tbh.commitmentBytes,
        { wallet, connection, isFirstVerification: true, relayerUrl: config.relayerUrl, relayerApiKey: config.relayerApiKey }
      );
    } else {
      submission = await submitViaWallet(solanaProof!, tbh.commitmentBytes, {
        wallet,
        connection,
        isFirstVerification: false,
        relayerUrl: config.relayerUrl,
        relayerApiKey: config.relayerApiKey,
      });
    }
  } else if (config.relayerUrl) {
    submission = await submitViaRelayer(
      solanaProof ?? { proofBytes: new Uint8Array(0), publicInputs: [] },
      tbh.commitmentBytes,
      { relayerUrl: config.relayerUrl, apiKey: config.relayerApiKey, isFirstVerification }
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
    await storeVerificationData({
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
    attestationTx: submission.attestationTx,
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

    // Acquire microphone permission within the user gesture context.
    // Awaited so the caller knows audio is ready before proceeding.
    // State transitions happen AFTER permission succeeds to avoid zombie state.
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });

    this.audioStageState = "capturing";
    this.audioController = new AbortController();
    this.audioPromise = captureAudio({
      signal: this.audioController.signal,
      onAudioLevel,
      stream,
    }).catch(() => {
      stream.getTracks().forEach((t) => t.stop());
      return null;
    });
  }

  async stopAudio(): Promise<AudioCapture | null> {
    if (this.audioStageState !== "capturing")
      throw new Error("Audio capture not active");
    this.audioController!.abort();
    this.audioData = await this.audioPromise!;
    this.audioStageState = "captured";
    return this.audioData;
  }

  // Audio is mandatory — no skipAudio() method.
  // If startAudio() fails, the verification cannot proceed.

  // --- Motion ---

  async startMotion(): Promise<void> {
    if (this.motionStageState !== "idle")
      throw new Error("Motion capture already started");

    // Request motion permission within the user gesture context (iOS 13+).
    // Awaited so the capture timer doesn't start before the user approves.
    const hasPermission = await requestMotionPermission();
    if (!hasPermission) {
      this.motionStageState = "skipped";
      return;
    }

    this.motionStageState = "capturing";
    this.motionController = new AbortController();
    this.motionPromise = captureMotion({
      signal: this.motionController.signal,
      permissionGranted: true,
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

  isMotionCapturing(): boolean {
    return this.motionStageState === "capturing";
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Solana types are optional peer deps
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

      // Motion first — requires user gesture on iOS (gesture expires after getUserMedia)
      try {
        await session.startMotion();
      } catch {
        /* unexpected error — motion already skipped or idle */
      }
      if (session.isMotionCapturing()) {
        stopPromises.push(
          new Promise<void>((r) => setTimeout(r, DEFAULT_CAPTURE_MS))
            .then(() => session.stopMotion())
            .then(() => {})
        );
      }

      // Audio second — getUserMedia works without a gesture on secure origins
      try {
        await session.startAudio();
        stopPromises.push(
          new Promise<void>((r) => setTimeout(r, DEFAULT_CAPTURE_MS))
            .then(() => session.stopAudio())
            .then(() => {})
        );
      } catch (err: any) {
        throw new Error(`Audio capture failed: ${err?.message ?? "microphone unavailable"}`);
      }

      // Touch
      if (touchElement) {
        try {
          await session.startTouch();
          stopPromises.push(
            new Promise<void>((r) => setTimeout(r, DEFAULT_CAPTURE_MS))
              .then(() => session.stopTouch())
              .then(() => {})
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
