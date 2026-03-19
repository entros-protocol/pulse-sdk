import type { PulseConfig } from "./config";
import { DEFAULT_THRESHOLD } from "./config";
import type { SensorData } from "./sensor/types";
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

const CAPTURE_DURATION_MS = 7000;

/**
 * PulseSDK — main entry point for IAM Protocol verification.
 *
 * Usage:
 *   const pulse = new PulseSDK({ cluster: 'devnet' });
 *   const result = await pulse.verify(touchElement);
 */
export class PulseSDK {
  private config: Required<
    Pick<PulseConfig, "cluster" | "threshold">
  > &
    PulseConfig;

  constructor(config: PulseConfig) {
    this.config = {
      threshold: DEFAULT_THRESHOLD,
      ...config,
    };
  }

  /**
   * Run a full verification: capture sensors, extract features, generate proof, submit.
   *
   * @param touchElement - DOM element for touch capture (user traces Lissajous curve here)
   * @param wallet - Wallet adapter (if null, uses walletless/relayer mode)
   * @param connection - Solana connection (required for wallet mode)
   */
  async verify(
    touchElement?: HTMLElement,
    wallet?: any,
    connection?: any
  ): Promise<VerificationResult> {
    try {
      // 1. Capture sensor data
      const sensorData = await this.captureSensors(touchElement);

      // 2. Extract features
      const features = this.extractFeatures(sensorData);

      // 3. Generate fingerprint via SimHash
      const fingerprint = simhash(features);

      // 4. Generate TBH (Poseidon commitment)
      const tbh = await generateTBH(fingerprint);

      // 5. Check for previous verification data
      const previousData = loadVerificationData();
      const isFirstVerification = !previousData;

      let solanaProof: SolanaProof | null = null;

      if (!isFirstVerification && previousData) {
        // Re-verification: generate ZK proof
        const previousTBH: TBH = {
          fingerprint: previousData.fingerprint,
          salt: BigInt(previousData.salt),
          commitment: BigInt(previousData.commitment),
          commitmentBytes: new Uint8Array(32), // Not needed for proof input
        };

        const circuitInput = prepareCircuitInput(
          tbh,
          previousTBH,
          this.config.threshold
        );

        const wasmPath = this.config.wasmUrl ?? "";
        const zkeyPath = this.config.zkeyUrl ?? "";

        const { proof, publicSignals } = await generateProof(
          circuitInput,
          wasmPath,
          zkeyPath
        );

        solanaProof = serializeProof(proof, publicSignals);
      }

      // 6. Submit
      let submission;

      if (wallet && connection) {
        // Wallet-connected mode
        if (isFirstVerification) {
          submission = await submitViaWallet(
            solanaProof ?? { proofBytes: new Uint8Array(0), publicInputs: [] },
            tbh.commitmentBytes,
            {
              wallet,
              connection,
              isFirstVerification: true,
            }
          );
        } else {
          submission = await submitViaWallet(solanaProof!, tbh.commitmentBytes, {
            wallet,
            connection,
            isFirstVerification: false,
          });
        }
      } else if (this.config.relayerUrl) {
        // Walletless mode
        submission = await submitViaRelayer(
          solanaProof ?? { proofBytes: new Uint8Array(0), publicInputs: [] },
          tbh.commitmentBytes,
          {
            relayerUrl: this.config.relayerUrl,
            isFirstVerification,
          }
        );
      } else {
        return {
          success: false,
          commitment: tbh.commitmentBytes,
          isFirstVerification,
          error: "No wallet or relayer configured",
        };
      }

      // 7. Store verification data locally for next re-verification
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
    } catch (err: any) {
      return {
        success: false,
        commitment: new Uint8Array(32),
        isFirstVerification: true,
        error: err.message ?? String(err),
      };
    }
  }

  /** Capture all available sensor modalities for 7 seconds */
  private async captureSensors(
    touchElement?: HTMLElement
  ): Promise<SensorData> {
    const promises: Promise<any>[] = [];

    // Audio capture
    let audioPromise: Promise<any>;
    try {
      audioPromise = captureAudio(CAPTURE_DURATION_MS);
    } catch {
      audioPromise = Promise.resolve(null);
    }
    promises.push(audioPromise);

    // Motion capture
    let motionPromise: Promise<any>;
    try {
      motionPromise = captureMotion(CAPTURE_DURATION_MS);
    } catch {
      motionPromise = Promise.resolve([]);
    }
    promises.push(motionPromise);

    // Touch capture
    let touchPromise: Promise<any>;
    if (touchElement) {
      try {
        touchPromise = captureTouch(touchElement, CAPTURE_DURATION_MS);
      } catch {
        touchPromise = Promise.resolve([]);
      }
    } else {
      touchPromise = Promise.resolve([]);
    }
    promises.push(touchPromise);

    const [audio, motion, touch] = await Promise.all(promises);

    return {
      audio,
      motion: motion ?? [],
      touch: touch ?? [],
      modalities: {
        audio: !!audio,
        motion: (motion?.length ?? 0) > 0,
        touch: (touch?.length ?? 0) > 0,
      },
    };
  }

  /** Extract features from sensor data and fuse into a single vector */
  private extractFeatures(data: SensorData): number[] {
    const audioFeatures = data.audio ? extractMFCC(data.audio) : new Array(156).fill(0);
    const motionFeatures = extractMotionFeatures(data.motion);
    const touchFeatures = extractTouchFeatures(data.touch);
    return fuseFeatures(audioFeatures, motionFeatures, touchFeatures);
  }
}
