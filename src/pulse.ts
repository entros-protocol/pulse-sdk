import type { PulseConfig } from "./config";
import { DEFAULT_THRESHOLD, DEFAULT_CAPTURE_MS, PROGRAM_IDS } from "./config";
import { setDebug, sdkLog, sdkWarn } from "./log";
import type { SensorData, AudioCapture, MotionSample, TouchSample, StageState } from "./sensor/types";
import type { TBH } from "./hashing/types";
import type { SolanaProof } from "./proof/types";
import type { SignedReceiptDto, VerificationResult } from "./submit/types";
import type { StoredVerificationData } from "./identity/types";

import { captureAudio } from "./sensor/audio";
import { encodeAudioAsBase64 } from "./sensor/encode";
import { captureMotion, requestMotionPermission } from "./sensor/motion";
import { captureTouch } from "./sensor/touch";
import { extractSpeakerFeaturesDetailed, SPEAKER_FEATURE_COUNT } from "./extraction/speaker";
import {
  extractMotionFeatures,
  extractTouchFeatures,
  extractMouseDynamics,
  extractAccelerationMagnitude,
  MOTION_FEATURE_COUNT,
  TOUCH_FEATURE_COUNT,
} from "./extraction/kinematic";
import { fuseFeatures, fuseRawFeatures } from "./extraction/statistics";
import { yieldToMainThread } from "./yield";
import { simhash, hammingDistance } from "./hashing/simhash";
import { generateTBH, bigintToBytes32 } from "./hashing/poseidon";
import { prepareCircuitInput, generateProof } from "./proof/prover";
import { serializeProof } from "./proof/serializer";
import { submitViaWallet, submitResetViaWallet } from "./submit/wallet";
import { submitViaRelayer } from "./submit/relayer";
import { bytesToHex } from "./submit/receipt";
import {
  storeVerificationData,
  loadVerificationData,
  setPrivacyFallback,
  recoverBaselineFromChain,
} from "./identity/anchor";
import {
  BaselineWallet,
  deriveEncryptedBaselinePda,
  encryptBaselineBlob,
  fingerprintToBytes,
  getOrDeriveBaselineKey,
} from "./identity/baseline";

// Build-time constant. Replaced by tsup `define` (true when IAM_INTERNAL_TEST=1)
// and by vitest `define`. In default builds (npm publish path) this is `false`
// and any test hook short-circuits to throw — guaranteeing the harness-only
// injection path is unreachable in published artifacts.
declare const __IAM_INTERNAL_TEST__: boolean;

type ResolvedConfig = Required<Pick<PulseConfig, "cluster" | "threshold">> &
  PulseConfig;

interface ExtractedFeatures {
  /** Raw features in physical units (Hz, ratios, dB, px/frame). For server-side validation. */
  raw: number[];
  /** Z-score normalized features. For SimHash fingerprint computation. */
  normalized: number[];
  /**
   * F0 (fundamental frequency) contour per audio frame (~10ms hop).
   * Sent to the validation service for cross-modal temporal analysis.
   * Empty array when audio is invalid or too short.
   */
  f0Contour: number[];
  /**
   * Acceleration magnitude (√(ax²+ay²+az²)) resampled to match the F0 frame count.
   * Paired with `f0Contour` for server-side analysis.
   * Empty array when motion data is absent.
   */
  accelMagnitude: number[];
}

/**
 * Extract features from sensor data. Returns both raw (physical units)
 * and normalized (z-scored) feature vectors.
 */
async function extractFeatures(data: SensorData): Promise<ExtractedFeatures> {
  if (!data.audio) {
    throw new Error(
      "Audio data missing. Capture audio via session.startAudio() before extracting features.",
    );
  }
  const { features: audioFeatures, f0Contour } = await extractSpeakerFeaturesDetailed(
    data.audio,
  );
  // The audio path is the dominant cost. Yield once it's done so the
  // verify-flow spinner gets a paint frame before motion/touch extraction
  // resumes the main-thread work.
  await yieldToMainThread();

  const hasMotion = data.motion.length >= MIN_MOTION_SAMPLES;
  const hasTouch = data.touch.length >= MIN_TOUCH_SAMPLES;

  const motionFeatures =
    hasMotion && hasTouch
      ? extractMouseDynamics(data.touch)
      : hasMotion
        ? extractMotionFeatures(data.motion)
        : extractMouseDynamics(data.touch);
  await yieldToMainThread();

  const touchFeatures = extractTouchFeatures(data.touch);
  await yieldToMainThread();

  // Align acceleration magnitude to the F0 frame count for direct cross-correlation.
  // Empty if motion absent or F0 extraction produced no frames (e.g. silent capture).
  const accelMagnitude =
    hasMotion && f0Contour.length > 0
      ? extractAccelerationMagnitude(data.motion, f0Contour.length)
      : [];

  return {
    raw: fuseRawFeatures(audioFeatures, motionFeatures, touchFeatures),
    normalized: fuseFeatures(audioFeatures, motionFeatures, touchFeatures),
    f0Contour,
    accelMagnitude,
  };
}

/**
 * Shared pipeline: features → simhash → TBH → proof → submit.
 * Used by both PulseSDK.verify() and PulseSession.complete().
 */
// Minimum sample counts for meaningful feature extraction.
// Exported so consumers (including the internal-build-only red team harness)
// can enforce the same thresholds upstream and surface clearer errors than
// the SDK's data-quality gate would.
export const MIN_AUDIO_SAMPLES = 16000; // ~1 second at 16 kHz
export const MIN_MOTION_SAMPLES = 10;
export const MIN_TOUCH_SAMPLES = 10;

type ExtractionResult =
  | {
      ok: true;
      features: number[];
      f0Contour: number[];
      accelMagnitude: number[];
      fingerprint: number[];
      tbh: TBH;
      /**
       * Validator-signed mint receipt. Present only when the request
       * reached the validator with `commitment_new_hex` AND the validator
       * has a signing key configured. `undefined` indicates the SDK
       * should mint without an Ed25519 prefix; while the on-chain check
       * is log-only this is harmless, but once enforcement is enabled
       * missing receipts cause `mint_anchor` to hard-fail.
       */
      signedReceipt?: SignedReceiptDto;
    }
  | { ok: false; error: string; reason?: string };

/**
 * Shared front half of the verification pipeline, covering feature
 * extraction, server-side feature validation (if configured), and
 * TBH (Poseidon commitment) generation. Used by both the normal
 * verify path and the reset path — the back half diverges after this
 * point (proof generation + update_anchor for verify, direct
 * reset_identity_state for reset).
 *
 * `walletAddress` is the base58-encoded public key sent to the
 * validator's `/validate-features` endpoint as `wallet_id`. Pass
 * `undefined` for walletless mode to skip server validation.
 */
async function extractFingerprintAndValidate(
  sensorData: SensorData,
  config: ResolvedConfig,
  walletAddress: string | undefined,
  onProgress?: (stage: string) => void,
): Promise<ExtractionResult> {
  onProgress?.("Extracting features...");
  // Let React render the new stage label before we re-enter the heavy
  // synchronous extraction path. Without this, the host UI sets the
  // string but the main thread is captured by extractFeatures before
  // the spinner can repaint, and the user sees the previous stage's
  // label until extraction completes.
  await yieldToMainThread();
  const {
    raw: features,
    normalized: normalizedFeatures,
    f0Contour,
    accelMagnitude,
  } = await extractFeatures(sensorData);

  // Diagnostic: log feature vector composition. Block boundaries follow the
  // v2 layout, derived from the canonical per-modality counts so any future
  // modality bump propagates automatically (no hand-sync drift).
  const AUDIO_END = SPEAKER_FEATURE_COUNT;
  const MOTION_END = AUDIO_END + MOTION_FEATURE_COUNT;
  const TOUCH_END = MOTION_END + TOUCH_FEATURE_COUNT;
  const nonZero = features.filter((v) => v !== 0).length;
  sdkLog(
    `[Entros SDK] Feature vector: ${features.length} dimensions, ${nonZero} non-zero. ` +
    `Audio[0..${AUDIO_END - 1}]: ${features.slice(0, AUDIO_END).filter((v) => v !== 0).length} non-zero. ` +
    `Motion/Mouse[${AUDIO_END}..${MOTION_END - 1}]: ${features.slice(AUDIO_END, MOTION_END).filter((v) => v !== 0).length} non-zero. ` +
    `Touch[${MOTION_END}..${TOUCH_END - 1}]: ${features.slice(MOTION_END, TOUCH_END).filter((v) => v !== 0).length} non-zero.`
  );

  // Compute the SimHash fingerprint and Poseidon TBH commitment BEFORE the
  // validation POST. The validator signs a (wallet, commitment, validated_at)
  // receipt that the SDK bundles before `mint_anchor` in the same atomic
  // transaction; for the validator to sign the right commitment, we must
  // transmit it in the request. SimHash + Poseidon together cost ~20ms —
  // trivial overhead even on rejection paths.
  const fingerprint = simhash(normalizedFeatures);
  const tbh = await generateTBH(fingerprint);

  let signedReceipt: SignedReceiptDto | undefined;

  onProgress?.("Validating...");
  // Same rationale as the "Extracting features..." yield above — give
  // React a paint opportunity before we encode the audio buffer to base64
  // (~16k samples), which is the next synchronous chunk on the main thread.
  await yieldToMainThread();
  if (config.relayerUrl && walletAddress) {
    try {
      const baseUrl = new URL(config.relayerUrl);
      const validateUrl = `${baseUrl.origin}/validate-features`;
      const validateHeaders: Record<string, string> = { "Content-Type": "application/json" };
      if (config.relayerApiKey) {
        validateHeaders["X-API-Key"] = config.relayerApiKey;
      }

      // Encode captured audio for server-side phrase verification. The
      // validator transcribes the audio and matches it against the
      // server-issued challenge phrase (which the executor looks up by
      // nonce). If audio is absent, the validation service skips the
      // phrase check — preserving backward compatibility for older SDKs.
      //
      // We also transmit the actual `sampleRate` from the capture — browsers
      // occasionally ignore the 16kHz AudioContext request (Safari with
      // Bluetooth codec negotiation, some Android devices) and deliver 44.1k
      // or 48k. The validator resamples to 16kHz internally before feeding
      // Whisper, so transmitting the true rate avoids silent transcription
      // quality loss.
      const audioSamplesB64 = sensorData.audio?.samples
        ? encodeAudioAsBase64(sensorData.audio.samples)
        : undefined;
      const audioSampleRateHz = sensorData.audio?.sampleRate;

      // Hex-encode the 32-byte commitment for the validator's signing
      // input. The validator only signs when this field is present AND
      // its own signing key is configured; the SDK only consumes the
      // receipt on first-verification, so sending it on every
      // wallet-connected request is harmless on the re-verify path
      // (validator signs cheaply, executor passes through, SDK ignores
      // the field for `update_anchor`).
      const commitmentNewHex = bytesToHex(tbh.commitmentBytes);

      // Server-side transcription adds ~1s to the validation round trip.
      // Extend timeout from 10s to 15s to tolerate cold-start model load
      // without aborting on legitimate requests.
      const validateController = new AbortController();
      const validateTimer = setTimeout(() => validateController.abort(), 15_000);

      const validateResponse = await fetch(validateUrl, {
        method: "POST",
        headers: validateHeaders,
        body: JSON.stringify({
          features,
          f0_contour: f0Contour,
          accel_magnitude: accelMagnitude,
          wallet_id: walletAddress,
          audio_samples_b64: audioSamplesB64,
          audio_sample_rate_hz: audioSampleRateHz,
          commitment_new_hex: commitmentNewHex,
        }),
        signal: validateController.signal,
      });

      clearTimeout(validateTimer);

      if (!validateResponse.ok) {
        const errorBody = await validateResponse.json().catch(() => ({}));
        sdkWarn("[Entros SDK] Feature validation rejected by server");
        return {
          ok: false,
          error: (errorBody as Record<string, string>).error || "Feature validation failed",
          reason: (errorBody as Record<string, string>).reason,
        };
      }

      // Parse the validator's success body for the signed receipt. Older
      // validator deploys omit the field entirely — the SDK proceeds
      // without a receipt and the on-chain log-only check writes "no
      // preceding instruction" to the tx logs. Once on-chain enforcement
      // is enabled, missing receipts will hard-fail mint_anchor; the
      // executor + validator deploys must therefore be brought up to
      // receipt-supporting versions before the enforcement flag flips.
      try {
        const successBody = (await validateResponse.json()) as {
          signed_receipt?: SignedReceiptDto;
        };
        if (successBody.signed_receipt) {
          signedReceipt = successBody.signed_receipt;
        }
      } catch (err) {
        // Body wasn't JSON — typically an older validator returning an
        // empty 200, or a proxy mangling the response. Surface a warn
        // so operators can distinguish "validator-too-old" from a real
        // validator misconfiguration. Treat as no-receipt and proceed.
        const msg = err instanceof Error ? err.message : String(err);
        sdkWarn(
          `[Entros SDK] /validate-features returned 200 but body was not parseable JSON; proceeding without receipt: ${msg}`
        );
      }
    } catch (err) {
      // Network failure / timeout / abort. Previously this silently
      // continued and skipped server-side validation, which let a
      // network-failure attacker bypass server-side checks entirely.
      // Return as a recoverable error instead;
      // the host app can surface a retry CTA. The reason category
      // `validation_unavailable` is client-side only (distinct from
      // any server-side `ReasonCode`) and is intended for soft-fail
      // UX similar to a transient network error.
      const msg = err instanceof Error ? err.message : String(err);
      sdkWarn(`[Entros SDK] Feature validation unavailable: ${msg}`);
      return {
        ok: false,
        error: "Validation service unreachable. Please check your connection and try again.",
        reason: "validation_unavailable",
      };
    }
  }

  return { ok: true, features, f0Contour, accelMagnitude, fingerprint, tbh, signedReceipt };
}

/**
 * Resolve a `BaselineWallet` (just the `{ publicKey, signMessage }` surface
 * required for the encrypted-baseline AES key derivation) from the wallet
 * shape the host app supplies. Returns `null` when the wallet can't sign
 * messages — e.g., some Ledger firmware versions, or any wallet adapter
 * without `signMessage`. Callers gracefully skip the encrypted-baseline
 * path in that case.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- Wallet shape is supplied by host app
function resolveBaselineWallet(wallet: any): BaselineWallet | null {
  if (!wallet) return null;
  const adapter = wallet.adapter ?? wallet;
  if (!adapter?.publicKey || typeof adapter.signMessage !== "function") {
    return null;
  }
  return {
    publicKey: adapter.publicKey,
    signMessage: adapter.signMessage.bind(adapter),
  };
}

/**
 * Build the 96-byte encrypted-baseline blob for the wallet's next on-chain
 * write, best-effort: returns `undefined` (rather than throwing) when the
 * wallet can't `signMessage`, AES key derivation fails, or any crypto
 * primitive errors out. The submit path skips bundling the
 * `set_encrypted_baseline` instruction in that case; the local-only
 * baseline tier still works.
 */
async function buildEncryptedBaselineBlobBestEffort(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Wallet shape is supplied by host app
  wallet: any,
  fingerprint: number[],
  salt: bigint,
  commitmentBytes: Uint8Array,
): Promise<Uint8Array | undefined> {
  const baselineWallet = resolveBaselineWallet(wallet);
  if (!baselineWallet) return undefined;
  try {
    const key = await getOrDeriveBaselineKey(baselineWallet);
    const [baselinePda] = await deriveEncryptedBaselinePda(baselineWallet.publicKey);
    const simhashBytes = fingerprintToBytes(fingerprint);
    const saltBytes = bigintToBytes32(salt);
    return await encryptBaselineBlob(
      simhashBytes,
      saltBytes,
      key,
      baselineWallet.publicKey,
      baselinePda,
      commitmentBytes,
    );
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    sdkWarn(
      `[Entros SDK] Encrypted-baseline build skipped (cross-device recovery unavailable this session): ${msg}`,
    );
    return undefined;
  }
}

async function processSensorData(
  sensorData: SensorData,
  config: ResolvedConfig,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Solana types are optional peer deps
  wallet?: any,
  connection?: any,
  onProgress?: (stage: string) => void,
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
        const programId = new PublicKey(PROGRAM_IDS.entrosAnchor);
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

  const walletAddress = wallet?.adapter?.publicKey?.toBase58?.()
    ?? wallet?.publicKey?.toBase58?.();
  const extraction = await extractFingerprintAndValidate(
    sensorData,
    config,
    walletAddress,
    onProgress,
  );
  if (!extraction.ok) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: false,
      error: extraction.error,
      reason: extraction.reason,
    };
  }
  const { fingerprint, tbh, features, signedReceipt } = extraction;

  // Determine if this is a first verification.
  // Wallet-connected: check on-chain IdentityState PDA (source of truth).
  // Walletless: check localStorage for stored fingerprint.
  let isFirstVerification: boolean;
  let previousData = await loadVerificationData();

  if (wallet && connection) {
    const walletPubkey = wallet.adapter?.publicKey ?? wallet.publicKey;
    if (walletPubkey) {
      // Check if IdentityState PDA exists on-chain (simple existence check, no IDL needed)
      try {
        const { PublicKey } = await import("@solana/web3.js");
        const programId = new PublicKey(PROGRAM_IDS.entrosAnchor);
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
  // (cleared browser data, new device, different browser). Before forcing a
  // reset, attempt to recover the baseline from the on-chain
  // `EncryptedBaseline` PDA (master-list #98). Recovery requires the wallet
  // to support `signMessage` for AES key derivation; on success, the
  // recovered baseline is written to local storage and the flow continues
  // into normal re-verification.
  if (!isFirstVerification && !previousData && wallet && connection) {
    const baselineWallet = resolveBaselineWallet(wallet);
    if (baselineWallet) {
      onProgress?.("Recovering baseline from chain...");
      const recovery = await recoverBaselineFromChain(baselineWallet, connection);
      if (recovery.recovered) {
        previousData = await loadVerificationData();
        sdkLog("[Entros SDK] On-chain encrypted baseline recovered");
      } else {
        sdkLog(
          `[Entros SDK] On-chain encrypted baseline recovery not available (${recovery.reason ?? "unknown"})`,
        );
      }
    }
  }

  if (!isFirstVerification && !previousData) {
    return {
      success: false,
      commitment: tbh.commitmentBytes,
      isFirstVerification: false,
      error: "Previous behavioral fingerprint not found on this device. Your Entros Anchor exists on-chain but the local baseline is missing. Reset your baseline to re-enroll from this device, or verify from the device that has the original baseline.",
    };
  }

  let solanaProof: SolanaProof | null = null;

  if (!isFirstVerification && previousData) {
    onProgress?.("Computing proof...");
    const previousTBH: TBH = {
      fingerprint: previousData.fingerprint,
      salt: BigInt(previousData.salt),
      commitment: BigInt(previousData.commitment),
      commitmentBytes: bigintToBytes32(BigInt(previousData.commitment)),
    };

    const distance = hammingDistance(fingerprint, previousData.fingerprint);
    sdkLog(
      `[Entros SDK] Re-verification: Hamming distance = ${distance} / 256 bits (threshold = ${config.threshold})`
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
        error: "Re-verification requires wasmUrl and zkeyUrl in PulseConfig. Host the entros_hamming.wasm and entros_hamming_final.zkey circuit artifacts at public URLs.",
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
      // Include diagnostics in error for mobile debugging (no devtools).
      // Block boundaries derived from extractor constants so they stay in
      // sync with the v2 layout if any modality count shifts.
      const motionStart = SPEAKER_FEATURE_COUNT;
      const touchStart = motionStart + MOTION_FEATURE_COUNT;
      const touchEnd = touchStart + TOUCH_FEATURE_COUNT;
      const audioNZ = features.slice(0, motionStart).filter((v) => v !== 0).length;
      const motionNZ = features.slice(motionStart, touchStart).filter((v) => v !== 0).length;
      const touchNZ = features.slice(touchStart, touchEnd).filter((v) => v !== 0).length;
      const rawAudio = sensorData.audio?.samples.length ?? 0;
      const rawMotion = sensorData.motion.length;
      const rawTouch = sensorData.touch.length;
      // First 3 feature values as a fingerprint to detect identical data
      const sig = features.slice(0, 3).map((v) => v.toFixed(4)).join(",");
      return {
        success: false,
        commitment: tbh.commitmentBytes,
        isFirstVerification: false,
        error: `Proof generation failed: ${proofErr?.message ?? proofErr}. Check wasmUrl/zkeyUrl reachability. Diagnostics: dist=${distance}, nz=${audioNZ}/${motionNZ}/${touchNZ}, raw=${rawAudio}/${rawMotion}/${rawTouch}, sig=${sig}`,
      };
    }
  }

  // Submit
  onProgress?.("Submitting to Solana...");
  let submission;

  if (wallet && connection) {
    // Best-effort: build the encrypted-baseline blob bound to the NEW
    // commitment so `submitViaWallet` can bundle a `set_encrypted_baseline`
    // ix into the same atomic transaction. Returns undefined when the
    // wallet adapter lacks `signMessage` (e.g., some Ledger firmware) —
    // the user falls back to local-only baseline storage gracefully.
    const encryptedBaselineBlob = await buildEncryptedBaselineBlobBestEffort(
      wallet,
      tbh.fingerprint,
      tbh.salt,
      tbh.commitmentBytes,
    );

    if (isFirstVerification) {
      // Pass the validator-signed receipt (when present) so submitViaWallet
      // can bundle an `Ed25519Program::verify` instruction before
      // `mint_anchor` in the same atomic transaction. Re-verification
      // doesn't need the receipt — the binding is already enforced via
      // the VerificationResult PDA path that `update_anchor` consumes.
      submission = await submitViaWallet(
        solanaProof ?? { proofBytes: new Uint8Array(0), publicInputs: [] },
        tbh.commitmentBytes,
        {
          wallet,
          connection,
          isFirstVerification: true,
          relayerUrl: config.relayerUrl,
          relayerApiKey: config.relayerApiKey,
          signedReceipt,
          encryptedBaselineBlob,
        }
      );
    } else {
      submission = await submitViaWallet(solanaProof!, tbh.commitmentBytes, {
        wallet,
        connection,
        isFirstVerification: false,
        relayerUrl: config.relayerUrl,
        relayerApiKey: config.relayerApiKey,
        encryptedBaselineBlob,
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
      error: "No submission path available. Pass wallet+connection to verify() for wallet-connected mode, or set relayerUrl in PulseConfig for walletless mode.",
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
 * Reset pipeline: features → simhash → TBH → reset_identity_state → store.
 * Mirrors `processSensorData()` but skips the Hamming ZK proof (there is no
 * prior fingerprint to bind against) and substitutes `submitResetViaWallet`
 * for the wallet submission path.
 *
 * Humanness is enforced server-side: the /validate-features and /attest
 * endpoints on the executor reject synthetic captures identically to the
 * normal verify flow.
 */
async function processResetSensorData(
  sensorData: SensorData,
  config: ResolvedConfig,
  wallet: any,
  connection: any,
  onProgress?: (stage: string) => void,
): Promise<VerificationResult> {
  const audioSamples = sensorData.audio?.samples.length ?? 0;
  const motionSamples = sensorData.motion.length;
  const touchSamples = sensorData.touch.length;

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

  // Reset requires the full multi-modal capture just like a fresh mint, so
  // the on-chain baseline is established from a meaningful fingerprint.
  if (!hasMotion && !hasTouch) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: true,
      error: "Insufficient sensor data for baseline reset. Please trace the curve and allow motion access.",
    };
  }

  const walletAddress = wallet.adapter?.publicKey?.toBase58?.()
    ?? wallet.publicKey?.toBase58?.();
  const extraction = await extractFingerprintAndValidate(
    sensorData,
    config,
    walletAddress,
    onProgress,
  );
  if (!extraction.ok) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: true,
      error: extraction.error,
      reason: extraction.reason,
    };
  }
  const { tbh } = extraction;

  // Best-effort: build the encrypted-baseline blob bound to the NEW
  // post-reset commitment so `submitResetViaWallet` can overwrite the
  // on-chain blob in the same atomic transaction. Without this, the prior
  // pre-reset blob would be stale on the next recovery attempt (auth-tag
  // mismatch under the new commitment in AAD).
  const encryptedBaselineBlob = await buildEncryptedBaselineBlobBestEffort(
    wallet,
    tbh.fingerprint,
    tbh.salt,
    tbh.commitmentBytes,
  );

  onProgress?.("Submitting reset to Solana...");
  const submission = await submitResetViaWallet(tbh.commitmentBytes, {
    wallet,
    connection,
    relayerUrl: config.relayerUrl,
    relayerApiKey: config.relayerApiKey,
    encryptedBaselineBlob,
  });

  // Persist the new local baseline on on-chain success. A throw here would
  // leave the user with an on-chain commitment they can't prove locally;
  // surface the failure explicitly instead of swallowing it so the UI can
  // prompt the user to reset again (after the 7-day cooldown) or transfer
  // the baseline from another device.
  if (submission.success) {
    try {
      await storeVerificationData({
        fingerprint: tbh.fingerprint,
        salt: tbh.salt.toString(),
        commitment: tbh.commitment.toString(),
        timestamp: Date.now(),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      sdkWarn(`[Entros SDK] Reset succeeded on chain but local baseline persistence failed: ${msg}`);
      return {
        success: false,
        commitment: tbh.commitmentBytes,
        txSignature: submission.txSignature,
        attestationTx: submission.attestationTx,
        isFirstVerification: true,
        error:
          "Reset confirmed on chain, but saving the new baseline to this device failed. " +
          "Re-verification from this device will not work. Try clearing site data and " +
          "resetting again after the 7-day cooldown, or transfer a baseline from another " +
          "device.",
      };
    }
  }

  return {
    success: submission.success,
    commitment: tbh.commitmentBytes,
    txSignature: submission.txSignature,
    attestationTx: submission.attestationTx,
    // Semantically this is a fresh baseline enrollment from the UX
    // perspective. `isFirstVerification: true` lets the caller render
    // success copy that matches first-time flows.
    isFirstVerification: true,
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
      throw new Error(
        "Audio capture already in progress. Call stopAudio() before starting a new capture.",
      );

    // Acquire microphone permission within the user gesture context.
    // Awaited so the caller knows audio is ready before proceeding.
    // State transitions happen AFTER permission succeeds to avoid zombie state.
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        // Capture constraints kept in lock-step with `sensor/audio.ts` —
        // the two entry points (standalone capture vs session-based
        // capture) must agree or the verify flow and direct-API
        // consumers diverge.
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        // @ts-expect-error -- W3C Media Capture Extensions property; not
        // yet in lib.dom.d.ts as of TypeScript 6.0.
        voiceIsolation: true,
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
      throw new Error(
        "No active audio capture to stop. Call startAudio() first.",
      );
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
      throw new Error(
        "Motion capture already in progress. Call stopMotion() before starting a new capture.",
      );

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
      throw new Error(
        "No active motion capture to stop. Call startMotion() first.",
      );
    this.motionController!.abort();
    this.motionData = await this.motionPromise!;
    this.motionStageState = "captured";
    return this.motionData;
  }

  skipMotion(): void {
    if (this.motionStageState !== "idle")
      throw new Error(
        "Cannot skip motion: capture already started. skipMotion() must be called before startMotion().",
      );
    this.motionStageState = "skipped";
  }

  isMotionCapturing(): boolean {
    return this.motionStageState === "capturing";
  }

  // --- Touch ---

  async startTouch(): Promise<void> {
    if (this.touchStageState !== "idle")
      throw new Error(
        "Touch capture already in progress. Call stopTouch() before starting a new capture.",
      );
    if (!this.touchElement)
      throw new Error(
        "No touch element provided to session. Pass an HTMLElement to createSession() to enable touch capture.",
      );
    this.touchStageState = "capturing";
    this.touchController = new AbortController();
    this.touchPromise = captureTouch(this.touchElement, {
      signal: this.touchController.signal,
    }).catch(() => []);
  }

  async stopTouch(): Promise<TouchSample[]> {
    if (this.touchStageState !== "capturing")
      throw new Error(
        "No active touch capture to stop. Call startTouch() first.",
      );
    this.touchController!.abort();
    this.touchData = await this.touchPromise!;
    this.touchStageState = "captured";
    return this.touchData;
  }

  skipTouch(): void {
    if (this.touchStageState !== "idle")
      throw new Error(
        "Cannot skip touch: capture already started. skipTouch() must be called before startTouch().",
      );
    this.touchStageState = "skipped";
  }

  // --- Test hooks (internal builds only) ---

  /**
   * @internal Test-only. Primes the session with pre-captured sensor data,
   * bypassing browser capture APIs. Throws unless built with IAM_INTERNAL_TEST=1.
   * Stripped from the published .d.ts so npm consumers never see it. Used by the
   * red team harness to drive the real verification pipeline (extraction →
   * SimHash → TBH → proof → submit) against synthetic sensor data — never
   * available to npm consumers.
   */
  __injectSensorData(data: {
    audio: AudioCapture;
    motion: MotionSample[];
    touch: TouchSample[];
  }): void {
    // typeof guard tolerates the constant being undeclared at runtime (e.g.
    // direct ts-node/tsx execution that bypasses tsup/vitest `define`).
    // Without this, a missing build-time replacement throws ReferenceError
    // before the user-facing message can fire.
    if (typeof __IAM_INTERNAL_TEST__ !== "boolean" || !__IAM_INTERNAL_TEST__) {
      throw new Error(
        "PulseSession.__injectSensorData is only available in internal test builds. " +
          "Set IAM_INTERNAL_TEST=1 when building pulse-sdk from source.",
      );
    }
    const conflicts: string[] = [];
    if (this.audioStageState === "capturing") conflicts.push("audio");
    if (this.motionStageState === "capturing") conflicts.push("motion");
    if (this.touchStageState === "capturing") conflicts.push("touch");
    if (conflicts.length > 0) {
      throw new Error(
        `__injectSensorData: cannot inject while stages are capturing: ${conflicts.join(", ")}. ` +
          `Create a fresh session via sdk.createSession() and inject before any startAudio/startMotion/startTouch call.`,
      );
    }
    if (!data.audio || data.audio.samples.length < MIN_AUDIO_SAMPLES) {
      throw new Error(
        `__injectSensorData: audio required, minimum ${MIN_AUDIO_SAMPLES} samples (got ${data.audio?.samples.length ?? 0}).`,
      );
    }
    if (data.motion.length < MIN_MOTION_SAMPLES) {
      throw new Error(
        `__injectSensorData: motion required, minimum ${MIN_MOTION_SAMPLES} samples (got ${data.motion.length}).`,
      );
    }
    if (data.touch.length < MIN_TOUCH_SAMPLES) {
      throw new Error(
        `__injectSensorData: touch required, minimum ${MIN_TOUCH_SAMPLES} samples (got ${data.touch.length}).`,
      );
    }
    this.audioData = data.audio;
    this.motionData = data.motion;
    this.touchData = data.touch;
    this.audioStageState = "captured";
    this.motionStageState = "captured";
    this.touchStageState = "captured";
  }

  // --- Complete ---

  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Solana types are optional peer deps
  async complete(wallet?: any, connection?: any, onProgress?: (stage: string) => void): Promise<VerificationResult> {
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

    return processSensorData(sensorData, this.config, wallet, connection, onProgress);
  }

  /**
   * Complete the session as a baseline RESET instead of a normal verify.
   *
   * Use when the wallet has an on-chain IdentityState but the device has
   * no recoverable local baseline (cleared site data, new device, etc).
   * Skips the Hamming ZK proof; submits `reset_identity_state` on chain,
   * which rotates the commitment and zeros verification history.
   *
   * Requires a connected wallet + Solana connection. Rejects if either
   * is missing — reset is a wallet-mode-only operation since it writes
   * to the user's on-chain account.
   */
  async completeReset(
    wallet: any,
    connection: any,
    onProgress?: (stage: string) => void
  ): Promise<VerificationResult> {
    const active: string[] = [];
    if (this.audioStageState === "capturing") active.push("audio");
    if (this.motionStageState === "capturing") active.push("motion");
    if (this.touchStageState === "capturing") active.push("touch");
    if (active.length > 0) {
      throw new Error(
        `Cannot complete reset: stages still capturing: ${active.join(", ")}`
      );
    }

    if (!wallet || !connection) {
      return {
        success: false,
        commitment: new Uint8Array(32),
        isFirstVerification: true,
        error:
          "Baseline reset requires a connected wallet and Solana connection. " +
          "Reset cannot be performed in walletless mode.",
      };
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

    return processResetSensorData(sensorData, this.config, wallet, connection, onProgress);
  }
}

/**
 * PulseSDK — main entry point for Entros Protocol verification.
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
    setDebug(config.debug ?? false);
    setPrivacyFallback(config.onPrivacyFallback);
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
        throw new Error(
          `Audio capture failed: ${err?.message ?? "microphone unavailable"}. Ensure microphone permission is granted and no other app is using it.`,
        );
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

  /**
   * Reset the wallet's on-chain baseline using a fresh capture.
   *
   * Convenience wrapper that mirrors `verify()` but routes the captured
   * sensor data through `reset_identity_state` instead of `update_anchor`.
   * Use when the wallet has an on-chain IdentityState but the local
   * encrypted baseline is unrecoverable.
   *
   * For fine-grained control, call `createSession()` and `completeReset()`
   * directly — the session API exposes per-stage start/stop hooks that
   * this convenience wrapper trades away for simplicity.
   */
  async resetBaseline(
    touchElement: HTMLElement | undefined,
    wallet: any,
    connection: any,
    onProgress?: (stage: string) => void
  ): Promise<VerificationResult> {
    try {
      const session = this.createSession(touchElement);
      const stopPromises: Promise<void>[] = [];

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

      try {
        await session.startAudio();
        stopPromises.push(
          new Promise<void>((r) => setTimeout(r, DEFAULT_CAPTURE_MS))
            .then(() => session.stopAudio())
            .then(() => {})
        );
      } catch (err: any) {
        throw new Error(
          `Audio capture failed: ${err?.message ?? "microphone unavailable"}. Ensure microphone permission is granted and no other app is using it.`,
        );
      }

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
      return session.completeReset(wallet, connection, onProgress);
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
