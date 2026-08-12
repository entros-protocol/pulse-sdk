import type { PulseConfig } from "./config";
import { DEFAULT_THRESHOLD, DEFAULT_MIN_DISTANCE, DEFAULT_CAPTURE_MS, AUDIO_READY_TIMEOUT_MS, PROGRAM_IDS, VALIDATE_UPLOAD_STALL_MS, VALIDATE_DEADLINE_MS } from "./config";
import { setDebug, sdkLog, sdkWarn } from "./log";
import type { SensorData, AudioCapture, MotionSample, TouchSample, StageState, CurveTracePoint } from "./sensor/types";
import type { TBH } from "./hashing/types";
import type { SolanaProof } from "./proof/types";
import type { SignedReceiptDto, VerificationResult, ProgressCallback } from "./submit/types";
import type { PostJsonResponse } from "./transport/post-json";
import { postJson, TransportError } from "./transport/post-json";
import type { StoredVerificationData } from "./identity/types";
import type { VerificationPhase } from "./phases";
import type { BaselineRecoveryReason } from "./identity/anchor";
import type { StudyContext, StudyRecordStatus } from "./study";

import { audioCaptureConstraints, captureAudio, analyzeAcousticRealism } from "./sensor/audio";
import { encodeAudioAsBase64 } from "./sensor/encode";
import { resampleCurveTrace } from "./sensor/curve";
import { captureMotion, requestMotionPermission } from "./sensor/motion";
import { captureTouch } from "./sensor/touch";
import { extractSpeakerFeaturesDetailed, SPEAKER_FEATURE_COUNT } from "./extraction/speaker";
import {
  extractMotionFeatures,
  extractTouchFeatures,
  extractMouseDynamics,
  extractAccelerationMagnitude,
  describeCaptureTiming,
  MOTION_FEATURE_COUNT,
  TOUCH_FEATURE_COUNT,
  type CaptureTiming,
} from "./extraction/kinematic";
import { fuseFeatures, fuseRawFeatures } from "./extraction/statistics";
import { yieldToMainThread } from "./yield";
import { collectClientSignals } from "./client-signals/automation";
import { simhash, hammingDistance } from "./hashing/simhash";
import {
  generateTBH,
  bigintToBytes32,
  computeCommitment,
} from "./hashing/poseidon";
import {
  prepareCircuitInput,
  generateProof,
  classifyHammingDistance,
  warmSnarkjs,
  prefetchCircuitArtifacts,
  takeCircuitArtifacts,
} from "./proof/prover";
import { warmMeyda } from "./extraction/mfcc";
import { serializeProof } from "./proof/serializer";
import {
  submitRebaselineViaWallet,
  submitResetViaWallet,
  submitViaWallet,
} from "./submit/wallet";
import { submitViaRelayer } from "./submit/relayer";
import { bytesToHex } from "./submit/receipt";
import {
  storeVerificationData,
  loadVerificationData,
  setPrivacyFallback,
  recoverBaselineFromChain,
  decodeIdentityState,
  fetchProjectionPolicy,
  localCommitmentMatchesChain,
} from "./identity/anchor";
import type { ProjectionPolicy } from "./identity/anchor";
import type { ProjectionPolicyConnection } from "./identity/anchor";
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
  /**
   * Observe-only summary of how motion sat against the audio window.
   * `undefined` when there was no motion to describe. Logged, never judged.
   */
  captureTiming?: CaptureTiming;
}

/**
 * Extract features from sensor data. Returns both raw (physical units)
 * and normalized (z-scored) feature vectors.
 */
async function extractFeatures(
  data: SensorData,
  projectionVersion: number,
): Promise<ExtractedFeatures> {
  if (!data.audio) {
    throw new Error(
      "Audio data missing. Capture audio via session.startAudio() before extracting features.",
    );
  }
  const { features: audioFeatures, f0Contour } = await extractSpeakerFeaturesDetailed(
    data.audio,
    projectionVersion,
  );
  // The audio path is the dominant cost. Yield once it's done so the
  // verify-flow spinner gets a paint frame before motion/touch extraction
  // resumes the main-thread work.
  await yieldToMainThread();

  const hasMotion = data.motion.length >= MIN_MOTION_SAMPLES;
  const hasTouch = data.touch.length >= MIN_TOUCH_SAMPLES;

  const motionFeatures =
    projectionVersion >= 1
      ? hasMotion
        ? extractMotionFeatures(data.motion, projectionVersion)
        : extractMouseDynamics(data.touch, projectionVersion)
      : hasMotion && hasTouch
        ? extractMouseDynamics(data.touch, projectionVersion)
        : hasMotion
          ? extractMotionFeatures(data.motion, projectionVersion)
          : extractMouseDynamics(data.touch, projectionVersion);
  await yieldToMainThread();

  const touchFeatures = extractTouchFeatures(data.touch, projectionVersion);
  await yieldToMainThread();

  // Resample acceleration magnitude onto the exact stretch of wall-clock time
  // the transmitted audio covers, at the F0 frame count, so the validator's
  // cross-correlation compares two views of one moment. Aligning by array
  // index instead is what broke mobile in 4.0.0, when the lead-in trim left
  // motion covering seconds that audio no longer did.
  // Empty if motion absent, F0 extraction produced no frames (e.g. a silent
  // capture), or motion does not span enough of the audio window.
  const accelMagnitude =
    hasMotion && f0Contour.length > 0
      ? extractAccelerationMagnitude(data.motion, f0Contour.length, {
          startMs: data.audio.windowStartMs,
          endMs: data.audio.windowEndMs,
        })
      : [];

  return {
    raw: fuseRawFeatures(audioFeatures, motionFeatures, touchFeatures),
    normalized: fuseFeatures(audioFeatures, motionFeatures, touchFeatures),
    f0Contour,
    accelMagnitude,
    // Described whenever motion exists, including when the contour above came
    // back empty. That case is exactly the one worth being able to read.
    captureTiming:
      data.motion.length > 0
        ? describeCaptureTiming(
            data.motion,
            { startMs: data.audio.windowStartMs, endMs: data.audio.windowEndMs },
            data.audio.inputLevel,
          )
        : undefined,
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
       * Validator-signed mint receipt. Present only when the request signaled
       * mint intent AND the validator has a signing key configured.
       * `undefined` means mint without an Ed25519 prefix — which the on-chain
       * `mint_anchor` rejects whenever the protocol's `validator_pubkey` is
       * configured (it is, on devnet): a missing or mismatched receipt
       * hard-fails the mint. The receipt binds the SERVER-derived commitment,
       * so the SDK must mint exactly that commitment (see the tbh swap below).
       */
      signedReceipt?: SignedReceiptDto;
      compositeRiskScore?: number;
      studyRecordStatus?: StudyRecordStatus;
    }
  | {
      ok: false;
      error: string;
      reason?: string;
      retryAfterSec?: number;
      failedAt: VerificationPhase;
      opaque?: boolean;
      studyRecordStatus?: StudyRecordStatus;
    };

/**
 * Turn a non-2xx `/validate-features` response into a rejection the host can
 * act on.
 *
 * Every non-2xx used to take one branch here, so a 413, a 429 and a 502 were
 * indistinguishable by the time they reached a host. The web app recovered
 * rate-limiting downstream by substring-matching the words "too many" in the
 * server's English prose, which meant any copy edit on the server silently
 * regressed the rate-limit UI to a generic failure. Reading the status makes
 * that unnecessary.
 *
 * Unrecognised server reasons pass through untouched. `reasonDisposition`
 * treats what it does not know as fatal, so a newer server can add a reason
 * without an old client mistakenly offering a retry for it.
 */
function rejectionFromStatus(response: PostJsonResponse): ExtractionResult {
  const body = response.body as {
    error?: unknown;
    reason?: unknown;
    retry_after?: unknown;
    study_record_status?: unknown;
  };
  const serverError = typeof body.error === "string" ? body.error : undefined;
  const serverReason = typeof body.reason === "string" ? body.reason : undefined;
  const studyRecordStatus = parseStudyRecordStatus(body.study_record_status);

  // Body before header. Cross-origin a browser only sees headers the server
  // lists in `Access-Control-Expose-Headers`, and the executor does not list
  // `retry-after`, which is why it puts the value in the body as well.
  const headerRetry = Number(response.header("retry-after"));
  const retryAfterSec =
    typeof body.retry_after === "number" && body.retry_after > 0
      ? body.retry_after
      : Number.isFinite(headerRetry) && headerRetry > 0
        ? headerRetry
        : undefined;

  // XHR reports 0 for a request that never produced a response. The spec
  // routes CORS, DNS and TLS failures to `onerror`, so the transport catches
  // them first and this is unreachable on the browser path. It is here so a
  // runtime that does surface a 0 cannot land in the final branch and tell
  // the user their capture failed validation when no server ever judged it.
  if (response.status === 0) {
    sdkWarn("[Entros SDK] Validation request produced no response");
    return {
      ok: false,
      error: "Validation service unreachable. Please check your connection and try again.",
      reason: "validation_unavailable",
      failedAt: "validation",
      studyRecordStatus,
    };
  }

  // The executor returns 408 when the request body stopped arriving and its
  // timeout layer reclaimed the connection. Distinct from 413: nothing was
  // wrong with the capture, so this must not consume a verification attempt.
  if (response.status === 408) {
    sdkWarn("[Entros SDK] Validation request body timed out in transit");
    return {
      ok: false,
      error:
        "The connection stalled while sending your verification. Move somewhere with better signal and try again.",
      reason: "validation_timeout",
      failedAt: "validation",
      studyRecordStatus,
    };
  }

  if (response.status === 413) {
    sdkWarn("[Entros SDK] Verification payload rejected as too large");
    return {
      ok: false,
      error:
        serverError ?? "Your verification data was too large to send. Please start over.",
      reason: "payload_too_large",
      failedAt: "validation",
      studyRecordStatus,
    };
  }

  if (response.status === 429) {
    sdkWarn("[Entros SDK] Verification rate limited");
    return {
      ok: false,
      error: serverError ?? "Too many requests. Please wait before trying again.",
      // `rate_limited`, `ip_rate_limited` and `cross_wallet_cooldown` all
      // arrive as 429 and mean different waits, so keep whichever the server
      // named rather than flattening them.
      reason: serverReason ?? "rate_limited",
      retryAfterSec,
      failedAt: "validation",
      studyRecordStatus,
    };
  }

  if (response.status >= 500) {
    // The server is unwell, not the capture. Transient, so retryable.
    sdkWarn(`[Entros SDK] Validation service returned HTTP ${response.status}`);
    return {
      ok: false,
      error: "Validation service is temporarily unavailable. Please try again.",
      reason: "validation_unavailable",
      failedAt: "validation",
      studyRecordStatus,
    };
  }

  sdkWarn("[Entros SDK] Feature validation rejected by server");
  return {
    ok: false,
    error: serverError ?? "Feature validation failed",
    reason: serverReason,
    retryAfterSec,
    failedAt: "validation",
    // A rejection the validator declined to label is an attack-signal
    // rejection: Sybil match, synthetic speech, or one of the checks it keeps
    // deliberately unnamed. The generic body it returns is all a host may
    // show. A labelled rejection names a capture-quality problem the user can
    // act on, so its text is safe.
    opaque: serverReason === undefined,
    studyRecordStatus,
  };
}

function parseStudyRecordStatus(value: unknown): StudyRecordStatus | undefined {
  return value === "recorded" ||
    value === "replayed" ||
    value === "technical_failure" ||
    value === "invalid_token" ||
    value === "disabled"
    ? value
    : undefined;
}

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
  onProgress?: ProgressCallback,
  studyContext?: StudyContext,
  projectionVersion = 0,
  receiptPurpose?: "mint" | "rebaseline" | "reset",
): Promise<ExtractionResult> {
  onProgress?.("Extracting features...");
  // Let React render the new stage label before we re-enter the heavy
  // synchronous extraction path. Without this, the host UI sets the
  // string but the main thread is captured by extractFeatures before
  // the spinner can repaint, and the user sees the previous stage's
  // label until extraction completes.
  await yieldToMainThread();
  // Extraction throws on malformed or unusable capture data, and that throw
  // used to escape `complete()` entirely, reaching the host as a bare string
  // with no phase, no reason and nothing to route on. Turning it into a
  // result keeps every failure on one typed surface.
  let extracted: ExtractedFeatures;
  try {
    extracted = await extractFeatures(sensorData, projectionVersion);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    sdkWarn(`[Entros SDK] Feature extraction failed: ${msg}`);
    return {
      ok: false,
      error:
        "We couldn't read the capture from your device. Start over and let the recording run to the end.",
      failedAt: "extraction",
    };
  }
  const {
    raw: features,
    normalized: normalizedFeatures,
    f0Contour,
    accelMagnitude,
    captureTiming,
  } = extracted;

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
  const fingerprint = simhash(normalizedFeatures, projectionVersion);
  // Local TBH with a client-random salt. This is the fallback used when the
  // validator doesn't return a server-derived commitment (older deploys); when
  // it does, we swap in the server's salt + commitment below (C2). The
  // fingerprint stays ours either way.
  let tbh = await generateTBH(fingerprint);

  let signedReceipt: SignedReceiptDto | undefined;
  let compositeRiskScore: number | undefined;
  let studyRecordStatus: StudyRecordStatus | undefined;

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
      // We also transmit the `sampleRate` of the buffer. Browsers treat the
      // 16kHz AudioContext request as a hint and some (Safari with Bluetooth
      // codec negotiation, some Android devices) deliver 44.1k or 48k
      // instead, so `sensor/audio.ts` decimates every capture to the
      // canonical 16kHz before extraction and before this encode. The field
      // therefore reads 16000 for any device that honoured the request and
      // any device that did not; it is transmitted rather than assumed so an
      // older client that predates that decimation still describes itself
      // accurately to the validator.
      const audioSamplesB64 = sensorData.audio?.samples
        ? encodeAudioAsBase64(sensorData.audio.samples)
        : undefined;
      const audioSampleRateHz = sensorData.audio?.sampleRate;

      // Touch-curve outline (wallet-connected verify only). Resampled to a
      // coarse, equal-time, timestamp-free outline — only the {x,y} points +
      // duration leave the device. Observe-only server-side; never affects the
      // decision. Absent (→ undefined → dropped) for reset/walletless.
      const curveTrace = sensorData.curveTrace
        ? resampleCurveTrace(sensorData.curveTrace)
        : undefined;

      // Hex-encode the 32-byte commitment for the validator's signing
      // input. The validator only signs when this field is present AND
      // its own signing key is configured; the SDK only consumes the
      // receipt on first-verification, so sending it on every
      // wallet-connected request is harmless on the re-verify path
      // (validator signs cheaply, executor passes through, SDK ignores
      // the field for `update_anchor`).
      const commitmentNewHex = bytesToHex(tbh.commitmentBytes);

      // Collect the observe-only client-signals envelope so the
      // executor can measure the bot-vs-human automation signal on real
      // traffic. Privacy-first — detects the automation harness driving the
      // page (Selenium/Puppeteer/Playwright/CDP), never the user; no
      // fingerprinting. The executor logs it and does NOT feed it into the
      // pass/fail decision. Non-browser runtimes (React Native) return a clean
      // marker. Synchronous + exception-safe, so it can never break submission.
      const clientSignals = collectClientSignals();
      if (sensorData.audio) {
        const acoustic = analyzeAcousticRealism(sensorData.audio.samples, sensorData.audio.sampleRate);
        clientSignals.capture = {
          virtual_device: !!sensorData.audio.virtualDevice,
          voice_isolation_applied: sensorData.audio.voiceIsolationApplied,
          flatness: parseFloat(acoustic.flatness.toFixed(4)),
          centroid: parseFloat(acoustic.centroid.toFixed(2)),
        };
      }

      const validateResponse = await postJson(
        validateUrl,
        {
          features,
          projection_version: projectionVersion,
          f0_contour: f0Contour,
          accel_magnitude: accelMagnitude,
          wallet_id: walletAddress,
          audio_samples_b64: audioSamplesB64,
          audio_sample_rate_hz: audioSampleRateHz,
          commitment_new_hex: commitmentNewHex,
          // Explicit mint-intent signal. New validators sign a receipt over a
          // commitment THEY derive from `features`; `commitment_new_hex` is
          // still sent so older validators (which trust it) keep working.
          request_receipt: receiptPurpose !== undefined,
          receipt_purpose: receiptPurpose,
          baseline_reset: receiptPurpose === "reset",
          // Observe-only automation-detection signal. Optional and
          // additive — the executor logs it; older executors ignore the
          // unknown field. Never affects the verification decision.
          client_signals: clientSignals,
          // Observe-only touch content-binding signal (curve-trace outline).
          // Optional + additive; older executors ignore it. `undefined` when no
          // outline was captured, and JSON.stringify then omits it entirely.
          curve_trace: curveTrace,
          // Observe-only capture-timing summary: how the motion stream sat
          // against the audio window `accel_magnitude` was resampled onto.
          // Optional and additive. Older validators ignore the unknown field.
          // Logged for calibration, never read by a check. See
          // `describeCaptureTiming`.
          capture_timing: captureTiming,
          study: studyContext,
        },
        {
          headers: validateHeaders,
          stallMs: VALIDATE_UPLOAD_STALL_MS,
          deadlineMs: VALIDATE_DEADLINE_MS,
          onUploadProgress: (loaded, total) => {
            // Same stage label as before. `popup-content.tsx` matches on
            // these strings to drive the embed wire protocol's heartbeat, so
            // the text is API. The progress argument is additive.
            onProgress?.("Validating...", total > 0 ? { loaded, total } : undefined);
          },
        },
      );

      if (validateResponse.status < 200 || validateResponse.status >= 300) {
        return rejectionFromStatus(validateResponse);
      }

      // Parse the validator's success body for the signed receipt and the
      // server-derived commitment + salt. A returning verification can use a
      // successful response without a mint receipt. The first-verification
      // wallet path fails closed before requesting a wallet transaction when
      // the receipt is absent.
      try {
        const successBody = validateResponse.body as {
          signed_receipt?: SignedReceiptDto;
          commitment_hex?: string;
          salt_hex?: string;
          composite_risk_score?: number;
          study_record_status?: unknown;
        };
        if (successBody.signed_receipt) {
          signedReceipt = successBody.signed_receipt;
        }
        if (successBody.composite_risk_score !== undefined) {
          compositeRiskScore = successBody.composite_risk_score;
          sdkLog(`[Entros SDK] Validation composite risk score: ${compositeRiskScore.toFixed(4)}`);
        }
        studyRecordStatus = parseStudyRecordStatus(successBody.study_record_status);
        // C2: adopt the validator-derived commitment + salt. The validator
        // signs — and the chain enforces — a commitment it computed from the
        // features we sent, not one we chose, so we must mint exactly that.
        // Our local `fingerprint` is unchanged and stays consistent with the
        // server commitment: the validator derived it from a byte-identical
        // fingerprint (parity-tested across SDK/validator/circuit) under this
        // salt, so the {fingerprint, salt, commitment} triple still opens for
        // future rotation proofs.
        if (successBody.commitment_hex && successBody.salt_hex) {
          const serverCommitment = BigInt("0x" + successBody.commitment_hex);
          const serverSalt = BigInt("0x" + successBody.salt_hex);
          tbh = {
            fingerprint,
            salt: serverSalt,
            commitment: serverCommitment,
            commitmentBytes: bigintToBytes32(serverCommitment),
          };
          if (config.debug) {
            // Runtime cross-check of the parity contract: the commitment we'd
            // compute locally from our fingerprint under the server salt must
            // equal the server's. A mismatch means the SDK and validator
            // SimHash/Poseidon have drifted (independently deployed) — future
            // rotation proofs would silently fail to open — so flag it loudly.
            const localCheck = await computeCommitment(fingerprint, serverSalt);
            if (localCheck !== serverCommitment) {
              sdkWarn(
                "[Entros SDK] Commitment parity check failed: the validator-derived commitment does not match a local recomputation. SDK and validator may be out of sync."
              );
            }
          }
        }
      } catch (err) {
        // Body was not JSON. Keep the validated feature result for returning
        // verification. A first-verification wallet submission will reject
        // the missing receipt before requesting a signature.
        const msg = err instanceof Error ? err.message : String(err);
        sdkWarn(
          `[Entros SDK] /validate-features returned 200 but body was not parseable JSON; no mint receipt is available: ${msg}`
        );
      }
    } catch (err) {
      // The request never produced a response. Previously this path silently
      // continued and skipped server-side validation, which let a
      // network-failure attacker bypass server-side checks entirely, so it
      // returns a recoverable error and the host surfaces a retry CTA.
      //
      // The four transport failures used to collapse into one string, which
      // is how a slow uplink came to be reported as an unreachable service.
      // A timeout says something different from an unreachable host, and the
      // user can act on the difference.
      const msg = err instanceof Error ? err.message : String(err);
      if (err instanceof TransportError && (err.kind === "stalled" || err.kind === "deadline")) {
        sdkWarn(`[Entros SDK] Feature validation timed out: ${msg}`);
        return {
          ok: false,
          error:
            "The connection stalled while sending your verification. Move somewhere with better signal and try again.",
          reason: "validation_timeout",
          failedAt: "validation",
        };
      }
      sdkWarn(`[Entros SDK] Feature validation unavailable: ${msg}`);
      return {
        ok: false,
        error: "Validation service unreachable. Please check your connection and try again.",
        reason: "validation_unavailable",
        failedAt: "validation",
      };
    }
  }

  return { ok: true, features, f0Contour, accelMagnitude, fingerprint, tbh, signedReceipt, compositeRiskScore, studyRecordStatus };
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

/**
 * Explain why a re-verification cannot proceed without a baseline.
 *
 * One screen used to cover every one of these, which was wrong for most of the
 * population it was shown to. Only 13 of 107 anchors on devnet have an
 * `EncryptedBaseline` PDA at all: the rest were minted before the feature
 * existed, and telling those users that a fingerprint was "not found on this
 * device" describes a search that could never have succeeded and points at a
 * device that was never at fault.
 *
 * Each branch names the actual obstacle and the way past it. The substrings
 * `different wallet signed`, `out of sync with your on-chain identity` and
 * `baseline is missing` are a routing contract with `entros.io`, which matches
 * them to pick a surface. They stay until every host reads
 * `VerificationResult.baselineRecovery` instead.
 */
function baselineFailureMessage(
  reason: BaselineRecoveryReason | undefined,
  localIsStale: boolean,
): string {
  switch (reason) {
    case "wallet-mismatch":
      // The on-chain baseline is intact. This one must not offer a reset.
      return (
        "A different wallet signed than the one connected. Another wallet extension likely intercepted the signature prompt. " +
        "Sign with your connected wallet, or disable other wallet extensions (or unset their default), then try again."
      );
    case "no-encrypted-baseline":
      return (
        "Your Entros Anchor was created before on-chain baseline storage existed, so the local baseline is missing and there is no encrypted copy on chain to restore it from. " +
        "Reset your baseline once from this device and it becomes recoverable on any device."
      );
    case "signing-unavailable":
      return (
        "This wallet cannot sign the message that unlocks your on-chain baseline, so the local baseline is missing and cannot be restored here. " +
        "Connect a wallet that supports message signing, or reset your baseline to re-enrol from this device."
      );
    case "stale-baseline":
      return (
        "Your on-chain baseline was written under an earlier verification and can no longer be unlocked, so the local baseline is missing. " +
        "Reset your baseline to re-enrol from this device."
      );
    default:
      // `no-on-chain-identity`, `unknown-error`, and the walletless path where
      // recovery was never attempted.
      return localIsStale
        ? "Your baseline is out of sync with your on-chain identity. It may have advanced on another browser or device. " +
            "Reset your baseline to re-sync from here, or verify from the device with the up-to-date baseline."
        : "Previous behavioral fingerprint not found on this device. Your Entros Anchor exists on-chain but the local baseline is missing. " +
            "Reset your baseline to re-enroll from this device, or verify from the device that has the original baseline.";
  }
}

async function processSensorData(
  sensorData: SensorData,
  config: ResolvedConfig,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Solana types are optional peer deps
  wallet?: any,
  connection?: any,
  onProgress?: ProgressCallback,
  studyContext?: StudyContext,
  projectionPolicy: ProjectionPolicy = { current: 0, minimum: 0 },
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
      failedAt: "capture",
    };
  }

  if (!hasAudio) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: true,
      error: "No voice data detected. Please speak the phrase clearly during capture.",
      failedAt: "capture",
    };
  }

  const walletAddress = wallet?.adapter?.publicKey?.toBase58?.()
    ?? wallet?.publicKey?.toBase58?.();

  let previousData = await loadVerificationData(walletAddress);
  let isFirstVerification = previousData === null;
  let onChainCommitment: Uint8Array | null = null;
  let onChainProjectionVersion: number | null = null;

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
        isFirstVerification = !accountInfo;
        const identity = accountInfo
          ? await decodeIdentityState(accountInfo.data)
          : null;
        if (accountInfo && !identity) {
          return {
            success: false,
            commitment: new Uint8Array(32),
            isFirstVerification: false,
            error: "The on-chain identity could not be decoded. Update the app and try again.",
            failedAt: "submission",
          };
        }
        onChainCommitment = identity?.currentCommitment ?? null;
        onChainProjectionVersion = identity?.projectionVersion ?? null;
      } catch {
        return {
          success: false,
          commitment: new Uint8Array(32),
          isFirstVerification: false,
          error: "The on-chain identity could not be read. Check your connection and try again.",
          failedAt: "submission",
        };
      }
    }
  }

  if (!isFirstVerification && onChainProjectionVersion === null && wallet && connection) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: false,
      error: "The on-chain identity projection version could not be read. Please try again.",
      failedAt: "submission",
    };
  }
  if (
    onChainProjectionVersion !== null &&
    onChainProjectionVersion > projectionPolicy.current
  ) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: false,
      error: "This identity uses a newer projection version. Update the app and try again.",
      failedAt: "submission",
    };
  }

  const needsProjectionMigration =
    !isFirstVerification &&
    onChainProjectionVersion !== null &&
    onChainProjectionVersion < projectionPolicy.current;

  // Re-verification requires audio + at least one other modality.
  // Audio-only fingerprints lack inter-session variance from motion/touch.
  if (!isFirstVerification && !hasMotion && !hasTouch) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: false,
      error: "Insufficient sensor data for re-verification. Please trace the curve and allow motion access.",
      failedAt: "capture",
    };
  }

  const extraction = await extractFingerprintAndValidate(
    sensorData,
    config,
    walletAddress,
    onProgress,
    studyContext,
    projectionPolicy.current,
    needsProjectionMigration ? "rebaseline" : isFirstVerification ? "mint" : undefined,
  );
  if (!extraction.ok) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: false,
      error: extraction.error,
      reason: extraction.reason,
      retryAfterSec: extraction.retryAfterSec,
      failedAt: extraction.failedAt,
      opaque: extraction.opaque,
      studyRecordStatus: extraction.studyRecordStatus,
    };
  }
  const { fingerprint, tbh, features, signedReceipt, compositeRiskScore, studyRecordStatus } = extraction;

  // A local baseline is STALE when it exists but its commitment no longer
  // equals the on-chain head — proving from it would revert with 6011. We
  // evaluate it both before and after recovery (which may rewrite
  // `previousData`), so the check lives in one helper.
  const isBaselineStale = (data: typeof previousData): boolean =>
    !isFirstVerification &&
    data !== null &&
    onChainCommitment !== null &&
    (!localCommitmentMatchesChain(data.commitment, onChainCommitment) ||
      data.projectionVersion !== onChainProjectionVersion);

  const localBaselineStale = isBaselineStale(previousData);

  // If the legacy baseline matched, migrate it to the keyed location and remove the legacy storage entry.
  const STORAGE_KEY = "entros-protocol-verification-data";
  if (
    !needsProjectionMigration &&
    previousData &&
    !localBaselineStale &&
    walletAddress &&
    typeof localStorage !== "undefined" &&
    !localStorage.getItem(`${STORAGE_KEY}_${walletAddress}`)
  ) {
    await storeVerificationData(previousData, walletAddress);
    localStorage.removeItem(STORAGE_KEY);
  }

  // Recover the in-sync baseline from the on-chain `EncryptedBaseline` PDA
  // when the local copy is MISSING (cleared data, new device) OR STALE (fell
  // behind the chain, e.g. a verify from another origin/device). Recovery
  // decrypts the blob — which is bound to the current on-chain commitment —
  // and rewrites local storage with the matching fingerprint + current
  // commitment, so the rebuilt proof's `commitment_prev` equals the chain
  // head. Requires the wallet to sign for AES key derivation; the
  // session-cached key short-circuits a second prompt (master-list #98).
  let recoveryReason: BaselineRecoveryReason | undefined;
  if (
    !isFirstVerification &&
    !needsProjectionMigration &&
    (previousData === null || localBaselineStale) &&
    wallet &&
    connection
  ) {
    const baselineWallet = resolveBaselineWallet(wallet);
    if (!baselineWallet) {
      // The adapter cannot sign a message, so the key that decrypts the
      // on-chain blob cannot be derived and recovery is never attempted.
      //
      // This branch used to fall through in total silence. Its only trace was
      // an `sdkLog` that never ran, because both `sdkLog` and `sdkWarn` are
      // gated on `debug` and `entros.io` derives `debug` from
      // `NODE_ENV === "development"`. In production it was indistinguishable
      // from every other reason recovery might not have happened, and the user
      // was told their baseline was missing with no way to learn why. A louder
      // log would not have fixed that either, since nobody reads a console:
      // the reason travels in the result instead.
      recoveryReason = "signing-unavailable";
    } else {
      onProgress?.(
        localBaselineStale
          ? "Re-syncing baseline with chain..."
          : "Recovering baseline from chain..."
      );
      const recovery = await recoverBaselineFromChain(baselineWallet, connection);
      if (recovery.recovered) {
        previousData = await loadVerificationData(walletAddress);
        sdkLog(
          `[Entros SDK] On-chain encrypted baseline ${localBaselineStale ? "re-synced" : "recovered"}`
        );
      } else {
        // All six reasons are kept, not just `wallet-mismatch`. Collapsing the
        // other five put three different situations on one screen: an anchor
        // that predates on-chain baselines entirely, a blob that can no longer
        // be decrypted, and a wallet that cannot sign. Each has a different
        // way out, and the user was shown none of them.
        recoveryReason = recovery.reason ?? "unknown-error";
        sdkLog(
          `[Entros SDK] On-chain encrypted baseline recovery not available (${recoveryReason})`,
        );
      }
    }
  }

  // Re-confirm against the chain head AFTER any recovery attempt. If local is
  // still missing or stale, proving would revert with 6011 — so fail HERE,
  // before a signature/fee is spent, and route the user to the right surface.
  const baselineStillStale = isBaselineStale(previousData);

  if (
    !isFirstVerification &&
    !needsProjectionMigration &&
    (previousData === null || baselineStillStale)
  ) {
    return {
      success: false,
      commitment: tbh.commitmentBytes,
      isFirstVerification: false,
      error: baselineFailureMessage(recoveryReason, baselineStillStale),
      failedAt: "baseline",
      baselineRecovery: recoveryReason,
    };
  }

  let solanaProof: SolanaProof | null = null;

  if (!isFirstVerification && !needsProjectionMigration && previousData) {
    onProgress?.("Computing proof...");
    const previousTBH: TBH = {
      fingerprint: previousData.fingerprint,
      salt: BigInt(previousData.salt),
      commitment: BigInt(previousData.commitment),
      commitmentBytes: bigintToBytes32(BigInt(previousData.commitment)),
    };

    const distance = hammingDistance(fingerprint, previousData.fingerprint);
    // Single source for the circuit's accept band: the same threshold and
    // min_distance feed BOTH the pre-check below and the proof input, so the
    // pre-check can never disagree with what entros_hamming.circom enforces.
    const threshold = config.threshold ?? DEFAULT_THRESHOLD;
    const minDistance = DEFAULT_MIN_DISTANCE;
    const verdict = classifyHammingDistance(distance, threshold, minDistance);
    sdkLog(
      `[Entros SDK] Re-verification: Hamming distance = ${distance} / 256 bits (threshold = ${threshold}, min = ${minDistance}) -> ${verdict}`
    );

    // The capture already violates the circuit's bounds — return a clean,
    // user-actionable result instead of attempting a proof that would throw a
    // raw circom assert. Skips proof setup AND the submit signature.
    if (verdict === "drift_too_high") {
      return {
        success: false,
        commitment: tbh.commitmentBytes,
        isFirstVerification: false,
        error:
          "This capture didn't closely match your usual pattern. That can happen when the recording is interrupted or your movements are rushed. Please try again with a steady, uninterrupted capture.",
        failedAt: "proving",
      };
    }
    if (verdict === "below_min_distance") {
      // Replay floor: the new capture is near-identical to the previous one.
      // Stay opaque (don't reveal "too similar") — route to the same
      // validation-rejected surface as other attack-signal rejections.
      return {
        success: false,
        commitment: tbh.commitmentBytes,
        isFirstVerification: false,
        error: "Verification rejected. Please try again.",
        failedAt: "proving",
        // The replay floor is an attack signal, so it must read exactly like a
        // validator rejection in `validation` and a program revert in
        // `confirmation`. An honest phase is only safe alongside this flag.
        opaque: true,
      };
    }

    const circuitInput = prepareCircuitInput(
      tbh,
      previousTBH,
      threshold,
      minDistance
    );

    const wasmPath = config.wasmUrl;
    const zkeyPath = config.zkeyUrl;

    if (!wasmPath || !zkeyPath) {
      return {
        success: false,
        commitment: tbh.commitmentBytes,
        isFirstVerification: false,
        error: "Re-verification requires wasmUrl and zkeyUrl in PulseConfig. Host the entros_hamming.wasm and entros_hamming_final.zkey circuit artifacts at public URLs.",
        failedAt: "proving",
      };
    }

    try {
      // Collected from the prefetch started at capture start. Null when it was
      // never started, was aborted, or failed, in which case snarkjs fetches
      // the URLs itself exactly as it always has.
      const artifacts = await takeCircuitArtifacts(wasmPath, zkeyPath);
      const { proof, publicSignals } = await generateProof(
        circuitInput,
        artifacts ? artifacts.wasm : wasmPath,
        artifacts ? artifacts.zkey : zkeyPath
      );
      solanaProof = serializeProof(proof, publicSignals);
    } catch (proofErr: any) {
      // Bounds violations (drift / replay floor) are handled by the pre-check
      // above, so reaching here means a genuine proving failure (artifact
      // fetch, snarkjs internal, OOM). Diagnostics go to gated dev logs only —
      // derived feature values must never reach the UI or default production
      // logs (privacy). Block boundaries derived from extractor constants so
      // they stay in sync with the v2 layout if any modality count shifts.
      const motionStart = SPEAKER_FEATURE_COUNT;
      const touchStart = motionStart + MOTION_FEATURE_COUNT;
      const touchEnd = touchStart + TOUCH_FEATURE_COUNT;
      const audioNZ = features.slice(0, motionStart).filter((v) => v !== 0).length;
      const motionNZ = features.slice(motionStart, touchStart).filter((v) => v !== 0).length;
      const touchNZ = features.slice(touchStart, touchEnd).filter((v) => v !== 0).length;
      const rawAudio = sensorData.audio?.samples.length ?? 0;
      const rawMotion = sensorData.motion.length;
      const rawTouch = sensorData.touch.length;
      // First 3 feature values as a fingerprint to detect identical data.
      const sig = features.slice(0, 3).map((v) => v.toFixed(4)).join(",");
      sdkWarn(
        `[Entros SDK] Proof generation failed: ${proofErr?.message ?? proofErr}. dist=${distance}, nz=${audioNZ}/${motionNZ}/${touchNZ}, raw=${rawAudio}/${rawMotion}/${rawTouch}, sig=${sig}`
      );
      return {
        success: false,
        commitment: tbh.commitmentBytes,
        isFirstVerification: false,
        error:
          "We couldn't generate the verification proof. Check your connection and try again.",
        failedAt: "proving",
      };
    }
  }

  // Submit
  onProgress?.("Submitting to Solana...");
  let submission;
  // Whether this verification also wrote the portable on-chain copy of the
  // baseline. Hoisted out of the wallet branch so the result can report it:
  // a verification that advances the commitment without rewriting the blob is
  // how an identity quietly stops being recoverable anywhere else, and the
  // user found out about it on their next device rather than here.
  let portableBaseline: boolean | undefined;

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
    portableBaseline = encryptedBaselineBlob !== undefined;

    if (needsProjectionMigration) {
      if (!signedReceipt) {
        return {
          success: false,
          commitment: tbh.commitmentBytes,
          isFirstVerification: false,
          error: "Projection migration requires a validator-signed receipt.",
          failedAt: "submission",
        };
      }
      if (!encryptedBaselineBlob) {
        return {
          success: false,
          commitment: tbh.commitmentBytes,
          isFirstVerification: false,
          error: "Projection migration requires a wallet that supports message signing.",
          failedAt: "submission",
        };
      }
      submission = await submitRebaselineViaWallet(
        tbh.commitmentBytes,
        projectionPolicy.current,
        {
          wallet,
          connection,
          signedReceipt,
          encryptedBaselineBlob,
        },
      );
    } else if (isFirstVerification) {
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
          onProgress: (stage) => onProgress?.(stage),
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
        onProgress: (stage) => onProgress?.(stage),
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
      failedAt: "submission",
    };
  }

  // Store verification data locally for next re-verification.
  //
  // A throw here must not discard a verification the chain already accepted.
  // Storage fails for reasons that have nothing to do with the capture: a full
  // quota, a private-browsing mode, a wiped keystore. Unlike the reset path,
  // which cannot leave the user with a usable identity if this fails, a verify
  // that loses its local copy is recoverable. The next attempt finds no local
  // baseline and pulls the encrypted one from chain.
  if (submission.success) {
    try {
      await storeVerificationData({
        fingerprint: tbh.fingerprint,
        salt: tbh.salt.toString(),
        commitment: tbh.commitment.toString(),
        timestamp: Date.now(),
        projectionVersion: projectionPolicy.current,
      }, walletAddress);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      sdkWarn(
        `[Entros SDK] Verification confirmed on chain but the local baseline could not be saved. The next verification will recover it from chain: ${msg}`,
      );
    }
  }

  return {
    success: submission.success,
    commitment: tbh.commitmentBytes,
    txSignature: submission.txSignature,
    attestationTx: submission.attestationTx,
    isFirstVerification,
    error: submission.error,
    // Only meaningful when the transaction landed. On a failure nothing was
    // written, so reporting `true` here would describe an instruction that was
    // built and then thrown away.
    portableBaseline: submission.success ? portableBaseline : undefined,
    failedAt: submission.failedAt,
    // A program revert has to read like a validator rejection, or which layer
    // caught the attempt becomes calibration information. A host may still
    // match `Custom 6011` and `Custom 6012` first: those name protocol state
    // the user has to act on, not the outcome of a detection check.
    //
    // Only `confirmation` qualifies, and by construction it is only ever set
    // when the cluster reported an execution failure. A declined prompt or a
    // network problem carries a message that is safe to show.
    opaque: submission.failedAt === "confirmation",
    compositeRiskScore,
    studyRecordStatus,
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
  onProgress?: ProgressCallback,
  projectionPolicy: ProjectionPolicy = { current: 0, minimum: 0 },
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
      failedAt: "capture",
    };
  }

  if (!hasAudio) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: true,
      error: "No voice data detected. Please speak the phrase clearly during capture.",
      failedAt: "capture",
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
      failedAt: "capture",
    };
  }

  const walletAddress = wallet.adapter?.publicKey?.toBase58?.()
    ?? wallet.publicKey?.toBase58?.();
  const extraction = await extractFingerprintAndValidate(
    sensorData,
    config,
    walletAddress,
    onProgress,
    undefined,
    projectionPolicy.current,
    projectionPolicy.current >= 1 ? "reset" : undefined,
  );
  if (!extraction.ok) {
    return {
      success: false,
      commitment: new Uint8Array(32),
      isFirstVerification: true,
      error: extraction.error,
      reason: extraction.reason,
      retryAfterSec: extraction.retryAfterSec,
      failedAt: extraction.failedAt,
      opaque: extraction.opaque,
    };
  }
  const { tbh, compositeRiskScore, signedReceipt } = extraction;

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
  const portableBaseline = encryptedBaselineBlob !== undefined;

  onProgress?.("Submitting reset to Solana...");
  const submission = await submitResetViaWallet(tbh.commitmentBytes, {
    wallet,
    connection,
    relayerUrl: config.relayerUrl,
    relayerApiKey: config.relayerApiKey,
    projectionVersion: projectionPolicy.current,
    signedReceipt,
    encryptedBaselineBlob,
    onProgress: (stage) => onProgress?.(stage),
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
        projectionVersion: projectionPolicy.current,
      }, walletAddress);
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
        // The chain write landed. What failed is local baseline storage, which
        // is what `baseline` covers, and the message is safe to show as-is.
        failedAt: "baseline",
        compositeRiskScore,
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
    portableBaseline: submission.success ? portableBaseline : undefined,
    failedAt: submission.failedAt,
    opaque: submission.failedAt === "confirmation",
    compositeRiskScore,
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
  private studyContext: StudyContext | undefined;

  private audioStageState: StageState = "idle";
  private motionStageState: StageState = "idle";
  private touchStageState: StageState = "idle";

  private audioController: AbortController | null = null;
  /** Fires the capture-window mark; see `markCaptureStart`. */
  private captureWindowController: AbortController | null = null;
  private motionController: AbortController | null = null;
  private touchController: AbortController | null = null;

  private audioPromise: Promise<AudioCapture | null> | null = null;
  private motionPromise: Promise<MotionSample[]> | null = null;
  private touchPromise: Promise<TouchSample[]> | null = null;

  private audioData: AudioCapture | null = null;
  private motionData: MotionSample[] = [];
  private touchData: TouchSample[] = [];
  private projectionPolicyPromise: Promise<ProjectionPolicy | null>;

  constructor(config: ResolvedConfig, touchElement?: HTMLElement, studyContext?: StudyContext) {
    this.config = config;
    this.touchElement = touchElement;
    this.studyContext = studyContext;
    this.projectionPolicyPromise = this.readProjectionPolicy().catch(() => null);
  }

  private async readProjectionPolicy(
    connection?: ProjectionPolicyConnection,
  ): Promise<ProjectionPolicy> {
    if (connection) return fetchProjectionPolicy(connection);
    const { clusterApiUrl, Connection } = await import("@solana/web3.js");
    const endpoint =
      this.config.rpcEndpoint ??
      (this.config.cluster === "localnet"
        ? "http://127.0.0.1:8899"
        : clusterApiUrl(this.config.cluster));
    return fetchProjectionPolicy(new Connection(endpoint, "confirmed"));
  }

  private async resolveProjectionPolicy(
    connection?: ProjectionPolicyConnection,
  ): Promise<ProjectionPolicy> {
    // The constructor read only warms the configured RPC path. Completion
    // always refreshes policy so a pre-cutover response cannot select the
    // projection after the administrator changes on-chain configuration.
    return this.readProjectionPolicy(connection);
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
    const prefetchedPolicy = await this.projectionPolicyPromise;
    const captureProjectionVersion = prefetchedPolicy?.current ?? 0;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: audioCaptureConstraints(captureProjectionVersion),
    });

    this.audioStageState = "capturing";
    this.audioController = new AbortController();
    this.captureWindowController = new AbortController();

    // Resolve startAudio() only once audio is actually flowing — i.e. the
    // first real frame has been delivered — so callers don't begin the
    // "speak now" prompt during the AudioContext/mic cold-start gap (which
    // dropped the start of the phrase on first attempts). The timeout caps
    // the wait so a non-delivering mic can't hang the flow indefinitely.
    let signalReady!: () => void;
    const ready = new Promise<void>((resolve) => {
      signalReady = resolve;
    });
    this.audioPromise = captureAudio({
      signal: this.audioController.signal,
      onAudioLevel,
      onReady: () => signalReady(),
      captureWindowSignal: this.captureWindowController.signal,
      stream,
      projectionVersion: captureProjectionVersion,
    }).catch(() => {
      stream.getTracks().forEach((t) => t.stop());
      signalReady(); // unblock startAudio if setup failed before the first frame
      return null;
    });
    let readyTimer: ReturnType<typeof setTimeout> | undefined;
    await Promise.race([
      ready,
      new Promise<void>((resolve) => {
        readyTimer = setTimeout(resolve, AUDIO_READY_TIMEOUT_MS);
      }),
    ]);
    // Clear the safety timer if the first frame won the race, so a resolved
    // startAudio() never leaves a dangling timeout pending.
    if (readyTimer !== undefined) clearTimeout(readyTimer);

    // Everything the rest of the flow initialises lazily, started now.
    //
    // None of it depends on the capture, and all of it currently lands on the
    // user's critical path: the Meyda bundle is needed the instant capture
    // ends, and snarkjs plus 2.64 MiB of circuit artifacts are fetched after
    // the validator returns. Started here they have the rest of the capture,
    // plus the validator's four-second floor, to finish in.
    //
    // Poseidon is absent from this list on purpose. It used to be the largest
    // entry at 381 ms, and swapping `circomlibjs` for `poseidon-lite` removed
    // the WASM compile rather than merely rescheduling it, so there is nothing
    // left to warm.
    //
    // Placed after the readiness race on purpose. These parse a sizable bundle
    // on the main thread, and the AudioContext and microphone cold-start is the
    // one moment in the flow least able to absorb that — dropping the start of
    // the phrase is a failure this handshake exists to prevent. The few hundred
    // milliseconds of runway given up here is bought back many times over by
    // not contending with mic startup.
    //
    // Fire and forget, deliberately. Each helper resolves rather than rejects,
    // and `void` documents that nothing awaits them. A warm-up that fails
    // leaves the original lazy path to pay the cost later, so the worst case is
    // today's behaviour.
    void warmMeyda();
    void warmSnarkjs();
    if (this.config.wasmUrl && this.config.zkeyUrl) {
      // No abort signal, deliberately. Every controller in scope here
      // (`audioController`, `captureWindowController`) is aborted on *normal*
      // completion — `stopAudio` fires the first to release the microphone — so
      // wiring any of them up would cancel the download at the exact moment the
      // proof is about to need it. There is no session teardown hook to attach
      // to instead, and none is worth inventing: the transfer is bounded by its
      // own timeout, the browser cancels in-flight requests on navigation, and
      // a first verification that never proves simply leaves the bytes cached
      // for the re-verification that follows, which is the next thing that user
      // does.
      prefetchCircuitArtifacts(this.config.wasmUrl, this.config.zkeyUrl);
    }
  }

  /**
   * Tell the SDK that the capture window has opened, i.e. the speak prompt is
   * on screen and the user is about to talk.
   *
   * `startAudio()` deliberately resolves as soon as real audio is flowing, so
   * the prompt never appears during the microphone's cold start. The cost is
   * that recording begins before the prompt does, and on a slow connection the
   * challenge fetch sits inside that gap too. Left unmarked, several seconds
   * of silence are fingerprinted and uploaded as though they were speech.
   *
   * Optional. Without it the whole recording is used, which is the previous
   * behaviour.
   */
  markCaptureStart(): void {
    if (this.audioStageState !== "capturing") {
      // Not thrown, because a host that calls this defensively should not lose
      // a verification over it. Warned, because the failure is otherwise
      // invisible: the capture still succeeds, it just silently transmits the
      // dead air this call exists to remove.
      sdkWarn(
        "[Entros SDK] markCaptureStart() ignored: no audio capture in progress. Call it between startAudio() and stopAudio().",
      );
      return;
    }
    this.captureWindowController?.abort();
  }

  async stopAudio(): Promise<AudioCapture | null> {
    if (this.audioStageState !== "capturing")
      throw new Error(
        "No active audio capture to stop. Call startAudio() first.",
      );
    this.audioController!.abort();
    this.audioData = await this.audioPromise!;
    this.captureWindowController = null;
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
  async complete(wallet?: any, connection?: any, onProgress?: ProgressCallback, outline?: CurveTracePoint[]): Promise<VerificationResult> {
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
      curveTrace: outline,
      modalities: {
        audio: this.audioData !== null,
        motion: this.motionData.length > 0,
        touch: this.touchData.length > 0,
      },
    };

    let projectionPolicy: ProjectionPolicy;
    try {
      projectionPolicy = await this.resolveProjectionPolicy(connection);
    } catch (err) {
      return {
        success: false,
        commitment: new Uint8Array(32),
        isFirstVerification: true,
        error: err instanceof Error ? err.message : String(err),
        failedAt: "submission",
      };
    }

    return processSensorData(
      sensorData,
      this.config,
      wallet,
      connection,
      onProgress,
      this.studyContext,
      projectionPolicy,
    );
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
    onProgress?: ProgressCallback
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
        failedAt: "submission",
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

    let projectionPolicy: ProjectionPolicy;
    try {
      projectionPolicy = await this.resolveProjectionPolicy(connection);
    } catch (err) {
      return {
        success: false,
        commitment: new Uint8Array(32),
        isFirstVerification: true,
        error: err instanceof Error ? err.message : String(err),
        failedAt: "submission",
      };
    }

    return processResetSensorData(
      sensorData,
      this.config,
      wallet,
      connection,
      onProgress,
      projectionPolicy,
    );
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
  createSession(touchElement?: HTMLElement, studyContext?: StudyContext): PulseSession {
    return new PulseSession(this.config, touchElement, studyContext);
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
        // This catch wraps the whole one-shot capture, so what reaches it is
        // a sensor that would not start or stop: a denied microphone, a denied
        // motion permission, a stream that never opened. It also nets a throw
        // out of `complete()`, which is a caller error rather than a capture
        // one, and rare enough not to justify a phase of its own.
        failedAt: "capture",
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
    onProgress?: ProgressCallback
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
        // This catch wraps the whole one-shot capture, so what reaches it is
        // a sensor that would not start or stop: a denied microphone, a denied
        // motion permission, a stream that never opened. It also nets a throw
        // out of `complete()`, which is a caller error rather than a capture
        // one, and rare enough not to justify a phase of its own.
        failedAt: "capture",
      };
    }
  }
}
