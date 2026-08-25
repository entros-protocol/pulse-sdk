// Main SDK
export { PulseSDK, PulseSession, MIN_AUDIO_SAMPLES, MIN_MOTION_SAMPLES, MIN_TOUCH_SAMPLES } from "./pulse";
export type { TouchStartOptions, ValidationChallengeOptions } from "./pulse";

// Configuration
export type { PulseConfig } from "./config";
export {
  PROGRAM_IDS,
  SAS_CONFIG,
  CLIENT_PROJECTION_VERSION,
  DEFAULT_THRESHOLD,
  DEFAULT_MIN_DISTANCE,
  FINGERPRINT_BITS,
  MIN_CAPTURE_MS,
  MAX_CAPTURE_MS,
  DEFAULT_CAPTURE_MS,
  MAX_TRANSMITTED_CAPTURE_MS,
} from "./config";
// Clocks. `MAX_VERIFICATION_MS` is the one a host with its own backstop timer
// needs: it is derived from the other two plus the validate deadline, so a
// host that reads it cannot fall behind a change here.
export { SIGNATURE_TIMEOUT_MS, CONFIRMATION_TIMEOUT_MS, MAX_VERIFICATION_MS } from "./config";

// Hashing
export type { TemporalFingerprint, TBH, PackedFingerprint } from "./hashing/types";
export { simhash, hammingDistance } from "./hashing/simhash";
export {
  computeCommitment,
  generateSalt,
  generateTBH,
  packBits,
  bigintToBytes32,
} from "./hashing/poseidon";

// Feature extraction
export type { StatsSummary, FeatureVector, FusedFeatureVector } from "./extraction/types";
export { fuseFeatures } from "./extraction/statistics";
export { extractSpeakerFeatures, extractSpeakerFeaturesDetailed, SPEAKER_FEATURE_COUNT } from "./extraction/speaker";
export { extractMotionFeatures, extractTouchFeatures, extractMouseDynamics, extractAccelerationMagnitude, MOTION_FEATURE_COUNT, TOUCH_FEATURE_COUNT } from "./extraction/kinematic";
export { fuseRawFeatures } from "./extraction/statistics";

// Proof generation
export type { SolanaProof, CircuitInput, ProofResult } from "./proof/types";
export { serializeProof, toBigEndian32 } from "./proof/serializer";
export { generateProof, generateSolanaProof, prepareCircuitInput } from "./proof/prover";

// Submission
export type { SubmissionResult, VerificationResult } from "./submit/types";
export type { StudyCaptureClass, StudyContext, StudyRecordStatus } from "./study";
export { createStudyContext, featureSchemaVersionForProjection } from "./study";
// `submitResetViaWallet` is exported for advanced integrators building
// their own reset UX. Most consumers should use `PulseSDK.resetBaseline()`
// or `PulseSession.completeReset()` which handle capture + validation.
export {
  submitViaWallet,
  submitResetViaWallet,
  submitRebaselineViaWallet,
} from "./submit/wallet";
export { submitViaRelayer } from "./submit/relayer";

// Attestation (SAS)
export type { EntrosAttestation } from "./attestation/sas";
export { verifyEntrosAttestation } from "./attestation/sas";

// Agent Anchor (Solana Agent Registry)
export type { AgentHumanOperator } from "./agent/anchor";
export { attestAgentOperator, getAgentHumanOperator } from "./agent/anchor";

// Identity
export type { IdentityState, StoredVerificationData } from "./identity/types";
export type {
  BaselineRecoveryReason,
  BaselineRecoveryResult,
  ProjectionPolicy,
} from "./identity/anchor";
export {
  fetchIdentityState,
  fetchProjectionPolicy,
  storeVerificationData,
  loadVerificationData,
  recoverBaselineFromChain,
} from "./identity/anchor";

// Wallet-keyed encrypted SimHash and salt persist in each wallet's
// EncryptedBaseline PDA. A deterministic wallet signature derives the key.
export type { BaselineWallet } from "./identity/baseline";
export {
  deriveBaselineKey,
  getOrDeriveBaselineKey,
  clearBaselineKeyCache,
  deriveEncryptedBaselinePda,
  encryptBaselineBlob,
  decryptBaselineBlob,
  fetchEncryptedBaseline,
  StaleEncryptedBaselineError,
  WalletSignatureMismatchError,
  ENCRYPTED_BASELINE_BLOB_BYTES,
  fingerprintToBytes,
  bytesToFingerprint,
  bytes32ToBigint,
} from "./identity/baseline";

// Sensor types
export type { AudioCapture, MotionSample, TouchSample, SensorData, CaptureOptions, CaptureStage, StageState, CurveTracePoint, CurveTraceOutline } from "./sensor/types";
export { CURVE_OUTLINE_POINTS } from "./sensor/curve";

// Challenge
export { generatePhrase, generatePhraseSequence } from "./challenge/phrase";
export { randomLissajousParams, generateLissajousPoints, generateLissajousSequence } from "./challenge/lissajous";
export type { LissajousParams, Point2D } from "./challenge/lissajous";
export { fetchChallenge } from "./challenge/fetch";
export type {
  ChallengeResponse,
  ChallengeWithDeadline,
} from "./challenge/fetch";

// Audio encoding helper (transmits captured PCM to the validation service
// for server-side verification).
export { encodeAudioAsBase64 } from "./sensor/encode";

// Canonical capture format. Exported so the red-team harness can band-limit its
// synthesized corpora exactly as a browser now does, rather than with the plain
// linear interpolation it used to, which left far more energy above the cutoff
// than any real client can produce and made the corpus distinguishable by
// resampling artefact rather than by anything a campaign is trying to measure.
export {
  CANONICAL_SAMPLE_RATE,
  toCanonicalCapture,
  resampleTo,
  type CanonicalCapture,
} from "./sensor/resample";
export { normalizeCaptureRMS } from "./sensor/audio";

// The verification-failure reason taxonomy. Hosts should classify with
// `reasonDisposition` rather than keeping a local list. Six local copies
// existed before this was exported and they had already drifted apart, to the
// point where the same rejection offered a retry on the web and dead-ended on
// mobile.
export {
  RETRYABLE_REASONS,
  COOLDOWN_REASONS,
  CLIENT_ORIGIN_REASONS,
  isVerificationReason,
  isClientOriginReason,
  reasonDisposition,
  type VerificationReason,
  type RetryableReason,
  type ReasonDisposition,
} from "./reasons";

// Which stage of a verification failed. Hosts should route on `failedAt`
// rather than on the wording of `error`, and must consult `opaque` before
// showing anything derived from it. Routing by prose put an on-chain revert on
// the screen that says validation rejected the attempt, because the matcher
// for a validator rejection also matched `custom program error`.
export {
  isVerificationPhase,
  phaseChargesAttempt,
  phaseSpend,
  type VerificationPhase,
  type PhaseSpend,
} from "./phases";

// Error-shape helpers. `isUserRejection` is exported because the SDK uses it to
// decide whether a failure was `signing` or `submission`, and a host that keeps
// its own copy can disagree with that decision: the SDK would report
// `submission`, and the host would then never consult its own matcher. Two
// copies of this list already existed before it was exported.
export { isUserRejection } from "./submit/errors";

// Request transport. Exported for hosts that want to talk to
// `/validate-features` directly and need the same stall-not-duration
// behaviour the SDK uses.
export { postJson, TransportError } from "./transport/post-json";
export type {
  PostJsonOptions,
  PostJsonResponse,
  TransportFailureKind,
} from "./transport/post-json";
export type { ProgressCallback, UploadProgress } from "./submit/types";
