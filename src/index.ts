// Main SDK
export { PulseSDK, PulseSession, MIN_AUDIO_SAMPLES, MIN_MOTION_SAMPLES, MIN_TOUCH_SAMPLES } from "./pulse";

// Configuration
export type { PulseConfig } from "./config";
export { PROGRAM_IDS, DEFAULT_THRESHOLD, DEFAULT_MIN_DISTANCE, FINGERPRINT_BITS, MIN_CAPTURE_MS, MAX_CAPTURE_MS, DEFAULT_CAPTURE_MS } from "./config";

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
export { mean, variance, skewness, kurtosis, condense, entropy, autocorrelation, fuseFeatures } from "./extraction/statistics";
export { extractSpeakerFeatures, extractSpeakerFeaturesDetailed, SPEAKER_FEATURE_COUNT } from "./extraction/speaker";
export { extractMotionFeatures, extractTouchFeatures, extractMouseDynamics, extractAccelerationMagnitude } from "./extraction/kinematic";
export { fuseRawFeatures } from "./extraction/statistics";

// Proof generation
export type { SolanaProof, CircuitInput, ProofResult } from "./proof/types";
export { serializeProof, toBigEndian32 } from "./proof/serializer";
export { generateProof, generateSolanaProof, prepareCircuitInput } from "./proof/prover";

// Submission
export type { SubmissionResult, VerificationResult } from "./submit/types";
// `submitResetViaWallet` is exported for advanced integrators building
// their own reset UX. Most consumers should use `PulseSDK.resetBaseline()`
// or `PulseSession.completeReset()` which handle capture + validation.
export { submitViaWallet, submitResetViaWallet } from "./submit/wallet";
export { submitViaRelayer } from "./submit/relayer";

// Attestation (SAS)
export type { EntrosAttestation } from "./attestation/sas";
export { verifyEntrosAttestation } from "./attestation/sas";

// Agent Anchor (Solana Agent Registry)
export type { AgentHumanOperator } from "./agent/anchor";
export { attestAgentOperator, getAgentHumanOperator } from "./agent/anchor";

// Identity
export type { IdentityState, StoredVerificationData } from "./identity/types";
export { fetchIdentityState, storeVerificationData, loadVerificationData } from "./identity/anchor";

// Sensor types
export type { AudioCapture, MotionSample, TouchSample, SensorData, CaptureOptions, CaptureStage, StageState } from "./sensor/types";

// Challenge
export { generatePhrase, generatePhraseSequence } from "./challenge/phrase";
export { randomLissajousParams, generateLissajousPoints, generateLissajousSequence } from "./challenge/lissajous";
export type { LissajousParams, Point2D } from "./challenge/lissajous";
export { fetchChallenge } from "./challenge/fetch";
export type { ChallengeResponse } from "./challenge/fetch";

// Audio encoding helper (transmits captured PCM to the validation service
// for phrase content binding; master-list #89).
export { encodeAudioAsBase64 } from "./sensor/encode";
