// Main SDK
export { PulseSDK } from "./pulse";

// Configuration
export type { PulseConfig } from "./config";
export { PROGRAM_IDS, DEFAULT_THRESHOLD, FINGERPRINT_BITS } from "./config";

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
export { mean, variance, skewness, kurtosis, condense, fuseFeatures } from "./extraction/statistics";

// Proof generation
export type { SolanaProof, CircuitInput, ProofResult } from "./proof/types";
export { serializeProof, toBigEndian32 } from "./proof/serializer";
export { generateProof, generateSolanaProof, prepareCircuitInput } from "./proof/prover";

// Submission
export type { SubmissionResult, VerificationResult } from "./submit/types";
export { submitViaWallet } from "./submit/wallet";
export { submitViaRelayer } from "./submit/relayer";

// Identity
export type { IdentityState, StoredVerificationData } from "./identity/types";
export { fetchIdentityState, storeVerificationData, loadVerificationData } from "./identity/anchor";

// Sensor types
export type { AudioCapture, MotionSample, TouchSample, SensorData } from "./sensor/types";

// Challenge
export { generatePhrase } from "./challenge/phrase";
export { randomLissajousParams, generateLissajousPoints } from "./challenge/lissajous";
export type { LissajousParams, Point2D } from "./challenge/lissajous";
