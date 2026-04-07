/** Result of a verification submission */
export interface SubmissionResult {
  success: boolean;
  txSignature?: string;
  attestationTx?: string;
  error?: string;
}

/** Result of a full Pulse verification */
export interface VerificationResult {
  success: boolean;
  commitment: Uint8Array;
  txSignature?: string;
  attestationTx?: string;
  isFirstVerification: boolean;
  error?: string;
}
