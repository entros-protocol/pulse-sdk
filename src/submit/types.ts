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
  /**
   * Safe-to-reveal validator reason label when validation rejected — one of
   * `variance_floor`, `entropy_bounds`, `temporal_coupling_low`,
   * `phrase_content_mismatch`. Surfaced for the soft-reject + retry UX
   * (master-list #94) so the UI can show a per-category hint.
   *
   * Absent on every other failure path (data-quality, on-chain submission,
   * baseline missing, etc.) and on attack-signal rejections (TTS detection,
   * Sybil match) and capture-shape bugs — the validator deliberately keeps
   * those opaque to prevent adversarial probing. UI must not assume reason
   * is present even when `success === false`.
   */
  reason?: string;
}
