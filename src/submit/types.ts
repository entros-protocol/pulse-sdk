/** Result of a verification submission */
export interface SubmissionResult {
  success: boolean;
  txSignature?: string;
  attestationTx?: string;
  error?: string;
}

/**
 * Validator-signed receipt binding (wallet, commitment, validated_at) for the
 * upcoming `mint_anchor` transaction (master-list #146 Phase 4). Returned in
 * the `/validate-features` response when the request includes
 * `commitment_new_hex` and the validator has a signing key configured.
 *
 * Wire fields are byte-identical to `entros_validation::SignedReceiptDto` and
 * the executor's local mirror at `executor-node::validation::SignedReceiptDto`.
 */
export interface SignedReceiptDto {
  /** Hex-encoded 32-byte Ed25519 public key of the validator. */
  validator_pubkey_hex: string;
  /**
   * Hex-encoded 72-byte message:
   *   wallet_pubkey (32) || commitment_new (32) || validated_at i64 LE (8)
   */
  message_hex: string;
  /** Hex-encoded 64-byte Ed25519 signature over `message_hex`. */
  signature_hex: string;
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
   * Reason label when verification fails. Two-source taxonomy:
   *
   * Server-side safe-reveal (validator → executor → SDK):
   *   - `variance_floor`, `entropy_bounds`, `temporal_coupling_low`,
   *     `phrase_content_mismatch`
   *   Surfaced for the soft-reject + retry UX (master-list #94) so the
   *   UI can render a per-category hint.
   *
   * Client-side (SDK-emitted):
   *   - `validation_unavailable` — the relayer's `/validate-features`
   *     endpoint was unreachable (network failure, timeout, abort).
   *     UI should treat as transient + offer retry. NOT a server-side
   *     ReasonCode; emitted directly by `extractFingerprintAndValidate`
   *     when the fetch promise rejects.
   *
   * Absent on every other failure path (data-quality, on-chain submission,
   * baseline missing, etc.) and on attack-signal rejections (TTS detection,
   * Sybil match) and capture-shape bugs — the validator deliberately keeps
   * those opaque to prevent adversarial probing. UI must not assume reason
   * is present even when `success === false`.
   */
  reason?: string;
}
