/** Result of a verification submission */
export interface SubmissionResult {
  success: boolean;
  txSignature?: string;
  attestationTx?: string;
  error?: string;
  compositeRiskScore?: number;
}

/**
 * Validator-signed receipt binding (wallet, commitment, validated_at) for the
 * upcoming `mint_anchor` transaction. Returned in the `/validate-features`
 * response when the request includes `commitment_new_hex` and the validator
 * has a signing key configured.
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
  compositeRiskScore?: number;
  /**
   * Reason label when verification fails. Two-source taxonomy:
   *
   * Server-side safe-reveal (validator → executor → SDK):
   *   - `variance_floor`, `entropy_bounds`, `temporal_coupling_low`,
   *     `phrase_content_mismatch`
   *   Surfaced for the soft-reject + retry UX so the UI can render a
   *   per-category hint.
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
   *
   * Values are narrowed by `isVerificationReason` and classified by
   * `reasonDisposition`, both exported from the SDK. Prefer those to a local
   * copy of the list. Six local copies existed before they were exported and
   * they had already drifted apart.
   */
  reason?: string;
  /**
   * Seconds to wait before another attempt can succeed. Present only
   * alongside a cooldown reason (`rate_limited`, `ip_rate_limited`,
   * `cross_wallet_cooldown`) and only when the server supplied a value.
   *
   * Read from the JSON body rather than the `Retry-After` header, because a
   * browser cannot see response headers cross-origin unless the server lists
   * them in `Access-Control-Expose-Headers`.
   */
  retryAfterSec?: number;
}

/** Bytes sent so far, and the total, for the one request big enough to matter. */
export interface UploadProgress {
  loaded: number;
  total: number;
}

/**
 * Stage notifications during `complete()` / `completeReset()`.
 *
 * The stage strings are part of the public contract, not just UI copy: the
 * embed popup matches `"submitting"` against them to drive the integrator
 * heartbeat on the wire. Treat them as API and do not reword them.
 *
 * `progress` accompanies the upload of the validate request, which on a phone
 * is the longest single step in the flow and the one worth showing a bar for.
 * It is absent for every other stage, and absent on runtimes with no
 * `XMLHttpRequest` (Node), where upload progress cannot be observed.
 */
export type ProgressCallback = (stage: string, progress?: UploadProgress) => void;
