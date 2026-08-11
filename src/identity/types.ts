/** On-chain identity state (mirrors the Anchor program's IdentityState) */
export interface IdentityState {
  owner: string;
  creationTimestamp: number;
  lastVerificationTimestamp: number;
  verificationCount: number;
  trustScore: number;
  currentCommitment: Uint8Array;
  mint: string;
  /**
   * Unix timestamp of the most recent `reset_identity_state` call.
   * Zero for accounts that have never been reset (freshly minted or
   * minted before the reset feature was deployed).
   */
  lastResetTimestamp: number;
  /** Projection generation used by `currentCommitment`. */
  projectionVersion: number;
}

/** Local storage of previous verification data (needed for re-verification) */
export interface StoredVerificationData {
  fingerprint: number[];
  salt: string;
  commitment: string;
  timestamp: number;
  /** Projection generation used by `fingerprint` and `commitment`. */
  projectionVersion: number;
}
