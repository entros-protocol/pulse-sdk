/** On-chain identity state (mirrors the Anchor program's IdentityState) */
export interface IdentityState {
  owner: string;
  creationTimestamp: number;
  lastVerificationTimestamp: number;
  verificationCount: number;
  trustScore: number;
  currentCommitment: Uint8Array;
  mint: string;
}

/** Local storage of previous verification data (needed for re-verification) */
export interface StoredVerificationData {
  fingerprint: number[];
  salt: string;
  commitment: string;
  timestamp: number;
}
