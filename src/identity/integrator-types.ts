import type { Connection } from "@solana/web3.js";

export interface IntegratorIdentityEvidence {
  walletPubkey: string;
  identityPda: string;
  creationTimestamp: number;
  lastVerificationTimestamp: number;
  verificationCount: number;
  trustScore: number;
  currentCommitment: string;
  projectionVersion: number;
  lastResetTimestamp: number;
  lastRebaselineTimestamp: number;
  mint: string;
}

export interface IntegratorTransactionEvidence {
  signature: string;
  slot: number;
  blockTime: number;
  commitment: string;
  kind: "mint" | "update" | "rebaseline";
}

export type IntegratorAttestationEvidence =
  | {
      status: "present";
      address: string;
      readContextSlot: number;
      expiresAt: number | null;
    }
  | { status: "missing" | "invalid" | "unavailable" };

export interface IntegratorEvidence {
  identity: IntegratorIdentityEvidence;
  transaction: IntegratorTransactionEvidence;
  readContextSlot: number;
  genesisHash: string;
  cluster: "devnet";
  assuranceTier: "browser_unattested";
  uniquenessStatus: "unmeasured";
  programIds: {
    anchor: string;
    verifier: string;
    registry: string;
    sas: string;
    credential: string;
    schema: string;
  };
  attestation: IntegratorAttestationEvidence;
}

export type IntegratorEvidenceFailureReason =
  | "invalid_request"
  | "wrong_cluster"
  | "identity_missing"
  | "identity_invalid"
  | "transaction_invalid"
  | "transaction_unavailable"
  | "snapshot_unavailable"
  | "rpc_unavailable";

export type IntegratorEvidenceReadResult =
  | { status: "available"; evidence: IntegratorEvidence }
  | {
      status: "invalid" | "unavailable";
      reason: IntegratorEvidenceFailureReason;
    };

export type IntegratorEvidenceConnection = Pick<
  Connection,
  | "getGenesisHash"
  | "getSignatureStatuses"
  | "getParsedTransaction"
  | "getAccountInfoAndContext"
>;

export interface ReadIntegratorEvidenceInput {
  walletPubkey: string;
  transactionSignature: string;
  connection: IntegratorEvidenceConnection;
  nowSeconds: number;
}
