/** Serialized proof ready for on-chain submission */
export interface SolanaProof {
  proofBytes: Uint8Array;
  publicInputs: Uint8Array[];
}

/** Raw snarkjs proof output */
export interface RawProof {
  pi_a: string[];
  pi_b: string[][];
  pi_c: string[];
  protocol: string;
  curve: string;
}

/** Circuit input for proof generation */
export interface CircuitInput {
  ft_new: number[];
  ft_prev: number[];
  salt_new: string;
  salt_prev: string;
  commitment_new: string;
  commitment_prev: string;
  threshold: string;
  min_distance: string;
  request_digest_hi?: string;
  request_digest_lo?: string;
}

/** Proof generation result */
export interface ProofResult {
  proof: RawProof;
  publicSignals: string[];
}
