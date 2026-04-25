import type { TBH } from "../hashing/types";
import type { CircuitInput, ProofResult, SolanaProof } from "./types";
import { serializeProof } from "./serializer";
import { DEFAULT_THRESHOLD, DEFAULT_MIN_DISTANCE } from "../config";

// Use dynamic import for snarkjs (it's a CJS module)
let snarkjsModule: any = null;

async function getSnarkjs(): Promise<any> {
  if (!snarkjsModule) {
    snarkjsModule = await import("snarkjs");
  }
  return snarkjsModule;
}

/**
 * Prepare circuit input from current and previous TBH data.
 */
export function prepareCircuitInput(
  current: TBH,
  previous: TBH,
  threshold: number = DEFAULT_THRESHOLD,
  minDistance: number = DEFAULT_MIN_DISTANCE
): CircuitInput {
  return {
    ft_new: current.fingerprint,
    ft_prev: previous.fingerprint,
    salt_new: current.salt.toString(),
    salt_prev: previous.salt.toString(),
    commitment_new: current.commitment.toString(),
    commitment_prev: previous.commitment.toString(),
    threshold: threshold.toString(),
    min_distance: minDistance.toString(),
  };
}

/**
 * Generate a Groth16 proof for the Hamming distance circuit.
 *
 * @param input - Circuit input (fingerprints, salts, commitments, threshold)
 * @param wasmPath - Path or URL to entros_hamming.wasm
 * @param zkeyPath - Path or URL to entros_hamming_final.zkey
 */
export async function generateProof(
  input: CircuitInput,
  wasmPath: string,
  zkeyPath: string
): Promise<ProofResult> {
  const snarkjs = await getSnarkjs();
  const { proof, publicSignals } = await snarkjs.groth16.fullProve(
    input,
    wasmPath,
    zkeyPath
  );
  return { proof, publicSignals };
}

/**
 * Generate a proof and serialize it for Solana submission.
 */
export async function generateSolanaProof(
  current: TBH,
  previous: TBH,
  wasmPath: string,
  zkeyPath: string,
  threshold?: number
): Promise<SolanaProof> {
  const input = prepareCircuitInput(current, previous, threshold);
  const { proof, publicSignals } = await generateProof(
    input,
    wasmPath,
    zkeyPath
  );
  return serializeProof(proof, publicSignals);
}

/**
 * Verify a proof locally using snarkjs (for debugging/testing).
 * Caller is responsible for loading the verification key.
 */
export async function verifyProofLocally(
  proof: any,
  publicSignals: string[],
  vkey: Record<string, unknown>
): Promise<boolean> {
  const snarkjs = await getSnarkjs();
  return snarkjs.groth16.verify(vkey, publicSignals, proof);
}
