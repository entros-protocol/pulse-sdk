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

export type HammingVerdict = "in_bounds" | "drift_too_high" | "below_min_distance";

/**
 * Classify a Hamming distance against the circuit's accept band, mirroring
 * entros_hamming.circom:54-66 exactly:
 *   - LessThan      enforces  distance <  threshold    (maximum allowed drift)
 *   - GreaterEqThan enforces  distance >= minDistance  (replay floor)
 * so the accept band is [minDistance, threshold).
 *
 * Pass the SAME `threshold`/`minDistance` here that are fed to
 * `prepareCircuitInput`. The parameters are required (no defaults) so a caller
 * cannot accidentally classify against different bounds than the proof enforces.
 * Computing this before proving lets the SDK return a clean, user-actionable
 * result for a capture that would otherwise throw a raw circom assert.
 */
export function classifyHammingDistance(
  distance: number,
  threshold: number,
  minDistance: number
): HammingVerdict {
  if (distance >= threshold) return "drift_too_high";
  if (distance < minDistance) return "below_min_distance";
  return "in_bounds";
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
  threshold?: number,
  minDistance?: number
): Promise<SolanaProof> {
  // Low-level primitive: performs NO bounds pre-check. An out-of-band Hamming
  // distance produces an unsatisfiable witness and throws a circuit assert —
  // call classifyHammingDistance first (as processSensorData does) or catch.
  const input = prepareCircuitInput(current, previous, threshold, minDistance);
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
