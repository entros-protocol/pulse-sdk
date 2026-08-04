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
 * Load the snarkjs bundle ahead of proving. Measured at 174 ms cold.
 *
 * Delegates to the same lazy `getSnarkjs` the proof path uses, so there is one
 * initialisation route. Never rejects — a failed warm-up leaves the lazy path
 * to pay the cost later, exactly as today.
 */
export async function warmSnarkjs(): Promise<boolean> {
  try {
    await getSnarkjs();
    return true;
  } catch {
    return false;
  }
}

/** The two circuit artifacts, in memory. */
export interface CircuitArtifacts {
  wasm: Uint8Array;
  zkey: Uint8Array;
}

/**
 * Circuit artifacts held in memory between capture start and proving.
 *
 * Keyed by the URL pair. A prefetch started for one pair must never satisfy a
 * proof that asked for another, so the key is compared before the buffers are
 * handed out.
 */
interface PrefetchedArtifacts {
  wasmUrl: string;
  zkeyUrl: string;
  ready: Promise<CircuitArtifacts | null>;
}

let prefetched: PrefetchedArtifacts | null = null;

/** `priority` is a Priority Hints field that predates its appearance in lib.dom. */
type PrioritizedRequestInit = RequestInit & {
  priority?: "high" | "low" | "auto";
};

/**
 * Ceiling on a prefetch, and it is load-bearing rather than tidiness.
 *
 * `takeCircuitArtifacts` awaits an in-flight transfer instead of abandoning it,
 * which is right for a slow connection — the download has a head start and
 * restarting would discard it. On a *hung* connection that would be worse than
 * shipping no prefetch at all, because today snarkjs issues its own request at
 * proof time. The timeout keeps the pathological case bounded: the prefetch
 * gives up, resolves null, and the proof path falls back to the URL exactly as
 * it does now. Generous enough that no real connection moving 2.64 MiB trips it.
 */
const PREFETCH_TIMEOUT_MS = 60_000;

function prefetchSignal(
  caller: AbortSignal | undefined
): AbortSignal | undefined {
  if (
    typeof AbortSignal === "undefined" ||
    typeof AbortSignal.timeout !== "function"
  ) {
    return caller;
  }
  const timeout = AbortSignal.timeout(PREFETCH_TIMEOUT_MS);
  if (!caller) return timeout;
  return typeof AbortSignal.any === "function"
    ? AbortSignal.any([caller, timeout])
    : caller;
}

async function fetchArtifact(
  url: string,
  signal: AbortSignal | undefined
): Promise<Uint8Array> {
  const init: PrioritizedRequestInit = {
    // Below the microphone and motion sampler, which are running while this
    // downloads. Engines without Priority Hints ignore the unknown key.
    priority: "low",
    // No `cache` override on purpose. Default handling honours the server's
    // headers, which is the only safe policy here: a forced cache hit would
    // survive a circuit upgrade and hand snarkjs a stale zkey, producing proofs
    // the on-chain verifier rejects for reasons nothing in the client explains.
  };
  if (signal) init.signal = signal;
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`artifact fetch failed: ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

/**
 * Start downloading the circuit artifacts so proving does not wait on them.
 *
 * The wasm and zkey total about 2.64 MiB and snarkjs fetches them at proof
 * time, after the validator round trip, with nothing overlapping the transfer.
 * Nothing about that fetch depends on the capture, so it can begin the moment
 * the microphone opens and complete under the twelve seconds of capture plus
 * the validator's four-second floor.
 *
 * **Deliberately unconditional.** Artifacts are only ever *used* by a
 * re-verification, so fetching them only when a stored baseline exists would be
 * the smaller download. It would also make a fixed-size 2.64 MiB transfer an
 * observable that separates returning users from first-time ones, visible
 * through TLS by size alone. `executor-node/src/timing.rs` already spends four
 * seconds of real latency to stop behaviour varying with user state, and
 * opening a state-dependent channel here to save bandwidth would work against
 * the control the protocol already pays for.
 *
 * Returns immediately. The download proceeds in the background and its result
 * is collected by `takeCircuitArtifacts`. This function never throws and never
 * produces an unhandled rejection: the failure is captured in the stored
 * promise, which resolves to `null`.
 */
export function prefetchCircuitArtifacts(
  wasmUrl: string,
  zkeyUrl: string,
  signal?: AbortSignal
): void {
  if (typeof fetch !== "function") return;
  if (
    prefetched &&
    prefetched.wasmUrl === wasmUrl &&
    prefetched.zkeyUrl === zkeyUrl
  ) {
    return;
  }

  const bounded = prefetchSignal(signal);
  const ready = Promise.all([
    fetchArtifact(wasmUrl, bounded),
    fetchArtifact(zkeyUrl, bounded),
  ])
    .then(([wasm, zkey]) => ({ wasm, zkey }))
    // Swallowed on purpose. An aborted capture, an offline user or a blocked
    // CDN must leave the proof path exactly as it is today, passing URLs to
    // snarkjs, rather than introducing a new way to fail.
    .catch(() => null);

  prefetched = { wasmUrl, zkeyUrl, ready };
}

/**
 * Collect a prefetch started earlier, or `null` if there is nothing usable.
 *
 * Awaits an in-flight download rather than skipping it. That is never worse
 * than the current behaviour: the transfer started seconds earlier, so it
 * finishes at least as soon as the fetch snarkjs would issue here instead.
 *
 * Clears the entry on the way out, so the 2.64 MiB is not held for the life of
 * the page. A retry after proving fails falls back to URLs, which is the
 * behaviour that ships today.
 */
export async function takeCircuitArtifacts(
  wasmUrl: string,
  zkeyUrl: string
): Promise<CircuitArtifacts | null> {
  const entry = prefetched;
  if (!entry || entry.wasmUrl !== wasmUrl || entry.zkeyUrl !== zkeyUrl) {
    return null;
  }
  prefetched = null;
  return entry.ready;
}

/** Drop any pending prefetch. Exposed for tests and for teardown. */
export function clearPrefetchedArtifacts(): void {
  prefetched = null;
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
 * Both artifacts accept either a location or the bytes themselves. snarkjs
 * resolves them through `fastfile.readExisting`, which treats a `Uint8Array` as
 * an in-memory file and `fetch`es a string in the browser. Passing prefetched
 * bytes produces identical `publicSignals` to passing the URL — verified
 * against the real circuit — so the two forms are interchangeable to the chain.
 * The proof bytes themselves differ between any two runs regardless, because
 * Groth16 proving is randomised.
 *
 * @param input - Circuit input (fingerprints, salts, commitments, threshold)
 * @param wasmPath - Path or URL to entros_hamming.wasm, or its bytes
 * @param zkeyPath - Path or URL to entros_hamming_final.zkey, or its bytes
 */
export async function generateProof(
  input: CircuitInput,
  wasmPath: string | Uint8Array,
  zkeyPath: string | Uint8Array
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
  wasmPath: string | Uint8Array,
  zkeyPath: string | Uint8Array,
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
