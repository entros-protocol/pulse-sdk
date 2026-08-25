import { HIGHEST_SUPPORTED_PROJECTION_VERSION } from "./projection";

// BN254 base field prime (for G1 point negation in proof_a)
export const BN254_BASE_FIELD = BigInt(
  "21888242871839275222246405745257275088696311157297823662689037894645226208583"
);

// BN254 scalar field prime (for salt generation, field element bounds)
export const BN254_SCALAR_FIELD = BigInt(
  "21888242871839275222246405745257275088548364400416034343698204186575808495617"
);

export const FINGERPRINT_BITS = 256;
export const MAX_THRESHOLD = 96;
export const MIN_DISTANCE_FLOOR = 3;
export const DEFAULT_THRESHOLD = MAX_THRESHOLD;
export const DEFAULT_MIN_DISTANCE = MIN_DISTANCE_FLOOR;
export const NUM_PUBLIC_INPUTS = 4;

export const PROOF_A_SIZE = 64;
export const PROOF_B_SIZE = 128;
export const PROOF_C_SIZE = 64;
export const TOTAL_PROOF_SIZE = 256;

/** Public seed for version-bound projection transcripts. */
export const SIMHASH_PUBLIC_SEED_HEX =
  "9ee9c02f3fc6a2abce703010e64378d4531f8bcb110f0bc4c177c36a60c75bb5";

/** Projection version 0 seed retained for pre-cutover identities. */
export const LEGACY_SIMHASH_SEED = "IAM-PROTOCOL-SIMHASH-V1";

// Capture duration bounds (ms)
export const MIN_CAPTURE_MS = 2000;
export const MAX_CAPTURE_MS = 60000;

/**
 * Hard ceiling on the audio that leaves the device, measured from the capture
 * mark. Independent of `MAX_CAPTURE_MS`, which bounds how long the recorder
 * runs rather than how much of it is used.
 *
 * Mirrors `entros-validation::phrase_binding::MAX_AUDIO_SAMPLES` (320_000,
 * i.e. 20 s at 16 kHz). The validator truncates anything longer to that bound
 * before transcribing, so overrunning costs a client the tail of its audio
 * and, with it, the phrase check.
 *
 * The two meet exactly rather than leaving room. Both comparisons are strict
 * `>`, so a capture of precisely 320,000 samples is untouched on both sides,
 * and loosening either to `>=` would truncate every maxed capture. That is
 * the invariant `test/audio.test.ts` pins against the server's constant, not
 * against this one.
 *
 * It also keeps the body inside the executor's 1 MiB limit by construction.
 * Measured against a real 320,000-sample capture, the full body is 910,172
 * bytes: 853,338 of base64 audio, 37,140 and 9,941 for the two contours,
 * 6,751 for the 308 features and 2,529 for the curve trace. That leaves
 * 138,404 bytes of headroom, and pushing every float to its longest
 * round-trip representation still leaves about 106 KB.
 */
export const MAX_TRANSMITTED_CAPTURE_MS = 20000;
export const DEFAULT_CAPTURE_MS = 12000;

/**
 * How long the validate request may go without upload progress before the SDK
 * gives up on it.
 *
 * This bounds silence, not duration. A slow uplink that is still delivering
 * bytes resets the clock on every progress event and is never cut off, which
 * is the failure this replaces: a fixed 15-second abort covering both upload
 * and server work killed a healthy 9.4-second mobile upload and reported the
 * service as unreachable.
 *
 * Mirrors `RequestBodyTimeoutLayer` on both Rust services, which reclaims a
 * stalled body on the same principle from the other end.
 */
export const VALIDATE_UPLOAD_STALL_MS = 20000;

/**
 * Absolute ceiling on a validate request, used when the caller has no better
 * information.
 *
 * Prefer deriving the ceiling from the challenge's own remaining validity:
 * `/challenge` returns `expires_in`, and an upload landing after that is
 * refused regardless, because the executor looks the phrase up under the same
 * TTL. This constant is the fallback for callers that never fetched a
 * challenge.
 */
export const VALIDATE_DEADLINE_MS = 120000;
/**
 * Max time startAudio() waits for the first real audio frame before resolving
 * anyway. Normal cold starts deliver the first frame in well under a second;
 * this cap only matters if the mic never produces samples (e.g. hardware
 * failure), so the verify flow degrades gracefully instead of hanging.
 */
export const AUDIO_READY_TIMEOUT_MS = 5000;

/**
 * How long to wait for the wallet to return a signed transaction.
 *
 * There was no clock here at all, which is why a wallet that never prompted
 * was reported as "Proof generation timed out": the only timer in the stack
 * was a single 120-second race in `entros.io` covering the whole of
 * `complete()`, and its message named the slowest step anyone expected rather
 * than the step that hung.
 *
 * Three minutes is long on purpose. A mobile approval leaves the browser,
 * opens the wallet app, authenticates, returns, and each hop can stall on a
 * cold app or a slow deep link. It is also unsafe to be brisk: nothing can
 * cancel a prompt once it is shown, so an approval that arrives after the
 * clock expires still broadcasts, and the local baseline is not stored for a
 * transaction the SDK has already given up on. That is the stale-baseline
 * state this release exists to remove, so the timeout is set where it fires
 * for a hung wallet and effectively never for a slow user.
 */
export const SIGNATURE_TIMEOUT_MS = 180000;

/**
 * How long to wait for the cluster to confirm a broadcast transaction.
 *
 * A blockhash is valid for 150 slots, about 60 seconds, and a transaction that
 * has not landed by then cannot land at all. Ninety seconds covers slot-time
 * variance and leaves the timeout meaning what it says: past this point there
 * is nothing left to wait for.
 */
export const CONFIRMATION_TIMEOUT_MS = 90000;

/**
 * How long to wait for the wallet to sign the SAS attestation message.
 *
 * The prompt appears after the verification transaction confirms. The
 * attestation is optional, so an invisible prompt must not delay success or
 * local baseline storage.
 *
 * Twenty seconds is enough for a prompt a user can see, and short enough that
 * one they cannot see costs them very little.
 */
export const ATTESTATION_SIGNATURE_TIMEOUT_MS = 20000;

/**
 * Work inside a verification that no SDK clock bounds.
 *
 * Feature extraction, Groth16 proving, the challenge fetch and the
 * best-effort SAS attestation. Proving dominates it and is the part that
 * varies most: seconds on a laptop, tens of seconds on a low-end phone.
 */
const UNCLOCKED_WORK_MS = 90000;

/**
 * Longest a single `complete()` can legitimately run.
 *
 * Exported for hosts that keep their own backstop timer. Set yours above this
 * or it pre-empts the SDK's per-step clocks, and the failure is reported
 * against whichever step the backstop's message happens to name rather than
 * the one that hung. That is exactly how a pending wallet prompt came to be
 * reported as a proof-generation timeout: three hosts each raced the whole of
 * `complete()` against 120 seconds, which is less than the validate deadline
 * on its own.
 *
 * Derived rather than written down, so raising any clock above raises this in
 * step and no host has to be told.
 */
export const MAX_VERIFICATION_MS =
  VALIDATE_DEADLINE_MS +
  SIGNATURE_TIMEOUT_MS +
  CONFIRMATION_TIMEOUT_MS +
  UNCLOCKED_WORK_MS;

/**
 * Highest feature-projection generation this SDK build supports.
 * The active on-chain policy selects the generation used for capture.
 */
export { HIGHEST_SUPPORTED_PROJECTION_VERSION as CLIENT_PROJECTION_VERSION } from "./projection";

/** @deprecated Use `CLIENT_PROJECTION_VERSION`. */
export const PROJECTION_VERSION = HIGHEST_SUPPORTED_PROJECTION_VERSION;

export const PROGRAM_IDS = {
  entrosAnchor: "GZYwTp2ozeuRA5Gof9vs4ya961aANcJBdUzB7LN6q4b2",
  entrosVerifier: "4F97jNoxQzT2qRbkWpW3ztC3Nz2TtKj3rnKG8ExgnrfV",
  entrosRegistry: "6VBs3zr9KrfFPGd6j7aGBPQWwZa5tajVfA7HN6MMV9VW",
} as const;

export const AGENT_REGISTRY_CONFIG = {
  programIdDevnet: "8oo4J9tBB3Hna1jRQ3rWvJjojqM5DYTDJo5cejUuJy3C",
  programIdMainnet: "8oo4dC4JvBLwy5tGgiH3WwK4B9PWxL9Z4XjA2jzkQMbQ",
  metadataKey: "entros:human-operator",
} as const;

export const SAS_CONFIG = {
  programId: "22zoJMtdu4tQc2PzL74ZUT7FrwgB1Udec8DdW4yw4BdG",
  entrosCredentialPda: "AMBtabCgRFwGLjoZ21Z2LhSKJ6c47NckxUkMogJ3Lpuw",
  entrosSchemaPda: "5LNc7syFW7USPLveVLcNcjjY1xqS7QTXVjHZ7CQCbAMQ",
} as const;

export interface PulseConfig {
  cluster: "devnet" | "mainnet-beta" | "localnet";
  rpcEndpoint?: string;
  relayerUrl?: string;
  relayerApiKey?: string;
  zkeyUrl?: string;
  wasmUrl?: string;
  threshold?: number;
  /** Enable console logging for diagnostics. Default: false. */
  debug?: boolean;
  /**
   * Optional callback invoked when the SDK detects that encrypted local
   * storage is unavailable (e.g. iOS Safari private browsing, Brave
   * shields, Firefox Total Cookie Protection). The host app can prompt
   * the user and resolve to:
   *   - `true`  → SDK stores verification data in plaintext localStorage.
   *               Convenient (baseline survives reload) but the
   *               256-bit fingerprint + salt + commitment sit unencrypted.
   *   - `false` → SDK stores in-memory only. Data is lost on reload;
   *               user must re-enroll each session.
   * If this callback is NOT provided, the SDK defaults to in-memory only —
   * never silently writes plaintext to localStorage. This default is the
   * safer choice; opt-in to plaintext via the callback when the host app
   * has surfaced the privacy tradeoff to the user.
   */
  onPrivacyFallback?: () => Promise<boolean>;
}
