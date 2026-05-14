// BN254 base field prime (for G1 point negation in proof_a)
export const BN254_BASE_FIELD = BigInt(
  "21888242871839275222246405745257275088696311157297823662689037894645226208583"
);

// BN254 scalar field prime (for salt generation, field element bounds)
export const BN254_SCALAR_FIELD = BigInt(
  "21888242871839275222246405745257275088548364400416034343698204186575808495617"
);

export const FINGERPRINT_BITS = 256;
export const DEFAULT_THRESHOLD = 96;
export const DEFAULT_MIN_DISTANCE = 3;
export const NUM_PUBLIC_INPUTS = 4;

export const PROOF_A_SIZE = 64;
export const PROOF_B_SIZE = 128;
export const PROOF_C_SIZE = 64;
export const TOTAL_PROOF_SIZE = 256;

// Frozen at the original v1 string for backward compatibility — every existing
// user's baseline projects features into bit positions derived from this seed,
// so changing it would invalidate every prior fingerprint and force a global
// baseline reset. Kerckhoffs-compliant either way (the seed is public).
export const SIMHASH_SEED = "IAM-PROTOCOL-SIMHASH-V1";

// Capture duration bounds (ms)
export const MIN_CAPTURE_MS = 2000;
export const MAX_CAPTURE_MS = 60000;
export const DEFAULT_CAPTURE_MS = 12000;

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
  entrosCredentialPda: "GaPTkZC6JEGds1G5h645qyUrogx7NWghR2JgjvKQwTDo",
  entrosSchemaPda: "EPkajiGQjycPwcc3pupqExVdAmSfxWd31tRYZezd8c5g",
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
