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

export const SIMHASH_SEED = "IAM-PROTOCOL-SIMHASH-V1";

// Capture duration bounds (ms)
export const MIN_CAPTURE_MS = 2000;
export const MAX_CAPTURE_MS = 60000;
export const DEFAULT_CAPTURE_MS = 7000;

export const PROGRAM_IDS = {
  iamAnchor: "GZYwTp2ozeuRA5Gof9vs4ya961aANcJBdUzB7LN6q4b2",
  iamVerifier: "4F97jNoxQzT2qRbkWpW3ztC3Nz2TtKj3rnKG8ExgnrfV",
  iamRegistry: "6VBs3zr9KrfFPGd6j7aGBPQWwZa5tajVfA7HN6MMV9VW",
} as const;

export const SAS_CONFIG = {
  programId: "22zoJMtdu4tQc2PzL74ZUT7FrwgB1Udec8DdW4yw4BdG",
  iamCredentialPda: "GaPTkZC6JEGds1G5h645qyUrogx7NWghR2JgjvKQwTDo",
  iamSchemaPda: "EPkajiGQjycPwcc3pupqExVdAmSfxWd31tRYZezd8c5g",
} as const;

export interface PulseConfig {
  cluster: "devnet" | "mainnet-beta" | "localnet";
  rpcEndpoint?: string;
  relayerUrl?: string;
  relayerApiKey?: string;
  zkeyUrl?: string;
  wasmUrl?: string;
  threshold?: number;
}
