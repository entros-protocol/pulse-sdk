/**
 * Wallet-keyed encrypted baseline storage (master-list #98).
 *
 * Architecture: the user's previous SimHash + salt (private witnesses for
 * the Hamming-distance ZK proof) are encrypted with AES-256-GCM under a
 * key derived from a deterministic `signMessage` on a domain-separated
 * payload. The 96-byte ciphertext blob is stored on chain in a per-wallet
 * `EncryptedBaseline` PDA, making the baseline wallet-portable across
 * devices and recoverable after a cache wipe.
 *
 * The GCM AAD binds (version, algorithm, wallet, baselinePda, current on-
 * chain commitment). After `reset_identity_state` cycles the commitment,
 * the old blob's auth-tag fails to verify under the new commitment — that
 * mismatch IS the staleness signal. No explicit version tracking needed.
 *
 * Plaintext biometric data never reaches chain at any point. This module
 * is the only place wallet-derived key material flows; it never leaves
 * memory and the key is non-extractable.
 */

import type { PublicKey } from "@solana/web3.js";
import { PROGRAM_IDS } from "../config";

// --- Constants ---

const BLOB_VERSION = 0x01;
const ALGORITHM_AES_256_GCM = 0x01;

/** Layout: version(1) + algo(1) + reserved(2) + IV(12) + ct+tag(80) */
export const ENCRYPTED_BASELINE_BLOB_BYTES = 96;
const IV_BYTES = 12;
const HEADER_BYTES = 4; // version + algo + 2 reserved
const PLAINTEXT_BYTES = 64; // simhash(32) || salt(32)

// --- Bit packing (256-bit fingerprint <-> 32 bytes) ---

/**
 * Pack a 256-bit fingerprint into 32 bytes. LSB-first within each byte —
 * the convention is internal to the encrypted-baseline blob; the only
 * requirement is that `bytesToFingerprint` is its exact inverse.
 */
export function fingerprintToBytes(bits: number[]): Uint8Array {
  if (bits.length !== 256) {
    throw new Error(`expected 256-bit fingerprint, got ${bits.length}`);
  }
  const out = new Uint8Array(32);
  for (let i = 0; i < 256; i++) {
    if (bits[i] === 1) {
      out[i >> 3] = (out[i >> 3] ?? 0) | (1 << (i & 7));
    }
  }
  return out;
}

/** Inverse of `fingerprintToBytes`. */
export function bytesToFingerprint(bytes: Uint8Array): number[] {
  if (bytes.length !== 32) {
    throw new Error(`expected 32 bytes, got ${bytes.length}`);
  }
  const bits: number[] = new Array(256);
  for (let i = 0; i < 256; i++) {
    bits[i] = ((bytes[i >> 3] ?? 0) >> (i & 7)) & 1;
  }
  return bits;
}

/** Big-endian 32-byte → BigInt. Inverse of `bigintToBytes32` in hashing/poseidon. */
export function bytes32ToBigint(bytes: Uint8Array): bigint {
  if (bytes.length !== 32) {
    throw new Error(`expected 32 bytes, got ${bytes.length}`);
  }
  let val = BigInt(0);
  for (let i = 0; i < 32; i++) {
    val = (val << BigInt(8)) | BigInt(bytes[i] ?? 0);
  }
  return val;
}

/**
 * Domain-separated signMessage payload. Wallet-friendly multi-line format
 * with an explicit "not a transaction" hint per Phantom/Solflare best
 * practice. The wallet pubkey is embedded so phishing across wallets is
 * visibly wrong to the signer (Ed25519 already binds the signature to one
 * wallet — this is belt-and-suspenders for the human-readable layer).
 */
function buildDomainMessage(walletPubkey: PublicKey): string {
  return [
    "Entros Protocol — Identity Baseline Key Derivation",
    "",
    "By signing this message, you authorize Entros Protocol to derive",
    "an encryption key for your on-chain identity baseline. This is",
    "not a transaction.",
    "",
    `Wallet: ${walletPubkey.toBase58()}`,
    "Version: 1",
    "Domain: entros.io",
  ].join("\n");
}

// --- PDA derivation ---

/**
 * The per-wallet `EncryptedBaseline` PDA address. Owned by entros-anchor.
 * Seeds: `[b"encrypted_baseline", walletPubkey.toBuffer()]`.
 */
export async function deriveEncryptedBaselinePda(
  walletPubkey: PublicKey
): Promise<[PublicKey, number]> {
  const { PublicKey: PK } = await import("@solana/web3.js");
  const programId = new PK(PROGRAM_IDS.entrosAnchor);
  return PK.findProgramAddressSync(
    [new TextEncoder().encode("encrypted_baseline"), walletPubkey.toBuffer()],
    programId
  );
}

// --- Key derivation ---

/**
 * Wallet adapter shape needed for key derivation.
 * Compatible with `@solana/wallet-adapter-base` `WalletAdapter`.
 */
export interface BaselineWallet {
  publicKey: PublicKey;
  signMessage: (msg: Uint8Array) => Promise<Uint8Array>;
}

/**
 * Derive an AES-256 key from the wallet's deterministic signature over
 * the domain-separated payload, via HKDF-SHA256.
 *
 *   IKM  = Ed25519 signature (64 bytes, pseudorandom under RoM)
 *   salt = wallet pubkey bytes (32 bytes — wallet-binding salt)
 *   info = "entros-protocol/identity-baseline/v1"
 *   L    = 32 bytes (AES-256 key length)
 *
 * Determinism follows from RFC 8032 Ed25519 signature determinism. Same
 * wallet + same message → same signature → same key, across any device.
 *
 * The derived `CryptoKey` is non-extractable; only WebCrypto can use it
 * for AES-GCM encrypt/decrypt.
 *
 * Throws if the wallet doesn't implement `signMessage` (e.g., some Ledger
 * firmware versions). Callers should catch and fall back gracefully.
 */
export async function deriveBaselineKey(
  wallet: BaselineWallet
): Promise<CryptoKey> {
  if (typeof wallet.signMessage !== "function") {
    throw new Error("wallet does not support signMessage");
  }
  const message = buildDomainMessage(wallet.publicKey);
  const signature = await wallet.signMessage(new TextEncoder().encode(message));
  if (!(signature instanceof Uint8Array) || signature.length !== 64) {
    throw new Error(
      `expected 64-byte Ed25519 signature, got ${signature?.length ?? "non-Uint8Array"}`
    );
  }

  const ikm = await crypto.subtle.importKey(
    "raw",
    signature as Uint8Array<ArrayBuffer>,
    "HKDF",
    false,
    ["deriveBits"]
  );

  const bits = await crypto.subtle.deriveBits(
    {
      name: "HKDF",
      hash: "SHA-256",
      salt: wallet.publicKey.toBytes() as Uint8Array<ArrayBuffer>,
      info: new TextEncoder().encode("entros-protocol/identity-baseline/v1"),
    },
    ikm,
    256
  );

  return crypto.subtle.importKey(
    "raw",
    bits,
    { name: "AES-GCM" },
    false,
    ["encrypt", "decrypt"]
  );
}

// --- Session-scoped key cache ---

// Promise-keyed cache: concurrent callers join the same in-flight derivation
// rather than each prompting `signMessage`. On rejection the entry is evicted
// before the error propagates, so a denied prompt can be retried on the next
// call (e.g., user re-approves after misclicking "reject").
const baselineKeyCache = new Map<string, Promise<CryptoKey>>();

/**
 * Derive the baseline AES key for `wallet`, reusing the cached derivation
 * when one is in-flight or resolved for this wallet pubkey in the current
 * page lifetime. The key is deterministic per wallet, so caching is purely
 * a UX optimization (avoids re-prompting `signMessage` on every verification
 * within a session). Cache survives a single page load and is dropped on
 * reload or via `clearBaselineKeyCache`.
 *
 * Concurrent calls for the same wallet share one `signMessage` prompt; a
 * failed derivation evicts the cache entry so subsequent calls re-derive.
 */
export async function getOrDeriveBaselineKey(
  wallet: BaselineWallet
): Promise<CryptoKey> {
  const pubkeyBase58 = wallet.publicKey.toBase58();
  const cached = baselineKeyCache.get(pubkeyBase58);
  if (cached) return cached;
  const pending = (async () => {
    try {
      return await deriveBaselineKey(wallet);
    } catch (err) {
      // Don't poison the cache with a rejected promise. The next call
      // re-prompts, which is correct behavior after a transient wallet
      // error or a user-cancelled prompt.
      baselineKeyCache.delete(pubkeyBase58);
      throw err;
    }
  })();
  baselineKeyCache.set(pubkeyBase58, pending);
  return pending;
}

/** Drop all cached baseline keys. Test hook + emergency wipe. */
export function clearBaselineKeyCache(): void {
  baselineKeyCache.clear();
}

// --- AAD ---

/**
 * Build the AAD for a given (wallet, baselinePda, commitment) tuple.
 * Layout: version(1) || algo(1) || wallet(32) || baselinePda(32) || commitment(32) = 98 bytes.
 *
 * The commitment binding gives cryptographic staleness detection: the SDK
 * decrypts using the CURRENT on-chain commitment in the AAD. After a
 * `reset_identity_state` cycle, the new commitment doesn't match what
 * the blob was sealed under, so the auth tag fails to verify.
 */
function buildAAD(
  walletPubkey: PublicKey,
  baselinePda: PublicKey,
  commitment: Uint8Array
): Uint8Array {
  if (commitment.length !== 32) {
    throw new Error(`commitment must be 32 bytes, got ${commitment.length}`);
  }
  const aad = new Uint8Array(98);
  aad[0] = BLOB_VERSION;
  aad[1] = ALGORITHM_AES_256_GCM;
  aad.set(walletPubkey.toBytes(), 2);
  aad.set(baselinePda.toBytes(), 34);
  aad.set(commitment, 66);
  return aad;
}

// --- Encrypt ---

/**
 * Encrypt (simhash || salt) into a 96-byte versioned envelope.
 *
 * @param simhash      32-byte SimHash fingerprint (packed 256 bits)
 * @param salt         32-byte Poseidon commitment salt
 * @param key          AES-256-GCM key from `deriveBaselineKey`
 * @param walletPubkey wallet that owns the EncryptedBaseline PDA
 * @param baselinePda  the EncryptedBaseline PDA address (bound in AAD)
 * @param commitment   the on-chain `current_commitment` at encryption time
 *                     (bound in AAD for staleness detection)
 */
export async function encryptBaselineBlob(
  simhash: Uint8Array,
  salt: Uint8Array,
  key: CryptoKey,
  walletPubkey: PublicKey,
  baselinePda: PublicKey,
  commitment: Uint8Array
): Promise<Uint8Array> {
  if (simhash.length !== 32) {
    throw new Error(`simhash must be 32 bytes, got ${simhash.length}`);
  }
  if (salt.length !== 32) {
    throw new Error(`salt must be 32 bytes, got ${salt.length}`);
  }

  const plaintext = new Uint8Array(PLAINTEXT_BYTES);
  plaintext.set(simhash, 0);
  plaintext.set(salt, 32);

  const iv = crypto.getRandomValues(new Uint8Array(IV_BYTES));
  const aad = buildAAD(walletPubkey, baselinePda, commitment);

  const ciphertextWithTag = new Uint8Array(
    await crypto.subtle.encrypt(
      {
        name: "AES-GCM",
        iv: iv as Uint8Array<ArrayBuffer>,
        additionalData: aad as Uint8Array<ArrayBuffer>,
      },
      key,
      plaintext as Uint8Array<ArrayBuffer>
    )
  ); // 80 bytes (64 ct + 16 GCM tag)

  const blob = new Uint8Array(ENCRYPTED_BASELINE_BLOB_BYTES);
  blob[0] = BLOB_VERSION;
  blob[1] = ALGORITHM_AES_256_GCM;
  // bytes 2-3 reserved (zero-init from constructor)
  blob.set(iv, HEADER_BYTES);
  blob.set(ciphertextWithTag, HEADER_BYTES + IV_BYTES);
  return blob;
}

// --- Decrypt ---

/**
 * Thrown when the on-chain blob's auth tag fails to verify under the
 * current on-chain commitment. Indicates the blob was sealed against a
 * different commitment (e.g., before a `reset_identity_state`), or the
 * AAD-binding is otherwise broken. The SDK should fall back to a fresh-
 * capture flow rather than attempting to recover.
 */
export class StaleEncryptedBaselineError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "StaleEncryptedBaselineError";
  }
}

/**
 * Decrypt a 96-byte versioned envelope back to (simhash, salt).
 *
 * Throws `StaleEncryptedBaselineError` on auth-tag mismatch — this is the
 * cryptographic staleness signal. Other errors indicate a malformed blob.
 */
export async function decryptBaselineBlob(
  blob: Uint8Array,
  key: CryptoKey,
  walletPubkey: PublicKey,
  baselinePda: PublicKey,
  commitment: Uint8Array
): Promise<{ simhash: Uint8Array; salt: Uint8Array }> {
  // Defensive input validation first (caller-error invariants),
  // then blob-shape checks (corruption invariants).
  if (commitment.length !== 32) {
    throw new Error(`commitment must be 32 bytes, got ${commitment.length}`);
  }
  if (blob.length !== ENCRYPTED_BASELINE_BLOB_BYTES) {
    throw new Error(
      `blob must be ${ENCRYPTED_BASELINE_BLOB_BYTES} bytes, got ${blob.length}`
    );
  }
  if (blob[0] !== BLOB_VERSION) {
    throw new Error(`unsupported blob version: ${blob[0]}`);
  }
  if (blob[1] !== ALGORITHM_AES_256_GCM) {
    throw new Error(`unsupported algorithm id: ${blob[1]}`);
  }

  const iv = blob.slice(HEADER_BYTES, HEADER_BYTES + IV_BYTES);
  const ciphertextWithTag = blob.slice(
    HEADER_BYTES + IV_BYTES,
    ENCRYPTED_BASELINE_BLOB_BYTES
  );
  const aad = buildAAD(walletPubkey, baselinePda, commitment);

  let plaintext: ArrayBuffer;
  try {
    plaintext = await crypto.subtle.decrypt(
      {
        name: "AES-GCM",
        iv: iv as Uint8Array<ArrayBuffer>,
        additionalData: aad as Uint8Array<ArrayBuffer>,
      },
      key,
      ciphertextWithTag as Uint8Array<ArrayBuffer>
    );
  } catch (err) {
    // WebCrypto throws OperationError on auth-tag mismatch. Differentiate
    // from non-crypto errors (which would indicate a programming bug).
    if (err instanceof Error && err.name === "OperationError") {
      throw new StaleEncryptedBaselineError(
        "encrypted baseline auth-tag verification failed — blob is stale or AAD does not match"
      );
    }
    throw err;
  }

  const bytes = new Uint8Array(plaintext);
  return {
    simhash: bytes.slice(0, 32),
    salt: bytes.slice(32, 64),
  };
}

// --- Fetch on-chain blob ---

/**
 * Fetch the user's `EncryptedBaseline` PDA from chain. Returns `null` if
 * the account has never been initialized (user hasn't called
 * `set_encrypted_baseline` yet).
 */
export async function fetchEncryptedBaseline(
  walletPubkey: PublicKey,
  connection: { getAccountInfo: (k: PublicKey) => Promise<unknown> }
): Promise<Uint8Array | null> {
  const [baselinePda] = await deriveEncryptedBaselinePda(walletPubkey);
  const accountInfo = (await connection.getAccountInfo(baselinePda)) as
    | { data: ArrayLike<number> | Uint8Array }
    | null;
  if (!accountInfo) return null;

  // Anchor account layout: 8-byte discriminator || blob(96) || bump(1) = 105 bytes.
  // solana-web3.js returns `data` as `Buffer` in Node and `Uint8Array` in browser;
  // both satisfy `ArrayLike<number>`, so we wrap defensively without depending on
  // a Node-only `Buffer` type at the SDK boundary.
  const raw =
    accountInfo.data instanceof Uint8Array
      ? accountInfo.data
      : new Uint8Array(accountInfo.data);
  if (raw.length < 8 + ENCRYPTED_BASELINE_BLOB_BYTES + 1) {
    return null;
  }
  return raw.slice(8, 8 + ENCRYPTED_BASELINE_BLOB_BYTES);
}
