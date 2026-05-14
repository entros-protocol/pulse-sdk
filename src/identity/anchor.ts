import type { Idl } from "@coral-xyz/anchor";
import { PROGRAM_IDS } from "../config";
import { sdkLog, sdkWarn } from "../log";
import { entrosAnchorIdl } from "../protocol/idl";
import type { IdentityState, StoredVerificationData } from "./types";
import {
  hasCryptoSupport,
  getOrCreateEncryptionKey,
  encrypt,
  decrypt,
} from "./crypto";
import {
  BaselineWallet,
  StaleEncryptedBaselineError,
  bytes32ToBigint,
  bytesToFingerprint,
  decryptBaselineBlob,
  deriveEncryptedBaselinePda,
  fetchEncryptedBaseline,
  getOrDeriveBaselineKey,
} from "./baseline";

const STORAGE_KEY = "entros-protocol-verification-data";
const ENCRYPTED_VERSION = 2;

// In-memory fallback for environments without localStorage (Node.js, SSR,
// private browsing on some browsers). Data is lost on page reload — users
// in private browsing mode must re-enroll on each session.
let inMemoryStore: StoredVerificationData | null = null;

// Module-level privacy-fallback callback. Set by PulseSDK constructor via
// `setPrivacyFallback`. Mirrors the `setDebug` pattern in `log.ts` so
// `storeVerificationData` can be called without threading the config
// through every layer.
let privacyFallbackCallback: (() => Promise<boolean>) | null = null;

export function setPrivacyFallback(
  cb: (() => Promise<boolean>) | null | undefined
): void {
  privacyFallbackCallback = cb ?? null;
}

// --- Envelope detection ---

interface EncryptedEnvelope {
  v: 2;
  iv: string;
  ct: string;
}

function isEncryptedEnvelope(obj: unknown): obj is EncryptedEnvelope {
  if (typeof obj !== "object" || obj === null) return false;
  const o = obj as Record<string, unknown>;
  return (
    o.v === ENCRYPTED_VERSION &&
    typeof o.iv === "string" &&
    o.iv.length > 0 &&
    typeof o.ct === "string" &&
    o.ct.length > 0
  );
}

function isPlaintextData(obj: unknown): obj is StoredVerificationData {
  if (typeof obj !== "object" || obj === null) return false;
  const o = obj as Record<string, unknown>;
  // Require all four fields with correct types. The previous `Array.isArray`
  // check on `fingerprint` alone passed envelopes with missing salt or
  // wrong-typed timestamp, which would crash later during use.
  if (!Array.isArray(o.fingerprint)) return false;
  if (!o.fingerprint.every((bit) => typeof bit === "number")) return false;
  if (typeof o.salt !== "string" || o.salt.length === 0) return false;
  if (typeof o.commitment !== "string" || o.commitment.length === 0) return false;
  if (typeof o.timestamp !== "number" || !Number.isFinite(o.timestamp)) return false;
  return true;
}

// --- Public API ---

/**
 * Fetch identity state from the on-chain IdentityState PDA.
 *
 * Uses the bundled `entros_anchor.json` IDL (copied verbatim from
 * `protocol-core/target/idl/`) instead of `Program.fetchIdl`, which
 * adds a 150-300ms RPC round-trip per call to fetch the IDL from chain.
 * Account decoding is identical; the only difference is that IDL changes
 * now require an SDK bump rather than a chain-side IDL upload — in
 * practice that's already true since on-chain Anchor changes need
 * matching SDK updates anyway.
 *
 * When the on-chain `entros_anchor` program changes, re-copy
 * `protocol-core/target/idl/entros_anchor.json` into `src/protocol/idl/`
 * and bump the SDK minor version.
 */
export async function fetchIdentityState(
  walletPubkey: string,
  connection: any
): Promise<IdentityState | null> {
  try {
    const { PublicKey } = await import("@solana/web3.js");
    const anchor = await import("@coral-xyz/anchor");

    const programId = new PublicKey(PROGRAM_IDS.entrosAnchor);
    const [identityPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("identity"), new PublicKey(walletPubkey).toBuffer()],
      programId
    );

    const accountInfo = await connection.getAccountInfo(identityPda);
    if (!accountInfo) return null;

    const coder = new anchor.BorshAccountsCoder(entrosAnchorIdl as Idl);
    // Anchor 0.30+ IDL spec: account names are PascalCase and the
    // BorshAccountsCoder lookup is strict — passing camelCase silently
    // throws "Account not found" which the catch below swallows as null.
    // The decoded object preserves the IDL's snake_case field names, so
    // we destructure with snake_case before mapping to the public
    // camelCase IdentityState type.
    const decoded = coder.decode("IdentityState", accountInfo.data);

    return {
      owner: decoded.owner.toBase58(),
      creationTimestamp: decoded.creation_timestamp.toNumber(),
      lastVerificationTimestamp: decoded.last_verification_timestamp.toNumber(),
      verificationCount: decoded.verification_count,
      trustScore: decoded.trust_score,
      currentCommitment: new Uint8Array(decoded.current_commitment),
      mint: decoded.mint.toBase58(),
      // Anchor's Borsh coder returns the raw BN for i64 fields; .toNumber()
      // is safe here because Unix timestamps fit in Number.MAX_SAFE_INTEGER
      // until year 275760.
      lastResetTimestamp: decoded.last_reset_timestamp?.toNumber?.() ?? 0,
    };
  } catch {
    return null;
  }
}

/**
 * Store verification data locally for re-verification.
 *
 * Storage tiers (preferred first):
 *   1. Encrypted localStorage envelope (Web Crypto available).
 *   2. If crypto unavailable AND `onPrivacyFallback` callback registered
 *      AND the callback resolves true, plaintext localStorage. The host
 *      app is responsible for surfacing the privacy tradeoff to the user
 *      before approving the fallback.
 *   3. Otherwise, in-memory only (lost on reload). Safer default —
 *      never silently writes plaintext to localStorage.
 */
export async function storeVerificationData(data: StoredVerificationData): Promise<void> {
  try {
    if (!hasCryptoSupport()) {
      // Crypto unavailable → consult the host-provided privacy callback.
      // No callback registered → default to in-memory only (safer than
      // the previous behavior of silently writing plaintext).
      const allowPlaintext = privacyFallbackCallback
        ? await privacyFallbackCallback().catch(() => false)
        : false;
      if (allowPlaintext) {
        sdkWarn(
          "[Entros SDK] Crypto unavailable; user-approved plaintext storage"
        );
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      } else {
        sdkWarn(
          "[Entros SDK] Crypto unavailable and no privacy-fallback approval — using in-memory storage (data lost on reload)"
        );
        inMemoryStore = data;
      }
      return;
    }

    const key = await getOrCreateEncryptionKey();
    if (!key) {
      // Encryption key unavailable for this session — same fallback flow.
      const allowPlaintext = privacyFallbackCallback
        ? await privacyFallbackCallback().catch(() => false)
        : false;
      if (allowPlaintext) {
        sdkWarn(
          "[Entros SDK] Encryption key unavailable; user-approved plaintext storage"
        );
        localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      } else {
        sdkWarn(
          "[Entros SDK] Encryption key unavailable and no privacy-fallback approval — using in-memory storage"
        );
        inMemoryStore = data;
      }
      return;
    }

    const { iv, ct } = await encrypt(JSON.stringify(data), key);
    const envelope: EncryptedEnvelope = { v: ENCRYPTED_VERSION, iv, ct };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(envelope));
  } catch {
    inMemoryStore = data;
  }
}

/**
 * Load previously stored verification data.
 * Decrypts if encrypted, migrates plaintext to encrypted on first load.
 */
export async function loadVerificationData(): Promise<StoredVerificationData | null> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return inMemoryStore;

    const parsed: unknown = JSON.parse(raw);

    // Encrypted envelope
    if (isEncryptedEnvelope(parsed)) {
      if (!hasCryptoSupport()) {
        sdkWarn("[Entros SDK] Encrypted data found but crypto unavailable");
        return inMemoryStore;
      }
      const key = await getOrCreateEncryptionKey();
      if (!key) {
        // Preserve the envelope. If the IndexedDB key is temporarily
        // unavailable (transient storage issue, permission prompt denied
        // once, etc.), a future load with a recovered key can still decrypt.
        // Silently deleting was the previous behavior and caused permanent
        // baseline loss for wallet-connected users whose IndexedDB got
        // corrupted into a post-patch recoverable state (DB_VERSION bump
        // in crypto.ts now self-heals the store, but the envelope must
        // survive the broken window to benefit).
        sdkWarn(
          "[Entros SDK] Encryption key unavailable — keeping envelope for recovery. " +
            "If this persists across reloads, check IndexedDB state via DevTools."
        );
        return inMemoryStore;
      }
      try {
        const plaintext = await decrypt(parsed.iv, parsed.ct, key);
        return JSON.parse(plaintext) as StoredVerificationData;
      } catch {
        // Same rationale as above: decrypt failure is often transient
        // (IndexedDB hiccup, key re-derivation edge case). Preserve the
        // envelope so the next successful decrypt can recover the data.
        // If the data truly cannot be decrypted by this device, a user-
        // triggered baseline reset (or manual "Clear site data") is the
        // right path — not a silent delete on the SDK's initiative.
        sdkWarn(
          "[Entros SDK] Decryption failed — keeping envelope for recovery. " +
            "Trigger a baseline reset or Clear site data if this is persistent."
        );
        return inMemoryStore;
      }
    }

    // Plaintext legacy data — migrate to encrypted
    if (isPlaintextData(parsed)) {
      await storeVerificationData(parsed);
      return parsed;
    }

    // Unrecognized format
    sdkWarn("[Entros SDK] Unrecognized verification data format — clearing");
    localStorage.removeItem(STORAGE_KEY);
    return inMemoryStore;
  } catch {
    return inMemoryStore;
  }
}

/**
 * Outcome of an attempt to recover the local baseline from the on-chain
 * encrypted blob (master-list #98 cache-clear / cross-device path).
 *
 * Reasons distinguish recoverable from terminal failures:
 *   - `no-on-chain-identity`: caller should treat as first-verification.
 *   - `no-encrypted-baseline`: identity exists but user has never written
 *     an encrypted baseline (pre-3.3.0 SDK or pre-#98 deploy). UX should
 *     surface the existing baseline-missing copy and offer reset.
 *   - `signing-unavailable`: AES key derivation failed because the wallet
 *     can't `signMessage` (no method on the adapter, e.g., older Ledger
 *     firmware) OR the user cancelled the prompt OR the wallet erred. The
 *     `detail` field carries the specific cause when present.
 *   - `stale-baseline`: blob predates a `reset_identity_state` cycle.
 *     Treat as terminal recovery failure; route to fresh-capture flow.
 *   - `unknown-error`: catch-all (RPC failure, malformed blob, etc.).
 */
export type BaselineRecoveryReason =
  | "no-on-chain-identity"
  | "no-encrypted-baseline"
  | "signing-unavailable"
  | "stale-baseline"
  | "unknown-error";

export interface BaselineRecoveryResult {
  recovered: boolean;
  reason?: BaselineRecoveryReason;
  detail?: string;
}

/**
 * Attempt to recover this wallet's local baseline from the on-chain
 * `EncryptedBaseline` PDA. On success, writes the recovered fingerprint /
 * salt / commitment / timestamp into the SDK's normal local storage tier
 * so the next `loadVerificationData()` call returns it transparently.
 *
 * Wallet flow on success: ONE `signMessage` prompt (for the AES key
 * derivation). The wallet's currently-cached key — if `getOrDeriveBaselineKey`
 * has been called earlier in the session — short-circuits the prompt.
 *
 * No-op when:
 *   - The on-chain `IdentityState` PDA does not exist (treat as first-verify).
 *   - The on-chain `EncryptedBaseline` PDA does not exist (pre-#98 or wallet
 *     never had `set_encrypted_baseline` written — UX should offer reset).
 *   - The wallet lacks `signMessage` (some Ledger firmware versions).
 *   - The blob's auth tag doesn't verify under the current on-chain
 *     commitment (stale blob from a prior reset cycle).
 */
export async function recoverBaselineFromChain(
  wallet: BaselineWallet,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- Connection is an optional peer dep; matches fetchIdentityState style
  connection: any
): Promise<BaselineRecoveryResult> {
  try {
    const identity = await fetchIdentityState(
      wallet.publicKey.toBase58(),
      connection
    );
    if (!identity) {
      return { recovered: false, reason: "no-on-chain-identity" };
    }

    const blob = await fetchEncryptedBaseline(wallet.publicKey, connection);
    if (!blob) {
      return { recovered: false, reason: "no-encrypted-baseline" };
    }

    let key: CryptoKey;
    try {
      key = await getOrDeriveBaselineKey(wallet);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      return {
        recovered: false,
        reason: "signing-unavailable",
        detail,
      };
    }

    const [baselinePda] = await deriveEncryptedBaselinePda(wallet.publicKey);

    let plaintext: { simhash: Uint8Array; salt: Uint8Array };
    try {
      plaintext = await decryptBaselineBlob(
        blob,
        key,
        wallet.publicKey,
        baselinePda,
        identity.currentCommitment
      );
    } catch (err) {
      if (err instanceof StaleEncryptedBaselineError) {
        return { recovered: false, reason: "stale-baseline" };
      }
      const detail = err instanceof Error ? err.message : String(err);
      return { recovered: false, reason: "unknown-error", detail };
    }

    const fingerprint = bytesToFingerprint(plaintext.simhash);
    const saltBigint = bytes32ToBigint(plaintext.salt);
    const commitmentBigint = bytes32ToBigint(identity.currentCommitment);

    await storeVerificationData({
      fingerprint,
      salt: saltBigint.toString(),
      commitment: commitmentBigint.toString(),
      timestamp:
        identity.lastVerificationTimestamp > 0
          ? identity.lastVerificationTimestamp * 1000
          : Date.now(),
    });
    sdkLog(
      "[Entros SDK] Recovered local baseline from on-chain EncryptedBaseline PDA"
    );
    return { recovered: true };
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    return { recovered: false, reason: "unknown-error", detail };
  }
}
