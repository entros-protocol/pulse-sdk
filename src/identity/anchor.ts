import { PROGRAM_IDS } from "../config";
import { sdkWarn } from "../log";
import type { IdentityState, StoredVerificationData } from "./types";
import {
  hasCryptoSupport,
  getOrCreateEncryptionKey,
  encrypt,
  decrypt,
} from "./crypto";

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

    const idl = await anchor.Program.fetchIdl(programId, {
      connection,
    } as any);
    if (!idl) return null;

    const coder = new anchor.BorshAccountsCoder(idl);
    const decoded = coder.decode("identityState", accountInfo.data);

    return {
      owner: decoded.owner.toBase58(),
      creationTimestamp: decoded.creationTimestamp.toNumber(),
      lastVerificationTimestamp: decoded.lastVerificationTimestamp.toNumber(),
      verificationCount: decoded.verificationCount,
      trustScore: decoded.trustScore,
      currentCommitment: new Uint8Array(decoded.currentCommitment),
      mint: decoded.mint.toBase58(),
      // Anchor's Borsh coder returns the raw BN for i64 fields; .toNumber()
      // is safe here because Unix timestamps fit in Number.MAX_SAFE_INTEGER
      // until year 275760.
      lastResetTimestamp: decoded.lastResetTimestamp?.toNumber?.() ?? 0,
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
