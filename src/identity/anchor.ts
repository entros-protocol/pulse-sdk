import { PROGRAM_IDS } from "../config";
import type { IdentityState, StoredVerificationData } from "./types";
import {
  hasCryptoSupport,
  getOrCreateEncryptionKey,
  encrypt,
  decrypt,
} from "./crypto";

const STORAGE_KEY = "iam-protocol-verification-data";
const ENCRYPTED_VERSION = 2;

// In-memory fallback for environments without localStorage (Node.js, SSR,
// private browsing on some browsers). Data is lost on page reload — users
// in private browsing mode must re-enroll on each session.
let inMemoryStore: StoredVerificationData | null = null;

// --- Envelope detection ---

interface EncryptedEnvelope {
  v: 2;
  iv: string;
  ct: string;
}

function isEncryptedEnvelope(obj: unknown): obj is EncryptedEnvelope {
  return (
    typeof obj === "object" &&
    obj !== null &&
    (obj as Record<string, unknown>).v === ENCRYPTED_VERSION &&
    typeof (obj as Record<string, unknown>).iv === "string" &&
    typeof (obj as Record<string, unknown>).ct === "string"
  );
}

function isPlaintextData(obj: unknown): obj is StoredVerificationData {
  return (
    typeof obj === "object" &&
    obj !== null &&
    Array.isArray((obj as Record<string, unknown>).fingerprint)
  );
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

    const programId = new PublicKey(PROGRAM_IDS.iamAnchor);
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
    };
  } catch {
    return null;
  }
}

/**
 * Store verification data locally for re-verification.
 * Encrypts with AES-256-GCM when Web Crypto is available.
 * Falls back to plaintext with a warning otherwise.
 */
export async function storeVerificationData(data: StoredVerificationData): Promise<void> {
  try {
    if (!hasCryptoSupport()) {
      console.warn("[IAM SDK] Crypto unavailable — verification data stored unencrypted");
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      return;
    }

    const key = await getOrCreateEncryptionKey();
    if (!key) {
      console.warn("[IAM SDK] Encryption key unavailable — storing unencrypted");
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
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
        console.warn("[IAM SDK] Encrypted data found but crypto unavailable");
        return inMemoryStore;
      }
      const key = await getOrCreateEncryptionKey();
      if (!key) {
        console.warn("[IAM SDK] Encryption key lost — clearing stale data");
        localStorage.removeItem(STORAGE_KEY);
        return inMemoryStore;
      }
      try {
        const plaintext = await decrypt(parsed.iv, parsed.ct, key);
        return JSON.parse(plaintext) as StoredVerificationData;
      } catch {
        console.warn("[IAM SDK] Decryption failed — clearing corrupted data");
        localStorage.removeItem(STORAGE_KEY);
        return inMemoryStore;
      }
    }

    // Plaintext legacy data — migrate to encrypted
    if (isPlaintextData(parsed)) {
      await storeVerificationData(parsed);
      return parsed;
    }

    // Unrecognized format
    console.warn("[IAM SDK] Unrecognized verification data format — clearing");
    localStorage.removeItem(STORAGE_KEY);
    return inMemoryStore;
  } catch {
    return inMemoryStore;
  }
}
