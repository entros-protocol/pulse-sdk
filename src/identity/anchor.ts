import { PROGRAM_IDS } from "../config";
import type { IdentityState, StoredVerificationData } from "./types";

const STORAGE_KEY = "iam-protocol-verification-data";

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
      [Buffer.from("identity"), new PublicKey(walletPubkey).toBuffer()],
      programId
    );

    const accountInfo = await connection.getAccountInfo(identityPda);
    if (!accountInfo) return null;

    // Decode using Anchor's BorshAccountsCoder
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
 * Uses localStorage (browser) or in-memory fallback (Node.js).
 */
export function storeVerificationData(data: StoredVerificationData): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  } catch {
    // localStorage not available (Node.js or private browsing)
    inMemoryStore = data;
  }
}

/**
 * Load previously stored verification data.
 */
export function loadVerificationData(): StoredVerificationData | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return inMemoryStore;
    return JSON.parse(raw);
  } catch {
    return inMemoryStore;
  }
}

let inMemoryStore: StoredVerificationData | null = null;
