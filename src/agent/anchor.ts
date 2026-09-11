import type { Connection, PublicKey, Transaction } from "@solana/web3.js";

import { AGENT_REGISTRY_CONFIG, PROGRAM_IDS } from "../config";
import type { PulseConfig } from "../config";
import { canonicalKeyBytes } from "./base58";

/**
 * Historical `entros:human-operator` registry metadata.
 *
 * Any agent owner can write this entry with any content, and the registry checks only who owned
 * the asset at write time. It grants no permission and says nothing about the current owner.
 * Use Agent Operator Permits for a current decision.
 */
export interface AgentOperatorSnapshot {
  kind: "historical_snapshot";
  anchorPda: string;
  trustScore: number;
  verifiedAt: number;
  wallet: string;
  immutable: boolean;
}

/** @deprecated Use `AgentOperatorSnapshot`. The entry grants no current permission. */
export type AgentHumanOperator = AgentOperatorSnapshot;

export type AgentOperatorSnapshotReadResult =
  | { status: "present"; snapshot: AgentOperatorSnapshot }
  | { status: "absent" | "invalid" | "unavailable" };

export type AgentSnapshotConnection = Pick<Connection, "getAccountInfo">;

interface AgentAnchorSigner {
  publicKey?: PublicKey | null;
  signTransaction?: (transaction: Transaction) => Promise<Transaction>;
}

/** A wallet adapter, or a wallet context that exposes its adapter. */
export type AgentAnchorWallet = AgentAnchorSigner & { adapter?: AgentAnchorSigner };

export type AgentAnchorConnection = Pick<
  Connection,
  "getAccountInfo" | "getLatestBlockhash" | "sendRawTransaction" | "confirmTransaction"
>;

const METADATA_ENTRY_DISCRIMINATOR = Uint8Array.from([
  0x30, 0x91, 0x0c, 0xf9, 0xb0, 0x8d, 0xc5, 0xbb,
]);
const SET_METADATA_DISCRIMINATOR = Uint8Array.from([236, 60, 23, 48, 138, 69, 196, 153]);
const MAX_KEY_LENGTH = 32;
const MAX_VALUE_LENGTH = 250;
const SNAPSHOT_KEYS = ["anchorPda", "trustScore", "verifiedAt", "wallet"] as const;

function getRegistryProgramId(cluster?: PulseConfig["cluster"]): string {
  return cluster === "mainnet-beta"
    ? AGENT_REGISTRY_CONFIG.programIdMainnet
    : AGENT_REGISTRY_CONFIG.programIdDevnet;
}

async function sha256(data: Uint8Array): Promise<Uint8Array> {
  const buffer = new ArrayBuffer(data.length);
  new Uint8Array(buffer).set(data);
  return new Uint8Array(await crypto.subtle.digest("SHA-256", buffer));
}

async function metadataEntryAddress(
  agent: PublicKey,
  registry: PublicKey,
): Promise<PublicKey> {
  const { PublicKey } = await import("@solana/web3.js");
  const keyHash = (await sha256(new TextEncoder().encode(AGENT_REGISTRY_CONFIG.metadataKey))).slice(
    0,
    16,
  );
  const [entry] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("agent_meta"), agent.toBytes(), keyHash],
    registry,
  );
  return entry;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseSnapshotValue(bytes: Uint8Array, immutable: boolean): AgentOperatorSnapshot | null {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch {
    return null;
  }
  if (
    !record(value) ||
    Object.keys(value).length !== SNAPSHOT_KEYS.length ||
    !SNAPSHOT_KEYS.every((key) => Object.prototype.hasOwnProperty.call(value, key))
  )
    return null;
  const { anchorPda, trustScore, verifiedAt, wallet } = value;
  if (
    typeof anchorPda !== "string" ||
    canonicalKeyBytes(anchorPda) === null ||
    typeof wallet !== "string" ||
    canonicalKeyBytes(wallet) === null ||
    typeof trustScore !== "number" ||
    !Number.isSafeInteger(trustScore) ||
    trustScore < 0 ||
    trustScore > 10000 ||
    typeof verifiedAt !== "number" ||
    !Number.isSafeInteger(verifiedAt) ||
    verifiedAt < 0
  )
    return null;
  return { kind: "historical_snapshot", anchorPda, trustScore, verifiedAt, wallet, immutable };
}

/**
 * Reads the historical `entros:human-operator` entry of an agent with strict decoding.
 * A `present` result reports what someone wrote. It grants no permission.
 */
export async function readAgentOperatorSnapshot(
  agent: string,
  options: { connection?: AgentSnapshotConnection; cluster?: PulseConfig["cluster"] } = {},
): Promise<AgentOperatorSnapshotReadResult> {
  const agentBytes = canonicalKeyBytes(agent);
  if (!agentBytes) return { status: "invalid" };
  const { Connection, PublicKey } = await import("@solana/web3.js");
  const registry = new PublicKey(getRegistryProgramId(options.cluster));
  const entry = await metadataEntryAddress(new PublicKey(agentBytes), registry);
  const connection =
    options.connection ?? new Connection("https://api.devnet.solana.com", "confirmed");
  let account: Awaited<ReturnType<AgentSnapshotConnection["getAccountInfo"]>>;
  try {
    account = await connection.getAccountInfo(entry, "confirmed");
  } catch {
    return { status: "unavailable" };
  }
  if (!account) return { status: "absent" };
  const data = account.data;
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  if (
    account.executable ||
    !account.owner.equals(registry) ||
    data.length < 50 ||
    !METADATA_ENTRY_DISCRIMINATOR.every((byte, index) => data[index] === byte) ||
    !agentBytes.every((byte, index) => data[8 + index] === byte) ||
    (data[40] !== 0 && data[40] !== 1)
  )
    return { status: "invalid" };
  const keyLength = view.getUint32(42, true);
  if (keyLength > MAX_KEY_LENGTH || 46 + keyLength + 4 > data.length) return { status: "invalid" };
  const key = new TextDecoder().decode(data.subarray(46, 46 + keyLength));
  const valueOffset = 46 + keyLength + 4;
  const valueLength = view.getUint32(46 + keyLength, true);
  if (
    key !== AGENT_REGISTRY_CONFIG.metadataKey ||
    valueLength > MAX_VALUE_LENGTH ||
    valueOffset + valueLength > data.length
  )
    return { status: "invalid" };
  const snapshot = parseSnapshotValue(
    data.subarray(valueOffset, valueOffset + valueLength),
    data[40] === 1,
  );
  return snapshot ? { status: "present", snapshot } : { status: "invalid" };
}

/**
 * @deprecated Use `readAgentOperatorSnapshot`, which separates an absent entry from a failed read.
 * The entry grants no current permission.
 */
export async function getAgentHumanOperator(
  agentAsset: string,
  connection?: AgentSnapshotConnection,
  cluster?: PulseConfig["cluster"],
): Promise<AgentHumanOperator | null> {
  const result = await readAgentOperatorSnapshot(agentAsset, { connection, cluster });
  return result.status === "present" ? result.snapshot : null;
}

/**
 * @deprecated Writes permanent registry metadata that grants no permission and cannot follow a
 * transfer. Use Agent Operator Permits.
 *
 * The wallet must own both the Entros Anchor and the agent's Metaplex Core asset.
 */
export async function attestAgentOperator(
  agentAsset: string,
  options: {
    wallet: AgentAnchorWallet;
    connection: AgentAnchorConnection;
    cluster?: PulseConfig["cluster"];
  },
): Promise<{ success: boolean; signature?: string; error?: string }> {
  if (options.cluster && options.cluster !== "devnet") {
    return {
      success: false,
      error: "Agent Anchor attestation is currently available on devnet only.",
    };
  }

  try {
    const { PublicKey, Transaction, TransactionInstruction, SystemProgram } =
      await import("@solana/web3.js");
    const { Buffer } = await import("buffer");

    const walletPubkey = options.wallet.adapter?.publicKey ?? options.wallet.publicKey;
    if (!walletPubkey) {
      return {
        success: false,
        error: "Wallet not connected. Call wallet.connect() before attestAgentOperator().",
      };
    }

    const programId = new PublicKey(PROGRAM_IDS.entrosAnchor);
    const [identityPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("identity"), walletPubkey.toBytes()],
      programId,
    );
    const accountInfo = await options.connection.getAccountInfo(identityPda);
    if (!accountInfo || accountInfo.data.length < 62) {
      return {
        success: false,
        error: "No Entros Anchor found. Complete a verification first.",
      };
    }
    const view = new DataView(
      accountInfo.data.buffer,
      accountInfo.data.byteOffset,
      accountInfo.data.byteLength,
    );
    const lastVerificationTimestamp = Number(view.getBigInt64(48, true));
    const trustScore = view.getUint16(60, true);

    const metadataValue = JSON.stringify({
      anchorPda: identityPda.toBase58(),
      trustScore,
      verifiedAt: lastVerificationTimestamp,
      wallet: walletPubkey.toBase58(),
    });

    const registryProgramId = new PublicKey(getRegistryProgramId(options.cluster));
    const assetPubkey = new PublicKey(agentAsset);
    const [agentPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("agent"), assetPubkey.toBytes()],
      registryProgramId,
    );
    const keyBytes = new TextEncoder().encode(AGENT_REGISTRY_CONFIG.metadataKey);
    const keyHash = (await sha256(keyBytes)).slice(0, 16);
    const metadataEntryPda = await metadataEntryAddress(assetPubkey, registryProgramId);

    // Borsh arguments: key_hash [u8; 16], key String, value Vec<u8>, immutable bool.
    const valueBytes = new TextEncoder().encode(metadataValue);
    const ixData = new Uint8Array(8 + 16 + 4 + keyBytes.length + 4 + valueBytes.length + 1);
    const ixView = new DataView(ixData.buffer);
    let offset = 0;
    ixData.set(SET_METADATA_DISCRIMINATOR, offset);
    offset += 8;
    ixData.set(keyHash, offset);
    offset += 16;
    ixView.setUint32(offset, keyBytes.length, true);
    offset += 4;
    ixData.set(keyBytes, offset);
    offset += keyBytes.length;
    ixView.setUint32(offset, valueBytes.length, true);
    offset += 4;
    ixData.set(valueBytes, offset);
    offset += valueBytes.length;
    ixData[offset] = 1;

    const instruction = new TransactionInstruction({
      programId: registryProgramId,
      keys: [
        { pubkey: metadataEntryPda, isSigner: false, isWritable: true },
        { pubkey: agentPda, isSigner: false, isWritable: false },
        { pubkey: assetPubkey, isSigner: false, isWritable: false },
        { pubkey: walletPubkey, isSigner: true, isWritable: true },
        { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
      ],
      data: Buffer.from(ixData),
    });

    const tx = new Transaction().add(instruction);
    tx.feePayer = walletPubkey;
    const { blockhash } = await options.connection.getLatestBlockhash("confirmed");
    tx.recentBlockhash = blockhash;

    const signTransaction =
      options.wallet.adapter?.signTransaction ?? options.wallet.signTransaction;
    if (!signTransaction) {
      return {
        success: false,
        error:
          "Wallet adapter does not expose signTransaction. Use a wallet that implements the standard Solana Wallet Adapter interface (Phantom, Solflare, Backpack).",
      };
    }
    const signed = await signTransaction.call(options.wallet.adapter ?? options.wallet, tx);
    const sig = await options.connection.sendRawTransaction(signed.serialize(), {
      skipPreflight: false,
      preflightCommitment: "confirmed",
    });
    await options.connection.confirmTransaction(sig, "confirmed");
    return { success: true, signature: sig };
  } catch (err: unknown) {
    return {
      success: false,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}
