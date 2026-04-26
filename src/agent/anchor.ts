/* eslint-disable @typescript-eslint/no-explicit-any */
import { PROGRAM_IDS, AGENT_REGISTRY_CONFIG } from "../config";
import type { PulseConfig } from "../config";

function getRegistryProgramId(cluster?: PulseConfig["cluster"]): string {
  return cluster === "mainnet-beta"
    ? AGENT_REGISTRY_CONFIG.programIdMainnet
    : AGENT_REGISTRY_CONFIG.programIdDevnet;
}

/** Metadata written to an AI agent linking it to a verified human operator */
export interface AgentHumanOperator {
  anchorPda: string;
  trustScore: number;
  verifiedAt: number;
  wallet: string;
}

/**
 * Compute SHA256 hash, browser-compatible via SubtleCrypto.
 */
async function sha256(data: Uint8Array): Promise<Uint8Array> {
  const ab = new ArrayBuffer(data.length);
  new Uint8Array(ab).set(data);
  return new Uint8Array(await crypto.subtle.digest("SHA-256", ab));
}

/**
 * Attest that a verified Entros human operates an AI agent on the Solana Agent Registry.
 *
 * Reads the user's on-chain IdentityState PDA, builds metadata JSON, and writes
 * it to the agent's metadata via a manually constructed set_metadata_pda instruction.
 * The metadata is immutable once set, permanently linking the agent to its human operator.
 *
 * The wallet must own both the Entros Anchor and the agent's Metaplex Core NFT.
 *
 * @param agentAsset - Base58 pubkey of the agent's Metaplex Core NFT
 * @param options - Wallet adapter and Solana connection
 * @returns Transaction signature on success
 */
export async function attestAgentOperator(
  agentAsset: string,
  options: {
    wallet: any;
    connection: any;
    cluster?: PulseConfig["cluster"];
  }
): Promise<{ success: boolean; signature?: string; error?: string }> {
  try {
    const { PublicKey, Transaction, TransactionInstruction, SystemProgram } =
      await import("@solana/web3.js");

    const walletPubkey =
      options.wallet.adapter?.publicKey ?? options.wallet.publicKey;
    if (!walletPubkey) {
      return {
        success: false,
        error: "Wallet not connected. Call wallet.connect() before attestAgentOperator().",
      };
    }

    // 1. Read Entros IdentityState PDA
    const programId = new PublicKey(PROGRAM_IDS.entrosAnchor);
    const [identityPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("identity"), walletPubkey.toBuffer()],
      programId
    );

    const accountInfo = await options.connection.getAccountInfo(identityPda);
    if (!accountInfo || accountInfo.data.length < 62) {
      return {
        success: false,
        error: "No Entros Anchor found. Complete a verification first.",
      };
    }

    // 2. Deserialize trust_score and last_verification_timestamp
    const data = accountInfo.data;
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const lastVerificationTimestamp = Number(view.getBigInt64(48, true));
    const trustScore = view.getUint16(60, true);

    // 3. Build metadata JSON value
    const metadata: AgentHumanOperator = {
      anchorPda: identityPda.toBase58(),
      trustScore,
      verifiedAt: lastVerificationTimestamp,
      wallet: walletPubkey.toBase58(),
    };
    const metadataValue = JSON.stringify(metadata);
    const metadataKey = AGENT_REGISTRY_CONFIG.metadataKey;

    // 4. Derive PDAs for the 8004 Agent Registry
    const registryProgramId = new PublicKey(
      getRegistryProgramId(options.cluster)
    );
    const assetPubkey = new PublicKey(agentAsset);

    // Agent PDA: ["agent", asset]
    const [agentPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("agent"), assetPubkey.toBuffer()],
      registryProgramId
    );

    // Key hash: SHA256(key)[0..16]
    const keyBytes = new TextEncoder().encode(metadataKey);
    const keyHashFull = await sha256(keyBytes);
    const keyHash = keyHashFull.slice(0, 16);

    // Metadata entry PDA: ["agent_meta", asset, keyHash[0..16]]
    const [metadataEntryPda] = PublicKey.findProgramAddressSync(
      [
        new TextEncoder().encode("agent_meta"),
        assetPubkey.toBuffer(),
        keyHash,
      ],
      registryProgramId
    );

    // 5. Build set_metadata_pda instruction data
    // Discriminator: [236, 60, 23, 48, 138, 69, 196, 153]
    // Args: key_hash ([u8; 16]), key (String), value (Vec<u8>), immutable (bool)
    const valueBytes = new TextEncoder().encode(metadataValue);
    const discriminator = new Uint8Array([236, 60, 23, 48, 138, 69, 196, 153]);

    const ixDataSize =
      8 + // discriminator
      16 + // key_hash [u8; 16]
      4 + keyBytes.length + // key (Borsh string: 4-byte len + utf8)
      4 + valueBytes.length + // value (Borsh Vec<u8>: 4-byte len + bytes)
      1; // immutable (bool)

    const ixData = new Uint8Array(ixDataSize);
    const ixView = new DataView(ixData.buffer);
    let offset = 0;

    // Discriminator
    ixData.set(discriminator, offset);
    offset += 8;

    // key_hash: [u8; 16]
    ixData.set(keyHash, offset);
    offset += 16;

    // key: Borsh String (4-byte LE len + UTF-8)
    ixView.setUint32(offset, keyBytes.length, true);
    offset += 4;
    ixData.set(keyBytes, offset);
    offset += keyBytes.length;

    // value: Borsh Vec<u8> (4-byte LE len + bytes)
    ixView.setUint32(offset, valueBytes.length, true);
    offset += 4;
    ixData.set(valueBytes, offset);
    offset += valueBytes.length;

    // immutable: bool
    ixData[offset] = 1; // true
    offset += 1;

    // 6. Build instruction with 5 accounts
    const { Buffer: SolBuffer } = await import("buffer");
    const instruction = new TransactionInstruction({
      programId: registryProgramId,
      keys: [
        { pubkey: metadataEntryPda, isSigner: false, isWritable: true },
        { pubkey: agentPda, isSigner: false, isWritable: false },
        { pubkey: assetPubkey, isSigner: false, isWritable: false },
        { pubkey: walletPubkey, isSigner: true, isWritable: true },
        {
          pubkey: SystemProgram.programId,
          isSigner: false,
          isWritable: false,
        },
      ],
      data: SolBuffer.from(ixData),
    });

    // 7. Build transaction, sign, send
    const tx = new Transaction().add(instruction);
    tx.feePayer = walletPubkey;
    const { blockhash } = await options.connection.getLatestBlockhash(
      "confirmed"
    );
    tx.recentBlockhash = blockhash;

    const signFn =
      options.wallet.adapter?.signTransaction ??
      options.wallet.signTransaction;
    if (!signFn) {
      return {
        success: false,
        error: "Wallet adapter does not expose signTransaction. Use a wallet that implements the standard Solana Wallet Adapter interface (Phantom, Solflare, Backpack).",
      };
    }
    const signed = await signFn.call(
      options.wallet.adapter ?? options.wallet,
      tx
    );

    const sig = await options.connection.sendRawTransaction(
      signed.serialize(),
      { skipPreflight: false, preflightCommitment: "confirmed" }
    );

    await options.connection.confirmTransaction(sig, "confirmed");

    return { success: true, signature: sig };
  } catch (err: any) {
    return { success: false, error: err.message ?? String(err) };
  }
}

/**
 * Query whether an AI agent has a verified human operator via Entros.
 *
 * Reads the "entros:human-operator" metadata from the agent's on-chain record
 * and returns the operator's Entros Anchor details.
 *
 * @param agentAsset - Base58 pubkey of the agent's Metaplex Core NFT
 * @param connection - Solana connection (optional, defaults to devnet)
 * @returns Operator metadata or null if no Entros attestation exists
 */
export async function getAgentHumanOperator(
  agentAsset: string,
  connection?: any,
  cluster?: PulseConfig["cluster"],
): Promise<AgentHumanOperator | null> {
  try {
    const { PublicKey } = await import("@solana/web3.js");

    const registryProgramId = new PublicKey(
      getRegistryProgramId(cluster)
    );
    const assetPubkey = new PublicKey(agentAsset);
    const metadataKey = AGENT_REGISTRY_CONFIG.metadataKey;

    // Derive metadata entry PDA
    const keyBytes = new TextEncoder().encode(metadataKey);
    const keyHashFull = await sha256(keyBytes);
    const keyHash = keyHashFull.slice(0, 16);

    const [metadataEntryPda] = PublicKey.findProgramAddressSync(
      [
        new TextEncoder().encode("agent_meta"),
        assetPubkey.toBuffer(),
        keyHash,
      ],
      registryProgramId
    );

    // Read account directly (no SDK dependency)
    const conn =
      connection ??
      new (await import("@solana/web3.js")).Connection(
        "https://api.devnet.solana.com",
        "confirmed"
      );

    const accountInfo = await conn.getAccountInfo(metadataEntryPda);
    if (!accountInfo) return null;

    // Deserialize MetadataEntryPda
    // Layout: 8 (disc) + 32 (asset) + 1 (immutable) + 1 (bump) + 4+N (metadata_key) + 4+M (metadata_value)
    const raw = accountInfo.data;
    if (raw.length < 46) return null; // minimum size

    let offset = 8 + 32 + 1 + 1; // skip disc, asset, immutable, bump = 42

    // Read metadata_key (Borsh String: 4-byte LE len + UTF-8)
    const keyLen = new DataView(
      raw.buffer,
      raw.byteOffset + offset,
      4
    ).getUint32(0, true);
    offset += 4 + keyLen;

    // Read metadata_value (Borsh Vec<u8>: 4-byte LE len + bytes)
    if (offset + 4 > raw.length) return null;
    const valueLen = new DataView(
      raw.buffer,
      raw.byteOffset + offset,
      4
    ).getUint32(0, true);
    offset += 4;

    if (offset + valueLen > raw.length) return null;
    const valueBytes = raw.slice(offset, offset + valueLen);
    const valueStr = new TextDecoder().decode(valueBytes);

    return JSON.parse(valueStr) as AgentHumanOperator;
  } catch {
    return null;
  }
}
