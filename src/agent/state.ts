import type { AccountInfo, Connection, PublicKey } from "@solana/web3.js";

import { AGENT_REGISTRY_CONFIG } from "../config";
import { boundedRpc, INTEGRATOR_DEVNET_GENESIS_HASH } from "../identity/integrator";
import { base58Encode, canonicalKeyBytes } from "./base58";

export type AgentWalletStatus = "bound" | "unbound" | "stale";

/** Current state of one 8004 registry agent, read at one context slot. */
export interface AgentStateEvidence {
  cluster: "devnet";
  genesisHash: string;
  coreProgram: string;
  agentRegistryProgram: string;
  agentCollection: string;
  agent: string;
  agentAccount: string;
  /** Owner field of the Metaplex Core asset. The only owner source. */
  owner: string;
  /** Registry cache. The agent wallet counts only when this matches `owner`. */
  cachedOwner: string;
  agentWallet: string | null;
  agentWalletStatus: AgentWalletStatus;
  readContextSlot: number;
}

export type AgentStateFailureReason =
  | "invalid_request"
  | "wrong_cluster"
  | "agent_missing"
  | "agent_invalid"
  | "agent_unregistered"
  | "rpc_unavailable";

export type AgentStateReadResult =
  | { status: "available"; evidence: AgentStateEvidence }
  | { status: "invalid" | "unavailable"; reason: AgentStateFailureReason };

export type AgentStateConnection = Pick<
  Connection,
  "getGenesisHash" | "getMultipleAccountsInfoAndContext"
>;

export interface ReadAgentStateInput {
  /** Metaplex Core asset address of the agent. */
  agent: string;
  connection: AgentStateConnection;
  minContextSlot?: number;
}

const AGENT_ACCOUNT_DISCRIMINATOR = Uint8Array.from([
  0xf1, 0x77, 0x45, 0x8c, 0xe9, 0x09, 0x70, 0x32,
]);
const AGENT_ACCOUNT_LENGTH = 748;
const CORE_KEY_UNINITIALIZED = 0;
const CORE_KEY_ASSET = 1;
const CORE_UPDATE_AUTHORITY_COLLECTION = 2;

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  return left.length === right.length && left.every((byte, index) => byte === right[index]);
}

function readLength(data: Uint8Array, offset: number): number | null {
  if (offset + 4 > data.length) return null;
  return new DataView(data.buffer, data.byteOffset + offset, 4).getUint32(0, true);
}

type CoreAssetDecoding =
  | { owner: Uint8Array }
  | { reason: "agent_missing" | "agent_invalid" | "agent_unregistered" };

/** Decodes the `AssetV1` prefix. Plugin data after the prefix stays unread. */
function decodeCoreAsset(
  account: AccountInfo<Uint8Array> | null,
  coreProgram: PublicKey,
  collection: Uint8Array,
): CoreAssetDecoding {
  if (!account) return { reason: "agent_missing" };
  if (account.executable || !account.owner.equals(coreProgram))
    return { reason: "agent_invalid" };
  const data = account.data;
  // Core shrinks a burned asset to one uninitialized byte so nobody can recreate it.
  if (data.length === 1 && data[0] === CORE_KEY_UNINITIALIZED)
    return { reason: "agent_missing" };
  if (data.length < 34 || data[0] !== CORE_KEY_ASSET) return { reason: "agent_invalid" };
  const owner = data.subarray(1, 33);
  const updateAuthority = data[33];
  if (updateAuthority === 0 || updateAuthority === 1) return { reason: "agent_unregistered" };
  if (updateAuthority !== CORE_UPDATE_AUTHORITY_COLLECTION || data.length < 66)
    return { reason: "agent_invalid" };
  if (!equalBytes(data.subarray(34, 66), collection))
    return { reason: "agent_unregistered" };
  let offset = 66;
  for (let field = 0; field < 2; field++) {
    const length = readLength(data, offset);
    if (length === null || offset + 4 + length > data.length)
      return { reason: "agent_invalid" };
    offset += 4 + length;
  }
  const sequence = data[offset];
  if (sequence !== 0 && !(sequence === 1 && offset + 9 <= data.length))
    return { reason: "agent_invalid" };
  return { owner };
}

/** Decodes the fixed prefix of the registry `AgentAccount`, up to the agent wallet. */
function decodeAgentAccount(
  account: AccountInfo<Uint8Array> | null,
  registry: PublicKey,
  asset: Uint8Array,
  collection: Uint8Array,
  bump: number,
): { cachedOwner: Uint8Array; agentWallet: Uint8Array | null } | null {
  if (
    !account ||
    account.executable ||
    !account.owner.equals(registry) ||
    account.data.length !== AGENT_ACCOUNT_LENGTH
  )
    return null;
  const data = account.data;
  if (
    !equalBytes(data.subarray(0, 8), AGENT_ACCOUNT_DISCRIMINATOR) ||
    !equalBytes(data.subarray(8, 40), collection) ||
    !equalBytes(data.subarray(104, 136), asset) ||
    data[136] !== bump ||
    (data[137] !== 0 && data[137] !== 1)
  )
    return null;
  const walletTag = data[138];
  if (walletTag !== 0 && walletTag !== 1) return null;
  return {
    cachedOwner: data.subarray(72, 104),
    agentWallet: walletTag === 1 ? data.subarray(139, 171) : null,
  };
}

/**
 * Reads the Core asset and the registry account of an agent in one confirmed call.
 * The Core asset decides the owner. The registry decides the agent wallet only while its
 * cached owner matches. The reader never reads registry metadata entries.
 */
export async function readAgentState(
  input: ReadAgentStateInput,
): Promise<AgentStateReadResult> {
  const invalid = (reason: AgentStateFailureReason): AgentStateReadResult => ({
    status: "invalid",
    reason,
  });
  const assetBytes = canonicalKeyBytes(input.agent);
  if (!assetBytes) return invalid("invalid_request");
  const minContextSlot = input.minContextSlot;
  if (
    minContextSlot !== undefined &&
    (!Number.isSafeInteger(minContextSlot) || minContextSlot < 0)
  )
    return invalid("invalid_request");
  const { PublicKey } = await import("@solana/web3.js");
  const asset = new PublicKey(assetBytes);
  const registry = new PublicKey(AGENT_REGISTRY_CONFIG.programIdDevnet);
  const coreProgram = new PublicKey(AGENT_REGISTRY_CONFIG.coreProgramId);
  const collection = new PublicKey(AGENT_REGISTRY_CONFIG.collectionDevnet).toBytes();
  const [agentAccount, bump] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("agent"), assetBytes],
    registry,
  );
  let genesisHash: string;
  let slot: number;
  let accounts: (AccountInfo<Uint8Array> | null)[];
  try {
    const [genesis, read] = await Promise.all([
      boundedRpc(input.connection.getGenesisHash()),
      boundedRpc(
        input.connection.getMultipleAccountsInfoAndContext([asset, agentAccount], {
          commitment: "confirmed",
          ...(minContextSlot === undefined ? {} : { minContextSlot }),
        }),
      ),
    ]);
    genesisHash = genesis;
    slot = read.context.slot;
    accounts = read.value;
  } catch {
    return { status: "unavailable", reason: "rpc_unavailable" };
  }
  if (genesisHash !== INTEGRATOR_DEVNET_GENESIS_HASH) return invalid("wrong_cluster");
  if (
    !Number.isSafeInteger(slot) ||
    slot < (minContextSlot ?? 0) ||
    accounts.length !== 2
  )
    return invalid("agent_invalid");
  const core = decodeCoreAsset(accounts[0] ?? null, coreProgram, collection);
  if ("reason" in core) return invalid(core.reason);
  const registered = decodeAgentAccount(
    accounts[1] ?? null,
    registry,
    assetBytes,
    collection,
    bump,
  );
  if (!registered) return invalid("agent_invalid");
  const owner = base58Encode(core.owner);
  const cachedOwner = base58Encode(registered.cachedOwner);
  const agentWallet = registered.agentWallet ? base58Encode(registered.agentWallet) : null;
  return {
    status: "available",
    evidence: {
      cluster: "devnet",
      genesisHash,
      coreProgram: AGENT_REGISTRY_CONFIG.coreProgramId,
      agentRegistryProgram: AGENT_REGISTRY_CONFIG.programIdDevnet,
      agentCollection: AGENT_REGISTRY_CONFIG.collectionDevnet,
      agent: input.agent,
      agentAccount: agentAccount.toBase58(),
      owner,
      cachedOwner,
      agentWallet,
      agentWalletStatus:
        cachedOwner !== owner ? "stale" : agentWallet === null ? "unbound" : "bound",
      readContextSlot: slot,
    },
  };
}
