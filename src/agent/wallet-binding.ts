import { ed25519 } from "@noble/curves/ed25519";
import type { TransactionInstruction } from "@solana/web3.js";

import { AGENT_REGISTRY_CONFIG } from "../config";
import { canonicalKeyBytes } from "./base58";

/** The registry accepts a deadline at most this many seconds past its clock. */
export const AGENT_WALLET_BINDING_MAX_DEADLINE_SECONDS = 300;

/** What an agent hands its owner so the owner can bind the agent wallet in the registry. */
export interface AgentWalletBindingRequest {
  version: 1;
  agent: string;
  agentWallet: string;
  owner: string;
  /** Unix seconds. The registry rejects the binding after this time. */
  deadline: number;
  /** Agent wallet signature over `agentWalletBindingMessage`, as 128 lowercase hex. */
  signature: string;
}

const PREFIX = new TextEncoder().encode("8004_WALLET_SET:");
const SET_AGENT_WALLET_DISCRIMINATOR = Uint8Array.from([
  0x9a, 0x57, 0xfb, 0x17, 0x33, 0x0c, 0x04, 0x96,
]);
const KEYS = ["version", "agent", "agentWallet", "owner", "deadline", "signature"] as const;
const HEX128 = /^[0-9a-f]{128}$/;

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hexToBytes(value: string): Uint8Array {
  return Uint8Array.from(value.match(/../g) ?? [], (byte) => Number.parseInt(byte, 16));
}

function strictPublicKey(bytes: Uint8Array): boolean {
  try {
    return !ed25519.Point.fromBytes(bytes, false).isSmallOrder();
  } catch {
    return false;
  }
}

function address(value: unknown): value is string {
  return canonicalKeyBytes(value) !== null;
}

function signingKey(value: unknown): value is string {
  const bytes = canonicalKeyBytes(value);
  return bytes !== null && strictPublicKey(bytes);
}

function deadline(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0;
}

/**
 * The registry's own wallet-set message:
 * `"8004_WALLET_SET:" || asset || agent wallet || owner || deadline as little-endian i64`.
 */
export function agentWalletBindingMessage(input: {
  agent: string;
  agentWallet: string;
  owner: string;
  deadline: number;
}): Uint8Array {
  const agent = canonicalKeyBytes(input.agent);
  const agentWallet = canonicalKeyBytes(input.agentWallet);
  const owner = canonicalKeyBytes(input.owner);
  if (!agent || !agentWallet || !owner || !deadline(input.deadline))
    throw new Error("Invalid agent wallet binding fields");
  const message = new Uint8Array(PREFIX.length + 32 * 3 + 8);
  message.set(PREFIX, 0);
  message.set(agent, PREFIX.length);
  message.set(agentWallet, PREFIX.length + 32);
  message.set(owner, PREFIX.length + 64);
  new DataView(message.buffer).setBigInt64(PREFIX.length + 96, BigInt(input.deadline), true);
  return message;
}

/** Strict parse of a binding request given as an object or its JSON text. */
export function parseAgentWalletBindingRequest(
  input: unknown,
): AgentWalletBindingRequest | null {
  let value: unknown = input;
  if (typeof input === "string") {
    if (input.length > 1024) return null;
    try {
      value = JSON.parse(input);
    } catch {
      return null;
    }
  }
  if (
    !record(value) ||
    Object.keys(value).length !== KEYS.length ||
    !KEYS.every((name) => Object.prototype.hasOwnProperty.call(value, name))
  )
    return null;
  const { version, agent, agentWallet, owner, signature } = value;
  if (
    version !== 1 ||
    !address(agent) ||
    !signingKey(agentWallet) ||
    !address(owner) ||
    !deadline(value.deadline) ||
    typeof signature !== "string" ||
    !HEX128.test(signature)
  )
    return null;
  return { version: 1, agent, agentWallet, owner, deadline: value.deadline, signature };
}

/** Canonical JSON with a fixed key order. */
export function encodeAgentWalletBindingRequest(request: AgentWalletBindingRequest): string {
  const value = parseAgentWalletBindingRequest(request);
  if (!value) throw new Error("Invalid agent wallet binding request");
  return JSON.stringify({
    version: 1,
    agent: value.agent,
    agentWallet: value.agentWallet,
    owner: value.owner,
    deadline: value.deadline,
    signature: value.signature,
  });
}

function base64UrlEncode(text: string): string {
  return btoa(text).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64UrlDecode(value: string): string | null {
  if (!/^[A-Za-z0-9_-]*$/.test(value) || value.length % 4 === 1) return null;
  try {
    const padded = value.replace(/-/g, "+").replace(/_/g, "/");
    return atob(padded + "=".repeat((4 - (padded.length % 4)) % 4));
  } catch {
    return null;
  }
}

/** URL fragment for the Entros signing page, without the leading `#`. */
export function encodeAgentWalletBindingFragment(request: AgentWalletBindingRequest): string {
  return `bind=${base64UrlEncode(encodeAgentWalletBindingRequest(request))}`;
}

export function parseAgentWalletBindingFragment(
  fragment: string,
): AgentWalletBindingRequest | null {
  const body = fragment.startsWith("#") ? fragment.slice(1) : fragment;
  if (!body.startsWith("bind=") || body.length > 2048) return null;
  const json = base64UrlDecode(body.slice("bind=".length));
  return json === null ? null : parseAgentWalletBindingRequest(json);
}

/** Strict Ed25519 check of the agent wallet signature. */
export function verifyAgentWalletBindingRequest(request: AgentWalletBindingRequest): boolean {
  const value = parseAgentWalletBindingRequest(request);
  const walletKey = value ? canonicalKeyBytes(value.agentWallet) : null;
  if (!value || !walletKey) return false;
  try {
    return ed25519.verify(hexToBytes(value.signature), agentWalletBindingMessage(value), walletKey, {
      zip215: false,
    });
  } catch {
    return false;
  }
}

/**
 * Builds the two instructions of the binding transaction: the Ed25519 check of the agent
 * wallet signature, then the registry's `set_agent_wallet`. The registry reads the Ed25519
 * instruction at the index directly before its own, so keep them adjacent and in this order.
 * The owner signs the transaction.
 */
export async function buildSetAgentWalletInstructions(
  request: AgentWalletBindingRequest,
): Promise<TransactionInstruction[]> {
  if (!verifyAgentWalletBindingRequest(request))
    throw new Error("Agent wallet binding signature is invalid");
  const { Ed25519Program, PublicKey, SYSVAR_INSTRUCTIONS_PUBKEY, TransactionInstruction } =
    await import("@solana/web3.js");
  const { Buffer } = await import("buffer");
  const registry = new PublicKey(AGENT_REGISTRY_CONFIG.programIdDevnet);
  const asset = new PublicKey(request.agent);
  const agentWallet = new PublicKey(request.agentWallet);
  const [agentAccount] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("agent"), asset.toBytes()],
    registry,
  );
  const verify = Ed25519Program.createInstructionWithPublicKey({
    publicKey: agentWallet.toBytes(),
    message: agentWalletBindingMessage(request),
    signature: hexToBytes(request.signature),
  });
  const data = new Uint8Array(8 + 32 + 8);
  data.set(SET_AGENT_WALLET_DISCRIMINATOR, 0);
  data.set(agentWallet.toBytes(), 8);
  new DataView(data.buffer).setBigInt64(40, BigInt(request.deadline), true);
  const setAgentWallet = new TransactionInstruction({
    programId: registry,
    keys: [
      { pubkey: new PublicKey(request.owner), isSigner: true, isWritable: false },
      { pubkey: agentAccount, isSigner: false, isWritable: true },
      { pubkey: asset, isSigner: false, isWritable: false },
      { pubkey: SYSVAR_INSTRUCTIONS_PUBKEY, isSigner: false, isWritable: false },
    ],
    data: Buffer.from(data),
  });
  return [verify, setAgentWallet];
}
