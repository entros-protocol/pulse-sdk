import { readFileSync } from "node:fs";
import { ed25519 } from "@noble/curves/ed25519";
import {
  Ed25519Program,
  PublicKey,
  SYSVAR_INSTRUCTIONS_PUBKEY,
  VersionedMessage,
} from "@solana/web3.js";
import { describe, expect, it } from "vitest";

import {
  agentWalletBindingMessage,
  buildSetAgentWalletInstructions,
  encodeAgentWalletBindingFragment,
  encodeAgentWalletBindingRequest,
  parseAgentWalletBindingFragment,
  parseAgentWalletBindingRequest,
  verifyAgentWalletBindingRequest,
  type AgentWalletBindingRequest,
} from "../src/agent/wallet-binding";

interface BindingVector {
  agent: string;
  agentWallet: string;
  owner: string;
  deadline: number;
  messageHex: string;
  signature: string;
  bindingRequest: string;
  fragment: string;
}

const vectors = JSON.parse(
  readFileSync(new URL("./fixtures/agent-permit-vectors.json", import.meta.url), "utf8"),
) as {
  binding: BindingVector[];
  keys: Record<string, { seedHex?: string; publicKey: string }>;
  valid: { message: string; presentation: string }[];
};
const vector = vectors.binding[0] as BindingVector;
const request = JSON.parse(vector.bindingRequest) as AgentWalletBindingRequest;
const hex = (bytes: Uint8Array) => Buffer.from(bytes).toString("hex");

describe("8004 agent wallet binding", () => {
  it("reproduces the registry message and request encoding from the vectors", () => {
    expect(hex(agentWalletBindingMessage(request))).toBe(vector.messageHex);
    expect(parseAgentWalletBindingRequest(vector.bindingRequest)).toEqual(request);
    expect(encodeAgentWalletBindingRequest(request)).toBe(vector.bindingRequest);
    expect(encodeAgentWalletBindingFragment(request)).toBe(vector.fragment);
    expect(parseAgentWalletBindingFragment(`#${vector.fragment}`)).toEqual(request);
    expect(verifyAgentWalletBindingRequest(request)).toBe(true);
  });

  it("rejects a changed field, another signer and a small-order agent wallet", () => {
    expect(verifyAgentWalletBindingRequest({ ...request, deadline: request.deadline + 1 })).toBe(false);
    expect(verifyAgentWalletBindingRequest({ ...request, owner: vectors.keys.other!.publicKey })).toBe(false);
    const otherSignature = Buffer.from(
      ed25519.sign(agentWalletBindingMessage(request), Buffer.from(vectors.keys.other!.seedHex!, "hex")),
    ).toString("hex");
    expect(verifyAgentWalletBindingRequest({ ...request, signature: otherSignature })).toBe(false);
    expect(
      parseAgentWalletBindingRequest({ ...request, agentWallet: vectors.keys.smallOrder!.publicKey }),
    ).toBeNull();
    for (const value of [
      { ...request, extra: 1 },
      { ...request, version: 2 },
      { ...request, deadline: 0 },
      { ...request, deadline: 1.5 },
      { ...request, signature: request.signature.toUpperCase() },
      { ...request, agent: "1" + request.agent },
      "not json",
    ]) {
      expect(parseAgentWalletBindingRequest(value)).toBeNull();
    }
    expect(parseAgentWalletBindingFragment("request=abc")).toBeNull();
    expect(parseAgentWalletBindingFragment("bind=***")).toBeNull();
  });

  it("builds the Ed25519 check directly before set_agent_wallet", async () => {
    const [verify, setAgentWallet] = await buildSetAgentWalletInstructions(request);
    expect(verify!.programId.equals(Ed25519Program.programId)).toBe(true);
    const data = verify!.data;
    expect(data[0]).toBe(1);
    // Signature, key and message indices of u16::MAX point at this same instruction.
    for (const offset of [4, 8, 14]) expect(data.readUInt16LE(offset)).toBe(0xffff);
    const keyOffset = data.readUInt16LE(6);
    const signatureOffset = data.readUInt16LE(2);
    const messageOffset = data.readUInt16LE(10);
    const messageSize = data.readUInt16LE(12);
    expect(new PublicKey(data.subarray(keyOffset, keyOffset + 32)).toBase58()).toBe(request.agentWallet);
    expect(hex(data.subarray(signatureOffset, signatureOffset + 64))).toBe(request.signature);
    expect(hex(data.subarray(messageOffset, messageOffset + messageSize))).toBe(vector.messageHex);

    const registry = new PublicKey("8oo4J9tBB3Hna1jRQ3rWvJjojqM5DYTDJo5cejUuJy3C");
    const [agentAccount] = PublicKey.findProgramAddressSync(
      [Buffer.from("agent"), new PublicKey(request.agent).toBuffer()],
      registry,
    );
    expect(setAgentWallet!.programId.equals(registry)).toBe(true);
    expect(
      setAgentWallet!.keys.map((key) => [key.pubkey.toBase58(), key.isSigner, key.isWritable]),
    ).toEqual([
      [request.owner, true, false],
      [agentAccount.toBase58(), false, true],
      [request.agent, false, false],
      [SYSVAR_INSTRUCTIONS_PUBKEY.toBase58(), false, false],
    ]);
    const expected = Buffer.alloc(48);
    Buffer.from("9a57fb17330c0496", "hex").copy(expected, 0);
    new PublicKey(request.agentWallet).toBuffer().copy(expected, 8);
    expected.writeBigInt64LE(BigInt(request.deadline), 40);
    expect(hex(setAgentWallet!.data)).toBe(expected.toString("hex"));
  });

  it("refuses to build a transaction around an invalid signature", async () => {
    await expect(
      buildSetAgentWalletInstructions({ ...request, deadline: request.deadline + 1 }),
    ).rejects.toThrow("Agent wallet binding signature is invalid");
  });
});

describe("permit texts cannot pass as Solana transactions", () => {
  it("fails to deserialize every permit and presentation text", () => {
    for (const valid of vectors.valid) {
      for (const text of [valid.message, valid.presentation]) {
        const bytes = new TextEncoder().encode(text);
        // A legacy header needs fewer read-only signed accounts than required signatures.
        expect(bytes[1]).toBeGreaterThanOrEqual(bytes[0]!);
        expect(() => VersionedMessage.deserialize(bytes)).toThrow();
      }
    }
  });
});
