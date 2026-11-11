import { Keypair, PublicKey, type Transaction } from "@solana/web3.js";
import { describe, expect, it } from "vitest";

import { PROGRAM_IDS } from "../src/config";

import { attestAgentOperator, type AgentAnchorConnection } from "../src/agent/anchor";

describe("Agent Anchor cluster boundary", () => {
  it("rejects mainnet before wallet or RPC work", async () => {
    const wallet = {
      get publicKey(): never {
        throw new Error("wallet access must not occur");
      },
    };
    const unreachable = (): never => {
      throw new Error("RPC access must not occur");
    };
    const connection: AgentAnchorConnection = {
      getAccountInfo: unreachable,
      getLatestBlockhash: unreachable,
      sendRawTransaction: unreachable,
      confirmTransaction: unreachable,
    };

    const result = await attestAgentOperator("unused", {
      wallet,
      connection,
      cluster: "mainnet-beta",
    });

    expect(result).toEqual({
      success: false,
      error: "Agent Anchor attestation is currently available on devnet only.",
    });
  });
});

describe("Agent Anchor attestation result", () => {
  const owner = Keypair.generate();
  const asset = Keypair.generate().publicKey.toBase58();

  function connection(confirmation: unknown): AgentAnchorConnection {
    return {
      getAccountInfo: async () => ({
        data: Buffer.alloc(593, 1),
        owner: new PublicKey(PROGRAM_IDS.entrosAnchor),
        executable: false,
        lamports: 1,
      }),
      getLatestBlockhash: async () => ({
        blockhash: "11111111111111111111111111111111",
        lastValidBlockHeight: 1,
      }),
      sendRawTransaction: async () => "attestation-signature",
      confirmTransaction: async () => confirmation,
    } as unknown as AgentAnchorConnection;
  }

  const wallet = {
    publicKey: owner.publicKey,
    signTransaction: async (transaction: Transaction) => {
      transaction.sign(owner);
      return transaction;
    },
  };

  it("reports success for a transaction that confirmed without an error", async () => {
    const result = await attestAgentOperator(asset, {
      wallet,
      connection: connection({ value: { err: null } }),
    });
    expect(result).toEqual({ success: true, signature: "attestation-signature" });
  });

  // A transaction confirms once a block holds it. That says nothing about whether it
  // succeeded: a failed transaction confirms too, and carries its error in the result.
  it("reports failure for a transaction that confirmed with an error", async () => {
    const result = await attestAgentOperator(asset, {
      wallet,
      connection: connection({ value: { err: { InstructionError: [0, { Custom: 6001 }] } } }),
    });
    expect(result.success).toBe(false);
    expect(result.signature).toBeUndefined();
    expect(result.error).toContain("6001");
  });
});
