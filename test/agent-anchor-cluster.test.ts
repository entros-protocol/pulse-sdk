import { describe, expect, it } from "vitest";

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
