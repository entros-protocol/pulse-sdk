import { describe, expect, it } from "vitest";

import { attestAgentOperator } from "../src/agent/anchor";

describe("Agent Anchor cluster boundary", () => {
  it("rejects mainnet before wallet or RPC work", async () => {
    const wallet = {
      get publicKey(): never {
        throw new Error("wallet access must not occur");
      },
    };
    const connection = {
      getAccountInfo(): never {
        throw new Error("RPC access must not occur");
      },
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
