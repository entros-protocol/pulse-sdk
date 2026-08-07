import { describe, expect, it } from "vitest";
import { submitViaWallet } from "../src/submit/wallet";

describe("first-verification mint receipt", () => {
  it("fails before wallet submission when the receipt is absent", async () => {
    let walletCalled = false;
    const result = await submitViaWallet(
      { proofBytes: new Uint8Array(), publicInputs: [] },
      new Uint8Array(32),
      {
        isFirstVerification: true,
        wallet: {
          sendTransaction: async () => {
            walletCalled = true;
            return "unexpected";
          },
        },
        connection: {},
      },
    );

    expect(result.success).toBe(false);
    expect(result.failedAt).toBe("submission");
    expect(result.error).toContain("validator-signed mint receipt");
    expect(walletCalled).toBe(false);
  });
});
