import { describe, it, expect } from "vitest";
import { Keypair } from "@solana/web3.js";
import { submitResetViaWallet } from "../src/submit/wallet";

/**
 * Phase attribution across the signing / submission / confirmation seam.
 *
 * Wallet adapters merge signing and sending into one `sendTransaction` call,
 * and that seam is where both 2026-07-31 production failures lived: a wallet
 * prompt that never appeared was reported as a proving timeout, and an
 * on-chain revert was reported as a validator rejection. The rules below are
 * what let a host describe each outcome correctly, so they are pinned here
 * rather than left to the call site.
 *
 * The reset path is the vehicle because it is the shortest route through
 * `sendAndConfirm`: one instruction, no proof, no receipt. The rules it
 * exercises are shared with the mint and re-verify paths.
 *
 * The signature timeout is not driven here. Its two halves are covered
 * separately and completely: `withTimeout` is tested directly, and the timeout
 * message is asserted not to look like a user rejection, which is the whole of
 * what decides its phase.
 */

const AUTHORITY = Keypair.generate().publicKey;

function fakeConnection(overrides: Record<string, unknown> = {}) {
  return {
    rpcEndpoint: "http://localhost:8899",
    commitment: "confirmed",
    getLatestBlockhash: async () => ({
      blockhash: "11111111111111111111111111111111",
      lastValidBlockHeight: 1,
    }),
    confirmTransaction: async () => ({ value: { err: null } }),
    ...overrides,
  };
}

function fakeWallet(sendTransaction: () => Promise<string>) {
  return {
    publicKey: AUTHORITY,
    sendTransaction,
    signTransaction: async (tx: unknown) => tx,
    signAllTransactions: async (txs: unknown) => txs,
  };
}

describe("sendAndConfirm phase attribution", () => {
  it("reports a confirmed reset without a phase", async () => {
    const result = await submitResetViaWallet(new Uint8Array(32), {
      wallet: fakeWallet(async () => "sig-ok"),
      connection: fakeConnection(),
    });
    expect(result.success).toBe(true);
    expect(result.txSignature).toBe("sig-ok");
    expect(result.failedAt).toBeUndefined();
  });

  it("attributes a declined prompt to signing", async () => {
    // The one outcome that is certainly not on the wire, and the only reason
    // `signing` exists as a phase distinct from `submission`.
    const result = await submitResetViaWallet(new Uint8Array(32), {
      wallet: fakeWallet(async () => {
        throw new Error("User rejected the request.");
      }),
      connection: fakeConnection(),
    });
    expect(result.success).toBe(false);
    expect(result.failedAt).toBe("signing");
  });

  it("attributes any other send failure to submission", async () => {
    // The adapter may have signed and failed while sending, so the outcome is
    // unknown. `submission` says so; `signing` would claim nothing was sent.
    const result = await submitResetViaWallet(new Uint8Array(32), {
      wallet: fakeWallet(async () => {
        throw new Error("failed to send transaction: Node is behind by 200 slots");
      }),
      connection: fakeConnection(),
    });
    expect(result.success).toBe(false);
    expect(result.failedAt).toBe("submission");
  });

  it("attributes a cluster-reported revert to confirmation", async () => {
    const result = await submitResetViaWallet(new Uint8Array(32), {
      wallet: fakeWallet(async () => "sig-reverted"),
      connection: fakeConnection({
        confirmTransaction: async () => ({
          value: { err: { InstructionError: [1, "InstructionDidNotDeserialize"] } },
        }),
      }),
    });
    expect(result.success).toBe(false);
    expect(result.failedAt).toBe("confirmation");
    // The revert text has to survive, because a host still routes `Custom 6011`
    // and `Custom 6012` to their own surfaces before falling back to the
    // opaque one.
    expect(result.error).toContain("InstructionDidNotDeserialize");
  });

  it("does not claim a revert when the RPC stopped answering", async () => {
    // `confirmation` carries a spend of `certain`. An RPC that threw proves
    // nothing about whether the transaction landed, so it must not land there.
    const result = await submitResetViaWallet(new Uint8Array(32), {
      wallet: fakeWallet(async () => "sig-unknown"),
      connection: fakeConnection({
        confirmTransaction: async () => {
          throw new Error("failed to get confirmation: connection reset");
        },
      }),
    });
    expect(result.success).toBe(false);
    expect(result.failedAt).toBe("submission");
  });

  it("attributes a failure to build the transaction to submission", async () => {
    // Nothing was spent here, and `submission` reports `possible` rather than
    // `none`. That is the one direction it is safe to be wrong in.
    const result = await submitResetViaWallet(new Uint8Array(32), {
      wallet: fakeWallet(async () => "unreachable"),
      connection: fakeConnection({
        getLatestBlockhash: async () => {
          throw new Error("429 Too Many Requests");
        },
      }),
    });
    expect(result.success).toBe(false);
    expect(result.failedAt).toBe("submission");
  });
});
