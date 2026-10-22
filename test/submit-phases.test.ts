import { describe, it, expect } from "vitest";
import { Keypair } from "@solana/web3.js";
import { submitResetViaWallet } from "../src/submit/wallet";
import type { SignedReceiptDto } from "../src/submit/types";
import { phaseChargesAttempt, phaseSpend } from "../src/phases";
import { fundedPreflightConnection } from "./helpers/funding";

/**
 * Phase attribution across the signing / submission / confirmation seam.
 *
 * Wallet adapters merge signing and sending into one `sendTransaction` call,
 * and that seam is where misattributed failures live: a wallet prompt that
 * never appears can be reported as a proving timeout, and an on-chain revert
 * as a validator rejection. The rules below are
 * what let a host describe each outcome correctly, so they are pinned here
 * rather than left to the call site.
 *
 * The reset path is the vehicle because it is the shortest route through
 * `sendAndConfirm`: one instruction, no proof, and a reset receipt. The rules
 * it exercises are shared with the mint and re-verify paths.
 *
 * The signature timeout is not driven here. Its two halves are covered
 * separately and completely: `withTimeout` is tested directly, and the timeout
 * message is asserted not to look like a user rejection, which is the whole of
 * what decides its phase.
 */

const AUTHORITY = Keypair.generate().publicKey;
const COMMITMENT = new Uint8Array(32);

function resetReceipt(): SignedReceiptDto {
  const message = Buffer.alloc(103);
  Buffer.from("entros-validator-receipt-v2\0", "ascii").copy(message, 0);
  message[28] = 3;
  message.writeUInt16LE(0, 29);
  message.set(AUTHORITY.toBytes(), 31);
  message.set(COMMITMENT, 63);
  message.writeBigInt64LE(1_786_310_400n, 95);
  return {
    validator_pubkey_hex: "11".repeat(32),
    signature_hex: "22".repeat(64),
    message_hex: message.toString("hex"),
  };
}

function fakeConnection(overrides: Record<string, unknown> = {}) {
  return {
    ...fundedPreflightConnection(),
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
  it("refuses insufficient network fees before asking the wallet", async () => {
    let walletCalls = 0;
    const result = await submitResetViaWallet(COMMITMENT, {
      wallet: fakeWallet(async () => { walletCalls += 1; return "unreachable"; }),
      connection: fakeConnection({ getBalance: async () => 0 }),
      signedReceipt: resetReceipt(),
    });
    expect(result).toMatchObject({ success: false, failedAt: "signing" });
    expect(result.error).toContain("Insufficient funds");
    expect(walletCalls).toBe(0);
    expect(phaseSpend(result.failedAt)).toBe("none");
    expect(phaseChargesAttempt(result.failedAt)).toBe(false);
  });

  it("refuses insufficient account rent before a wallet prompt", async () => {
    let walletCalls = 0;
    const result = await submitResetViaWallet(COMMITMENT, {
      wallet: fakeWallet(async () => { walletCalls += 1; return "unreachable"; }),
      connection: fakeConnection({
        simulateTransaction: async () => ({
          value: { err: { InsufficientFundsForRent: { account_index: 0 } }, logs: null },
        }),
      }),
      signedReceipt: resetReceipt(),
    });
    expect(result).toMatchObject({ success: false, failedAt: "signing" });
    expect(result.error).toContain("account rent");
    expect(walletCalls).toBe(0);
  });

  it("does not ask the wallet when funding readback is unavailable", async () => {
    let walletCalls = 0;
    const result = await submitResetViaWallet(COMMITMENT, {
      wallet: fakeWallet(async () => { walletCalls += 1; return "unreachable"; }),
      connection: fakeConnection({
        getBalance: async () => { throw new Error("RPC unavailable"); },
      }),
      signedReceipt: resetReceipt(),
    });
    expect(result).toMatchObject({ success: false, failedAt: "signing" });
    expect(result.error).toContain("could not check");
    expect(result.error).not.toContain("Insufficient funds");
    expect(walletCalls).toBe(0);
  });

  it("reports a confirmed reset without a phase", async () => {
    const result = await submitResetViaWallet(COMMITMENT, {
      wallet: fakeWallet(async () => "sig-ok"),
      connection: fakeConnection(),
      signedReceipt: resetReceipt(),
    });
    expect(result.success).toBe(true);
    expect(result.txSignature).toBe("sig-ok");
    expect(result.failedAt).toBeUndefined();
  });

  it("attributes a declined prompt to signing", async () => {
    // Among wallet-send failures, a declined prompt proves no broadcast.
    // The funding preflight can also stop before that send call.
    const result = await submitResetViaWallet(COMMITMENT, {
      wallet: fakeWallet(async () => {
        throw new Error("User rejected the request.");
      }),
      connection: fakeConnection(),
      signedReceipt: resetReceipt(),
    });
    expect(result.success).toBe(false);
    expect(result.failedAt).toBe("signing");
  });

  it("attributes any other send failure to submission", async () => {
    // The adapter may have signed and failed while sending, so the outcome is
    // unknown. `submission` says so; `signing` would claim nothing was sent.
    const result = await submitResetViaWallet(COMMITMENT, {
      wallet: fakeWallet(async () => {
        throw new Error("failed to send transaction: Node is behind by 200 slots");
      }),
      connection: fakeConnection(),
      signedReceipt: resetReceipt(),
    });
    expect(result.success).toBe(false);
    expect(result.failedAt).toBe("submission");
  });

  it("attributes a cluster-reported revert to confirmation", async () => {
    const result = await submitResetViaWallet(COMMITMENT, {
      wallet: fakeWallet(async () => "sig-reverted"),
      connection: fakeConnection({
        confirmTransaction: async () => ({
          value: { err: { InstructionError: [1, "InstructionDidNotDeserialize"] } },
        }),
      }),
      signedReceipt: resetReceipt(),
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
    const result = await submitResetViaWallet(COMMITMENT, {
      wallet: fakeWallet(async () => "sig-unknown"),
      connection: fakeConnection({
        confirmTransaction: async () => {
          throw new Error("failed to get confirmation: connection reset");
        },
      }),
      signedReceipt: resetReceipt(),
    });
    expect(result.success).toBe(false);
    expect(result.failedAt).toBe("submission");
  });

  it("attributes a failure to build the transaction to submission", async () => {
    // Nothing was spent here, and `submission` reports `possible` rather than
    // `none`. That is the one direction it is safe to be wrong in.
    const result = await submitResetViaWallet(COMMITMENT, {
      wallet: fakeWallet(async () => "unreachable"),
      connection: fakeConnection({
        getLatestBlockhash: async () => {
          throw new Error("429 Too Many Requests");
        },
      }),
      signedReceipt: resetReceipt(),
    });
    expect(result.success).toBe(false);
    expect(result.failedAt).toBe("submission");
  });
});
