import { describe, expect, it } from "vitest";
import {
  ComputeBudgetProgram,
  Ed25519Program,
  Keypair,
  SYSVAR_INSTRUCTIONS_PUBKEY,
  type Transaction,
} from "@solana/web3.js";
import { submitResetViaWallet } from "../src/submit/wallet";
import type { SignedReceiptDto } from "../src/submit/types";

function receipt(
  wallet: Uint8Array,
  commitment: Uint8Array,
  purpose: number,
): SignedReceiptDto {
  const message = Buffer.alloc(103);
  Buffer.from("entros-validator-receipt-v2\0", "ascii").copy(message, 0);
  message[28] = purpose;
  message.writeUInt16LE(1, 29);
  message.set(wallet, 31);
  message.set(commitment, 63);
  message.writeBigInt64LE(1_786_310_400n, 95);
  return {
    validator_pubkey_hex: "11".repeat(32),
    signature_hex: "22".repeat(64),
    message_hex: message.toString("hex"),
  };
}

function connection() {
  return {
    rpcEndpoint: "http://localhost:8899",
    commitment: "confirmed",
    getLatestBlockhash: async () => ({
      blockhash: "11111111111111111111111111111111",
      lastValidBlockHeight: 1,
    }),
    confirmTransaction: async () => ({ value: { err: null } }),
  };
}

describe("versioned reset receipts", () => {
  it("rejects a missing or wrong-purpose receipt before wallet signing", async () => {
    const publicKey = Keypair.generate().publicKey;
    const commitment = new Uint8Array(32).fill(7);
    let walletCalls = 0;
    const wallet = {
      publicKey,
      sendTransaction: async () => {
        walletCalls += 1;
        return "unexpected";
      },
    };

    const missing = await submitResetViaWallet(commitment, {
      wallet,
      connection: connection(),
      projectionVersion: 1,
    });
    const wrongPurpose = await submitResetViaWallet(commitment, {
      wallet,
      connection: connection(),
      projectionVersion: 1,
      signedReceipt: receipt(publicKey.toBytes(), commitment, 1),
    });

    expect(missing.success).toBe(false);
    expect(wrongPurpose.success).toBe(false);
    expect(walletCalls).toBe(0);
  });

  it("places the reset receipt before the reset instruction", async () => {
    const publicKey = Keypair.generate().publicKey;
    const commitment = new Uint8Array(32).fill(7);
    let submitted: Transaction | undefined;
    const wallet = {
      publicKey,
      sendTransaction: async (transaction: Transaction) => {
        submitted = transaction;
        return "reset-signature";
      },
      signTransaction: async (transaction: Transaction) => transaction,
      signAllTransactions: async (transactions: Transaction[]) => transactions,
    };

    const result = await submitResetViaWallet(commitment, {
      wallet,
      connection: connection(),
      projectionVersion: 1,
      signedReceipt: receipt(publicKey.toBytes(), commitment, 3),
    });

    expect(result.success, result.error).toBe(true);
    expect(submitted?.instructions).toHaveLength(3);
    expect(submitted!.instructions[0]!.programId.equals(ComputeBudgetProgram.programId)).toBe(
      true,
    );
    expect(submitted!.instructions[1]!.programId.equals(Ed25519Program.programId)).toBe(true);
    expect(
      submitted!.instructions[2]!.keys.some(({ pubkey }) =>
        pubkey.equals(SYSVAR_INSTRUCTIONS_PUBKEY),
      ),
    ).toBe(true);
  });

  it("keeps the version 0 reset layout free of remaining accounts", async () => {
    const publicKey = Keypair.generate().publicKey;
    let submitted: Transaction | undefined;
    const wallet = {
      publicKey,
      sendTransaction: async (transaction: Transaction) => {
        submitted = transaction;
        return "reset-signature";
      },
      signTransaction: async (transaction: Transaction) => transaction,
      signAllTransactions: async (transactions: Transaction[]) => transactions,
    };

    const result = await submitResetViaWallet(new Uint8Array(32), {
      wallet,
      connection: connection(),
      projectionVersion: 0,
    });

    expect(result.success, result.error).toBe(true);
    expect(submitted?.instructions).toHaveLength(2);
    expect(
      submitted!.instructions[1]!.keys.some(({ pubkey }) =>
        pubkey.equals(SYSVAR_INSTRUCTIONS_PUBKEY),
      ),
    ).toBe(false);
  });
});
