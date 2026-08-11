import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  ComputeBudgetProgram,
  Ed25519Program,
  Keypair,
  PublicKey,
  type Transaction,
} from "@solana/web3.js";
import { PROGRAM_IDS } from "../src/config";
import { submitRebaselineViaWallet } from "../src/submit/wallet";
import type { SignedReceiptDto } from "../src/submit/types";

function discriminator(name: string): Buffer {
  return createHash("sha256").update(`global:${name}`).digest().subarray(0, 8);
}

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

describe("projection rebaseline transaction", () => {
  it("binds the receipt, version assertion, and encrypted baseline atomically", async () => {
    const authority = Keypair.generate().publicKey;
    const commitment = new Uint8Array(32).fill(7);
    let submitted: Transaction | undefined;
    const wallet = {
      publicKey: authority,
      sendTransaction: async (transaction: Transaction) => {
        submitted = transaction;
        return "rebaseline-signature";
      },
      signTransaction: async (transaction: Transaction) => transaction,
      signAllTransactions: async (transactions: Transaction[]) => transactions,
    };
    const connection = {
      rpcEndpoint: "http://localhost:8899",
      commitment: "confirmed",
      getLatestBlockhash: async () => ({
        blockhash: "11111111111111111111111111111111",
        lastValidBlockHeight: 1,
      }),
      confirmTransaction: async () => ({ value: { err: null } }),
    };

    const result = await submitRebaselineViaWallet(commitment, 1, {
      wallet,
      connection,
      signedReceipt: receipt(authority.toBytes(), commitment, 2),
      encryptedBaselineBlob: new Uint8Array(96).fill(4),
    });

    expect(result.success, result.error).toBe(true);
    expect(result.txSignature).toBe("rebaseline-signature");
    expect(submitted).toBeDefined();
    const instructions = submitted!.instructions;
    expect(instructions).toHaveLength(4);
    expect(instructions[0]!.programId.equals(ComputeBudgetProgram.programId)).toBe(true);
    expect(instructions[1]!.programId.equals(Ed25519Program.programId)).toBe(true);

    const anchorProgramId = new PublicKey(PROGRAM_IDS.entrosAnchor);
    expect(instructions[2]!.programId.equals(anchorProgramId)).toBe(true);
    expect(instructions[2]!.data.subarray(0, 8)).toEqual(discriminator("rebaseline_anchor"));
    expect(instructions[2]!.data.readUInt16LE(40)).toBe(1);
    expect(instructions[3]!.programId.equals(anchorProgramId)).toBe(true);
    expect(instructions[3]!.data.subarray(0, 8)).toEqual(
      discriminator("set_encrypted_baseline"),
    );
  });

  it("rejects a wrong-purpose receipt before wallet signing", async () => {
    const authority = Keypair.generate().publicKey;
    const commitment = new Uint8Array(32).fill(7);
    let walletCalls = 0;
    const result = await submitRebaselineViaWallet(commitment, 1, {
      wallet: {
        publicKey: authority,
        sendTransaction: async () => {
          walletCalls += 1;
          return "unexpected";
        },
        signTransaction: async (transaction: Transaction) => transaction,
        signAllTransactions: async (transactions: Transaction[]) => transactions,
      },
      connection: {
        getLatestBlockhash: async () => ({
          blockhash: "11111111111111111111111111111111",
          lastValidBlockHeight: 1,
        }),
        confirmTransaction: async () => ({ value: { err: null } }),
      },
      signedReceipt: receipt(authority.toBytes(), commitment, 3),
      encryptedBaselineBlob: new Uint8Array(96).fill(4),
    });

    expect(result.success).toBe(false);
    expect(walletCalls).toBe(0);
  });
});
