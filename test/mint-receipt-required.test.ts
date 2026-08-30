import { Keypair, PublicKey, Transaction } from "@solana/web3.js";
import { describe, expect, it } from "vitest";

import { PROGRAM_IDS } from "../src/config";
import {
  ASSOCIATED_TOKEN_PROGRAM_ADDRESS,
  deriveToken2022AssociatedAddress,
  TOKEN_2022_PROGRAM_ADDRESS,
} from "../src/submit/associated-token";
import { submitViaWallet } from "../src/submit/wallet";
import type { SignedReceiptDto } from "../src/submit/types";

const AUTHORITY = Keypair.fromSeed(new Uint8Array(32).fill(23)).publicKey;

const receipt = (commitment: Uint8Array): SignedReceiptDto => {
  const message = Buffer.alloc(103);
  Buffer.from("entros-validator-receipt-v2\0", "ascii").copy(message, 0);
  message[28] = 1;
  message.writeUInt16LE(1, 29);
  Buffer.from(AUTHORITY.toBytes()).copy(message, 31);
  Buffer.from(commitment).copy(message, 63);
  message.writeBigInt64LE(1_788_044_400n, 95);
  return {
    validator_pubkey_hex: "8c".repeat(32),
    signature_hex: "ab".repeat(64),
    message_hex: message.toString("hex"),
  };
};

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

  it("wires the Token-2022 associated account into the first mint", async () => {
    const commitment = new Uint8Array(32).fill(7);
    let captured: Transaction | undefined;
    const result = await submitViaWallet(
      { proofBytes: new Uint8Array(), publicInputs: [] },
      commitment,
      {
        isFirstVerification: true,
        signedReceipt: receipt(commitment),
        wallet: {
          publicKey: AUTHORITY,
          sendTransaction: async (transaction: Transaction) => {
            captured = transaction;
            return "mint-signature";
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
      },
    );

    expect(result).toMatchObject({ success: true, txSignature: "mint-signature" });
    expect(captured?.instructions).toHaveLength(3);

    const mintInstruction = captured!.instructions[2]!;
    const anchorProgramId = new PublicKey(PROGRAM_IDS.entrosAnchor);
    const [mint] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("mint"), AUTHORITY.toBuffer()],
      anchorProgramId,
    );
    const expectedAssociatedAccount = deriveToken2022AssociatedAddress(
      mint,
      AUTHORITY,
      PublicKey,
    );

    expect(mintInstruction.programId.equals(anchorProgramId)).toBe(true);
    expect(mintInstruction.keys[4]!.pubkey.equals(expectedAssociatedAccount)).toBe(true);
    expect(mintInstruction.keys[5]!.pubkey.toBase58()).toBe(ASSOCIATED_TOKEN_PROGRAM_ADDRESS);
    expect(mintInstruction.keys[6]!.pubkey.toBase58()).toBe(TOKEN_2022_PROGRAM_ADDRESS);
  });
});
