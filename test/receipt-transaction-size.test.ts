import { describe, expect, it } from "vitest";
import { Keypair, Transaction } from "@solana/web3.js";
import { ENCRYPTED_BASELINE_BLOB_BYTES } from "../src/identity/baseline";
import { submitViaWallet } from "../src/submit/wallet";
import type { SignedReceiptDto } from "../src/submit/types";

/** The largest serialized transaction the network accepts, in bytes. */
const PACKET_DATA_SIZE = 1232;

const AUTHORITY = Keypair.fromSeed(new Uint8Array(32).fill(23)).publicKey;

function v3Receipt(): SignedReceiptDto {
  const message = Buffer.alloc(136);
  Buffer.from("entros-validator-receipt-v3\0", "ascii").copy(message, 0);
  message[28] = 1;
  message.writeUInt16LE(1, 29);
  AUTHORITY.toBuffer().copy(message, 31);
  Buffer.alloc(32, 7).copy(message, 63);
  message.writeBigInt64LE(1_790_000_000n, 95);
  Buffer.alloc(32, 9).copy(message, 103);
  message[135] = 2;
  return {
    validator_pubkey_hex: "8c".repeat(32),
    signature_hex: "ab".repeat(64),
    message_hex: message.toString("hex"),
  };
}

function connectionFixture() {
  return {
    rpcEndpoint: "http://localhost:8899",
    commitment: "confirmed",
    getLatestBlockhash: async () => ({
      blockhash: "11111111111111111111111111111111",
      lastValidBlockHeight: 1,
    }),
    confirmTransaction: async () => ({ value: { err: null } }),
    getAccountInfo: async () => null,
  };
}

describe("receipt transaction size", () => {
  it("fits the largest first mint, a v3 receipt with an encrypted baseline", async () => {
    let captured: Transaction | undefined;
    const result = await submitViaWallet(
      { proofBytes: new Uint8Array(256), publicInputs: [] },
      new Uint8Array(32).fill(7),
      {
        wallet: {
          publicKey: AUTHORITY,
          sendTransaction: async (transaction: Transaction) => {
            captured = transaction;
            return "mint-signature";
          },
          signTransaction: async (transaction: Transaction) => transaction,
          signAllTransactions: async (transactions: Transaction[]) => transactions,
        },
        connection: connectionFixture(),
        isFirstVerification: true,
        signedReceipt: v3Receipt(),
        encryptedBaselineBlob: new Uint8Array(ENCRYPTED_BASELINE_BLOB_BYTES),
      },
    );
    expect(result.success).toBe(true);
    const size = captured!.serialize({
      requireAllSignatures: false,
      verifySignatures: false,
    }).length;
    // The Ed25519 instruction carries 16 header bytes, the key, the signature and the message.
    expect(captured!.instructions[1]!.data.length).toBe(16 + 32 + 64 + 136);
    expect(size).toBeLessThanOrEqual(PACKET_DATA_SIZE);
    console.info(`first mint with a v3 receipt: ${size} bytes, ${PACKET_DATA_SIZE - size} to spare`);
  });
});
