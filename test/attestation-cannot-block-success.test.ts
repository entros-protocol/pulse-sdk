import { describe, it, expect, vi, afterEach } from "vitest";
import { Keypair } from "@solana/web3.js";
import { submitViaWallet } from "../src/submit/wallet";
import { ATTESTATION_SIGNATURE_TIMEOUT_MS } from "../src/config";
import { toBigEndian32 } from "../src/proof/serializer";
import { fundedPreflightConnection } from "./helpers/funding";

/**
 * A confirmed verification must survive anything that happens afterwards.
 *
 * The SAS attestation runs last, is best-effort, and needs a wallet signature.
 * An invisible prompt must not hold the submission open. Local baseline
 * storage depends on the submission result.
 */

const AUTHORITY = Keypair.generate().publicKey;

function fakeConnection() {
  return {
    ...fundedPreflightConnection(),
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

/** A wallet that signs transactions but never answers a `signMessage`. */
function walletWithHangingSignMessage() {
  return {
    publicKey: AUTHORITY,
    sendTransaction: async () => "sig-confirmed-on-chain",
    signTransaction: async (tx: unknown) => tx,
    signAllTransactions: async (txs: unknown) => txs,
    signMessage: () => new Promise<Uint8Array>(() => {}),
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("a hanging attestation signature", () => {
  it("cannot hold a confirmed verification open", async () => {
    // The executor hands out a server nonce, which is what makes the SDK
    // attempt an attestation at all.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ nonce: Array.from({ length: 32 }, (_, i) => i) }),
      })),
    );

    const commitment = new Uint8Array(32).fill(17);
    const updateMessage = Buffer.alloc(136);
    updateMessage.set(Buffer.from("entros-validator-receipt-v4\0", "ascii"));
    updateMessage[28] = 4;
    updateMessage.writeUInt16LE(1, 29);
    updateMessage.set(AUTHORITY.toBytes(), 31);
    updateMessage.set(commitment, 63);
    updateMessage.writeBigInt64LE(1_790_000_000n, 95);
    updateMessage[135] = 0;

    const promise = submitViaWallet(
      {
        proofBytes: new Uint8Array(256),
        publicInputs: [
          commitment,
          new Uint8Array(32).fill(34),
          toBigEndian32("30"),
          toBigEndian32("3"),
        ],
      },
      commitment,
      {
        wallet: walletWithHangingSignMessage(),
        connection: fakeConnection(),
        isFirstVerification: false,
        signedReceipt: {
          validator_pubkey_hex: "11".repeat(32),
          signature_hex: "22".repeat(64),
          message_hex: updateMessage.toString("hex"),
        },
        relayerUrl: "https://relayer.example/verify",
      },
    );

    // Nothing resolves this but the attestation timeout. Before it existed the
    // assertion below never ran, which is exactly what the user saw.
    const settled = await Promise.race([
      promise,
      new Promise((resolve) =>
        setTimeout(() => resolve("STILL_HANGING"), ATTESTATION_SIGNATURE_TIMEOUT_MS + 10_000),
      ),
    ]);

    expect(settled, "the submission never settled").not.toBe("STILL_HANGING");
    const result = settled as Awaited<typeof promise>;
    expect(result.success, "a confirmed transaction was reported as a failure").toBe(true);
    expect(result.txSignature).toBe("sig-confirmed-on-chain");
    // Best-effort, so its absence is the correct outcome rather than an error.
    expect(result.attestationTx).toBeUndefined();
    expect(result.failedAt).toBeUndefined();
  }, 60_000);
});
