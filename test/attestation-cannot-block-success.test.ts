import { describe, it, expect, vi, afterEach } from "vitest";
import { Keypair } from "@solana/web3.js";
import { submitViaWallet } from "../src/submit/wallet";
import { ATTESTATION_SIGNATURE_TIMEOUT_MS } from "../src/config";

/**
 * A confirmed verification must survive anything that happens afterwards.
 *
 * The SAS attestation runs last, is best-effort, and needs a wallet signature.
 * That signature was unbounded, so a wallet that never surfaced the prompt held
 * the whole submission open. On 2026-07-31 a mobile verification confirmed on
 * chain at 22:38:36, the user dismissed the wallet after seeing "Sent!", the
 * fourth prompt never appeared, and the page sat on "Submitting to Solana..."
 * with no end. The executor logs show `/attest` was never called, while the two
 * desktop runs minutes earlier reached it fine.
 *
 * The cost was not only the spinner. `storeVerificationData` runs on
 * `submission.success`, so the device's local baseline was never written and
 * fell behind the chain it had just advanced.
 */

const AUTHORITY = Keypair.generate().publicKey;

function fakeConnection() {
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

    const promise = submitViaWallet(
      { proofBytes: new Uint8Array(256), publicInputs: [] },
      new Uint8Array(32),
      {
        wallet: walletWithHangingSignMessage(),
        connection: fakeConnection(),
        isFirstVerification: false,
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
