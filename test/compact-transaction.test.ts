import { afterEach, describe, expect, it, vi } from "vitest";
import { Keypair, Transaction } from "@solana/web3.js";
import type { SolanaProof } from "../src/proof/types";
import { toBigEndian32 } from "../src/proof/serializer";
import { submitViaWallet } from "../src/submit/wallet";

const AUTHORITY = Keypair.fromSeed(new Uint8Array(32).fill(19)).publicKey;

afterEach(() => {
  vi.restoreAllMocks();
});

function proofFixture(): SolanaProof {
  return {
    proofBytes: new Uint8Array(256),
    publicInputs: [
      new Uint8Array(32).fill(17),
      new Uint8Array(32).fill(34),
      toBigEndian32("30"),
      toBigEndian32("3"),
    ],
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

describe("compact re-verification transaction", () => {
  it("builds the pinned transaction shape", async () => {
    let captured: Transaction | undefined;
    const proof = proofFixture();
    const result = await submitViaWallet(
      proof,
      proof.publicInputs[0]!,
      {
        wallet: {
          publicKey: AUTHORITY,
          sendTransaction: async (transaction: Transaction) => {
            captured = transaction;
            return "compact-signature";
          },
          signTransaction: async (transaction: Transaction) => transaction,
          signAllTransactions: async (transactions: Transaction[]) => transactions,
        },
        connection: connectionFixture(),
        isFirstVerification: false,
        encryptedBaselineBlob: new Uint8Array(96),
      },
    );

    expect(result).toMatchObject({
      success: true,
      txSignature: "compact-signature",
    });
    expect(captured).toBeDefined();
    const transaction = captured!;
    expect(
      transaction.instructions.map((instruction) => instruction.data.length),
    ).toEqual([5, 40, 364, 40, 104]);
    expect(
      transaction.serialize({
        requireAllSignatures: false,
        verifySignatures: false,
      }).length,
    ).toBe(1040);
  });

  it("rejects a commitment mismatch before asking the wallet", async () => {
    let walletCalled = false;
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const result = await submitViaWallet(proofFixture(), new Uint8Array(32), {
      wallet: {
        publicKey: AUTHORITY,
        sendTransaction: async () => {
          walletCalled = true;
          return "unexpected";
        },
      },
      connection: connectionFixture(),
      isFirstVerification: false,
      relayerUrl: "https://executor.invalid",
    });

    expect(result).toMatchObject({
      success: false,
      failedAt: "submission",
    });
    expect(result.error).toContain("proof commitment does not match");
    expect(walletCalled).toBe(false);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  const invalidProofCases: {
    name: string;
    mutate: (proof: SolanaProof) => void;
    expected: string;
  }[] = [
    {
      name: "short proof",
      mutate: (proof) => {
        proof.proofBytes = new Uint8Array(255);
      },
      expected: "proof_bytes must contain 256 bytes",
    },
    {
      name: "long proof",
      mutate: (proof) => {
        proof.proofBytes = new Uint8Array(257);
      },
      expected: "proof_bytes must contain 256 bytes",
    },
    {
      name: "wrong public-input count",
      mutate: (proof) => {
        proof.publicInputs = proof.publicInputs.slice(0, 3);
      },
      expected: "proof must contain 4 public inputs",
    },
    {
      name: "non-byte proof value",
      mutate: (proof) => {
        const bytes = new Array<number>(256).fill(0);
        bytes[0] = 256;
        proof.proofBytes = bytes as unknown as Uint8Array;
      },
      expected: "proof_bytes must contain only bytes",
    },
    {
      name: "zero new commitment",
      mutate: (proof) => {
        proof.publicInputs[0] = new Uint8Array(32);
      },
      expected: "commitment_new must not be zero",
    },
    {
      name: "zero previous commitment",
      mutate: (proof) => {
        proof.publicInputs[1] = new Uint8Array(32);
      },
      expected: "commitment_prev must not be zero",
    },
    {
      name: "oversized threshold field",
      mutate: (proof) => {
        proof.publicInputs[2]![0] = 1;
      },
      expected: "threshold does not fit in u16",
    },
    {
      name: "threshold above the deployed limit",
      mutate: (proof) => {
        proof.publicInputs[2] = toBigEndian32("97");
      },
      expected: "threshold must be at most 96",
    },
    {
      name: "minimum distance below the deployed floor",
      mutate: (proof) => {
        proof.publicInputs[3] = toBigEndian32("2");
      },
      expected: "min_distance must be at least 3",
    },
    {
      name: "empty acceptance interval",
      mutate: (proof) => {
        proof.publicInputs[2] = toBigEndian32("3");
        proof.publicInputs[3] = toBigEndian32("3");
      },
      expected: "min_distance must be less than threshold",
    },
  ];

  it.each(invalidProofCases)(
    "rejects $name before external work",
    async ({ mutate, expected }) => {
      const proof = proofFixture();
      mutate(proof);
      let blockhashRequests = 0;
      let walletCalls = 0;
      const fetchMock = vi.fn();
      vi.stubGlobal("fetch", fetchMock);
      const connection = {
        ...connectionFixture(),
        getLatestBlockhash: async () => {
          blockhashRequests += 1;
          return {
            blockhash: "11111111111111111111111111111111",
            lastValidBlockHeight: 1,
          };
        },
      };

      const result = await submitViaWallet(proof, proof.publicInputs[0]!, {
        wallet: {
          publicKey: AUTHORITY,
          sendTransaction: async () => {
            walletCalls += 1;
            return "unexpected";
          },
          signTransaction: async (transaction: Transaction) => transaction,
          signAllTransactions: async (transactions: Transaction[]) => transactions,
        },
        connection,
        isFirstVerification: false,
        relayerUrl: "https://executor.invalid",
      });

      expect(result).toMatchObject({ success: false, failedAt: "submission" });
      expect(result.error).toContain(expected);
      expect(fetchMock).not.toHaveBeenCalled();
      expect(blockhashRequests).toBe(0);
      expect(walletCalls).toBe(0);
    },
  );

  it.each([
    { name: "zero", nonce: new Array<number>(32).fill(0) },
    {
      name: "non-byte",
      nonce: [...new Array<number>(31).fill(1), -1],
    },
  ])("replaces a $name server nonce with one canonical nonce", async ({ nonce }) => {
    let captured: Transaction | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ nonce }),
      })),
    );
    const proof = proofFixture();

    const result = await submitViaWallet(proof, proof.publicInputs[0]!, {
      wallet: {
        publicKey: AUTHORITY,
        sendTransaction: async (transaction: Transaction) => {
          captured = transaction;
          return "compact-signature";
        },
        signTransaction: async (transaction: Transaction) => transaction,
        signAllTransactions: async (transactions: Transaction[]) => transactions,
      },
      connection: connectionFixture(),
      isFirstVerification: false,
      relayerUrl: "https://executor.invalid",
    });

    expect(result.success).toBe(true);
    expect(captured).toBeDefined();
    const challengeNonce = captured!.instructions[1]!.data.subarray(8, 40);
    const proofNonce = captured!.instructions[2]!.data.subarray(8, 40);
    expect(challengeNonce).toEqual(proofNonce);
    expect(challengeNonce.some((byte) => byte !== 0)).toBe(true);
  });
});
