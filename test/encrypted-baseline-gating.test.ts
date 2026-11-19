import { ed25519 } from "@noble/curves/ed25519";
import { Keypair, PublicKey } from "@solana/web3.js";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { bigintToBytes32, computeCommitment } from "../src/hashing/poseidon";
import {
  clearBaselineKeyCache,
  deriveEncryptedBaselinePda,
  ENCRYPTED_BASELINE_BLOB_BYTES,
  encryptBaselineBlob,
  fingerprintToBytes,
  getOrDeriveBaselineKey,
  hasCachedBaselineKey,
  hasValidEncryptedBaselineEnvelope,
  shouldWriteEncryptedBaseline,
} from "../src/identity/baseline";

const SALT = 9876543210987654321098765432109876543210n;

function makeFingerprint(seed: number): number[] {
  return Array.from({ length: 256 }, (_, index) => ((index * 5 + seed) % 3 === 0 ? 1 : 0));
}

function makeWallet() {
  const keypair = Keypair.generate();
  const signMessage = vi.fn(async (message: Uint8Array) =>
    ed25519.sign(message, keypair.secretKey.slice(0, 32)),
  );
  return {
    keypair,
    wallet: {
      publicKey: keypair.publicKey,
      signMessage,
    },
    signMessage,
  };
}

function encodeAccountWithBlob(blob: Uint8Array): Uint8Array {
  const raw = new Uint8Array(8 + blob.length + 1);
  raw.set(blob, 8);
  return raw;
}

beforeEach(() => {
  clearBaselineKeyCache();
});

describe("encrypted baseline write gating", () => {
  it("always writes when commitmentAdvancing is true without extra RPC reads", async () => {
    const { keypair } = makeWallet();
    const commitment = new Uint8Array(32).fill(11);
    const getAccountInfo = vi.fn(async () => null);

    const decision = await shouldWriteEncryptedBaseline({
      walletPubkey: keypair.publicKey,
      connection: { getAccountInfo },
      targetCommitment: commitment,
      onChainCommitment: commitment,
      commitmentAdvancing: true,
    });

    expect(decision).toEqual({
      shouldWrite: true,
      reason: "commitment-advancing",
    });
    expect(getAccountInfo).not.toHaveBeenCalled();
  });

  it("always writes on a commitment-advancing returning update so second-device recovery stays intact", async () => {
    const { keypair } = makeWallet();
    const prevCommitment = new Uint8Array(32).fill(11);
    const nextCommitment = new Uint8Array(32).fill(22);
    const getAccountInfo = vi.fn(async () => null);

    const decision = await shouldWriteEncryptedBaseline({
      walletPubkey: keypair.publicKey,
      connection: { getAccountInfo },
      targetCommitment: nextCommitment,
      onChainCommitment: prevCommitment,
    });

    expect(decision).toEqual({
      shouldWrite: true,
      reason: "commitment-advancing",
    });
    expect(getAccountInfo).not.toHaveBeenCalled();
  });

  it("skips cold-session signMessage and write when commitment is unchanged and on-chain blob is valid", async () => {
    const { keypair, wallet, signMessage } = makeWallet();
    const fp = makeFingerprint(1);
    const commitment = bigintToBytes32(await computeCommitment(fp, SALT));
    const [baselinePda] = await deriveEncryptedBaselinePda(keypair.publicKey);
    const blob = await encryptBaselineBlob(
      fingerprintToBytes(fp),
      bigintToBytes32(SALT),
      await getOrDeriveBaselineKey(wallet),
      keypair.publicKey,
      baselinePda,
      commitment,
    );

    // Simulate a cold page reload: drop in-memory key cache and verify zero
    // IndexedDB persistence holds the derived key.
    clearBaselineKeyCache();
    expect(hasCachedBaselineKey(keypair.publicKey)).toBe(false);
    signMessage.mockClear();

    const getAccountInfo = vi.fn(async (address: PublicKey) =>
      address.equals(baselinePda) ? { data: encodeAccountWithBlob(blob) } : null,
    );

    const decision = await shouldWriteEncryptedBaseline({
      walletPubkey: keypair.publicKey,
      connection: { getAccountInfo },
      targetCommitment: commitment,
      onChainCommitment: commitment,
    });

    expect(decision).toEqual({
      shouldWrite: false,
      reason: "already-current",
    });
    expect(getAccountInfo).toHaveBeenCalledTimes(1);
    expect(signMessage).not.toHaveBeenCalled();
  });

  it("writes when the on-chain EncryptedBaseline PDA is missing", async () => {
    const { keypair } = makeWallet();
    const commitment = new Uint8Array(32).fill(33);
    const getAccountInfo = vi.fn(async () => null);

    const decision = await shouldWriteEncryptedBaseline({
      walletPubkey: keypair.publicKey,
      connection: { getAccountInfo },
      targetCommitment: commitment,
      onChainCommitment: commitment,
    });

    expect(decision).toEqual({
      shouldWrite: true,
      reason: "missing-on-chain-blob",
    });
    expect(getAccountInfo).toHaveBeenCalledTimes(1);
  });

  it("writes when the on-chain blob is all zeros or has an invalid header", async () => {
    const { keypair } = makeWallet();
    const commitment = new Uint8Array(32).fill(44);
    const [baselinePda] = await deriveEncryptedBaselinePda(keypair.publicKey);

    const emptyBlob = new Uint8Array(ENCRYPTED_BASELINE_BLOB_BYTES);
    emptyBlob[0] = 0x01;
    emptyBlob[1] = 0x01;
    expect(hasValidEncryptedBaselineEnvelope(emptyBlob)).toBe(false);

    const emptyDecision = await shouldWriteEncryptedBaseline({
      walletPubkey: keypair.publicKey,
      connection: {
        getAccountInfo: async (address: PublicKey) =>
          address.equals(baselinePda) ? { data: encodeAccountWithBlob(emptyBlob) } : null,
      },
      targetCommitment: commitment,
      onChainCommitment: commitment,
    });
    expect(emptyDecision).toEqual({
      shouldWrite: true,
      reason: "malformed-on-chain-blob",
    });

    const wrongVersionBlob = new Uint8Array(ENCRYPTED_BASELINE_BLOB_BYTES).fill(7);
    wrongVersionBlob[0] = 0x99;
    expect(hasValidEncryptedBaselineEnvelope(wrongVersionBlob)).toBe(false);

    const wrongVersionDecision = await shouldWriteEncryptedBaseline({
      walletPubkey: keypair.publicKey,
      connection: {
        getAccountInfo: async (address: PublicKey) =>
          address.equals(baselinePda) ? { data: encodeAccountWithBlob(wrongVersionBlob) } : null,
      },
      targetCommitment: commitment,
      onChainCommitment: commitment,
    });
    expect(wrongVersionDecision).toEqual({
      shouldWrite: true,
      reason: "malformed-on-chain-blob",
    });
  });

  it("fails safe toward writing when the RPC account read throws", async () => {
    const { keypair } = makeWallet();
    const commitment = new Uint8Array(32).fill(55);

    const decision = await shouldWriteEncryptedBaseline({
      walletPubkey: keypair.publicKey,
      connection: {
        getAccountInfo: async () => {
          throw new Error("RPC timeout");
        },
      },
      targetCommitment: commitment,
      onChainCommitment: commitment,
    });

    expect(decision).toEqual({
      shouldWrite: true,
      reason: "rpc-check-failed",
    });
  });

  it("uses a warm in-memory cached key to detect a stale on-chain blob with zero extra prompts", async () => {
    const { keypair, wallet, signMessage } = makeWallet();
    const fp = makeFingerprint(2);
    const currentCommitment = bigintToBytes32(await computeCommitment(fp, SALT));
    const olderCommitment = bigintToBytes32(await computeCommitment(makeFingerprint(9), SALT));
    const [baselinePda] = await deriveEncryptedBaselinePda(keypair.publicKey);

    const key = await getOrDeriveBaselineKey(wallet);
    expect(hasCachedBaselineKey(keypair.publicKey)).toBe(true);
    signMessage.mockClear();

    // Seal blob under an older commitment so AAD authentication fails.
    const staleBlob = await encryptBaselineBlob(
      fingerprintToBytes(fp),
      bigintToBytes32(SALT),
      key,
      keypair.publicKey,
      baselinePda,
      olderCommitment,
    );

    const staleDecision = await shouldWriteEncryptedBaseline({
      walletPubkey: keypair.publicKey,
      connection: {
        getAccountInfo: async (address: PublicKey) =>
          address.equals(baselinePda) ? { data: encodeAccountWithBlob(staleBlob) } : null,
      },
      targetCommitment: currentCommitment,
      onChainCommitment: currentCommitment,
    });

    expect(staleDecision).toEqual({
      shouldWrite: true,
      reason: "stale-on-chain-blob",
    });
    expect(signMessage).not.toHaveBeenCalled();

    // Now seal validly under currentCommitment and verify warm check skips write.
    const freshBlob = await encryptBaselineBlob(
      fingerprintToBytes(fp),
      bigintToBytes32(SALT),
      key,
      keypair.publicKey,
      baselinePda,
      currentCommitment,
    );
    const freshDecision = await shouldWriteEncryptedBaseline({
      walletPubkey: keypair.publicKey,
      connection: {
        getAccountInfo: async (address: PublicKey) =>
          address.equals(baselinePda) ? { data: encodeAccountWithBlob(freshBlob) } : null,
      },
      targetCommitment: currentCommitment,
      onChainCommitment: currentCommitment,
    });

    expect(freshDecision).toEqual({
      shouldWrite: false,
      reason: "already-current",
    });
    expect(signMessage).not.toHaveBeenCalled();
  });
});
