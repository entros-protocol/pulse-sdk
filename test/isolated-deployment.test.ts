import { afterEach, describe, expect, it, vi } from "vitest";
import { ed25519 } from "@noble/curves/ed25519";
import { PublicKey, type Transaction } from "@solana/web3.js";
import { BorshAccountsCoder, type Idl } from "@coral-xyz/anchor";
import { PROGRAM_IDS } from "../src/config";
import type { RequestBoundManifest } from "../src/proof/request";
import { resolveDeployment, deploymentIdl } from "../src/protocol/deployment";
import { entrosAnchorIdl, entrosVerifierIdl } from "../src/protocol/idl";
import {
  deriveEncryptedBaselinePda,
  encryptBaselineBlob,
  decryptBaselineBlob,
  getOrDeriveBaselineKey,
} from "../src/identity/baseline";
import {
  recoverBaselineFromChain,
  fetchIdentityState,
  loadVerificationData,
  storeVerificationData,
  setPrivacyFallback,
} from "../src/identity/anchor";
import { readIntegratorEvidence } from "../src/identity/integrator";
import { upgradeIdentityLayoutViaWallet } from "../src/submit/wallet";
import { prepareWalletProofRequest } from "../src/submit/request";
import { createEvidenceFixture } from "./fixtures/integrator-evidence";

const wallet = new PublicKey(new Uint8Array(32).fill(61));
const manifest: RequestBoundManifest = {
  generation: "request-bound-v1",
  deploymentDomain: "11".repeat(32),
  genesisHash: "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG",
  consumerProgram: new PublicKey(new Uint8Array(32).fill(62)).toBase58(),
  verifierProgram: new PublicKey(new Uint8Array(32).fill(63)).toBase58(),
  wasm: { url: "synthetic.wasm", sha256: "12".repeat(32) },
  zkey: { url: "synthetic.zkey", sha256: "13".repeat(32) },
};
const data = (commitment: string) => ({
  fingerprint: Array<number>(256).fill(0),
  salt: "3",
  commitment,
  timestamp: 1,
  projectionVersion: 1,
});
afterEach(() => {
  vi.unstubAllGlobals();
  setPrivacyFallback(undefined);
});

describe("trusted isolated deployment", () => {
  it("preserves official defaults and rejects mixed pairs or other networks", () => {
    expect(resolveDeployment().consumerProgram).toBe(PROGRAM_IDS.entrosAnchor);
    for (const change of [
      { consumerProgram: PROGRAM_IDS.entrosAnchor },
      { verifierProgram: PROGRAM_IDS.entrosVerifier },
      { verifierProgram: manifest.consumerProgram },
      { genesisHash: "mainnet" },
    ])
      expect(() => resolveDeployment({ ...manifest, ...change })).toThrow();
  });
  it("derives independent baseline PDAs and captures configuration before awaiting", async () => {
    const official = await deriveEncryptedBaselinePda(wallet);
    const mutable = { ...manifest };
    const pending = deriveEncryptedBaselinePda(wallet, mutable);
    mutable.consumerProgram = PROGRAM_IDS.entrosAnchor;
    const isolated = await pending;
    expect(isolated[0].equals(official[0])).toBe(false);
    expect(
      isolated[0].equals(
        (await deriveEncryptedBaselinePda(wallet, manifest))[0],
      ),
    ).toBe(true);
  });
  it.each([false, true])(
    "preserves independent baselines with localStorage=%s",
    async (persistent) => {
      const values = new Map<string, string>();
      vi.stubGlobal(
        "localStorage",
        persistent
          ? {
              getItem: (key: string) => values.get(key) ?? null,
              setItem: (key: string, value: string) => values.set(key, value),
              removeItem: (key: string) => values.delete(key),
            }
          : undefined,
      );
      if (persistent) {
        vi.stubGlobal("crypto", undefined);
        setPrivacyFallback(async () => true);
      }
      await storeVerificationData(data("1"), wallet.toBase58());
      await storeVerificationData(data("2"), wallet.toBase58(), manifest);
      expect((await loadVerificationData(wallet.toBase58()))?.commitment).toBe(
        "1",
      );
      expect(
        (await loadVerificationData(wallet.toBase58(), manifest))?.commitment,
      ).toBe("2");
      expect(
        await loadVerificationData(wallet.toBase58(), {
          ...manifest,
          deploymentDomain: "22".repeat(32),
        }),
      ).toBeNull();
      expect(
        await loadVerificationData(
          new PublicKey(new Uint8Array(32).fill(64)).toBase58(),
          manifest,
        ),
      ).toBeNull();
      await expect(loadVerificationData(undefined, manifest)).rejects.toThrow(
        "wallet address",
      );
      if (persistent)
        expect(
          values.has(`entros-protocol-verification-data_${wallet.toBase58()}`),
        ).toBe(true);
    },
  );
  it("binds encrypted baseline authentication to the isolated PDA", async () => {
    const key = await crypto.subtle.generateKey(
      { name: "AES-GCM", length: 256 },
      false,
      ["encrypt", "decrypt"],
    );
    const [official] = await deriveEncryptedBaselinePda(wallet);
    const [isolated] = await deriveEncryptedBaselinePda(wallet, manifest);
    const commitment = new Uint8Array(32).fill(3);
    const blob = await encryptBaselineBlob(
      new Uint8Array(32),
      new Uint8Array(32).fill(1),
      key,
      wallet,
      isolated,
      commitment,
    );
    expect(
      (await decryptBaselineBlob(blob, key, wallet, isolated, commitment))
        .simhash,
    ).toEqual(new Uint8Array(32));
    await expect(
      decryptBaselineBlob(blob, key, wallet, official, commitment),
    ).rejects.toThrow();
  });
  it("keeps shared IDLs immutable across concurrent program selections", async () => {
    const before = JSON.stringify([entrosAnchorIdl, entrosVerifierIdl]);
    const values = await Promise.all(
      Array.from({ length: 64 }, async (_, index) =>
        deploymentIdl(
          (index % 2 ? entrosAnchorIdl : entrosVerifierIdl) as Idl,
          manifest,
        ),
      ),
    );
    for (const idl of values) {
      expect([manifest.consumerProgram, manifest.verifierProgram]).toContain(
        idl.address,
      );
      const text = JSON.stringify(idl);
      expect(text).not.toContain(
        JSON.stringify(
          Array.from(new PublicKey(PROGRAM_IDS.entrosAnchor).toBytes()),
        ),
      );
      expect(text).not.toContain(
        JSON.stringify(
          Array.from(new PublicKey(PROGRAM_IDS.entrosVerifier).toBytes()),
        ),
      );
    }
    expect(JSON.stringify([entrosAnchorIdl, entrosVerifierIdl])).toBe(before);
    expect(await deploymentIdl(entrosAnchorIdl as Idl)).toBe(entrosAnchorIdl);
  });
  it("reads and prepares only the selected identity and rejects a wrong chain", async () => {
    const f = await createEvidenceFixture();
    const program = new PublicKey(manifest.consumerProgram);
    const [identity, bump] = PublicKey.findProgramAddressSync(
      [Buffer.from("identity"), f.wallet.toBytes()],
      program,
    );
    const [mint] = PublicKey.findProgramAddressSync(
      [Buffer.from("mint"), f.wallet.toBytes()],
      program,
    );
    const coder = new BorshAccountsCoder(entrosAnchorIdl as Idl);
    const decoded = coder.decode<Record<string, unknown>>(
      "IdentityState",
      f.identityAccount.data,
    );
    const encoded = await coder.encode("IdentityState", {
      ...decoded,
      mint,
      bump,
    });
    const account = { ...f.identityAccount, owner: program, data: encoded };
    const clockData = Buffer.alloc(40);
    clockData.writeBigInt64LE(1800000000n, 32);
    const read = vi.fn(async () => account);
    const reads = vi.fn(async () => ({
      context: { slot: 100 },
      value: [
        account,
        null,
        {
          ...account,
          owner: new PublicKey("Sysvar1111111111111111111111111111111111111"),
          data: clockData,
        },
      ],
    }));
    const connection = {
      getGenesisHash: async () => manifest.genesisHash,
      getAccountInfo: read,
      getMultipleAccountsInfoAndContext: reads,
    };
    expect(
      (await fetchIdentityState(f.wallet.toBase58(), connection, manifest))
        ?.mint,
    ).toBe(mint.toBase58());
    expect(read.mock.calls[0]).toEqual([identity]);
    const request = await prepareWalletProofRequest(
      connection,
      f.wallet,
      manifest,
      {
        commitmentNew: new Uint8Array(32).fill(1),
        commitmentPrevious: new Uint8Array(32).fill(3),
        threshold: 30,
        minDistance: 3,
        nonce: new Uint8Array(32).fill(9),
      },
    );
    expect(request.action.identity).toBe(
      Buffer.from(identity.toBytes()).toString("hex"),
    );
    expect(request.action.projectionVersion).toBe(1);
    await expect(
      fetchIdentityState(
        f.wallet.toBase58(),
        { ...connection, getGenesisHash: async () => "wrong" },
        manifest,
      ),
    ).rejects.toThrow("Connected chain");
    expect(read).toHaveBeenCalledTimes(1);
  });
  it("recovers an isolated encrypted baseline without changing the official baseline", async () => {
    vi.stubGlobal("localStorage", undefined);
    const seed = new Uint8Array(32).fill(65);
    const signer = {
      publicKey: new PublicKey(ed25519.getPublicKey(seed)),
      signMessage: async (message: Uint8Array) => ed25519.sign(message, seed),
    };
    const f = await createEvidenceFixture();
    const program = new PublicKey(manifest.consumerProgram);
    const [identity, bump] = PublicKey.findProgramAddressSync(
      [Buffer.from("identity"), signer.publicKey.toBytes()],
      program,
    );
    const [baseline, baselineBump] = await deriveEncryptedBaselinePda(
      signer.publicKey,
      manifest,
    );
    const coder = new BorshAccountsCoder(entrosAnchorIdl as Idl);
    const decoded = coder.decode<Record<string, unknown>>(
      "IdentityState",
      f.identityAccount.data,
    );
    const identityData = await coder.encode("IdentityState", {
      ...decoded,
      owner: signer.publicKey,
      bump,
    });
    const commitment = new Uint8Array(32).fill(3);
    const blob = await encryptBaselineBlob(
      new Uint8Array(32),
      new Uint8Array(32).fill(1),
      await getOrDeriveBaselineKey(signer),
      signer.publicKey,
      baseline,
      commitment,
    );
    const encodedBlob = new Uint8Array(105);
    encodedBlob.set(blob, 8);
    encodedBlob[104] = baselineBump;
    const read = vi.fn(async (address: PublicKey) => {
      if (address.equals(identity))
        return { ...f.identityAccount, data: identityData, owner: program };
      if (address.equals(baseline))
        return { ...f.identityAccount, data: encodedBlob, owner: program };
      throw new Error("Unexpected official account read");
    });
    await storeVerificationData(data("99"), signer.publicKey.toBase58());
    const recovered = await recoverBaselineFromChain(
      signer,
      {
        getGenesisHash: async () => manifest.genesisHash,
        getAccountInfo: read,
      },
      manifest,
    );
    expect(recovered).toEqual({ recovered: true });
    expect(
      (await loadVerificationData(signer.publicKey.toBase58()))?.commitment,
    ).toBe("99");
    expect(
      (await loadVerificationData(signer.publicKey.toBase58(), manifest))
        ?.commitment,
    ).toBe(BigInt(`0x${"03".repeat(32)}`).toString());
    expect(read).toHaveBeenCalledTimes(2);
  });
  it("upgrades only the selected identity layout", async () => {
    let submitted: Transaction | undefined;
    const result = await upgradeIdentityLayoutViaWallet({
      requestBoundManifest: manifest,
      wallet: {
        publicKey: wallet,
        signTransaction: async (tx: Transaction) => tx,
        signAllTransactions: async (txs: Transaction[]) => txs,
        sendTransaction: async (tx: Transaction) => {
          submitted = tx;
          return "synthetic-upgrade";
        },
      },
      connection: {
        getGenesisHash: async () => manifest.genesisHash,
        getLatestBlockhash: async () => ({
          blockhash: "11111111111111111111111111111111",
          lastValidBlockHeight: 1,
        }),
        confirmTransaction: async () => ({ value: { err: null } }),
      },
    });
    expect(result.success, result.error).toBe(true);
    expect(submitted?.instructions[0]?.programId.toBase58()).toBe(
      manifest.consumerProgram,
    );
  });
  it("fails explicitly before reading shared policy or SAS evidence", async () => {
    const f = await createEvidenceFixture();
    expect(
      await readIntegratorEvidence({ ...f.input, deployment: manifest }),
    ).toEqual({ status: "invalid", reason: "unsupported_deployment" });
    expect(f.calls).toHaveLength(0);
  });
});
