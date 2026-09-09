import { describe, expect, it, vi } from "vitest";
import { PublicKey, Transaction } from "@solana/web3.js";
import { sha256 } from "@noble/hashes/sha256";
import {
  prepareWalletProofRequest,
  IdentityLayoutUpgradeRequiredError,
} from "../src/submit/request";
import { bytesHex } from "../src/proof/request";
import type { RequestBoundManifest } from "../src/proof/request";
import { PROGRAM_IDS } from "../src/config";
import { createEvidenceFixture } from "./fixtures/integrator-evidence";
import { submitViaRelayer } from "../src/submit/relayer";
import {
  submitViaWallet,
  upgradeIdentityLayoutViaWallet,
} from "../src/submit/wallet";
import { toBigEndian32 } from "../src/proof/serializer";

async function setup(present = true) {
  const fixture = await createEvidenceFixture();
  const manifest: RequestBoundManifest = {
    generation: "request-bound-v1",
    deploymentDomain: "11".repeat(32),
    genesisHash: "synthetic",
    verifierProgram: PROGRAM_IDS.entrosVerifier,
    consumerProgram: PROGRAM_IDS.entrosAnchor,
    wasm: { url: "synthetic.wasm", sha256: "12".repeat(32) },
    zkey: { url: "synthetic.zkey", sha256: "13".repeat(32) },
  };
  const [, bump] = PublicKey.findProgramAddressSync(
    [Buffer.from("proof_request_state"), fixture.wallet.toBytes()],
    new PublicKey(PROGRAM_IDS.entrosAnchor),
  );
  const stateData = Buffer.alloc(50);
  stateData.set(sha256(Buffer.from("account:ProofRequestState")).slice(0, 8));
  stateData[8] = 1;
  stateData.set(fixture.wallet.toBytes(), 9);
  stateData.writeBigUInt64LE(7n, 41);
  stateData[49] = bump;
  const state = { ...fixture.identityAccount, data: stateData };
  const clockData = Buffer.alloc(40);
  clockData.writeBigInt64LE(1800000000n, 32);
  const clock = {
    ...fixture.identityAccount,
    owner: new PublicKey("Sysvar1111111111111111111111111111111111111"),
    data: clockData,
  };
  const connection = {
    getGenesisHash: vi.fn(async () => "synthetic"),
    getMultipleAccountsInfoAndContext: vi.fn(async () => ({
      context: { slot: 100 },
      value: [fixture.identityAccount, present ? state : null, clock],
    })),
  };
  const options = {
    commitmentNew: new Uint8Array(32).fill(1),
    commitmentPrevious: new Uint8Array(32).fill(3),
    threshold: 30,
    minDistance: 3,
    nonce: new Uint8Array(32).fill(9),
  };
  return { fixture, manifest, state, clock, connection, options };
}
describe("pre-proof request preparation", () => {
  it("freezes the coherent account snapshot and chain expiry before proving", async () => {
    const { connection, fixture, manifest, options, state } = await setup();
    const request = await prepareWalletProofRequest(
      connection,
      fixture.wallet,
      manifest,
      options,
    );
    expect(request.action.counter).toBe(7n);
    expect(request.action.validUntil).toBe(1800000180n);
    expect(request.nonce).toBe(bytesHex(options.nonce));
    state.data.writeBigUInt64LE(8n, 41);
    options.nonce[0] = 10;
    expect(request.action.counter).toBe(7n);
    expect(request.nonce).toBe("09".repeat(32));
    expect(connection.getMultipleAccountsInfoAndContext).toHaveBeenCalledTimes(
      1,
    );
  });
  it("preserves an executor nonce from preparation through the SAS request", async () => {
    const f = await setup();
    const calls: { url: string; body?: unknown }[] = [];
    const serverNonce = Array<number>(32).fill(25);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init?: RequestInit) => {
        calls.push({
          url,
          body: init?.body ? JSON.parse(String(init.body)) : undefined,
        });
        if (url.includes("/challenge?"))
          return new Response(JSON.stringify({ nonce: serverNonce }));
        if (url.endsWith("/attest"))
          return new Response(
            JSON.stringify({
              success: true,
              attestation_tx: "attestation-signature",
            }),
          );
        throw new Error("Unexpected request");
      }),
    );
    try {
      const request = await prepareWalletProofRequest(
        f.connection,
        f.fixture.wallet,
        f.manifest,
        {
          ...f.options,
          nonce: undefined,
          relayerUrl: "https://executor.invalid/verify",
        },
      );
      expect(request.nonceSource).toBe("executor");
      expect(calls).toHaveLength(1);
      const proof = {
        proofBytes: new Uint8Array(256),
        publicInputs: [
          BigInt(`0x${request.action.commitmentNew}`).toString(),
          BigInt(`0x${request.action.commitmentPrevious}`).toString(),
          "30",
          "3",
          request.digestHi,
          request.digestLo,
        ].map(toBigEndian32),
      };
      const result = await submitViaWallet(proof, f.options.commitmentNew, {
        wallet: {
          publicKey: f.fixture.wallet,
          sendTransaction: async () => "bound-signature",
          signTransaction: async (tx: Transaction) => tx,
          signAllTransactions: async (txs: Transaction[]) => txs,
          signMessage: async () => new Uint8Array(64),
        },
        connection: {
          ...f.connection,
          getLatestBlockhash: async () => ({
            blockhash: "11111111111111111111111111111111",
            lastValidBlockHeight: 1,
          }),
          confirmTransaction: async () => ({ value: { err: null } }),
        },
        isFirstVerification: false,
        preparedRequest: request,
        requestBoundManifest: f.manifest,
        relayerUrl: "https://executor.invalid/verify",
      });
      expect(result).toMatchObject({
        success: true,
        attestationTx: "attestation-signature",
      });
      expect(calls).toHaveLength(2);
      expect(calls[1]!.url).toBe("https://executor.invalid/attest");
      expect(calls[1]!.body).toMatchObject({ nonce: serverNonce });
      expect(request.nonce).toBe("19".repeat(32));
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it.each(["offline", "zero", "wrong-length"])(
    "falls back before proving for a %s executor nonce",
    async (mode) => {
      const f = await setup();
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => {
          if (mode === "offline") throw new Error("Synthetic offline response");
          return new Response(
            JSON.stringify({
              nonce: mode === "zero" ? Array<number>(32).fill(0) : [1],
            }),
          );
        }),
      );
      try {
        const request = await prepareWalletProofRequest(
          f.connection,
          f.fixture.wallet,
          f.manifest,
          {
            ...f.options,
            nonce: undefined,
            relayerUrl: "https://executor.invalid/verify",
          },
        );
        expect(request.nonceSource).toBe("client");
        expect(request.nonce).not.toBe("00".repeat(32));
      } finally {
        vi.unstubAllGlobals();
      }
    },
  );

  it.each([543, 551, 583])(
    "requires a signed upgrade for a known %i-byte identity, then preserves its baseline and counter",
    async (length) => {
      const f = await setup();
      f.fixture.identityAccount.data = f.fixture.identityAccount.data.subarray(
        0,
        length,
      );
      const original = f.fixture.identityAccount.data.slice();
      await expect(
        prepareWalletProofRequest(
          f.connection,
          f.fixture.wallet,
          f.manifest,
          f.options,
        ),
      ).rejects.toBeInstanceOf(IdentityLayoutUpgradeRequiredError);
      let captured: Transaction | undefined;
      const result = await upgradeIdentityLayoutViaWallet({
        wallet: {
          publicKey: f.fixture.wallet,
          signTransaction: async (tx: Transaction) => tx,
          signAllTransactions: async (txs: Transaction[]) => txs,
          sendTransaction: async (tx: Transaction) => {
            captured = tx;
            const grown = Buffer.alloc(593);
            grown.set(f.fixture.identityAccount.data);
            f.fixture.identityAccount.data = grown;
            return "upgrade-signature";
          },
        },
        connection: {
          ...f.connection,
          getLatestBlockhash: async () => ({
            blockhash: "11111111111111111111111111111111",
            lastValidBlockHeight: 1,
          }),
          confirmTransaction: async () => ({ value: { err: null } }),
        },
        requestBoundManifest: f.manifest,
      });
      expect(result.success).toBe(true);
      expect(captured?.instructions).toHaveLength(1);
      expect(f.fixture.identityAccount.data.subarray(0, length)).toEqual(
        original,
      );
      const request = await prepareWalletProofRequest(
        f.connection,
        f.fixture.wallet,
        f.manifest,
        f.options,
      );
      expect(request.action.counter).toBe(7n);
      expect(request.action.commitmentPrevious).toBe(
        bytesHex(f.options.commitmentPrevious),
      );
    },
  );

  it("rejects unknown old identity layouts without requesting an upgrade", async () => {
    const f = await setup();
    f.fixture.identityAccount.data = f.fixture.identityAccount.data.subarray(
      0,
      207,
    );
    await expect(
      prepareWalletProofRequest(
        f.connection,
        f.fixture.wallet,
        f.manifest,
        f.options,
      ),
    ).rejects.not.toBeInstanceOf(IdentityLayoutUpgradeRequiredError);
  });

  it("prepares a funded empty system account at counter zero", async () => {
    const f = await setup();
    f.state.owner = PublicKey.default;
    f.state.data = Buffer.alloc(0);
    f.state.lamports = 1000;
    const request = await prepareWalletProofRequest(
      f.connection,
      f.fixture.wallet,
      f.manifest,
      f.options,
    );
    expect(request.action.counter).toBe(0n);
  });

  it.each(["wrong-owner", "nonempty", "executable"])(
    "rejects invalid uninitialized request state %s",
    async (kind) => {
      const f = await setup();
      f.state.owner = PublicKey.default;
      f.state.data = Buffer.alloc(0);
      if (kind === "wrong-owner")
        f.state.owner = new PublicKey(PROGRAM_IDS.entrosVerifier);
      if (kind === "nonempty") f.state.data = Buffer.alloc(1);
      if (kind === "executable") f.state.executable = true;
      await expect(
        prepareWalletProofRequest(
          f.connection,
          f.fixture.wallet,
          f.manifest,
          f.options,
        ),
      ).rejects.toThrow("request state");
    },
  );

  it("prepares missing state at zero for atomic initialization", async () => {
    const { connection, fixture, manifest, options } = await setup(false);
    expect(
      (
        await prepareWalletProofRequest(
          connection,
          fixture.wallet,
          manifest,
          options,
        )
      ).action.counter,
    ).toBe(0n);
  });
  it.each([
    "owner",
    "wallet",
    "version",
    "length",
    "bump",
    "discriminator",
    "overflow",
  ])("rejects invalid request state %s", async (kind) => {
    const { connection, fixture, manifest, options, state } = await setup();
    if (kind === "owner") state.owner = PublicKey.default;
    if (kind === "wallet") state.data[9] = state.data[9]! ^ 1;
    if (kind === "version") state.data[8] = 2;
    if (kind === "length")
      state.data = Buffer.concat([state.data, Buffer.alloc(1)]);
    if (kind === "bump") state.data[49] = state.data[49]! ^ 1;
    if (kind === "discriminator") state.data[0] = state.data[0]! ^ 1;
    if (kind === "overflow") state.data.writeBigUInt64LE((1n << 64n) - 1n, 41);
    await expect(
      prepareWalletProofRequest(connection, fixture.wallet, manifest, options),
    ).rejects.toThrow();
  });
  it("rejects a wrong chain, stale baseline, and invalid clock", async () => {
    const { connection, fixture, manifest, options, clock } = await setup();
    await expect(
      prepareWalletProofRequest(
        connection,
        fixture.wallet,
        { ...manifest, genesisHash: "different" },
        options,
      ),
    ).rejects.toThrow("chain");
    await expect(
      prepareWalletProofRequest(connection, fixture.wallet, manifest, {
        ...options,
        commitmentPrevious: new Uint8Array(32).fill(4),
      }),
    ).rejects.toThrow("baseline");
    clock.owner = PublicKey.default;
    await expect(
      prepareWalletProofRequest(connection, fixture.wallet, manifest, options),
    ).rejects.toThrow("clock");
  });
  it("rejects bound proofs before any walletless API request", async () => {
    const fetch = vi.spyOn(globalThis, "fetch");
    try {
      const result = await submitViaRelayer(
        {
          proofBytes: new Uint8Array(256),
          publicInputs: Array.from({ length: 6 }, () => new Uint8Array(32)),
        },
        new Uint8Array(32),
        { relayerUrl: "https://example.invalid", isFirstVerification: false },
      );
      expect(result.success).toBe(false);
      expect(result.error).toContain("connected wallet");
      expect(fetch).not.toHaveBeenCalled();
    } finally {
      fetch.mockRestore();
    }
  });
});
