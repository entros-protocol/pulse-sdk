import { readFileSync } from "node:fs";
import {
  PublicKey,
  SystemProgram,
  type AccountInfo,
  type ParsedTransactionWithMeta,
} from "@solana/web3.js";
import { afterEach, describe, expect, it, vi } from "vitest";

import { readAgentState, type AgentStateConnection } from "../src/agent/state";
import { verifyEntrosAttestation } from "../src/attestation/sas";
import { decodeIdentityState } from "../src/identity/anchor";
import {
  CLOCK_CATCH_UP_SECONDS,
  readIntegratorEvidence,
} from "../src/identity/integrator";
import type { IntegratorEvidenceConnection } from "../src/identity/integrator-types";
import { readMigrationRecord } from "../src/identity/migration";

const vectors = JSON.parse(
  readFileSync(
    new URL("./fixtures/solana-parity-vectors.json", import.meta.url),
    "utf8",
  ),
);

const fromB64 = (value: string): Buffer => Buffer.from(value, "base64");
const anchorProgram = new PublicKey(vectors.programs.anchor);
const verifierProgram = new PublicKey(vectors.programs.verifier);
const sasProgram = new PublicKey(vectors.programs.sas);
const agentRegistryProgram = new PublicKey(vectors.programs.agentRegistry);
const coreProgram = new PublicKey(vectors.programs.core);
const nonceBytes = Buffer.from(vectors.nonceHex, "hex");

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

function buildTx(
  kind: "update" | "mint" | "rebaseline" | "compact" | "bound" | "reset",
  blockTime = vectors.blockTime,
): ParsedTransactionWithMeta {
  const wallet = new PublicKey(vectors.wallet);
  const identityPda = new PublicKey(vectors.pdas.identity[0]);
  const proofReqPda = new PublicKey(vectors.pdas.proofRequestState[0]);
  const sys = SystemProgram.programId;

  const instructions: ParsedTransactionWithMeta["transaction"]["message"]["instructions"] =
    [];

  if (kind === "mint") {
    instructions.push({
      programId: anchorProgram,
      accounts: Array.from({ length: 11 }, (_, i) =>
        i === 0 ? wallet : i === 1 ? identityPda : sys,
      ),
      data: vectors.instructions.mint,
    });
  } else if (kind === "rebaseline") {
    instructions.push({
      programId: anchorProgram,
      accounts: Array.from({ length: 6 }, (_, i) =>
        i === 0 ? wallet : i === 1 ? identityPda : sys,
      ),
      data: vectors.instructions.rebaseline,
    });
  } else if (kind === "compact") {
    const verPda = new PublicKey(vectors.pdas.verificationCompact[0]);
    instructions.push(
      {
        programId: verifierProgram,
        accounts: [wallet, verPda, sys],
        data: vectors.instructions.compact.verifier,
      },
      {
        programId: anchorProgram,
        accounts: Array.from({ length: 8 }, (_, i) =>
          i === 0 ? wallet : i === 1 ? identityPda : i === 2 ? verPda : sys,
        ),
        data: vectors.instructions.compact.anchor,
      },
    );
  } else if (kind === "bound") {
    const verPda = new PublicKey(vectors.pdas.verificationBound[0]);
    instructions.push(
      {
        programId: verifierProgram,
        accounts: [wallet, verPda, identityPda, proofReqPda, sys],
        data: vectors.instructions.bound.verifier,
      },
      {
        programId: anchorProgram,
        accounts: Array.from({ length: 9 }, (_, i) =>
          i === 0
            ? wallet
            : i === 1
              ? identityPda
              : i === 2
                ? proofReqPda
                : i === 3
                  ? verPda
                  : sys,
        ),
        data: vectors.instructions.bound.anchor,
      },
    );
  } else if (kind === "reset") {
    instructions.push({
      programId: anchorProgram,
      accounts: Array.from({ length: 5 }, (_, i) =>
        i === 0 ? wallet : i === 1 ? identityPda : sys,
      ),
      data: vectors.instructions.reset,
    });
  } else {
    instructions.push({
      programId: anchorProgram,
      accounts: Array.from({ length: 8 }, (_, i) =>
        i === 0 ? wallet : i === 1 ? identityPda : sys,
      ),
      data: vectors.instructions.update,
    });
  }

  return {
    slot: 100,
    blockTime,
    version: "legacy",
    meta: { err: null, fee: 0, preBalances: [], postBalances: [] },
    transaction: {
      signatures: [vectors.signature],
      message: {
        accountKeys: [{ pubkey: wallet, signer: true, writable: true }],
        recentBlockhash: PublicKey.default.toBase58(),
        instructions,
      },
    },
  };
}

function makeEvidenceConnection(opts: {
  identityData?: Buffer | null;
  sasData?: Buffer | null;
  sasOwner?: PublicKey;
  migrationData?: Buffer | null;
  migrationOwner?: PublicKey;
  tx?: ParsedTransactionWithMeta;
  genesisHash?: string;
}): IntegratorEvidenceConnection {
  const idPda = vectors.pdas.identity[0];
  const attPda = vectors.pdas.sasAttestation[0];
  const migPda = vectors.pdas.migration[0];
  const tx = opts.tx ?? buildTx("update");

  return {
    getGenesisHash: async () => opts.genesisHash ?? vectors.genesisHash,
    getSignatureStatuses: async () => ({
      context: { slot: 101 },
      value: [
        {
          slot: 100,
          confirmations: 1,
          err: null,
          confirmationStatus: "confirmed",
        },
      ],
    }),
    getParsedTransaction: async () => tx,
    getAccountInfoAndContext: async (address) => {
      const key = address.toBase58();
      let value: AccountInfo<Buffer> | null = null;
      if (key === idPda) {
        const data =
          opts.identityData === undefined
            ? fromB64(vectors.identityBuffers.activeBase64)
            : opts.identityData;
        value = data
          ? {
              data,
              executable: false,
              lamports: 1,
              owner: anchorProgram,
              rentEpoch: 0,
            }
          : null;
      } else if (key === attPda) {
        const data =
          opts.sasData === undefined
            ? fromB64(vectors.sasBuffers.activeBase64)
            : opts.sasData;
        value = data
          ? {
              data,
              executable: false,
              lamports: 1,
              owner: opts.sasOwner ?? sasProgram,
              rentEpoch: 0,
            }
          : null;
      } else if (key === migPda && opts.migrationData) {
        value = {
          data: opts.migrationData,
          executable: false,
          lamports: 1,
          owner: opts.migrationOwner ?? anchorProgram,
          rentEpoch: 0,
        };
      }
      return { context: { slot: 102 }, value };
    },
  };
}

describe("canonical shared-vector parity (@entros/pulse-sdk)", () => {
  it("matches every PDA and bump byte-for-byte", () => {
    const wallet = new PublicKey(vectors.wallet);
    const cred = new PublicKey(vectors.programs.credential);
    const schema = new PublicKey(vectors.programs.schema);
    const toTuple = ([pk, bump]: [PublicKey, number]) => [pk.toBase58(), bump];

    expect(
      toTuple(
        PublicKey.findProgramAddressSync(
          [Buffer.from("identity"), wallet.toBuffer()],
          anchorProgram,
        ),
      ),
    ).toEqual(vectors.pdas.identity);
    expect(
      toTuple(
        PublicKey.findProgramAddressSync(
          [Buffer.from("mint"), wallet.toBuffer()],
          anchorProgram,
        ),
      ),
    ).toEqual(vectors.pdas.mint);
    expect(
      toTuple(
        PublicKey.findProgramAddressSync(
          [Buffer.from("migration"), wallet.toBuffer()],
          anchorProgram,
        ),
      ),
    ).toEqual(vectors.pdas.migration);
    expect(
      toTuple(
        PublicKey.findProgramAddressSync(
          [Buffer.from("proof_request_state"), wallet.toBuffer()],
          anchorProgram,
        ),
      ),
    ).toEqual(vectors.pdas.proofRequestState);
    expect(
      toTuple(
        PublicKey.findProgramAddressSync(
          [
            Buffer.from("attestation"),
            cred.toBuffer(),
            schema.toBuffer(),
            wallet.toBuffer(),
          ],
          sasProgram,
        ),
      ),
    ).toEqual(vectors.pdas.sasAttestation);
    expect(
      toTuple(
        PublicKey.findProgramAddressSync(
          [Buffer.from("verification"), wallet.toBuffer(), nonceBytes],
          verifierProgram,
        ),
      ),
    ).toEqual(vectors.pdas.verificationCompact);
    expect(
      toTuple(
        PublicKey.findProgramAddressSync(
          [Buffer.from("verification_bound"), wallet.toBuffer(), nonceBytes],
          verifierProgram,
        ),
      ),
    ).toEqual(vectors.pdas.verificationBound);
    expect(
      toTuple(
        PublicKey.findProgramAddressSync(
          [
            Buffer.from("agent"),
            new PublicKey(vectors.agentFixtures.devnet.assetAddress).toBuffer(),
          ],
          agentRegistryProgram,
        ),
      ),
    ).toEqual(vectors.pdas.agentAccount);
  });

  it("matches IdentityState prefix layouts and MigrationRecord boundaries", async () => {
    const d593 = await decodeIdentityState(
      fromB64(vectors.identityBuffers.full593Base64),
    );
    const d583 = await decodeIdentityState(
      fromB64(vectors.identityBuffers.prefix583Base64),
    );
    const d551 = await decodeIdentityState(
      fromB64(vectors.identityBuffers.prefix551Base64),
    );
    const d543 = await decodeIdentityState(
      fromB64(vectors.identityBuffers.prefix543Base64),
    );
    expect(d593?.projectionVersion).toBe(2);
    expect(d593?.lastRebaselineTimestamp).toBe(vectors.nowSeconds - 200);
    expect(d583?.projectionVersion).toBe(0);
    expect(d551?.lastResetTimestamp).toBe(vectors.nowSeconds - 500);
    expect(d543?.lastResetTimestamp).toBe(0);
    expect(
      await decodeIdentityState(fromB64(vectors.identityBuffers.legacy207Base64)),
    ).toBeNull();

    expect(
      readMigrationRecord(
        {
          data: fromB64(vectors.migrationBuffers.atBlockTimeBase64),
          executable: false,
          owner: anchorProgram,
        },
        anchorProgram,
      ),
    ).toEqual({ status: "migrated", migratedAt: vectors.blockTime });
    expect(
      readMigrationRecord(
        {
          data: fromB64(vectors.migrationBuffers.extended65Base64),
          executable: false,
          owner: anchorProgram,
        },
        anchorProgram,
      ),
    ).toEqual({ status: "migrated", migratedAt: vectors.blockTime - 1 });
    expect(
      readMigrationRecord(
        {
          data: fromB64(vectors.migrationBuffers.truncated56Base64),
          executable: false,
          owner: anchorProgram,
        },
        anchorProgram,
      ),
    ).toEqual({ status: "invalid" });
  });

  it("matches active, expired, reset, migrated, and SAS mismatch evidence cases", async () => {
    for (const [kind, idKey, expectedKind] of [
      ["update", "activeBase64", "update"],
      ["mint", "mintBase64", "mint"],
      ["compact", "activeBase64", "update"],
      ["bound", "activeBase64", "update"],
      ["rebaseline", "rebaselineBase64", "rebaseline"],
    ] as const) {
      const res = await readIntegratorEvidence({
        walletPubkey: vectors.wallet,
        transactionSignature: vectors.signature,
        connection: makeEvidenceConnection({
          identityData: fromB64(vectors.identityBuffers[idKey]),
          tx: buildTx(kind),
        }),
        nowSeconds: vectors.nowSeconds,
      });
      expect(res.status, kind).toBe("available");
      if (res.status === "available") {
        expect(res.evidence.transaction.kind).toBe(expectedKind);
        expect(res.evidence.attestation.status).toBe("present");
      }
    }

    expect(
      await readIntegratorEvidence({
        walletPubkey: vectors.wallet,
        transactionSignature: vectors.signature,
        connection: makeEvidenceConnection({
          sasData: fromB64(vectors.sasBuffers.expiredBase64),
        }),
        nowSeconds: vectors.nowSeconds,
      }),
    ).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "missing" } },
    });

    expect(
      await readIntegratorEvidence({
        walletPubkey: vectors.wallet,
        transactionSignature: vectors.signature,
        connection: makeEvidenceConnection({
          identityData: fromB64(vectors.identityBuffers.resetBase64),
          sasData: fromB64(vectors.sasBuffers.preResetBase64),
        }),
        nowSeconds: vectors.nowSeconds,
      }),
    ).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "missing" } },
    });

    expect(
      await readIntegratorEvidence({
        walletPubkey: vectors.wallet,
        transactionSignature: vectors.signature,
        connection: makeEvidenceConnection({
          migrationData: fromB64(vectors.migrationBuffers.atBlockTimeBase64),
        }),
        nowSeconds: vectors.nowSeconds,
      }),
    ).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "missing" } },
    });
    expect(
      await readIntegratorEvidence({
        walletPubkey: vectors.wallet,
        transactionSignature: vectors.signature,
        connection: makeEvidenceConnection({
          migrationData: fromB64(vectors.migrationBuffers.truncated56Base64),
        }),
        nowSeconds: vectors.nowSeconds,
      }),
    ).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "invalid" } },
    });
    expect(
      await readIntegratorEvidence({
        walletPubkey: vectors.wallet,
        transactionSignature: vectors.signature,
        connection: makeEvidenceConnection({
          migrationData: fromB64(vectors.migrationBuffers.atBlockTimeBase64),
          migrationOwner: SystemProgram.programId,
        }),
        nowSeconds: vectors.nowSeconds,
      }),
    ).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "present" } },
    });

    for (const key of [
      "wrongWalletBase64",
      "wrongSchemaBase64",
      "falseHumanBase64",
      "zeroSignerBase64",
      "nonZeroTokenBase64",
    ] as const) {
      expect(
        await readIntegratorEvidence({
          walletPubkey: vectors.wallet,
          transactionSignature: vectors.signature,
          connection: makeEvidenceConnection({
            sasData: fromB64(vectors.sasBuffers[key]),
          }),
          nowSeconds: vectors.nowSeconds,
        }),
        key,
      ).toMatchObject({
        status: "available",
        evidence: { attestation: { status: "invalid" } },
      });
    }

    expect(
      await readIntegratorEvidence({
        walletPubkey: vectors.wallet,
        transactionSignature: vectors.signature,
        connection: makeEvidenceConnection({
          sasData: Buffer.alloc(0),
          sasOwner: SystemProgram.programId,
        }),
        nowSeconds: vectors.nowSeconds,
      }),
    ).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "missing" } },
    });
  });

  it("matches 2s slot tolerance, 15s clock catch-up, SAS batch read, and captured Core agent states", async () => {
    const idPlus2 = fromB64(vectors.identityBuffers.activeBase64);
    idPlus2.writeBigInt64LE(BigInt(vectors.blockTime + 2), 48);
    expect(
      (
        await readIntegratorEvidence({
          walletPubkey: vectors.wallet,
          transactionSignature: vectors.signature,
          connection: makeEvidenceConnection({ identityData: idPlus2 }),
          nowSeconds: vectors.nowSeconds,
        })
      ).status,
    ).toBe("available");

    const idMinus3 = fromB64(vectors.identityBuffers.activeBase64);
    idMinus3.writeBigInt64LE(BigInt(vectors.blockTime - 3), 48);
    expect(
      await readIntegratorEvidence({
        walletPubkey: vectors.wallet,
        transactionSignature: vectors.signature,
        connection: makeEvidenceConnection({ identityData: idMinus3 }),
        nowSeconds: vectors.nowSeconds,
      }),
    ).toEqual({ status: "invalid", reason: "transaction_invalid" });

    // Pre-warm dynamic imports before enabling fake timers.
    const futureBlock = vectors.nowSeconds + 4;
    const futureId = fromB64(vectors.identityBuffers.activeBase64);
    futureId.writeBigInt64LE(BigInt(futureBlock), 48);
    const futureSas = fromB64(vectors.sasBuffers.activeBase64);
    futureSas.writeBigInt64LE(BigInt(futureBlock), 104);
    const catchupConn = makeEvidenceConnection({
      identityData: futureId,
      sasData: futureSas,
      tx: buildTx("update", futureBlock),
    });
    await readIntegratorEvidence({
      walletPubkey: vectors.wallet,
      transactionSignature: vectors.signature,
      connection: catchupConn,
      nowSeconds: vectors.nowSeconds,
    });
    vi.useFakeTimers({
      toFake: ["setTimeout", "clearTimeout", "Date", "performance"],
    });
    vi.setSystemTime(vectors.nowSeconds * 1000);
    const pending = readIntegratorEvidence({
      walletPubkey: vectors.wallet,
      transactionSignature: vectors.signature,
      connection: catchupConn,
      nowSeconds: () => Math.floor(Date.now() / 1000),
    });
    let settled = false;
    void pending.finally(() => {
      settled = true;
    });
    for (let round = 0; round < 4 && !settled; round += 1) {
      await vi.dynamicImportSettled();
      if (settled) break;
      await vi.runAllTimersAsync();
    }
    expect((await pending).status).toBe("available");
    expect(performance.now()).toBe(4000);
    expect(CLOCK_CATCH_UP_SECONDS).toBe(15);
    vi.useRealTimers();

    // verifyEntrosAttestation parity
    vi.useFakeTimers();
    vi.setSystemTime(vectors.nowSeconds * 1000);
    expect(
      await verifyEntrosAttestation(vectors.wallet, {
        getMultipleAccountsInfo: async () => [
          { data: fromB64(vectors.sasBuffers.activeBase64), owner: sasProgram },
          {
            data: fromB64(vectors.identityBuffers.activeBase64),
            owner: anchorProgram,
          },
          null,
        ],
      }),
    ).toEqual({
      isHuman: true,
      trustScore: 300,
      verifiedAt: vectors.blockTime,
      mode: "wallet-connected",
      expired: false,
    });
    vi.useRealTimers();

    // Captured devnet & localnet Core + 8004 agent states
    const devnetAgent = vectors.agentFixtures.devnet;
    const devnetConn: AgentStateConnection = {
      getGenesisHash: async () => vectors.genesisHash,
      getMultipleAccountsInfoAndContext: async () => ({
        context: { slot: devnetAgent.contextSlot },
        value: [
          {
            data: fromB64(devnetAgent.assetBase64),
            executable: false,
            lamports: 1,
            owner: coreProgram,
            rentEpoch: 0,
          },
          {
            data: fromB64(devnetAgent.agentAccountBase64),
            executable: false,
            lamports: 1,
            owner: agentRegistryProgram,
            rentEpoch: 0,
          },
        ],
      }),
    };
    expect(
      await readAgentState({
        agent: devnetAgent.assetAddress,
        connection: devnetConn,
      }),
    ).toEqual({
      status: "available",
      evidence: {
        cluster: "devnet",
        genesisHash: vectors.genesisHash,
        coreProgram: vectors.programs.core,
        agentRegistryProgram: vectors.programs.agentRegistry,
        agentCollection: vectors.programs.agentCollection,
        agent: devnetAgent.assetAddress,
        agentAccount: devnetAgent.agentAccountAddress,
        owner: devnetAgent.owner,
        cachedOwner: devnetAgent.owner,
        agentWallet: null,
        agentWalletStatus: "unbound",
        readContextSlot: devnetAgent.contextSlot,
      },
    });

    const localnet = vectors.agentFixtures.localnet;
    for (const [stage, expected] of [
      ["registered", [localnet.ownerA, localnet.ownerA, null, "unbound"]],
      [
        "bound",
        [localnet.ownerA, localnet.ownerA, localnet.agentWallets.first, "bound"],
      ],
      [
        "coreTransfer",
        [localnet.ownerB, localnet.ownerA, localnet.agentWallets.first, "stale"],
      ],
      [
        "reboundByB",
        [
          localnet.ownerB,
          localnet.ownerB,
          localnet.agentWallets.successor,
          "bound",
        ],
      ],
      ["registryTransfer", [localnet.ownerA, localnet.ownerA, null, "unbound"]],
    ] as const) {
      const stageBytes = localnet.stages[stage];
      const conn: AgentStateConnection = {
        getGenesisHash: async () => vectors.genesisHash,
        getMultipleAccountsInfoAndContext: async () => ({
          context: { slot: 1 },
          value: [
            {
              data: fromB64(stageBytes.asset),
              executable: false,
              lamports: 1,
              owner: coreProgram,
              rentEpoch: 0,
            },
            {
              data: fromB64(stageBytes.agentAccount),
              executable: false,
              lamports: 1,
              owner: agentRegistryProgram,
              rentEpoch: 0,
            },
          ],
        }),
      };
      const res = await readAgentState({ agent: localnet.agent, connection: conn });
      expect(res, stage).toMatchObject({
        status: "available",
        evidence: {
          agentAccount: localnet.agentAccount,
          owner: expected[0],
          cachedOwner: expected[1],
          agentWallet: expected[2],
          agentWalletStatus: expected[3],
        },
      });
    }
  });
});
