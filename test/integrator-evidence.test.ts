import { afterEach, describe, expect, it, vi } from "vitest";
import { BorshInstructionCoder, utils, type Idl } from "@coral-xyz/anchor";
import { PublicKey } from "@solana/web3.js";
import {
  INTEGRATOR_DEVNET_GENESIS_HASH,
  readIntegratorEvidence,
} from "../src/identity/integrator";
import { entrosAnchorIdl } from "../src/protocol/idl";
import { createEvidenceFixture } from "./fixtures/integrator-evidence";
import sasGolden from "./fixtures/sas-attestation-golden.json";

afterEach(() => vi.useRealTimers());

describe("strict integrator evidence", () => {
  it("matches the pinned Rust SAS client's Borsh serialization", async () => {
    const fixture = await createEvidenceFixture();
    expect(fixture.sasAccount.data.toString("hex")).toBe(
      sasGolden.serializedHex,
    );
  });
  it("pins the complete RPC genesis hash rather than a shortened chain reference", () => {
    expect(INTEGRATOR_DEVNET_GENESIS_HASH).toBe(
      "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG",
    );
    expect(
      new PublicKey(INTEGRATOR_DEVNET_GENESIS_HASH).toBytes(),
    ).toHaveLength(32);
  });
  it.each(["mint", "update", "compact", "rebaseline"] as const)(
    "accepts the confirmed %s path",
    async (kind) => {
      const fixture = await createEvidenceFixture({ kind });
      const result = await readIntegratorEvidence(fixture.input);
      expect(result.status).toBe("available");
      if (result.status !== "available") return;
      expect(result.evidence.transaction.kind).toBe(
        kind === "compact" ? "update" : kind,
      );
      expect(result.evidence.attestation.status).toBe("present");
      expect(result.evidence.identity.verificationCount).toBe(
        kind === "mint" ? 0 : 7,
      );
    },
  );

  it.each([543, 551, 583, 593])(
    "accepts the supported %i-byte prefix through the shared decoder",
    async (length) => {
      const fixture = await createEvidenceFixture();
      fixture.identityAccount.data = fixture.identityAccount.data.subarray(
        0,
        length,
      );
      expect((await readIntegratorEvidence(fixture.input)).status).toBe(
        "available",
      );
    },
  );

  it.each([0, 207, 542, 544, 550, 552, 582, 584, 592, 594, 1000])(
    "rejects unknown identity length %i",
    async (length) => {
      const fixture = await createEvidenceFixture();
      const changed = Buffer.alloc(length);
      fixture.identityAccount.data.copy(changed);
      fixture.identityAccount.data = changed;
      expect(await readIntegratorEvidence(fixture.input)).toEqual({
        status: "invalid",
        reason: "identity_invalid",
      });
    },
  );

  const accountMutations: [
    string,
    (fixture: Awaited<ReturnType<typeof createEvidenceFixture>>) => void,
  ][] = [
    [
      "owner program",
      (f) => {
        f.identityAccount.owner = PublicKey.default;
      },
    ],
    [
      "executable",
      (f) => {
        f.identityAccount.executable = true;
      },
    ],
    [
      "discriminator",
      (f) => {
        f.identityAccount.data[0] = 0;
      },
    ],
    [
      "wallet",
      (f) => {
        f.identityAccount.data.fill(0, 8, 40);
      },
    ],
    [
      "mint",
      (f) => {
        f.identityAccount.data.fill(0, 94, 126);
      },
    ],
    [
      "score",
      (f) => {
        f.identityAccount.data.writeUInt16LE(10001, 60);
      },
    ],
    [
      "unsafe timestamp",
      (f) => {
        f.identityAccount.data.writeBigInt64LE(9007199254740992n, 40);
      },
    ],
    [
      "future timestamp",
      (f) => {
        f.identityAccount.data.writeBigInt64LE(
          BigInt(f.input.nowSeconds + 1),
          48,
        );
      },
    ],
    [
      "reversed timestamps",
      (f) => {
        f.identityAccount.data.writeBigInt64LE(1n, 48);
      },
    ],
    [
      "negative reset",
      (f) => {
        f.identityAccount.data.writeBigInt64LE(-1n, 543);
      },
    ],
    [
      "negative rebaseline",
      (f) => {
        f.identityAccount.data.writeBigInt64LE(-1n, 585);
      },
    ],
    [
      "unknown projection",
      (f) => {
        f.identityAccount.data.writeUInt16LE(65535, 583);
      },
    ],
    [
      "zero commitment",
      (f) => {
        f.identityAccount.data.fill(0, 62, 94);
      },
    ],
    [
      "out-of-field commitment",
      (f) => {
        f.identityAccount.data.fill(255, 62, 94);
      },
    ],
  ];
  it.each(accountMutations)(
    "rejects invalid identity %s",
    async (_, mutate) => {
      const fixture = await createEvidenceFixture();
      mutate(fixture);
      expect(await readIntegratorEvidence(fixture.input)).toEqual({
        status: "invalid",
        reason: "identity_invalid",
      });
    },
  );

  it.each(["missing", "unavailable"] as const)(
    "preserves identity %s",
    async (identity) => {
      const fixture = await createEvidenceFixture({ identity });
      expect(await readIntegratorEvidence(fixture.input)).toEqual(
        identity === "missing"
          ? { status: "invalid", reason: "identity_missing" }
          : { status: "unavailable", reason: "rpc_unavailable" },
      );
    },
  );

  it.each(["missing", "unavailable"] as const)(
    "preserves optional SAS %s without an issued address",
    async (attestation) => {
      const fixture = await createEvidenceFixture({ attestation });
      const result = await readIntegratorEvidence(fixture.input);
      expect(result.status).toBe("available");
      if (result.status === "available")
        expect(result.evidence.attestation).toEqual({ status: attestation });
    },
  );

  const sasMutations: [
    string,
    (fixture: Awaited<ReturnType<typeof createEvidenceFixture>>) => void,
  ][] = [
    [
      "owner",
      (f) => {
        f.sasAccount.owner = PublicKey.default;
      },
    ],
    [
      "executable",
      (f) => {
        f.sasAccount.executable = true;
      },
    ],
    [
      "discriminator",
      (f) => {
        f.sasAccount.data[0] = 0;
      },
    ],
    [
      "wallet nonce",
      (f) => {
        f.sasAccount.data[1] = 0;
      },
    ],
    [
      "credential",
      (f) => {
        f.sasAccount.data[33] = 0;
      },
    ],
    [
      "schema",
      (f) => {
        f.sasAccount.data[65] = 0;
      },
    ],
    [
      "data length",
      (f) => {
        f.sasAccount.data.writeUInt32LE(0xffffffff, 97);
      },
    ],
    [
      "truncated layout",
      (f) => {
        f.sasAccount.data = f.sasAccount.data.subarray(0, 203);
      },
    ],
    [
      "trailing bytes",
      (f) => {
        f.sasAccount.data = Buffer.concat([
          f.sasAccount.data,
          Buffer.from([0]),
        ]);
      },
    ],
    [
      "false assertion",
      (f) => {
        f.sasAccount.data[101] = 0;
      },
    ],
    [
      "invalid boolean",
      (f) => {
        f.sasAccount.data[101] = 2;
      },
    ],
    [
      "score",
      (f) => {
        f.sasAccount.data.writeUInt16LE(10001, 102);
      },
    ],
    [
      "unsafe verified time",
      (f) => {
        f.sasAccount.data.writeBigInt64LE(9007199254740992n, 104);
      },
    ],
    [
      "future verified time",
      (f) => {
        f.sasAccount.data.writeBigInt64LE(BigInt(f.input.nowSeconds + 1), 104);
      },
    ],
    [
      "mode length",
      (f) => {
        f.sasAccount.data.writeUInt32LE(15, 112);
      },
    ],
    [
      "mode",
      (f) => {
        f.sasAccount.data[116] = 0;
      },
    ],
    [
      "signer",
      (f) => {
        f.sasAccount.data.fill(0, 132, 164);
      },
    ],
    [
      "token account",
      (f) => {
        f.sasAccount.data[172] = 1;
      },
    ],
    [
      "expired",
      (f) => {
        f.sasAccount.data.writeBigInt64LE(BigInt(f.input.nowSeconds), 164);
      },
    ],
    [
      "negative expiry",
      (f) => {
        f.sasAccount.data.writeBigInt64LE(-1n, 164);
      },
    ],
    [
      "unsafe expiry",
      (f) => {
        f.sasAccount.data.writeBigInt64LE(9007199254740992n, 164);
      },
    ],
  ];
  it.each(sasMutations)("rejects SAS %s", async (_, mutate) => {
    const fixture = await createEvidenceFixture();
    mutate(fixture);
    const result = await readIntegratorEvidence(fixture.input);
    expect(result.status).toBe("available");
    if (result.status === "available")
      expect(result.evidence.attestation).toEqual({ status: "invalid" });
  });

  it("accepts exactly the issuer's zero expiry sentinel", async () => {
    const fixture = await createEvidenceFixture({ expiry: 0 });
    const result = await readIntegratorEvidence(fixture.input);
    if (result.status !== "available")
      throw new Error("Fixture did not decode");
    expect(result.evidence.attestation).toMatchObject({
      status: "present",
      expiresAt: null,
    });
  });

  it("cannot use a transaction after a reset or incompatible rebaseline", async () => {
    for (const options of [
      { lastResetTimestamp: 1_799_999_990 },
      {
        lastRebaselineTimestamp: 1_799_999_991,
        lastVerificationTimestamp: 1_799_999_991,
      },
    ]) {
      const fixture = await createEvidenceFixture(options);
      expect(await readIntegratorEvidence(fixture.input)).toEqual({
        status: "invalid",
        reason: "transaction_invalid",
      });
    }
  });

  it("rejects an output commitment that no longer matches current state", async () => {
    const fixture = await createEvidenceFixture();
    fixture.identityAccount.data[62] = 1;
    expect(await readIntegratorEvidence(fixture.input)).toEqual({
      status: "invalid",
      reason: "transaction_invalid",
    });
  });

  it("limits the zero-count exception to fresh zero-score mint evidence", async () => {
    for (const options of [
      { kind: "update" as const, verificationCount: 0 },
      { kind: "rebaseline" as const, verificationCount: 0 },
      { kind: "mint" as const, trustScore: 1 },
      { kind: "mint" as const, lastVerificationTimestamp: 1_799_999_999 },
    ]) {
      const fixture = await createEvidenceFixture(options);
      expect((await readIntegratorEvidence(fixture.input)).status).toBe(
        "invalid",
      );
    }
  });

  it("rejects a rebaseline projection mismatch", async () => {
    const fixture = await createEvidenceFixture({ kind: "rebaseline" });
    fixture.identityAccount.data.writeUInt16LE(0, 583);
    expect(await readIntegratorEvidence(fixture.input)).toEqual({
      status: "invalid",
      reason: "transaction_invalid",
    });
  });

  it("rejects failed, unrelated, non-signer, wrong-program, malformed, and reset transactions", async () => {
    const mutations: ((
      f: Awaited<ReturnType<typeof createEvidenceFixture>>,
    ) => void)[] = [
      (f) => {
        if (f.transaction.meta)
          f.transaction.meta.err = { InstructionError: [0, "InvalidArgument"] };
      },
      (f) => {
        f.transaction.transaction.message.instructions = [];
      },
      (f) => {
        f.transaction.transaction.message.accountKeys[0]!.signer = false;
      },
      (f) => {
        f.transaction.transaction.message.instructions[0]!.programId =
          PublicKey.default;
      },
      (f) => {
        const ix = f.transaction.transaction.message.instructions[0]!;
        if ("data" in ix) ix.data = "garbage";
      },
      (f) => {
        const ix = f.transaction.transaction.message.instructions[0]!;
        if ("accounts" in ix) ix.accounts[1] = PublicKey.default;
      },
      (f) => {
        f.transaction.transaction.signatures = [
          utils.bytes.bs58.encode(Buffer.alloc(64, 5)),
        ];
      },
      (f) => {
        const ix = f.transaction.transaction.message.instructions[0]!;
        if ("data" in ix)
          ix.data = utils.bytes.bs58.encode(
            Buffer.concat([utils.bytes.bs58.decode(ix.data), Buffer.from([0])]),
          );
      },
      (f) => {
        const ix = f.transaction.transaction.message.instructions[0]!;
        if ("data" in ix)
          ix.data = utils.bytes.bs58.encode(
            new BorshInstructionCoder(entrosAnchorIdl as Idl).encode(
              "reset_identity_state",
              {},
            ),
          );
      },
    ];
    for (const mutate of mutations) {
      const fixture = await createEvidenceFixture();
      mutate(fixture);
      expect(await readIntegratorEvidence(fixture.input)).toEqual({
        status: "invalid",
        reason: "transaction_invalid",
      });
    }
  });

  it("requires matching proof evidence before a compact update", async () => {
    const fixture = await createEvidenceFixture({ kind: "compact" });
    fixture.transaction.transaction.message.instructions.shift();
    expect(await readIntegratorEvidence(fixture.input)).toEqual({
      status: "invalid",
      reason: "transaction_invalid",
    });
  });

  it("retries propagation once and preserves unavailable block time", async () => {
    const fixture = await createEvidenceFixture();
    fixture.transaction.blockTime = null;
    expect(await readIntegratorEvidence(fixture.input)).toEqual({
      status: "unavailable",
      reason: "transaction_unavailable",
    });
    expect(fixture.calls.filter((call) => call === "transaction")).toHaveLength(
      2,
    );
    expect(fixture.calls).not.toContain("identity");
  });

  it("does not read accounts from an unconfirmed transaction or wrong genesis", async () => {
    const fixture = await createEvidenceFixture();
    fixture.connection.getSignatureStatuses = async () => ({
      context: { slot: 101 },
      value: [{ ...fixture.status, confirmationStatus: "processed" }],
    });
    expect((await readIntegratorEvidence(fixture.input)).status).toBe(
      "unavailable",
    );
    expect(fixture.calls).not.toContain("identity");
    fixture.connection.getGenesisHash = async () =>
      PublicKey.default.toBase58();
    expect(await readIntegratorEvidence(fixture.input)).toEqual({
      status: "invalid",
      reason: "wrong_cluster",
    });
  });

  it("retries a lagging identity snapshot and rejects persistent lag", async () => {
    const fixture = await createEvidenceFixture();
    fixture.connection.getAccountInfoAndContext = vi.fn(async () => ({
      context: { slot: 99 },
      value: fixture.identityAccount,
    }));
    expect(await readIntegratorEvidence(fixture.input)).toEqual({
      status: "unavailable",
      reason: "snapshot_unavailable",
    });
    expect(fixture.connection.getAccountInfoAndContext).toHaveBeenCalledTimes(
      2,
    );
  });

  it("bounds hanging initial RPC and starts no later account reads", async () => {
    const fixture = await createEvidenceFixture();
    vi.useFakeTimers();
    fixture.connection.getGenesisHash = () => new Promise(() => {});
    const pending = readIntegratorEvidence(fixture.input);
    await vi.waitFor(() => expect(fixture.calls).toContain("transaction"));
    await vi.advanceTimersByTimeAsync(3001);
    expect(await pending).toEqual({
      status: "unavailable",
      reason: "rpc_unavailable",
    });
    await vi.advanceTimersByTimeAsync(10000);
    expect(fixture.calls).not.toContain("identity");
  });

  it("bounds hanging optional SAS while retaining valid identity evidence", async () => {
    const fixture = await createEvidenceFixture();
    vi.useFakeTimers();
    let sasStarted = false;
    fixture.connection.getAccountInfoAndContext = (address) => {
      if (address.equals(fixture.identityPda))
        return Promise.resolve({
          context: { slot: 102 },
          value: fixture.identityAccount,
        });
      sasStarted = true;
      return new Promise(() => {});
    };
    const pending = readIntegratorEvidence(fixture.input);
    await vi.waitFor(() => expect(sasStarted).toBe(true));
    await vi.advanceTimersByTimeAsync(3001);
    const result = await pending;
    expect(result.status).toBe("available");
    if (result.status === "available")
      expect(result.evidence.attestation).toEqual({ status: "unavailable" });
  });

  it("handles bounded synthetic concurrent reads", async () => {
    const fixture = await createEvidenceFixture();
    const results = await Promise.all(
      Array.from({ length: 30 }, () => readIntegratorEvidence(fixture.input)),
    );
    expect(results.every((result) => result.status === "available")).toBe(true);
  });
});
