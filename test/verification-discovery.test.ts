import type { ConfirmedSignatureInfo, Connection, ParsedTransactionWithMeta } from "@solana/web3.js";
import { utils } from "@coral-xyz/anchor";
import { describe, expect, it } from "vitest";

import { findLatestVerificationTransaction } from "../src/identity/integrator";
import { createEvidenceFixture } from "./fixtures/integrator-evidence";

type DiscoveryConnection = Pick<Connection, "getSignaturesForAddress" | "getParsedTransaction">;

function info(signature: string, err: ConfirmedSignatureInfo["err"] = null): ConfirmedSignatureInfo {
  return { signature, slot: 100, err, memo: null, blockTime: 1, confirmationStatus: "confirmed" };
}

async function setup(entries: ConfirmedSignatureInfo[], transactions: Record<string, ParsedTransactionWithMeta | null>) {
  const fixture = await createEvidenceFixture();
  const parsed: string[] = [];
  const limits: (number | undefined)[] = [];
  const connection: DiscoveryConnection = {
    async getSignaturesForAddress(address, options) {
      expect(address.equals(fixture.identityPda)).toBe(true);
      limits.push(options?.limit);
      return entries;
    },
    async getParsedTransaction(signature) {
      parsed.push(signature as string);
      return transactions[signature as string] ?? null;
    },
  };
  return { fixture, connection, parsed, limits };
}

describe("verification transaction discovery", () => {
  it("returns the newest qualifying transaction and skips failed or unrelated ones", async () => {
    const fixture = await createEvidenceFixture();
    const unrelated: ParsedTransactionWithMeta = {
      ...fixture.transaction,
      transaction: {
        ...fixture.transaction.transaction,
        signatures: ["unrelated"],
        message: { ...fixture.transaction.transaction.message, instructions: [] },
      },
    };
    const qualifying = fixture.input.transactionSignature;
    const { connection, parsed } = await setup(
      [info("failed", { InstructionError: [0, "Custom"] } as ConfirmedSignatureInfo["err"]), info("unrelated"), info("missing"), info(qualifying)],
      { unrelated, [qualifying]: fixture.transaction },
    );
    expect(
      await findLatestVerificationTransaction({ walletPubkey: fixture.input.walletPubkey, connection }),
    ).toEqual({ status: "found", signature: qualifying });
    expect(parsed).toEqual(["unrelated", "missing", qualifying]);
  });

  it("reports none when nothing qualifies within the limit", async () => {
    const { fixture, connection, limits } = await setup([info("a"), info("b")], {});
    expect(
      await findLatestVerificationTransaction({ walletPubkey: fixture.input.walletPubkey, connection, limit: 2 }),
    ).toEqual({ status: "none" });
    expect(limits).toEqual([2]);
  });

  it("rejects bad input and reports RPC failure", async () => {
    const { fixture, connection } = await setup([], {});
    for (const limit of [0, 26, 1.5]) {
      expect(
        await findLatestVerificationTransaction({ walletPubkey: fixture.input.walletPubkey, connection, limit }),
      ).toEqual({ status: "invalid" });
    }
    expect(
      await findLatestVerificationTransaction({ walletPubkey: "1" + fixture.input.walletPubkey, connection }),
    ).toEqual({ status: "invalid" });
    const failing: DiscoveryConnection = {
      getSignaturesForAddress: async () => {
        throw new Error("Synthetic RPC failure");
      },
      getParsedTransaction: async () => null,
    };
    expect(
      await findLatestVerificationTransaction({ walletPubkey: fixture.input.walletPubkey, connection: failing }),
    ).toEqual({ status: "unavailable" });
  });
});

/** A history served newest first, in pages, as the RPC serves it. */
async function paged(entries: ConfirmedSignatureInfo[], transactions: Record<string, ParsedTransactionWithMeta | null>) {
  const fixture = await createEvidenceFixture();
  const parsed: string[] = [];
  const pages: { limit?: number; before?: string }[] = [];
  const connection: DiscoveryConnection = {
    async getSignaturesForAddress(address, options) {
      expect(address.equals(fixture.identityPda)).toBe(true);
      pages.push({ limit: options?.limit, before: options?.before });
      const start = options?.before ? entries.findIndex((entry) => entry.signature === options.before) + 1 : 0;
      return entries.slice(start, start + (options?.limit ?? 1_000));
    },
    async getParsedTransaction(signature) {
      parsed.push(signature as string);
      return transactions[signature as string] ?? null;
    },
  };
  return { fixture, connection, parsed, pages };
}

const at = (signature: string, blockTime: number | null, err: ConfirmedSignatureInfo["err"] = null): ConfirmedSignatureInfo => ({
  signature,
  slot: 100,
  err,
  memo: null,
  blockTime,
  confirmationStatus: "confirmed",
});

describe("verification transaction discovery from the verification time", () => {
  const VERIFIED_AT = 1_799_999_990;

  it("finds the verification behind more than a page of later transactions", async () => {
    const fixture = await createEvidenceFixture();
    const qualifying = fixture.input.transactionSignature;
    // Any party can send transactions that name the identity account. 1,500 of them follow the verification.
    const later = Array.from({ length: 1_500 }, (_, index) => at(`later-${index}`, VERIFIED_AT + 3 + index));
    const { connection, parsed, pages } = await paged(
      [...later, at(qualifying, VERIFIED_AT), at("older", VERIFIED_AT - 500)],
      { [qualifying]: fixture.transaction },
    );
    expect(
      await findLatestVerificationTransaction({
        walletPubkey: fixture.input.walletPubkey,
        connection,
        verifiedAt: VERIFIED_AT,
      }),
    ).toEqual({ status: "found", signature: qualifying });
    // Nothing outside the window of the verification time is fetched.
    expect(parsed).toEqual([qualifying]);
    expect(pages).toEqual([
      { limit: 1_000, before: undefined },
      { limit: 1_000, before: "later-999" },
    ]);
  });

  it("parses the window newest first and skips failed and undated entries", async () => {
    const fixture = await createEvidenceFixture();
    const qualifying = fixture.input.transactionSignature;
    const { connection, parsed } = await paged(
      [
        at("after-window", VERIFIED_AT + 3),
        at("in-window-unrelated", VERIFIED_AT + 2),
        at("in-window-failed", VERIFIED_AT + 1, { InstructionError: [0, "Custom"] } as ConfirmedSignatureInfo["err"]),
        at("undated", null),
        at(qualifying, VERIFIED_AT - 2),
      ],
      { [qualifying]: fixture.transaction },
    );
    expect(
      await findLatestVerificationTransaction({
        walletPubkey: fixture.input.walletPubkey,
        connection,
        verifiedAt: VERIFIED_AT,
      }),
    ).toEqual({ status: "found", signature: qualifying });
    expect(parsed).toEqual(["in-window-unrelated", qualifying]);
  });

  it("reports none once the history is older than the verification time", async () => {
    const older = Array.from({ length: 1_200 }, (_, index) => at(`older-${index}`, VERIFIED_AT - 3 - index));
    const { fixture, connection, parsed, pages } = await paged([at("later", VERIFIED_AT + 50), ...older], {});
    expect(
      await findLatestVerificationTransaction({
        walletPubkey: fixture.input.walletPubkey,
        connection,
        verifiedAt: VERIFIED_AT,
      }),
    ).toEqual({ status: "none" });
    expect(parsed).toEqual([]);
    expect(pages).toHaveLength(1);
  });

  it("reports none when the whole history holds no verification", async () => {
    const { fixture, connection } = await paged([at("later", VERIFIED_AT + 50)], {});
    expect(
      await findLatestVerificationTransaction({
        walletPubkey: fixture.input.walletPubkey,
        connection,
        verifiedAt: VERIFIED_AT,
      }),
    ).toEqual({ status: "none" });
  });

  it("reports the search as unfinished when it reaches its bounds", async () => {
    // More later transactions than the search reads.
    const flood = Array.from({ length: 5_001 }, (_, index) => at(`later-${index}`, VERIFIED_AT + 10));
    const far = await paged(flood, {});
    expect(
      await findLatestVerificationTransaction({
        walletPubkey: far.fixture.input.walletPubkey,
        connection: far.connection,
        verifiedAt: VERIFIED_AT,
      }),
    ).toEqual({ status: "unavailable" });
    expect(far.pages).toHaveLength(5);

    // More transactions inside the window than the search parses.
    const crowd = Array.from({ length: 51 }, (_, index) => at(`window-${index}`, VERIFIED_AT));
    const near = await paged(crowd, {});
    expect(
      await findLatestVerificationTransaction({
        walletPubkey: near.fixture.input.walletPubkey,
        connection: near.connection,
        verifiedAt: VERIFIED_AT,
      }),
    ).toEqual({ status: "unavailable" });
    expect(near.parsed).toHaveLength(50);
  });

  it("rejects a verification time that is not a positive whole number", async () => {
    const { fixture, connection } = await paged([], {});
    for (const verifiedAt of [0, -1, 1.5, Number.NaN]) {
      expect(
        await findLatestVerificationTransaction({ walletPubkey: fixture.input.walletPubkey, connection, verifiedAt }),
      ).toEqual({ status: "invalid" });
    }
  });

  describe("with candidate signatures", () => {
    const flood = () => Array.from({ length: 5_001 }, (_, index) => at(`later-${index}`, VERIFIED_AT + 10));

    it("returns a candidate that is the verification without searching the history", async () => {
      const fixture = await createEvidenceFixture();
      const qualifying = fixture.input.transactionSignature;
      // More later transactions than the search reads: the history cannot reach the verification.
      const { connection, parsed, pages } = await paged([...flood(), at(qualifying, VERIFIED_AT)], {
        [qualifying]: fixture.transaction,
      });
      expect(
        await findLatestVerificationTransaction({
          walletPubkey: fixture.input.walletPubkey,
          connection,
          verifiedAt: VERIFIED_AT,
          candidateSignatures: [qualifying],
        }),
      ).toEqual({ status: "found", signature: qualifying });
      expect(parsed).toEqual([qualifying]);
      expect(pages).toEqual([]);
    });

    it("skips candidates that are not the verification and falls back to the search", async () => {
      const fixture = await createEvidenceFixture();
      const qualifying = fixture.input.transactionSignature;
      const named = (fill: number) => utils.bytes.bs58.encode(Buffer.alloc(64, fill));
      // The verification's own transaction under another signature, at another time or failed.
      const variant = (
        signature: string,
        blockTime: number | null,
        err: unknown = null,
      ): ParsedTransactionWithMeta => ({
        ...fixture.transaction,
        blockTime,
        meta: { ...fixture.transaction.meta!, err: err as never },
        transaction: { ...fixture.transaction.transaction, signatures: [signature] },
      });
      const early = named(20);
      const late = named(21);
      const failed = named(22);
      const unrelated = named(23);
      const missing = named(24);
      const candidates: Record<string, ParsedTransactionWithMeta | null> = {
        [early]: variant(early, VERIFIED_AT - 3),
        [late]: variant(late, VERIFIED_AT + 3),
        [failed]: variant(failed, VERIFIED_AT, { InstructionError: [0, "Custom"] }),
        [unrelated]: {
          ...variant(unrelated, VERIFIED_AT),
          transaction: {
            ...fixture.transaction.transaction,
            signatures: [unrelated],
            message: { ...fixture.transaction.transaction.message, instructions: [] },
          },
        },
        [missing]: null,
      };
      // Inside the window the same transaction is the verification, which shows that
      // the window, and not the transaction, refused the early and late variants.
      expect(
        await findLatestVerificationTransaction({
          walletPubkey: fixture.input.walletPubkey,
          connection: (await paged([], { [late]: variant(late, VERIFIED_AT + 2) })).connection,
          verifiedAt: VERIFIED_AT,
          candidateSignatures: [late],
        }),
      ).toEqual({ status: "found", signature: late });

      const { connection, parsed, pages } = await paged([at(qualifying, VERIFIED_AT)], {
        ...candidates,
        [qualifying]: fixture.transaction,
      });
      expect(
        await findLatestVerificationTransaction({
          walletPubkey: fixture.input.walletPubkey,
          connection,
          verifiedAt: VERIFIED_AT,
          candidateSignatures: [early, late, failed, unrelated, missing],
        }),
      ).toEqual({ status: "found", signature: qualifying });
      expect(parsed).toEqual([early, late, failed, unrelated, missing, qualifying]);
      expect(pages).toHaveLength(1);
    });

    it("needs the verification time and well-formed signatures", async () => {
      const { fixture, connection } = await paged([], {});
      const signature = fixture.input.transactionSignature;
      for (const input of [
        { candidateSignatures: [signature] },
        { verifiedAt: VERIFIED_AT, candidateSignatures: ["not a signature"] },
        { verifiedAt: VERIFIED_AT, candidateSignatures: Array.from({ length: 11 }, () => signature) },
      ]) {
        expect(
          await findLatestVerificationTransaction({ walletPubkey: fixture.input.walletPubkey, connection, ...input }),
        ).toEqual({ status: "invalid" });
      }
    });
  });
});
