import type { ConfirmedSignatureInfo, Connection, ParsedTransactionWithMeta } from "@solana/web3.js";
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
