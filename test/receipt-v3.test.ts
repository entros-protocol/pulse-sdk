import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { decodeSignedReceipt, receiptMatchesBinding } from "../src/submit/receipt";
import type { SignedReceiptDto } from "../src/submit/types";

const VECTORS_SHA256 = "a1926388062f2dad25fa476b9ab2c3cb194ba0f5d042dccd14dc343ba94d31eb";
const vectorsText = readFileSync(
  resolve(__dirname, "fixtures/paired-round-vectors.json"),
  "utf8",
);
const vectors = JSON.parse(vectorsText);
const receipts = vectors.receipts;

function dto(messageHex: string): SignedReceiptDto {
  return {
    validator_pubkey_hex: "8c".repeat(32),
    signature_hex: "ab".repeat(64),
    message_hex: messageHex,
  };
}

const hex = (value: string) => Uint8Array.from(Buffer.from(value, "hex"));

describe("version 3 receipts", () => {
  it("reads the pinned vector generation", () => {
    expect(createHash("sha256").update(vectorsText).digest("hex")).toBe(VECTORS_SHA256);
  });

  it("decodes every v3 vector with its session and tier", () => {
    for (const entry of receipts.v3) {
      const decoded = decodeSignedReceipt(dto(entry.messageHex));
      expect(decoded?.version).toBe(3);
      expect(decoded?.message).toHaveLength(136);
      expect(decoded?.assuranceTier).toBe(entry.assuranceTier);
      expect(Buffer.from(decoded!.finalDigest!).toString("hex")).toBe(
        receipts.finalDigestHex,
      );
    }
  });

  it("still decodes a v2 receipt without a session", () => {
    const decoded = decodeSignedReceipt(dto(receipts.v2.messageHex));
    expect(decoded?.version).toBe(2);
    expect(decoded?.finalDigest).toBeNull();
    expect(decoded?.assuranceTier).toBeNull();
  });

  it("refuses every malformed vector", () => {
    for (const entry of receipts.invalid) {
      expect(decodeSignedReceipt(dto(entry.messageHex)), entry.name).toBeNull();
    }
  });

  it("binds a paired session's digest only through a v3 receipt", () => {
    const binding = {
      purpose: 1 as const,
      projectionVersion: receipts.projectionVersion,
      wallet: hex(receipts.walletHex),
      commitment: hex(receipts.commitmentHex),
    };
    const mintV3 = receipts.v3.find(
      (entry: { purpose: number; assuranceTier: number }) =>
        entry.purpose === 1 && entry.assuranceTier === 2,
    );
    const finalDigest = hex(receipts.finalDigestHex);

    expect(receiptMatchesBinding(dto(mintV3.messageHex), { ...binding, finalDigest })).toBe(
      true,
    );
    expect(
      receiptMatchesBinding(dto(mintV3.messageHex), {
        ...binding,
        finalDigest: new Uint8Array(32),
      }),
    ).toBe(false);
    expect(
      receiptMatchesBinding(dto(receipts.v2.messageHex), { ...binding, finalDigest }),
    ).toBe(false);
    expect(receiptMatchesBinding(dto(receipts.v2.messageHex), binding)).toBe(true);
    expect(
      receiptMatchesBinding(dto(mintV3.messageHex), { ...binding, purpose: 3 }),
    ).toBe(false);
  });
});
