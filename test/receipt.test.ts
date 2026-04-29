import { describe, it, expect } from "vitest";
import {
  buildEd25519ReceiptIx,
  bytesToHex,
  decodeSignedReceipt,
} from "../src/submit/receipt";
import type { SignedReceiptDto } from "../src/submit/types";

// 32 bytes = 64 hex chars
const VALIDATOR_PUBKEY_HEX = "8c".repeat(32);
// 64 bytes = 128 hex chars
const SIGNATURE_HEX = "ab".repeat(64);
const MESSAGE_HEX =
  "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" + // wallet (32B)
  "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" + // commitment (32B)
  "0102030405060708"; // validated_at i64 LE (8B)

const VALID_RECEIPT: SignedReceiptDto = {
  validator_pubkey_hex: VALIDATOR_PUBKEY_HEX,
  signature_hex: SIGNATURE_HEX,
  message_hex: MESSAGE_HEX,
};

describe("bytesToHex", () => {
  it("encodes a typical commitment without prefix", () => {
    const bytes = new Uint8Array(32);
    for (let i = 0; i < 32; i += 1) bytes[i] = i;
    expect(bytesToHex(bytes)).toBe(
      "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
    );
  });

  it("pads single-digit bytes", () => {
    expect(bytesToHex(new Uint8Array([0, 1, 15, 16, 255]))).toBe("00010f10ff");
  });

  it("returns empty string for empty input", () => {
    expect(bytesToHex(new Uint8Array(0))).toBe("");
  });
});

describe("decodeSignedReceipt", () => {
  it("decodes a well-formed receipt to raw byte arrays", () => {
    const decoded = decodeSignedReceipt(VALID_RECEIPT);
    expect(decoded).not.toBeNull();
    expect(decoded!.publicKey).toHaveLength(32);
    expect(decoded!.signature).toHaveLength(64);
    expect(decoded!.message).toHaveLength(72);
  });

  it("accepts hex with leading 0x prefix on any field", () => {
    const decoded = decodeSignedReceipt({
      validator_pubkey_hex: `0x${VALIDATOR_PUBKEY_HEX}`,
      signature_hex: `0X${SIGNATURE_HEX}`,
      message_hex: MESSAGE_HEX,
    });
    expect(decoded).not.toBeNull();
  });

  it("rejects pubkey of wrong byte length", () => {
    expect(
      decodeSignedReceipt({
        ...VALID_RECEIPT,
        validator_pubkey_hex: "ab".repeat(31),
      })
    ).toBeNull();
  });

  it("rejects signature of wrong byte length", () => {
    expect(
      decodeSignedReceipt({
        ...VALID_RECEIPT,
        signature_hex: "ab".repeat(63),
      })
    ).toBeNull();
  });

  it("rejects message of wrong byte length", () => {
    expect(
      decodeSignedReceipt({
        ...VALID_RECEIPT,
        message_hex: "ab".repeat(71),
      })
    ).toBeNull();
  });

  it("rejects non-hex characters", () => {
    expect(
      decodeSignedReceipt({
        ...VALID_RECEIPT,
        validator_pubkey_hex: "zz".repeat(32),
      })
    ).toBeNull();
  });

  it("rejects odd-length hex (caught by length check, not character regex)", () => {
    expect(
      decodeSignedReceipt({
        ...VALID_RECEIPT,
        validator_pubkey_hex: VALIDATOR_PUBKEY_HEX.slice(0, 63),
      })
    ).toBeNull();
  });

  it("rejects uppercase hex to pin the wire-format contract", () => {
    // Rust `hex::encode` (used by the validator) is canonically lowercase.
    // Accepting uppercase here would mask a future validator regression
    // that drifts to mixed/upper case. Surface it as a decode failure so
    // the SDK fails fast instead of silently bundling a possibly-incorrect
    // receipt encoding.
    expect(
      decodeSignedReceipt({
        ...VALID_RECEIPT,
        validator_pubkey_hex: VALIDATOR_PUBKEY_HEX.toUpperCase(),
      })
    ).toBeNull();
    expect(
      decodeSignedReceipt({
        ...VALID_RECEIPT,
        signature_hex: SIGNATURE_HEX.toUpperCase(),
      })
    ).toBeNull();
    expect(
      decodeSignedReceipt({
        ...VALID_RECEIPT,
        message_hex: MESSAGE_HEX.toUpperCase(),
      })
    ).toBeNull();
  });
});

describe("buildEd25519ReceiptIx", () => {
  it("constructs an Ed25519Program::verify instruction with correct programId", async () => {
    const ix = await buildEd25519ReceiptIx(VALID_RECEIPT);
    expect(ix).not.toBeNull();
    expect(ix!.programId.toBase58()).toBe(
      "Ed25519SigVerify111111111111111111111111111"
    );
    // Ed25519Program::verify is a precompile — no AccountMeta entries.
    expect(ix!.keys).toHaveLength(0);
  });

  it("returns null for malformed receipts", async () => {
    const ix = await buildEd25519ReceiptIx({
      ...VALID_RECEIPT,
      message_hex: "ab".repeat(50),
    });
    expect(ix).toBeNull();
  });

  it("encodes data with single-signature header at offsets matching the on-chain parser", async () => {
    const ix = await buildEd25519ReceiptIx(VALID_RECEIPT);
    expect(ix).not.toBeNull();
    const data = ix!.data;

    // Layout (matches entros_anchor::lib::ED25519_*_OFFSET constants):
    //   [0]    num_signatures (u8) — must be 1
    //   [1]    padding (u8)
    //   [2..4] signature_offset (u16 LE)
    //   [4..6] signature_instruction_index (u16 LE) — must be 0xFFFF
    //   [6..8] public_key_offset (u16 LE)
    //   [8..10] public_key_instruction_index (u16 LE) — must be 0xFFFF
    //   [10..12] message_data_offset (u16 LE)
    //   [12..14] message_data_size (u16 LE) — must be 72
    //   [14..16] message_instruction_index (u16 LE) — must be 0xFFFF
    expect(data[0]).toBe(1); // num_signatures
    expect(data.readUInt16LE(4)).toBe(0xffff); // signature_instruction_index
    expect(data.readUInt16LE(8)).toBe(0xffff); // public_key_instruction_index
    expect(data.readUInt16LE(14)).toBe(0xffff); // message_instruction_index
    expect(data.readUInt16LE(12)).toBe(72); // message_data_size
  });

  it("places the message bytes verbatim at message_data_offset", async () => {
    const ix = await buildEd25519ReceiptIx(VALID_RECEIPT);
    expect(ix).not.toBeNull();
    const data = ix!.data;
    const messageOffset = data.readUInt16LE(10);
    const messageSize = data.readUInt16LE(12);
    const writtenMessage = data.subarray(messageOffset, messageOffset + messageSize);

    const decoded = decodeSignedReceipt(VALID_RECEIPT);
    expect(writtenMessage.equals(Buffer.from(decoded!.message))).toBe(true);
  });

  it("places the validator pubkey verbatim at public_key_offset", async () => {
    const ix = await buildEd25519ReceiptIx(VALID_RECEIPT);
    expect(ix).not.toBeNull();
    const data = ix!.data;
    const pubkeyOffset = data.readUInt16LE(6);
    const writtenPubkey = data.subarray(pubkeyOffset, pubkeyOffset + 32);

    const decoded = decodeSignedReceipt(VALID_RECEIPT);
    expect(writtenPubkey.equals(Buffer.from(decoded!.publicKey))).toBe(true);
  });
});
