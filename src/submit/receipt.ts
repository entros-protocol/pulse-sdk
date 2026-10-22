import { sha256 } from "@noble/hashes/sha2.js";
import type { SignedReceiptDto } from "./types";

/**
 * Expected byte lengths for the receipt's three hex-encoded fields. Values
 * are pinned at the wire format defined by `entros_validation::receipts`
 * and verified on-chain by the anchor program's receipt parser.
 *
 * Pubkey: Ed25519 public key (32B). Signature: Ed25519 signature (64B).
 * A version 2 message holds the domain (28), purpose (1), projection version
 * (2), wallet (32), commitment (32), and validation time (8). A version 3
 * message, issued for a paired session, keeps those offsets under its own
 * domain and appends the session's final digest (32) and its assurance
 * tier (1). A version 4 message, issued for an update, carries the same
 * 136-byte preimage layout under its own domain while signing the 32-byte
 * SHA-256 digest of that preimage so the Ed25519 instruction stays compact.
 */
const PUBKEY_BYTES = 32;
const SIGNATURE_BYTES = 64;
const V2_MESSAGE_BYTES = 103;
const V3_MESSAGE_BYTES = 136;
const RECEIPT_DOMAIN_V2 = new TextEncoder().encode("entros-validator-receipt-v2\0");
const RECEIPT_DOMAIN_V3 = new TextEncoder().encode("entros-validator-receipt-v3\0");
const RECEIPT_DOMAIN_V4 = new TextEncoder().encode("entros-validator-receipt-v4\0");
/** 0 open, 1 bound, 2 attested. */
const MAX_ASSURANCE_TIER = 2;

export type ReceiptPurpose = 1 | 2 | 3 | 4;
export type ReceiptVersion = 2 | 3 | 4;

export interface ReceiptBinding {
  purpose: ReceiptPurpose;
  projectionVersion: number;
  wallet: Uint8Array;
  commitment: Uint8Array;
  /**
   * The paired session this receipt must have consumed. When present, only a
   * version 3 or version 4 receipt carrying this digest matches.
   */
  finalDigest?: Uint8Array;
}

/**
 * Lowercase hex encoding without `0x` prefix. Matches the validator's
 * `hex::encode` output exactly (lowercase, no separators) so the receipt's
 * `commitment_new_hex` round-trips byte-identical between SDK and validator.
 */
export function bytesToHex(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i += 1) {
    out += (bytes[i] ?? 0).toString(16).padStart(2, "0");
  }
  return out;
}

/**
 * Decode a hex string into a Uint8Array of the expected byte length. Returns
 * `null` on malformed input (odd length, non-lowercase-hex characters, wrong
 * length). Permissive about a leading `0x` because some integrations may
 * strip or preserve it inconsistently. Strict on case so a future validator
 * regression that emits uppercase hex surfaces immediately rather than
 * silently accepting drift from the wire-format contract (Rust `hex::encode`
 * is canonically lowercase).
 */
function hexToBytes(hex: string, expectedLen: number): Uint8Array | null {
  const trimmed = hex.startsWith("0x") || hex.startsWith("0X") ? hex.slice(2) : hex;
  if (trimmed.length !== expectedLen * 2) return null;
  if (!/^[0-9a-f]+$/.test(trimmed)) return null;
  const out = new Uint8Array(expectedLen);
  for (let i = 0; i < expectedLen; i += 1) {
    out[i] = parseInt(trimmed.substr(i * 2, 2), 16);
  }
  return out;
}

/**
 * Decoded byte form of a `SignedReceiptDto`. `null` slots indicate the
 * caller should treat the receipt as unusable and fall back to the
 * no-receipt mint flow — which `mint_anchor` rejects whenever the protocol's
 * `validator_pubkey` is configured (so the fallback only mints on a
 * pre-migration / unconfigured program).
 */
export interface DecodedReceipt {
  publicKey: Uint8Array;
  signature: Uint8Array;
  message: Uint8Array;
  version: ReceiptVersion;
  /** Validation Unix timestamp (seconds) from bytes 95..103 (little-endian i64). */
  validatedAt: bigint;
  /** The paired session's final digest. Version 3 and 4 only. */
  finalDigest: Uint8Array | null;
  /** Non-zero session digest as a 32-byte number array for `UpdateReceiptArgs`, or `null` when all zeros. */
  sessionFinalDigest: number[] | null;
  /** 0 open, 1 bound, 2 attested. Version 3 and 4 only. */
  assuranceTier: number | null;
}

/**
 * Decode a `SignedReceiptDto` from hex strings into raw bytes. Returns `null`
 * if any field is malformed, including a domain that does not match the
 * message length or an unknown assurance tier. Callers should skip Ed25519 ix
 * construction in that case rather than building an ix the on-chain parser
 * will reject.
 */
export function decodeSignedReceipt(receipt: SignedReceiptDto): DecodedReceipt | null {
  const publicKey = hexToBytes(receipt.validator_pubkey_hex, PUBKEY_BYTES);
  const signature = hexToBytes(receipt.signature_hex, SIGNATURE_BYTES);
  const messageHexLength = receipt.message_hex.replace(/^0x/i, "").length;
  if (!publicKey || !signature) return null;
  if (messageHexLength === V2_MESSAGE_BYTES * 2) {
    const message = hexToBytes(receipt.message_hex, V2_MESSAGE_BYTES);
    if (!message || !equalBytes(message.subarray(0, 28), RECEIPT_DOMAIN_V2)) return null;
    const purpose = message[28] ?? 0;
    if (purpose < 1 || purpose > 3) return null;
    const view = new DataView(message.buffer, message.byteOffset, message.byteLength);
    return {
      publicKey,
      signature,
      message,
      version: 2,
      validatedAt: view.getBigInt64(95, true),
      finalDigest: null,
      sessionFinalDigest: null,
      assuranceTier: null,
    };
  }
  if (messageHexLength !== V3_MESSAGE_BYTES * 2) return null;
  const message = hexToBytes(receipt.message_hex, V3_MESSAGE_BYTES);
  if (!message) return null;
  const domain = message.subarray(0, 28);
  const version: ReceiptVersion | null = equalBytes(domain, RECEIPT_DOMAIN_V3)
    ? 3
    : equalBytes(domain, RECEIPT_DOMAIN_V4)
      ? 4
      : null;
  if (version === null) return null;
  const purpose = message[28] ?? 0;
  if (version === 3 && (purpose < 1 || purpose > 3)) return null;
  if (version === 4 && purpose !== 4) return null;
  const assuranceTier = message[135] ?? MAX_ASSURANCE_TIER + 1;
  if (assuranceTier > MAX_ASSURANCE_TIER) return null;
  const finalDigest = message.slice(103, 135);
  const hasNonZeroDigest = finalDigest.some((byte) => byte !== 0);
  const view = new DataView(message.buffer, message.byteOffset, message.byteLength);
  return {
    publicKey,
    signature,
    message,
    version,
    validatedAt: view.getBigInt64(95, true),
    finalDigest,
    sessionFinalDigest: hasNonZeroDigest ? Array.from(finalDigest) : null,
    assuranceTier,
  };
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
}

/** Check the signed message fields before a wallet submits the transition. */
export function receiptMatchesBinding(
  receipt: SignedReceiptDto,
  binding: ReceiptBinding,
): boolean {
  const decoded = decodeSignedReceipt(receipt);
  if (!decoded) return false;
  if (binding.purpose === 4 && decoded.version !== 4) return false;
  if (binding.purpose !== 4 && decoded.version === 4) return false;
  const { message } = decoded;
  const view = new DataView(message.buffer, message.byteOffset, message.byteLength);
  if (
    binding.finalDigest &&
    !(decoded.finalDigest && equalBytes(decoded.finalDigest, binding.finalDigest))
  ) {
    return false;
  }
  return (
    message[28] === binding.purpose &&
    view.getUint16(29, true) === binding.projectionVersion &&
    equalBytes(message.subarray(31, 63), binding.wallet) &&
    equalBytes(message.subarray(63, 95), binding.commitment)
  );
}

/**
 * Build the Ed25519 verification instruction for a validator receipt.
 *
 * Versions 2 and 3 verify the raw receipt message bytes (103B or 136B).
 * Version 4 (update receipts) verifies the 32-byte SHA-256 digest of the
 * 136-byte preimage so the instruction fits inside the 1,232-byte Solana
 * legacy transaction wire limit alongside the Groth16 proof.
 */
export async function buildEd25519ReceiptIx(
  receipt: SignedReceiptDto,
): Promise<import("@solana/web3.js").TransactionInstruction | null> {
  const decoded = decodeSignedReceipt(receipt);
  if (!decoded) return null;

  const { Ed25519Program } = await import("@solana/web3.js");
  return Ed25519Program.createInstructionWithPublicKey({
    publicKey: decoded.publicKey,
    message: decoded.version === 4 ? sha256(decoded.message) : decoded.message,
    signature: decoded.signature,
  });
}
