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
 * tier (1).
 */
const PUBKEY_BYTES = 32;
const SIGNATURE_BYTES = 64;
const V2_MESSAGE_BYTES = 103;
const V3_MESSAGE_BYTES = 136;
const RECEIPT_DOMAIN_V2 = new TextEncoder().encode("entros-validator-receipt-v2\0");
const RECEIPT_DOMAIN_V3 = new TextEncoder().encode("entros-validator-receipt-v3\0");
/** 0 open, 1 bound, 2 attested. */
const MAX_ASSURANCE_TIER = 2;

export type ReceiptPurpose = 1 | 2 | 3;
export type ReceiptVersion = 2 | 3;

export interface ReceiptBinding {
  purpose: ReceiptPurpose;
  projectionVersion: number;
  wallet: Uint8Array;
  commitment: Uint8Array;
  /**
   * The paired session this receipt must have consumed. When present, only a
   * version 3 receipt carrying this digest matches.
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
  /** The paired session's final digest. Version 3 only. */
  finalDigest: Uint8Array | null;
  /** 0 open, 1 bound, 2 attested. Version 3 only. */
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
  const version: ReceiptVersion | null =
    messageHexLength === V2_MESSAGE_BYTES * 2
      ? 2
      : messageHexLength === V3_MESSAGE_BYTES * 2
        ? 3
        : null;
  if (!publicKey || !signature || version === null) return null;
  const message = hexToBytes(
    receipt.message_hex,
    version === 2 ? V2_MESSAGE_BYTES : V3_MESSAGE_BYTES,
  );
  if (!message) return null;
  const domain = version === 2 ? RECEIPT_DOMAIN_V2 : RECEIPT_DOMAIN_V3;
  if (!equalBytes(message.subarray(0, 28), domain)) return null;
  if (version === 2) {
    return {
      publicKey,
      signature,
      message,
      version,
      finalDigest: null,
      assuranceTier: null,
    };
  }
  const assuranceTier = message[135] ?? MAX_ASSURANCE_TIER + 1;
  if (assuranceTier > MAX_ASSURANCE_TIER) return null;
  return {
    publicKey,
    signature,
    message,
    version,
    finalDigest: message.slice(103, 135),
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
 * Build the Ed25519 verification instruction for a mint or rebaseline receipt.
 *
 * Returns `null` if the receipt fails to decode. Callers must stop the
 * receipt-required transition before wallet submission.
 *
 * Web3.js's `Ed25519Program.createInstructionWithPublicKey` defaults the
 * three `*_instruction_index` fields to `0xFFFF`, which is the exact
 * "current instruction" sentinel the on-chain parser pins to. Cross-ix
 * substitution attacks are closed by that sentinel. We never build a
 * receipt that points at another ix's data.
 */
export async function buildEd25519ReceiptIx(
  receipt: SignedReceiptDto,
): Promise<import("@solana/web3.js").TransactionInstruction | null> {
  const decoded = decodeSignedReceipt(receipt);
  if (!decoded) return null;

  const { Ed25519Program } = await import("@solana/web3.js");
  return Ed25519Program.createInstructionWithPublicKey({
    publicKey: decoded.publicKey,
    message: decoded.message,
    signature: decoded.signature,
  });
}
