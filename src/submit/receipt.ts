import type { SignedReceiptDto } from "./types";

/**
 * Expected byte lengths for the receipt's three hex-encoded fields. Values
 * are pinned at the wire format defined by `entros_validation::receipts`
 * and verified on-chain in `entros_anchor::verify_mint_receipt`.
 *
 * Pubkey: Ed25519 public key (32B). Signature: Ed25519 signature (64B).
 * Message: `wallet_pubkey (32) || commitment_new (32) || validated_at i64 LE (8) = 72B`.
 */
const PUBKEY_BYTES = 32;
const SIGNATURE_BYTES = 64;
const MESSAGE_BYTES = 72;

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
 * no-receipt mint flow (Phase 3 on-chain check logs and proceeds).
 */
export interface DecodedReceipt {
  publicKey: Uint8Array;
  signature: Uint8Array;
  message: Uint8Array;
}

/**
 * Decode a `SignedReceiptDto` from hex strings into raw bytes. Returns `null`
 * if any field is malformed — callers should skip Ed25519 ix construction in
 * that case rather than building an ix the on-chain parser will reject.
 */
export function decodeSignedReceipt(receipt: SignedReceiptDto): DecodedReceipt | null {
  const publicKey = hexToBytes(receipt.validator_pubkey_hex, PUBKEY_BYTES);
  const signature = hexToBytes(receipt.signature_hex, SIGNATURE_BYTES);
  const message = hexToBytes(receipt.message_hex, MESSAGE_BYTES);
  if (!publicKey || !signature || !message) return null;
  return { publicKey, signature, message };
}

/**
 * Build the `Ed25519Program::verify` instruction that binds a validator-signed
 * mint receipt to the immediately-following `mint_anchor` instruction
 * (master-list #146 Phase 4).
 *
 * Returns `null` if the receipt fails to decode — caller should fall back to
 * sending `mint_anchor` without an Ed25519 prefix. Phase 3's on-chain check
 * is log-only, so the fallback still works on the deployed program; once
 * Phase 5 enforcement flips, missing receipts hard-fail and the SDK's no-op
 * fallback becomes a deliberate "no-receipt" path that mint_anchor rejects.
 *
 * Web3.js's `Ed25519Program.createInstructionWithPublicKey` defaults the
 * three `*_instruction_index` fields to `0xFFFF`, which is the exact
 * "current instruction" sentinel the on-chain parser pins to. Cross-ix
 * substitution attacks are closed by that sentinel — we never build a
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
