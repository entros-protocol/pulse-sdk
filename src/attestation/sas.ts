/* eslint-disable @typescript-eslint/no-explicit-any */
import { PROGRAM_IDS, SAS_CONFIG } from "../config";

const IDENTITY_DISCRIMINATOR = [156, 32, 87, 93, 52, 155, 248, 207] as const;
const IDENTITY_TRUST_SCORE_OFFSET = 60;
const IDENTITY_LAST_RESET_TIMESTAMP_OFFSET = 543;
const IDENTITY_MIN_SIZE = 62;
const IDENTITY_RESET_SIZE = 551;

/** Decoded Entros attestation from the Solana Attestation Service */
export interface EntrosAttestation {
  isHuman: boolean;
  trustScore: number;
  verifiedAt: number;
  mode: string;
  expired: boolean;
}

/**
 * Check if a wallet has a valid Entros attestation via SAS.
 *
 * Derives the attestation PDA and IdentityState PDA, fetches both accounts in a
 * single RPC batch, deserializes the attestation data, cross-checks against
 * IdentityState (`trustScore` clamp and post-reset invalidation), and checks expiry.
 *
 * @param walletAddress - Base58 Solana wallet address
 * @param connection - Solana web3.js Connection instance
 * @returns Decoded attestation or null if none exists
 */
export async function verifyEntrosAttestation(
  walletAddress: string,
  connection: any
): Promise<EntrosAttestation | null> {
  try {
    const { PublicKey } = await import("@solana/web3.js");

    const sasProgramId = new PublicKey(SAS_CONFIG.programId);
    const anchorProgramId = new PublicKey(PROGRAM_IDS.entrosAnchor);
    const credentialPda = new PublicKey(SAS_CONFIG.entrosCredentialPda);
    const schemaPda = new PublicKey(SAS_CONFIG.entrosSchemaPda);
    const userWallet = new PublicKey(walletAddress);

    // Derive attestation PDA: ["attestation", credential, schema, nonce(wallet)]
    const [attestationPda] = PublicKey.findProgramAddressSync(
      [
        new TextEncoder().encode("attestation"),
        credentialPda.toBuffer(),
        schemaPda.toBuffer(),
        userWallet.toBuffer(),
      ],
      sasProgramId
    );
    // Derive IdentityState PDA: ["identity", wallet]
    const [identityPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("identity"), userWallet.toBuffer()],
      anchorProgramId
    );

    const [accountInfo, identityInfo] =
      typeof connection.getMultipleAccountsInfo === "function"
        ? await connection.getMultipleAccountsInfo([attestationPda, identityPda])
        : await Promise.all([
            connection.getAccountInfo(attestationPda),
            connection.getAccountInfo(identityPda),
          ]);
    if (!accountInfo || !identityInfo) return null;

    const sas = deserializeSasAttestation(new Uint8Array(accountInfo.data));
    if (!sas) return null;

    const identity = deserializeIdentityStateCrossCheck(
      new Uint8Array(identityInfo.data)
    );
    if (!identity) return null;

    return {
      ...sas,
      trustScore: Math.min(sas.trustScore, identity.trustScore),
      expired: sas.expired || identity.lastResetTimestamp > sas.verifiedAt,
    };
  } catch {
    return null;
  }
}

function deserializeIdentityStateCrossCheck(
  raw: Uint8Array
): { trustScore: number; lastResetTimestamp: number } | null {
  if (raw.length < IDENTITY_MIN_SIZE) return null;
  for (let i = 0; i < IDENTITY_DISCRIMINATOR.length; i++) {
    if (raw[i] !== IDENTITY_DISCRIMINATOR[i]) return null;
  }
  const trustScore = readU16LE(raw, IDENTITY_TRUST_SCORE_OFFSET);
  const lastResetTimestamp =
    raw.length >= IDENTITY_RESET_SIZE
      ? readI64LE(raw, IDENTITY_LAST_RESET_TIMESTAMP_OFFSET)
      : 0;
  return { trustScore, lastResetTimestamp };
}

/** Read a u16 from a Uint8Array at the given offset (little-endian) */
function readU16LE(data: Uint8Array, offset: number): number {
  return data[offset]! | (data[offset + 1]! << 8);
}

/** Read a u32 from a Uint8Array at the given offset (little-endian) */
function readU32LE(data: Uint8Array, offset: number): number {
  return (
    data[offset]! |
    (data[offset + 1]! << 8) |
    (data[offset + 2]! << 16) |
    ((data[offset + 3]! << 24) >>> 0)
  );
}

/** Read an i64 from a Uint8Array at the given offset (little-endian, safe for JS number range) */
function readI64LE(data: Uint8Array, offset: number): number {
  const low = readU32LE(data, offset);
  const high = readU32LE(data, offset + 4);
  // Convert high word to signed for proper negative handling
  const signedHigh = high > 0x7FFFFFFF ? high - 0x100000000 : high;
  return signedHigh * 0x100000000 + low;
}

/**
 * Deserialize a SAS Attestation account into EntrosAttestation fields.
 *
 * SAS Attestation account layout (borsh):
 *   1 byte:  discriminator (u8)
 *  32 bytes: nonce (Pubkey)
 *  32 bytes: credential (Pubkey)
 *  32 bytes: schema (Pubkey)
 *   4 bytes: data length (u32 LE) + N bytes: data (Vec<u8>)
 *  32 bytes: signer (Pubkey)
 *   8 bytes: expiry (i64 LE)
 *  32 bytes: token_account (Pubkey)
 *
 * Entros attestation data layout (inside the data Vec):
 *   1 byte:  isHuman (bool)
 *   2 bytes: trustScore (u16 LE)
 *   8 bytes: verifiedAt (i64 LE)
 *   4 bytes: mode length (u32 LE) + N bytes: mode (UTF-8 string)
 */
function deserializeSasAttestation(raw: Uint8Array): EntrosAttestation | null {
  // Minimum account size: 1 + 32 + 32 + 32 + 4 + 0 + 32 + 8 + 32 = 173 bytes
  if (raw.length < 173) return null;

  let offset = 0;

  // Skip discriminator (1 byte)
  offset += 1;

  // Skip nonce, credential, schema (32 + 32 + 32 = 96 bytes)
  offset += 96;

  // Read data Vec<u8>: 4-byte LE length prefix
  const dataLen = readU32LE(raw, offset);
  offset += 4;

  if (raw.length < offset + dataLen + 32 + 8 + 32) return null;

  const attestationData = raw.slice(offset, offset + dataLen);
  offset += dataLen;

  // Skip signer (32 bytes)
  offset += 32;

  // Read expiry (i64 LE)
  const expiry = readI64LE(raw, offset);

  // Parse Entros attestation data: [bool, u16, i64, string]
  if (attestationData.length < 11) return null;

  let dataOffset = 0;

  // isHuman: bool (1 byte)
  const isHuman = attestationData[dataOffset] === 1;
  dataOffset += 1;

  // trustScore: u16 LE (2 bytes)
  const trustScore = readU16LE(attestationData, dataOffset);
  dataOffset += 2;

  // verifiedAt: i64 LE (8 bytes)
  const verifiedAt = readI64LE(attestationData, dataOffset);
  dataOffset += 8;

  // mode: string (4-byte LE length + UTF-8)
  let mode = "unknown";
  if (dataOffset + 4 <= attestationData.length) {
    const modeLen = readU32LE(attestationData, dataOffset);
    dataOffset += 4;
    if (dataOffset + modeLen <= attestationData.length) {
      mode = new TextDecoder().decode(
        attestationData.slice(dataOffset, dataOffset + modeLen)
      );
    }
  }

  const now = Math.floor(Date.now() / 1000);
  const expired = expiry > 0 && now >= expiry;

  return { isHuman, trustScore, verifiedAt, mode, expired };
}
