/* eslint-disable @typescript-eslint/no-explicit-any */
import type { PublicKeyInitData } from "@solana/web3.js";
import { PROGRAM_IDS, SAS_CONFIG } from "../config";
import { migrationRecordSeeds, readMigrationRecord } from "../identity/migration";

const IDENTITY_DISCRIMINATOR = [156, 32, 87, 93, 52, 155, 248, 207] as const;
const IDENTITY_CREATION_TIMESTAMP_OFFSET = 40;
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
 * Derives the attestation PDA, the IdentityState PDA and the wallet's migration
 * record PDA, fetches the three accounts in a single RPC batch, deserializes the
 * attestation data, cross-checks against IdentityState (`trustScore` clamp and
 * post-reset invalidation), and checks expiry.
 *
 * Returns null when the attestation was issued before the wallet's current
 * identity was created, or at or before that identity migrated to the wallet.
 * Such an attestation was issued for what the wallet held before.
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

    const [migrationPda] = PublicKey.findProgramAddressSync(
      migrationRecordSeeds(userWallet),
      anchorProgramId
    );

    const [accountInfo, identityInfo, migrationInfo] =
      typeof connection.getMultipleAccountsInfo === "function"
        ? await connection.getMultipleAccountsInfo([attestationPda, identityPda, migrationPda])
        : await Promise.all([
            connection.getAccountInfo(attestationPda),
            connection.getAccountInfo(identityPda),
            connection.getAccountInfo(migrationPda),
          ]);
    if (!accountInfo || !identityInfo) return null;
    // Each address is a PDA, so only its program can write data there. The owner
    // check keeps that true if an account is ever read from another source.
    const ownedBy = (
      account: { owner?: PublicKeyInitData },
      program: InstanceType<typeof PublicKey>,
    ): boolean => {
      try {
        return account.owner !== undefined && new PublicKey(account.owner).equals(program);
      } catch {
        return false;
      }
    };
    if (!ownedBy(accountInfo, sasProgramId) || !ownedBy(identityInfo, anchorProgramId)) {
      return null;
    }

    const sas = deserializeSasAttestation(new Uint8Array(accountInfo.data));
    if (!sas) return null;

    const identity = deserializeIdentityStateCrossCheck(
      new Uint8Array(identityInfo.data)
    );
    if (!identity) return null;

    // The wallet closed an earlier identity and created this one afterwards.
    if (sas.verifiedAt < identity.creationTimestamp) return null;
    // Anyone can send lamports to the migration address, so only a record the
    // program owns counts. A migration in the same second as the verification
    // counts as later, since the program clock cannot order them.
    const migration = readMigrationRecord(
      migrationInfo
        ? {
            data: new Uint8Array(migrationInfo.data),
            executable: migrationInfo.executable === true,
            owner: { equals: (program) => ownedBy(migrationInfo, program) },
          }
        : null,
      anchorProgramId
    );
    if (migration.status === "invalid") return null;
    if (migration.status === "migrated" && sas.verifiedAt <= migration.migratedAt) {
      return null;
    }

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
): { trustScore: number; lastResetTimestamp: number; creationTimestamp: number } | null {
  if (raw.length < IDENTITY_MIN_SIZE) return null;
  for (let i = 0; i < IDENTITY_DISCRIMINATOR.length; i++) {
    if (raw[i] !== IDENTITY_DISCRIMINATOR[i]) return null;
  }
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const creationTimestamp = readI64LE(view, IDENTITY_CREATION_TIMESTAMP_OFFSET);
  if (creationTimestamp === null) return null;
  const trustScore = view.getUint16(IDENTITY_TRUST_SCORE_OFFSET, true);
  const lastResetTimestamp =
    raw.length >= IDENTITY_RESET_SIZE
      ? readI64LE(view, IDENTITY_LAST_RESET_TIMESTAMP_OFFSET)
      : 0;
  if (lastResetTimestamp === null) return null;
  return { trustScore, lastResetTimestamp, creationTimestamp };
}

/**
 * Reads a little-endian i64 as a number, or null when the value is not a safe
 * integer. `DataView` reads the 64 bits as one value. Building it from two
 * 32-bit halves goes wrong once a half has its top bit set, which a time in
 * seconds does from 2038-01-19.
 */
function readI64LE(view: DataView, offset: number): number | null {
  const value = Number(view.getBigInt64(offset, true));
  return Number.isSafeInteger(value) ? value : null;
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
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);

  // Skip the discriminator (1 byte), then nonce, credential and schema (96 bytes).
  let offset = 1 + 96;

  // Read data Vec<u8>: 4-byte LE length prefix
  const dataLen = view.getUint32(offset, true);
  offset += 4;

  if (raw.length < offset + dataLen + 32 + 8 + 32) return null;

  const attestationData = raw.subarray(offset, offset + dataLen);
  offset += dataLen;

  // Skip signer (32 bytes)
  offset += 32;

  // Read expiry (i64 LE). Zero means no expiry. The issuer writes no negative value.
  const expiry = readI64LE(view, offset);
  if (expiry === null || expiry < 0) return null;

  // Parse Entros attestation data: [bool, u16, i64, string]
  if (attestationData.length < 11) return null;
  const data = new DataView(
    attestationData.buffer,
    attestationData.byteOffset,
    attestationData.byteLength,
  );

  let dataOffset = 0;

  // isHuman: bool (1 byte)
  const isHuman = attestationData[dataOffset] === 1;
  dataOffset += 1;

  // trustScore: u16 LE (2 bytes)
  const trustScore = data.getUint16(dataOffset, true);
  dataOffset += 2;

  // verifiedAt: i64 LE (8 bytes)
  const verifiedAt = readI64LE(data, dataOffset);
  if (verifiedAt === null) return null;
  dataOffset += 8;

  // mode: string (4-byte LE length + UTF-8)
  let mode = "unknown";
  if (dataOffset + 4 <= attestationData.length) {
    const modeLen = data.getUint32(dataOffset, true);
    dataOffset += 4;
    if (dataOffset + modeLen <= attestationData.length) {
      mode = new TextDecoder().decode(
        attestationData.subarray(dataOffset, dataOffset + modeLen)
      );
    }
  }

  const now = Math.floor(Date.now() / 1000);
  const expired = expiry > 0 && now >= expiry;

  return { isHuman, trustScore, verifiedAt, mode, expired };
}
