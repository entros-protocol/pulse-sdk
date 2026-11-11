import type { PublicKey } from "@solana/web3.js";

/**
 * The Entros Anchor program writes a migration record at seeds
 * `["migration", wallet]` each time an identity moves to that wallet. A
 * migrated identity keeps its verification times, so anything issued to the
 * wallet before the identity arrived belongs to whatever the wallet held then.
 */
const MIGRATION_RECORD_DISCRIMINATOR = [16, 36, 116, 243, 192, 61, 40, 1] as const;
/** Bytes of a record through `previous_wallet`. A later layout may add fields after it. */
const MIGRATION_RECORD_SIZE = 57;
const MIGRATED_AT_OFFSET = 8;

export type MigrationRecordRead =
  | { status: "none" }
  | { status: "migrated"; migratedAt: number }
  | { status: "invalid" };

/** The account fields the parser reads. */
export interface MigrationRecordAccount {
  data: Uint8Array;
  executable: boolean;
  owner: { equals(other: PublicKey): boolean };
}

/** The seeds of a wallet's migration record, under the Entros Anchor program. */
export function migrationRecordSeeds(wallet: PublicKey): Uint8Array[] {
  return [new TextEncoder().encode("migration"), wallet.toBytes()];
}

/**
 * Reads the migration record at a wallet's migration address. An absent
 * account, or one another program owns, means the identity never migrated to
 * the wallet: anyone can send lamports to the address. An account the program
 * owns that does not hold a record is invalid.
 */
export function readMigrationRecord(
  account: MigrationRecordAccount | null | undefined,
  anchorProgram: PublicKey,
): MigrationRecordRead {
  if (!account || !account.owner.equals(anchorProgram)) return { status: "none" };
  const data = account.data;
  if (
    account.executable ||
    data.length < MIGRATION_RECORD_SIZE ||
    !MIGRATION_RECORD_DISCRIMINATOR.every((byte, index) => data[index] === byte)
  ) {
    return { status: "invalid" };
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const migratedAt = Number(view.getBigInt64(MIGRATED_AT_OFFSET, true));
  if (!Number.isSafeInteger(migratedAt) || migratedAt < 1) return { status: "invalid" };
  return { status: "migrated", migratedAt };
}
