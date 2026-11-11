import { Keypair, PublicKey, SystemProgram } from "@solana/web3.js";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PROGRAM_IDS, SAS_CONFIG } from "../src/config";
import { verifyEntrosAttestation } from "../src/attestation/sas";
import { migrationRecordData } from "./fixtures/integrator-evidence";

function buildSasAttestationBuffer(opts: {
  trustScore: number;
  verifiedAt: bigint;
  expiry: bigint;
}): Buffer {
  const mode = Buffer.from("wallet-connected", "utf8");
  const dataLen = 1 + 2 + 8 + 4 + mode.length;
  const buf = Buffer.alloc(1 + 96 + 4 + dataLen + 32 + 8 + 32);
  buf[0] = 2;
  buf.writeUInt32LE(dataLen, 97);
  let dataOffset = 101;
  buf[dataOffset] = 1;
  dataOffset += 1;
  buf.writeUInt16LE(opts.trustScore, dataOffset);
  dataOffset += 2;
  buf.writeBigInt64LE(opts.verifiedAt, dataOffset);
  dataOffset += 8;
  buf.writeUInt32LE(mode.length, dataOffset);
  dataOffset += 4;
  mode.copy(buf, dataOffset);
  const expiryOffset = 101 + dataLen + 32;
  buf.writeBigInt64LE(opts.expiry, expiryOffset);
  return buf;
}

function buildIdentityStateBuffer(opts: {
  trustScore: number;
  lastResetTimestamp: bigint;
  creationTimestamp?: bigint;
}): Buffer {
  const buf = Buffer.alloc(593);
  Buffer.from([156, 32, 87, 93, 52, 155, 248, 207]).copy(buf, 0);
  buf.writeBigInt64LE(opts.creationTimestamp ?? 1_700_000_000n, 40);
  buf.writeUInt16LE(opts.trustScore, 60);
  buf.writeBigInt64LE(opts.lastResetTimestamp, 543);
  return buf;
}

const SAS_PROGRAM = new PublicKey(SAS_CONFIG.programId);
const ANCHOR_PROGRAM = new PublicKey(PROGRAM_IDS.entrosAnchor);

type Served = { data: Buffer; owner?: PublicKey; executable?: boolean } | null;

/** A connection that serves an attestation, an identity and, when given, a migration record. */
function accounts(sas: Served, identity: Served, migration?: Served) {
  return {
    getMultipleAccountsInfo: async (addresses: PublicKey[]) =>
      migration === undefined ? [sas, identity] : addresses.length === 3 ? [sas, identity, migration] : [],
  };
}

const liveIdentity = () => ({
  data: buildIdentityStateBuffer({ trustScore: 720, lastResetTimestamp: 0n }),
  owner: ANCHOR_PROGRAM,
});

afterEach(() => vi.useRealTimers());

describe("SAS configuration", () => {
  it("exports the reviewed Entros credential and schema", () => {
    expect(SAS_CONFIG).toEqual({
      programId: "22zoJMtdu4tQc2PzL74ZUT7FrwgB1Udec8DdW4yw4BdG",
      entrosCredentialPda: "AMBtabCgRFwGLjoZ21Z2LhSKJ6c47NckxUkMogJ3Lpuw",
      entrosSchemaPda: "5LNc7syFW7USPLveVLcNcjjY1xqS7QTXVjHZ7CQCbAMQ",
    });

    const addresses = Object.values(SAS_CONFIG).map((value) =>
      new PublicKey(value).toBase58(),
    );
    expect(new Set(addresses).size).toBe(addresses.length);
  });

  it("cross-checks SAS attestation against IdentityState trust score and reset timestamp", async () => {
    const wallet = Keypair.generate().publicKey.toBase58();
    const sasBuf = buildSasAttestationBuffer({
      trustScore: 720,
      verifiedAt: 1_780_000_000n,
      expiry: 4_000_000_000n,
    });
    const resetIdentityBuf = buildIdentityStateBuffer({
      trustScore: 0,
      lastResetTimestamp: 1_780_000_100n,
    });
    const decoded = await verifyEntrosAttestation(
      wallet,
      accounts(
        { data: sasBuf, owner: SAS_PROGRAM },
        { data: resetIdentityBuf, owner: ANCHOR_PROGRAM },
      ),
    );
    expect(decoded).toEqual({
      isHuman: true,
      trustScore: 0,
      verifiedAt: 1_780_000_000,
      mode: "wallet-connected",
      expired: true,
    });

    const missingIdentity = await verifyEntrosAttestation(
      wallet,
      accounts({ data: sasBuf, owner: SAS_PROGRAM }, null),
    );
    expect(missingIdentity).toBeNull();
  });

  // Seconds since the epoch pass 2^31 on 2038-01-19. A reader that builds a 64-bit value
  // from two signed 32-bit halves reads every later expiry as a negative number.
  it.each([2_147_483_648n, 4_000_000_000n, 4_294_967_295n, 4_294_967_296n])(
    "reads expiry %s as that time and expires the attestation after it",
    async (expiry) => {
      const wallet = Keypair.generate().publicKey.toBase58();
      const sas = {
        data: buildSasAttestationBuffer({ trustScore: 720, verifiedAt: 1_780_000_000n, expiry }),
        owner: SAS_PROGRAM,
      };
      vi.useFakeTimers();

      vi.setSystemTime(Number(expiry - 1n) * 1000);
      expect(await verifyEntrosAttestation(wallet, accounts(sas, liveIdentity()))).toMatchObject({
        expired: false,
        verifiedAt: 1_780_000_000,
      });

      vi.setSystemTime(Number(expiry) * 1000);
      expect(await verifyEntrosAttestation(wallet, accounts(sas, liveIdentity()))).toMatchObject({
        expired: true,
      });
    },
  );

  it("reads a verification time past 2^31 without wrapping", async () => {
    const wallet = Keypair.generate().publicKey.toBase58();
    const sas = {
      data: buildSasAttestationBuffer({
        trustScore: 720,
        verifiedAt: 3_000_000_000n,
        expiry: 0n,
      }),
      owner: SAS_PROGRAM,
    };
    expect(await verifyEntrosAttestation(wallet, accounts(sas, liveIdentity()))).toMatchObject({
      verifiedAt: 3_000_000_000,
      expired: false,
    });
  });

  it("refuses a negative expiry and a data length that runs past the account", async () => {
    const wallet = Keypair.generate().publicKey.toBase58();
    const negativeExpiry = buildSasAttestationBuffer({
      trustScore: 720,
      verifiedAt: 1_780_000_000n,
      expiry: -1n,
    });
    expect(
      await verifyEntrosAttestation(
        wallet,
        accounts({ data: negativeExpiry, owner: SAS_PROGRAM }, liveIdentity()),
      ),
    ).toBeNull();

    const hugeLength = buildSasAttestationBuffer({
      trustScore: 720,
      verifiedAt: 1_780_000_000n,
      expiry: 0n,
    });
    hugeLength.writeUInt32LE(0xffff_ffff, 97);
    expect(
      await verifyEntrosAttestation(
        wallet,
        accounts({ data: hugeLength, owner: SAS_PROGRAM }, liveIdentity()),
      ),
    ).toBeNull();
  });

  it("reads only accounts that their programs own", async () => {
    const wallet = Keypair.generate().publicKey.toBase58();
    const sasData = buildSasAttestationBuffer({
      trustScore: 720,
      verifiedAt: 1_780_000_000n,
      expiry: 0n,
    });
    const identityData = buildIdentityStateBuffer({ trustScore: 720, lastResetTimestamp: 0n });
    const other = Keypair.generate().publicKey;

    expect(
      await verifyEntrosAttestation(
        wallet,
        accounts({ data: sasData, owner: SAS_PROGRAM }, { data: identityData, owner: ANCHOR_PROGRAM }),
      ),
    ).toMatchObject({ isHuman: true, trustScore: 720 });
    for (const [sasOwner, identityOwner] of [
      [other, ANCHOR_PROGRAM],
      [SAS_PROGRAM, other],
      [undefined, ANCHOR_PROGRAM],
      [SAS_PROGRAM, undefined],
    ] as const) {
      expect(
        await verifyEntrosAttestation(
          wallet,
          accounts({ data: sasData, owner: sasOwner }, { data: identityData, owner: identityOwner }),
        ),
      ).toBeNull();
    }
  });

  describe("an attestation that does not stand for the wallet's current identity", () => {
    const VERIFIED_AT = 1_780_000_000;
    const sas = () => ({
      data: buildSasAttestationBuffer({ trustScore: 720, verifiedAt: BigInt(VERIFIED_AT), expiry: 0n }),
      owner: SAS_PROGRAM,
    });
    const identityCreatedAt = (creationTimestamp: number) => ({
      data: buildIdentityStateBuffer({
        trustScore: 720,
        lastResetTimestamp: 0n,
        creationTimestamp: BigInt(creationTimestamp),
      }),
      owner: ANCHOR_PROGRAM,
    });
    const migration = (migratedAt: number, owner = ANCHOR_PROGRAM) => ({
      data: migrationRecordData(migratedAt, Keypair.generate().publicKey),
      owner,
    });

    it("reads one issued before the identity was created as no attestation", async () => {
      const wallet = Keypair.generate().publicKey.toBase58();
      expect(
        await verifyEntrosAttestation(wallet, accounts(sas(), identityCreatedAt(VERIFIED_AT + 1))),
      ).toBeNull();
      expect(
        await verifyEntrosAttestation(wallet, accounts(sas(), identityCreatedAt(VERIFIED_AT))),
      ).toMatchObject({ verifiedAt: VERIFIED_AT, expired: false });
    });

    it("reads one issued at or before the identity's migration to the wallet as no attestation", async () => {
      const wallet = Keypair.generate().publicKey.toBase58();
      for (const migratedAt of [VERIFIED_AT + 5, VERIFIED_AT]) {
        expect(
          await verifyEntrosAttestation(wallet, accounts(sas(), liveIdentity(), migration(migratedAt))),
        ).toBeNull();
      }
      expect(
        await verifyEntrosAttestation(wallet, accounts(sas(), liveIdentity(), migration(VERIFIED_AT - 1))),
      ).toMatchObject({ verifiedAt: VERIFIED_AT, expired: false });
    });

    it("reads the migration record in the same batch as the attestation and the identity", async () => {
      const wallet = Keypair.generate().publicKey;
      const requested: PublicKey[][] = [];
      const connection = {
        getMultipleAccountsInfo: async (addresses: PublicKey[]) => {
          requested.push(addresses);
          return [sas(), liveIdentity(), migration(VERIFIED_AT)];
        },
      };
      expect(await verifyEntrosAttestation(wallet.toBase58(), connection)).toBeNull();
      const [migrationPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("migration"), wallet.toBuffer()],
        ANCHOR_PROGRAM,
      );
      expect(requested).toHaveLength(1);
      expect(requested[0]![2]!.equals(migrationPda)).toBe(true);

      const single = {
        getAccountInfo: async (address: PublicKey) =>
          address.equals(migrationPda) ? migration(VERIFIED_AT) : null,
      };
      expect(await verifyEntrosAttestation(wallet.toBase58(), single)).toBeNull();
    });

    it("reads a migration address another program owns as no migration", async () => {
      const wallet = Keypair.generate().publicKey.toBase58();
      for (const owner of [SystemProgram.programId, Keypair.generate().publicKey]) {
        expect(
          await verifyEntrosAttestation(
            wallet,
            accounts(sas(), liveIdentity(), migration(VERIFIED_AT, owner)),
          ),
        ).toMatchObject({ verifiedAt: VERIFIED_AT });
      }
    });

    it("reads no attestation beside a migration record that does not parse", async () => {
      const wallet = Keypair.generate().publicKey.toBase58();
      const malformed = migration(VERIFIED_AT - 1);
      malformed.data = malformed.data.subarray(0, 56);
      expect(await verifyEntrosAttestation(wallet, accounts(sas(), liveIdentity(), malformed))).toBeNull();
    });
  });
});
