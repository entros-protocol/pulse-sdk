import { Keypair, PublicKey } from "@solana/web3.js";
import { describe, expect, it } from "vitest";
import { SAS_CONFIG } from "../src/config";
import { verifyEntrosAttestation } from "../src/attestation/sas";

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
}): Buffer {
  const buf = Buffer.alloc(593);
  Buffer.from([156, 32, 87, 93, 52, 155, 248, 207]).copy(buf, 0);
  buf.writeUInt16LE(opts.trustScore, 60);
  buf.writeBigInt64LE(opts.lastResetTimestamp, 543);
  return buf;
}

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
    const decoded = await verifyEntrosAttestation(wallet, {
      getMultipleAccountsInfo: async () => [
        { data: sasBuf },
        { data: resetIdentityBuf },
      ],
    });
    expect(decoded).toEqual({
      isHuman: true,
      trustScore: 0,
      verifiedAt: 1_780_000_000,
      mode: "wallet-connected",
      expired: true,
    });

    const missingIdentity = await verifyEntrosAttestation(wallet, {
      getMultipleAccountsInfo: async () => [{ data: sasBuf }, null],
    });
    expect(missingIdentity).toBeNull();
  });
});
