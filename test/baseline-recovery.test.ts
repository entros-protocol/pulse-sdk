import { ed25519 } from "@noble/curves/ed25519";
import * as anchor from "@coral-xyz/anchor";
import { BN } from "@coral-xyz/anchor";
import { Keypair, PublicKey } from "@solana/web3.js";
import { beforeEach, describe, expect, it } from "vitest";
import { PROGRAM_IDS } from "../src/config";
import { bigintToBytes32, computeCommitment } from "../src/hashing/poseidon";
import {
  loadVerificationData,
  recoverBaselineFromChain,
  storeVerificationData,
} from "../src/identity/anchor";
import {
  clearBaselineKeyCache,
  deriveEncryptedBaselinePda,
  encryptBaselineBlob,
  fingerprintToBytes,
  getOrDeriveBaselineKey,
} from "../src/identity/baseline";
import { entrosAnchorIdl } from "../src/protocol/idl";

const ANCHOR_PROGRAM = new PublicKey(PROGRAM_IDS.entrosAnchor);
const SALT = 1234567890123456789012345678901234567890n;

function fingerprint(seed: number): number[] {
  return Array.from({ length: 256 }, (_, index) => ((index * 7 + seed) % 3 === 0 ? 1 : 0));
}

/** A wallet with an identity on chain, and an encrypted baseline that holds `sealed`. */
async function chainState(sealed: { fingerprint: number[]; salt: bigint }) {
  const keypair = Keypair.generate();
  const wallet = {
    publicKey: keypair.publicKey,
    signMessage: async (message: Uint8Array) =>
      ed25519.sign(message, keypair.secretKey.slice(0, 32)),
  };
  const open = fingerprint(1);
  const commitment = bigintToBytes32(await computeCommitment(open, SALT));
  const [identityPda, bump] = PublicKey.findProgramAddressSync(
    [Buffer.from("identity"), keypair.publicKey.toBuffer()],
    ANCHOR_PROGRAM,
  );
  const [mint] = PublicKey.findProgramAddressSync(
    [Buffer.from("mint"), keypair.publicKey.toBuffer()],
    ANCHOR_PROGRAM,
  );
  const identity = await new anchor.BorshAccountsCoder(entrosAnchorIdl as anchor.Idl).encode(
    "IdentityState",
    {
      owner: keypair.publicKey,
      creation_timestamp: new BN(1_790_000_000),
      last_verification_timestamp: new BN(1_790_000_500),
      verification_count: 3,
      trust_score: 120,
      current_commitment: Array.from(commitment),
      mint,
      bump,
      recent_timestamps: Array.from({ length: 52 }, () => new BN(0)),
      last_reset_timestamp: new BN(0),
      new_wallet: PublicKey.default,
      projection_version: 1,
      last_rebaseline_timestamp: new BN(0),
    },
  );
  const [baselinePda] = await deriveEncryptedBaselinePda(keypair.publicKey);
  const blob = await encryptBaselineBlob(
    fingerprintToBytes(sealed.fingerprint),
    bigintToBytes32(sealed.salt),
    await getOrDeriveBaselineKey(wallet),
    keypair.publicKey,
    baselinePda,
    commitment,
  );
  const baseline = new Uint8Array(8 + blob.length + 1);
  baseline.set(blob, 8);
  const connection = {
    getAccountInfo: async (address: PublicKey) => {
      if (address.equals(identityPda)) {
        return { data: identity, owner: ANCHOR_PROGRAM, executable: false, lamports: 1 };
      }
      if (address.equals(baselinePda)) {
        return { data: baseline, owner: ANCHOR_PROGRAM, executable: false, lamports: 1 };
      }
      return null;
    },
  };
  return { wallet, connection, open, commitment, address: keypair.publicKey.toBase58() };
}

beforeEach(() => clearBaselineKeyCache());

describe("baseline recovery from chain", () => {
  it("stores a baseline that opens the on-chain commitment", async () => {
    const state = await chainState({ fingerprint: fingerprint(1), salt: SALT });
    expect(await recoverBaselineFromChain(state.wallet, state.connection)).toEqual({
      recovered: true,
    });
    const stored = await loadVerificationData(state.address);
    expect(stored?.fingerprint).toEqual(state.open);
    expect(stored?.salt).toBe(SALT.toString());
  });

  // The blob authenticates under the current commitment, so it decrypts. Its
  // contents still have to be the fingerprint and salt that the commitment hides.
  // A stored baseline that does not open the commitment can never produce a proof.
  it.each([
    ["another fingerprint", { fingerprint: fingerprint(2), salt: SALT }],
    ["another salt", { fingerprint: fingerprint(1), salt: SALT + 1n }],
  ])("stores nothing when the blob holds %s", async (_label, sealed) => {
    const state = await chainState(sealed);
    // This device already holds a baseline. The recovery must not replace it.
    const held = {
      fingerprint: fingerprint(3),
      salt: "42",
      commitment: "7".repeat(64),
      timestamp: 1_790_000_000_000,
      projectionVersion: 1,
    };
    await storeVerificationData(held, state.address);
    expect(await loadVerificationData(state.address)).toEqual(held);
    const result = await recoverBaselineFromChain(state.wallet, state.connection);
    expect(result.recovered).toBe(false);
    expect(result.reason).toBe("stale-baseline");
    expect(await loadVerificationData(state.address)).toEqual(held);
  });
});
