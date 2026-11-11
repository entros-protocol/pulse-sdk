import * as anchor from "@coral-xyz/anchor";
import { BN } from "@coral-xyz/anchor";
import { Keypair, PublicKey, SystemProgram } from "@solana/web3.js";
import { describe, expect, it } from "vitest";
import { PROGRAM_IDS } from "../src/config";
import { type IdentityAccountInfo, readOnChainIdentity } from "../src/identity/anchor";
import { entrosAnchorIdl } from "../src/protocol/idl";

const ANCHOR_PROGRAM = new PublicKey(PROGRAM_IDS.entrosAnchor);
const ADDRESS = Keypair.generate().publicKey;

const serving = (account: IdentityAccountInfo | null) => ({ getAccountInfo: async () => account });

async function identityData(): Promise<Buffer> {
  const owner = Keypair.generate().publicKey;
  return new anchor.BorshAccountsCoder(entrosAnchorIdl as anchor.Idl).encode("IdentityState", {
    owner,
    creation_timestamp: new BN(1_790_000_000),
    last_verification_timestamp: new BN(1_790_000_500),
    verification_count: 3,
    trust_score: 120,
    current_commitment: Array.from(new Uint8Array(32).fill(5)),
    mint: Keypair.generate().publicKey,
    bump: 254,
    recent_timestamps: Array.from({ length: 52 }, () => new BN(0)),
    last_reset_timestamp: new BN(0),
    new_wallet: PublicKey.default,
    projection_version: 1,
    last_rebaseline_timestamp: new BN(0),
  });
}

describe("identity read before a verification", () => {
  it("reads no account as no identity", async () => {
    expect(await readOnChainIdentity(serving(null), ADDRESS)).toEqual({ status: "absent" });
  });

  // Anyone can send lamports to a wallet's identity address. That creates an account the
  // System Program owns, with no data. The program still creates the identity there on the
  // first mint, so the client must go on to mint.
  it("reads an address that holds lamports alone as no identity", async () => {
    const funded = {
      data: Buffer.alloc(0),
      owner: SystemProgram.programId,
      executable: false,
      lamports: 890_880,
    };
    expect(await readOnChainIdentity(serving(funded), ADDRESS)).toEqual({ status: "absent" });
  });

  it("reads a decodable identity as present", async () => {
    const account = { data: await identityData(), owner: ANCHOR_PROGRAM, executable: false, lamports: 1 };
    const read = await readOnChainIdentity(serving(account), ADDRESS);
    expect(read.status).toBe("present");
    if (read.status === "present") expect(read.identity.trustScore).toBe(120);
  });

  it("keeps every other account an error, so the client never mints over one", async () => {
    const others = [
      { data: Buffer.alloc(0), owner: ANCHOR_PROGRAM, executable: false, lamports: 1 },
      { data: Buffer.alloc(0), owner: SystemProgram.programId, executable: true, lamports: 1 },
      { data: Buffer.alloc(40, 1), owner: SystemProgram.programId, executable: false, lamports: 1 },
      { data: Buffer.alloc(593, 1), owner: ANCHOR_PROGRAM, executable: false, lamports: 1 },
      { data: Buffer.alloc(0), executable: false, lamports: 1 },
    ];
    for (const account of others) {
      expect(await readOnChainIdentity(serving(account), ADDRESS)).toEqual({ status: "undecodable" });
    }
  });
});
