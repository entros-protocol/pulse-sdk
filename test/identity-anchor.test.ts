/**
 * Tests for src/identity/anchor.ts—the on-chain IdentityState reader.
 *
 * Regression guard for the Anchor 0.30+ IDL convention: BorshAccountsCoder
 * requires PascalCase account names AND returns snake_case field names in
 * the decoded object. The previous camelCase usage silently failed (caught
 * + return null), which broke the encrypted-baseline recovery path because
 * `recoverBaselineFromChain` depends on a populated `currentCommitment`
 * for AAD-bound decryption.
 *
 * The round-trip pattern (encode known struct → decode through public
 * fetchIdentityState API → assert returned shape) catches any future
 * regression in either the IDL sync or the decoder name/field conventions.
 */

import { describe, it, expect } from "vitest";
import { Keypair } from "@solana/web3.js";
import * as anchor from "@coral-xyz/anchor";
import { BN } from "@coral-xyz/anchor";
import { entrosAnchorIdl } from "../src/protocol/idl";
import { fetchIdentityState } from "../src/identity/anchor";

describe("fetchIdentityState", () => {
  it("decodes a freshly-minted IdentityState account end-to-end", async () => {
    const owner = Keypair.generate().publicKey;
    const mint = Keypair.generate().publicKey;
    const knownCommitment = new Uint8Array(32).fill(0xab);

    const coder = new anchor.BorshAccountsCoder(entrosAnchorIdl as anchor.Idl);
    const encoded = await coder.encode("IdentityState", {
      owner,
      creation_timestamp: new BN(1_700_000_000),
      last_verification_timestamp: new BN(1_700_001_000),
      verification_count: 5,
      trust_score: 250,
      current_commitment: Array.from(knownCommitment),
      mint,
      bump: 254,
      recent_timestamps: new Array(20).fill(new BN(0)),
      last_reset_timestamp: new BN(0),
      new_wallet: undefined,
    });

    const mockConnection = {
      getAccountInfo: async () => ({ data: encoded }),
    };

    const result = await fetchIdentityState(owner.toBase58(), mockConnection);
    expect(result).not.toBeNull();
    expect(result!.owner).toBe(owner.toBase58());
    expect(result!.mint).toBe(mint.toBase58());
    expect(result!.verificationCount).toBe(5);
    expect(result!.trustScore).toBe(250);
    expect(result!.creationTimestamp).toBe(1_700_000_000);
    expect(result!.lastVerificationTimestamp).toBe(1_700_001_000);
    expect(result!.lastResetTimestamp).toBe(0);
    expect([...result!.currentCommitment]).toEqual([...knownCommitment]);
  });

  it("reflects a non-zero reset timestamp through the camelCase field", async () => {
    const owner = Keypair.generate().publicKey;
    const mint = Keypair.generate().publicKey;
    const coder = new anchor.BorshAccountsCoder(entrosAnchorIdl as anchor.Idl);
    const encoded = await coder.encode("IdentityState", {
      owner,
      creation_timestamp: new BN(1_700_000_000),
      last_verification_timestamp: new BN(1_700_005_000),
      verification_count: 7,
      trust_score: 400,
      current_commitment: Array(32).fill(0xcd),
      mint,
      bump: 253,
      recent_timestamps: new Array(20).fill(new BN(0)),
      last_reset_timestamp: new BN(1_700_002_000),
      new_wallet: undefined,
    });

    const mockConnection = {
      getAccountInfo: async () => ({ data: encoded }),
    };

    const result = await fetchIdentityState(owner.toBase58(), mockConnection);
    expect(result?.lastResetTimestamp).toBe(1_700_002_000);
  });

  it("returns null when the on-chain account doesn't exist", async () => {
    const mockConnection = {
      getAccountInfo: async () => null,
    };
    const result = await fetchIdentityState(
      Keypair.generate().publicKey.toBase58(),
      mockConnection
    );
    expect(result).toBeNull();
  });

  it("returns null when the account data fails to decode (wrong account type)", async () => {
    const mockConnection = {
      getAccountInfo: async () => ({ data: new Uint8Array(583).fill(0xff) }),
    };
    const result = await fetchIdentityState(
      Keypair.generate().publicKey.toBase58(),
      mockConnection
    );
    expect(result).toBeNull();
  });
});
