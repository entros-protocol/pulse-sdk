import { describe, it, expect } from "vitest";
import { createHash } from "node:crypto";
import { decodeIdentityState } from "../src/identity/anchor";

/**
 * Decoding accounts written before `IdentityState` last grew.
 *
 * The struct has only ever been appended to, so 543, 551 and 583 bytes are
 * exact prefixes of the current 593. Anchor's Borsh coder does not know that
 * and throws on any of them, which `decodeIdentityState` swallowed to `null`.
 *
 * The cost was not a missing field. `recoverBaselineFromChain` reads the
 * identity before it fetches the encrypted baseline, so a decode failure made
 * cross-device recovery impossible for every legacy anchor, including twelve
 * on devnet that hold a valid blob. Measured on 2026-07-31: 107 accounts, of
 * which 2 at 207 bytes, 56 at 543, 11 at 551, 37 at 583 and 1 at 593.
 */

const LEN = 593;
const DISCRIMINATOR = createHash("sha256")
  .update("account:IdentityState")
  .digest()
  .subarray(0, 8);

/** Field offsets, from the program's own struct order. */
const OFF = {
  owner: 8,
  creation: 40,
  lastVerification: 48,
  count: 56,
  trustScore: 60,
  commitment: 62,
  mint: 94,
  bump: 126,
  recentTimestamps: 127,
  lastReset: 543,
  newWallet: 551,
  projectionVersion: 583,
  lastRebaseline: 585,
} as const;

/**
 * Build an account of `length` bytes carrying recognisable values in every
 * field that exists at that length. Anything past `length` is simply absent,
 * exactly as it is on chain.
 */
function buildAccount(length: number): Uint8Array {
  const buf = new Uint8Array(LEN);
  buf.set(DISCRIMINATOR, 0);
  const view = new DataView(buf.buffer);

  buf.fill(0x11, OFF.owner, OFF.owner + 32);
  view.setBigInt64(OFF.creation, 1_700_000_000n, true);
  view.setBigInt64(OFF.lastVerification, 1_700_009_999n, true);
  view.setUint32(OFF.count, 7, true);
  view.setUint16(OFF.trustScore, 481, true);
  buf.fill(0x22, OFF.commitment, OFF.commitment + 32);
  buf.fill(0x33, OFF.mint, OFF.mint + 32);
  buf[OFF.bump] = 254;
  view.setBigInt64(OFF.recentTimestamps, 1_700_009_999n, true);
  view.setBigInt64(OFF.lastReset, 1_699_000_000n, true);
  buf.fill(0x44, OFF.newWallet, OFF.newWallet + 32);
  view.setUint16(OFF.projectionVersion, 3, true);
  view.setBigInt64(OFF.lastRebaseline, 1_699_500_000n, true);

  // Production passes `accountInfo.data`, which web3.js hands over as a Node
  // Buffer. Anchor's Borsh layouts call `readUIntLE`, so a plain Uint8Array
  // fails to decode at any length and would make this suite test the wrong
  // thing.
  return Buffer.from(buf.subarray(0, length));
}

describe("legacy IdentityState layouts", () => {
  it("decodes every length the struct has ever had", async () => {
    for (const length of [543, 551, 583, 593]) {
      const decoded = await decodeIdentityState(buildAccount(length));
      expect(decoded, `${length}-byte account failed to decode`).not.toBeNull();
      // Fields present in all four layouts must read identically, whatever the
      // account's length. These sit before the first divergence.
      expect(decoded!.verificationCount, `${length}`).toBe(7);
      expect(decoded!.trustScore, `${length}`).toBe(481);
      expect(decoded!.lastVerificationTimestamp, `${length}`).toBe(1_700_009_999);
      expect(new Uint8Array(decoded!.currentCommitment).every((b) => b === 0x22)).toBe(true);
    }
  });

  it("defaults an appended field that the account predates", async () => {
    // A 543-byte account was written before `last_reset_timestamp` existed.
    // Zero is what the program's own realloc writes there, and it reads as
    // "never reset", which is true.
    const oldest = await decodeIdentityState(buildAccount(543));
    expect(oldest!.lastResetTimestamp).toBe(0);

    // A 551-byte account has the field, so it must not be clobbered.
    const withReset = await decodeIdentityState(buildAccount(551));
    expect(withReset!.lastResetTimestamp).toBe(1_699_000_000);
  });

  it("refuses a layout whose fields have moved rather than guessing", async () => {
    // At 207 bytes `recent_timestamps` held ten slots, not fifty-two, so every
    // offset after `bump` shifts. Padding would read one field's bytes as
    // another's, which is worse than reporting nothing.
    expect(await decodeIdentityState(buildAccount(207))).toBeNull();
    expect(await decodeIdentityState(buildAccount(127))).toBeNull();
    expect(await decodeIdentityState(new Uint8Array(0))).toBeNull();
  });

  it("still rejects an account belonging to another program", async () => {
    // Padding must not weaken the discriminator check.
    const foreign = Buffer.from(buildAccount(593));
    foreign.set(new Uint8Array([9, 9, 9, 9, 9, 9, 9, 9]), 0);
    expect(await decodeIdentityState(foreign)).toBeNull();
  });
});
