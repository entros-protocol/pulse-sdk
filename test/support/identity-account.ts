import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";

/** Every length `IdentityState` has had since it last changed shape. */
export const IDENTITY_ACCOUNT_LENGTHS = [543, 551, 583, 593] as const;

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
export function buildIdentityAccount(length: number): Uint8Array {
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

  // Production passes `accountInfo.data`, which web3.js hands over as a Buffer.
  return Buffer.from(buf.subarray(0, length));
}
