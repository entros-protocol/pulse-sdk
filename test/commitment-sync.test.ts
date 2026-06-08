import { describe, it, expect } from "vitest";
import { localCommitmentMatchesChain } from "../src/identity/anchor";

// Build a 32-byte big-endian commitment from a hex string (right-aligned),
// the form the on-chain IdentityState.current_commitment holds.
function bytes32(hex: string): Uint8Array {
  const clean = hex.replace(/^0x/, "").padStart(64, "0");
  const out = new Uint8Array(32);
  for (let i = 0; i < 32; i++) {
    out[i] = parseInt(clean.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

// Decimal string of a 32-byte big-endian array — the form local storage holds
// the previous commitment in (StoredVerificationData.commitment).
function decimalOf(b: Uint8Array): string {
  let v = 0n;
  for (let i = 0; i < b.length; i++) v = (v << 8n) | BigInt(b[i]!);
  return v.toString();
}

// Exact commitments from the production 6011 incident (2026-06-06): the chain
// head had advanced to C6 but this origin's local copy was one link stale at C5.
const C6 = bytes32(
  "144557968f0e85ae02ab15f188b837bb1291656488efe2c03dfa4ec660b62e31",
);
const C5 = bytes32(
  "0378575c0a5de3f126bb6165e6b2d1f8219b702aac2d4a88d7ee589b481c2823",
);

describe("localCommitmentMatchesChain", () => {
  it("matches when the local decimal equals the on-chain big-endian commitment", () => {
    expect(localCommitmentMatchesChain(decimalOf(C6), C6)).toBe(true);
  });

  it("does NOT match a one-link-stale local commitment (the production 6011 case)", () => {
    // C5 (local) vs C6 (chain head) is precisely what submitted a doomed
    // commitment_prev and reverted on-chain with PrevCommitmentMismatch.
    expect(localCommitmentMatchesChain(decimalOf(C5), C6)).toBe(false);
  });

  it("matches a leading-zero commitment (decimal carries no leading zeros)", () => {
    const small = bytes32("45"); // 0x45 = 69
    expect(localCommitmentMatchesChain("69", small)).toBe(true);
  });

  it("returns false (never throws) on malformed local input", () => {
    expect(localCommitmentMatchesChain("not-a-number", C6)).toBe(false);
    expect(localCommitmentMatchesChain("", C6)).toBe(false);
  });
});
