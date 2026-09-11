import { describe, it, expect } from "vitest";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { decodeIdentityState } from "../src/identity/anchor";
import {
  IDENTITY_ACCOUNT_LENGTHS as LENGTHS,
  buildIdentityAccount as buildAccount,
} from "./support/identity-account";

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

describe("legacy IdentityState layouts", () => {
  it("decodes every length the struct has ever had", async () => {
    for (const length of LENGTHS) {
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

  it("decodes when the page has no global Buffer", () => {
    // Bundlers such as Next.js leave `globalThis.Buffer` undefined in the page.
    const script = fileURLToPath(
      new URL("./support/decode-without-global-buffer.ts", import.meta.url),
    );
    const output = execFileSync(process.execPath, ["--import", "tsx", script], {
      cwd: fileURLToPath(new URL("..", import.meta.url)),
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    expect(JSON.parse(output)).toEqual({ 543: 481, 551: 481, 583: 481, 593: 481 });
  });

  it("decodes a plain Uint8Array as well as a Buffer", async () => {
    for (const length of LENGTHS) {
      const decoded = await decodeIdentityState(new Uint8Array(buildAccount(length)));
      expect(decoded?.trustScore, `${length}-byte account failed to decode`).toBe(481);
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
