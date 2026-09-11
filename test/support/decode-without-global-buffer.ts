/**
 * Decode every `IdentityState` length with no `globalThis.Buffer`, then print
 * the Trust Score each one yields as JSON.
 *
 * Bundlers such as Next.js give each module its own `Buffer` binding and leave
 * the global undefined. Anchor gets its binding from the bundler, so this
 * script loads Anchor while the global exists and removes the global before it
 * decodes. It runs in its own process because the test runner needs the global.
 */
import { decodeIdentityState } from "../../src/identity/anchor";
import { IDENTITY_ACCOUNT_LENGTHS, buildIdentityAccount } from "./identity-account";

async function main(): Promise<void> {
  await import("@coral-xyz/anchor");
  delete (globalThis as { Buffer?: unknown }).Buffer;

  const scores: Record<number, number | null> = {};
  for (const length of IDENTITY_ACCOUNT_LENGTHS) {
    const decoded = await decodeIdentityState(buildIdentityAccount(length));
    scores[length] = decoded?.trustScore ?? null;
  }
  process.stdout.write(JSON.stringify(scores));
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
