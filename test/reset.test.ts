import { describe, it, expect } from "vitest";
import {
  PulseSDK,
  PulseSession,
  submitResetViaWallet,
  DEFAULT_CAPTURE_MS,
} from "../src/index";

describe("resetBaseline: public API surface", () => {
  it("exports submitResetViaWallet as a function", () => {
    expect(typeof submitResetViaWallet).toBe("function");
  });

  it("exposes resetBaseline on PulseSDK", () => {
    const sdk = new PulseSDK({
      relayerUrl: "http://localhost:3001/verify",
      cluster: "devnet",
    });
    expect(typeof (sdk as unknown as { resetBaseline: unknown }).resetBaseline).toBe(
      "function",
    );
  });

  it("exposes completeReset on PulseSession", () => {
    const sdk = new PulseSDK({ cluster: "devnet" });
    const session = sdk.createSession();
    expect(
      typeof (session as unknown as { completeReset: unknown }).completeReset,
    ).toBe("function");
  });
});

describe("PulseSession.completeReset: wallet requirement", () => {
  it("rejects when wallet is missing", async () => {
    const sdk = new PulseSDK({ cluster: "devnet" });
    const session: PulseSession = sdk.createSession();
    // Must skip all stages so completeReset's capture-state check passes.
    session.skipMotion();
    session.skipTouch();
    // Audio wasn't started, so stage remains idle — completeReset will
    // reject on insufficient-data before hitting a network call.
    const result = await session.completeReset(undefined, undefined);
    expect(result.success).toBe(false);
    // The wallet-requirement rejection fires before data-quality checks.
    expect(result.error).toMatch(/wallet and Solana connection/i);
  });

  it("rejects when connection is missing", async () => {
    const sdk = new PulseSDK({ cluster: "devnet" });
    const session: PulseSession = sdk.createSession();
    session.skipMotion();
    session.skipTouch();
    const fakeWallet = { publicKey: { toBase58: () => "fake" } };
    const result = await session.completeReset(fakeWallet, undefined);
    expect(result.success).toBe(false);
    expect(result.error).toMatch(/wallet and Solana connection/i);
  });
});

describe("baseline-missing error message: stable detection sentinel", () => {
  // The website FailedView detects the "baseline is missing" state by
  // matching on a stable substring of the error copy emitted from
  // pulse.ts:278-285. This test guards against silent copy drift that
  // would break the UI's ability to offer the reset CTA.
  it("contains 'baseline is missing' substring", async () => {
    // Full end-to-end would need a Solana connection. Instead, we verify
    // the literal string exists in the SDK source by grepping the module
    // output. Vitest resolves the module so the text is in the bundle.
    const mod = await import("../src/pulse");
    // The error string should appear in the module's string table. Most
    // straightforward guard: re-declare the expected phrase and assert
    // nothing in the module has drifted.
    const expectedPhrase =
      "Your Entros Anchor exists on-chain but the local baseline is missing.";
    const src = mod.toString?.() ?? "";
    // `toString()` on a module object returns `[object Module]` — that's
    // not useful. Read the TS source via fs instead.
    expect(src).toBeDefined();

    const fs = await import("node:fs");
    const path = await import("node:path");
    const pulseSource = fs.readFileSync(
      path.resolve(__dirname, "../src/pulse.ts"),
      "utf-8",
    );
    expect(pulseSource).toContain(expectedPhrase);
  });

  it("DEFAULT_CAPTURE_MS is exported (session capture cadence)", () => {
    expect(typeof DEFAULT_CAPTURE_MS).toBe("number");
    expect(DEFAULT_CAPTURE_MS).toBeGreaterThan(0);
  });
});
