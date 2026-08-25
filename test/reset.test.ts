import { afterEach, describe, it, expect, vi } from "vitest";
import {
  PulseSDK,
  PulseSession,
  submitResetViaWallet,
  DEFAULT_CAPTURE_MS,
} from "../src/index";

function oneShotSessionHarness() {
  const result = {
    success: false,
    commitment: new Uint8Array(32),
    isFirstVerification: true,
    error: "stubbed",
  };
  const session = {
    bindValidationChallenge: vi.fn(),
    startMotion: vi.fn().mockResolvedValue(undefined),
    isMotionCapturing: vi.fn().mockReturnValue(false),
    startAudio: vi.fn().mockResolvedValue(undefined),
    stopAudio: vi.fn().mockResolvedValue(null),
    startTouch: vi.fn().mockResolvedValue(undefined),
    stopTouch: vi.fn().mockResolvedValue(null),
    skipTouch: vi.fn(),
    complete: vi.fn().mockResolvedValue(result),
    completeReset: vi.fn().mockResolvedValue(result),
  };
  return session;
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

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

  it("rejects malformed validation challenge bindings", () => {
    const sdk = new PulseSDK({ cluster: "devnet" });
    expect(() =>
      sdk.createSession().bindValidationChallenge(new Uint8Array(31), 180),
    ).toThrow(/32-byte array/i);
    expect(() =>
      sdk.createSession().bindValidationChallenge(new Uint8Array(32), 0),
    ).toThrow(/future monotonic timestamp/i);
    expect(() =>
      sdk.createSession().bindValidationChallenge(new Uint8Array(32), Number.NaN),
    ).toThrow(/future monotonic timestamp/i);
  });

  it("binds the server challenge in the one-shot verify path", async () => {
    vi.useFakeTimers();
    const sdk = new PulseSDK({ cluster: "devnet" });
    const session = oneShotSessionHarness();
    vi.spyOn(sdk, "createSession").mockReturnValue(
      session as unknown as PulseSession,
    );
    const nonce = new Uint8Array(32).fill(0x41);
    const expiresAtMs = performance.now() + 180_000;

    const pending = sdk.verify(undefined, undefined, undefined, {
      validationChallengeNonce: nonce,
      validationChallengeExpiresAtMs: expiresAtMs,
    });
    await vi.runAllTimersAsync();
    await pending;

    expect(session.bindValidationChallenge).toHaveBeenCalledOnce();
    expect(session.bindValidationChallenge).toHaveBeenCalledWith(
      nonce,
      expiresAtMs,
    );
  });

  it("binds the server challenge in the one-shot reset path", async () => {
    vi.useFakeTimers();
    const sdk = new PulseSDK({ cluster: "devnet" });
    const session = oneShotSessionHarness();
    vi.spyOn(sdk, "createSession").mockReturnValue(
      session as unknown as PulseSession,
    );
    const nonce = new Uint8Array(32).fill(0x42);
    const expiresAtMs = performance.now() + 180_000;

    const pending = sdk.resetBaseline(
      undefined,
      undefined,
      undefined,
      undefined,
      {
        validationChallengeNonce: nonce,
        validationChallengeExpiresAtMs: expiresAtMs,
      },
    );
    await vi.runAllTimersAsync();
    await pending;

    expect(session.bindValidationChallenge).toHaveBeenCalledOnce();
    expect(session.bindValidationChallenge).toHaveBeenCalledWith(
      nonce,
      expiresAtMs,
    );
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

describe("capture constants", () => {
  it("DEFAULT_CAPTURE_MS is exported (session capture cadence)", () => {
    expect(typeof DEFAULT_CAPTURE_MS).toBe("number");
    expect(DEFAULT_CAPTURE_MS).toBeGreaterThan(0);
  });
});
