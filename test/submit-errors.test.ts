import { describe, it, expect, vi } from "vitest";
import {
  chainRevertError,
  errToString,
  isChainRevertError,
  isUserRejection,
  withTimeout,
} from "../src/submit/errors";

// Regression coverage for the "[object Object]" display bug. On an on-chain
// revert the wallet adapter / web3.js can throw a BARE object whose `.message`
// is undefined; the old `err.message ?? String(err)` collapsed it to the
// literal "[object Object]", destroying the `"Custom":<code>` substring that
// entros.io's failure categorizer routes on.
describe("errToString", () => {
  it("returns a string unchanged", () => {
    expect(errToString("boom")).toBe("boom");
  });

  it("uses Error.message when present", () => {
    expect(errToString(new Error("on-chain revert"))).toBe("on-chain revert");
  });

  it("JSON-stringifies a bare on-chain error object and preserves the Custom code", () => {
    const out = errToString({ InstructionError: [4, { Custom: 6011 }] });
    expect(out).not.toBe("[object Object]");
    expect(out).toContain('"Custom":6011');
    // Must match the exact regex step-views.tsx uses to route 6011.
    expect(/"Custom":\s*6011\b/.test(out)).toBe(true);
  });

  it("never returns [object Object] for a plain object", () => {
    expect(errToString({ a: 1, b: "two" })).not.toBe("[object Object]");
    // even a degenerate empty object becomes "{}" rather than "[object Object]"
    expect(errToString({})).toBe("{}");
  });

  it("falls back to a non-empty string for an Error with an empty message", () => {
    const out = errToString(new Error(""));
    expect(typeof out).toBe("string");
    expect(out.length).toBeGreaterThan(0);
    expect(out).not.toBe("[object Object]");
  });

  it("handles null and undefined without throwing", () => {
    expect(errToString(null)).toBe("null");
    expect(errToString(undefined)).toBe("undefined");
  });
});

// `isUserRejection` is the discriminator between a `signing` failure and a
// `submission` one. Wallet adapters merge signing and sending into a single
// `sendTransaction` call, and a declined prompt is the only outcome that is
// certainly not on the wire. A miss costs nothing: the failure is attributed
// to `submission`, which reports the outcome as unknown.
describe("isUserRejection", () => {
  it("recognises the phrasings the major adapters emit", () => {
    for (const message of [
      "User rejected the request.",
      "Transaction rejected by user",
      "user denied transaction signature",
      "The user rejected the request through the wallet",
    ]) {
      expect(isUserRejection(new Error(message))).toBe(true);
    }
  });

  it("does not claim a rejection for anything else", () => {
    for (const message of [
      "Blockhash not found",
      "failed to send transaction",
      "Your wallet did not respond to the signature request.",
      "Transaction failed on chain: {\"InstructionError\":[0,{\"Custom\":6011}]}",
    ]) {
      expect(isUserRejection(new Error(message))).toBe(false);
    }
  });

  it("reads a bare thrown object rather than collapsing it", () => {
    expect(isUserRejection({ message: "User rejected the request." })).toBe(true);
    expect(isUserRejection(null)).toBe(false);
  });
});

describe("chainRevertError", () => {
  it("marks only a cluster-reported execution failure", () => {
    const revert = chainRevertError('Transaction failed on chain: {"Custom":6011}');
    expect(isChainRevertError(revert)).toBe(true);
    // Only this shape may be attributed to `confirmation`, whose spend is
    // `certain`. A confirmation timeout is not one: the transaction may still
    // land, so it reports `submission` instead.
    expect(isChainRevertError(new Error("The network did not confirm your transaction in time."))).toBe(false);
    expect(isChainRevertError("not an error")).toBe(false);
  });

  it("preserves the Custom code the host routes on", () => {
    const revert = chainRevertError('Transaction failed on chain: {"InstructionError":[0,{"Custom":6012}]}');
    expect(/"Custom":\s*6012\b/.test(errToString(revert))).toBe(true);
  });
});

describe("withTimeout", () => {
  it("passes a value through when the work settles first", async () => {
    await expect(withTimeout(Promise.resolve("ok"), 50_000, "late")).resolves.toBe("ok");
  });

  it("passes a rejection through unchanged", async () => {
    const boom = new Error("boom");
    await expect(withTimeout(Promise.reject(boom), 50_000, "late")).rejects.toBe(boom);
  });

  it("rejects with the given message once the clock expires", async () => {
    vi.useFakeTimers();
    try {
      const pending = withTimeout(new Promise(() => {}), 1_000, "wallet never answered");
      const assertion = expect(pending).rejects.toThrow("wallet never answered");
      await vi.advanceTimersByTimeAsync(1_000);
      await assertion;
    } finally {
      vi.useRealTimers();
    }
  });

  it("clears the timer when the work wins the race", async () => {
    // An uncleared timer holds the Node event loop open for the full duration
    // after a fast success, which is a hung test run rather than a hung
    // verification, but is a defect either way.
    vi.useFakeTimers();
    try {
      await expect(withTimeout(Promise.resolve("ok"), 300_000, "late")).resolves.toBe("ok");
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("does not raise an unhandled rejection when the work loses and then fails", async () => {
    // `Promise.race` subscribes to both promises, so a late rejection from the
    // losing side is already handled. Asserted rather than assumed: an
    // unhandled rejection here would crash a host app, not just the SDK.
    vi.useFakeTimers();
    const unhandled = vi.fn();
    process.on("unhandledRejection", unhandled);
    try {
      let failLate: (err: Error) => void = () => {};
      const work = new Promise<never>((_, reject) => {
        failLate = reject;
      });
      const pending = withTimeout(work, 1_000, "gave up");
      const assertion = expect(pending).rejects.toThrow("gave up");
      await vi.advanceTimersByTimeAsync(1_000);
      await assertion;
      failLate(new Error("adapter threw after we stopped waiting"));
      await vi.advanceTimersByTimeAsync(0);
      expect(unhandled).not.toHaveBeenCalled();
    } finally {
      process.off("unhandledRejection", unhandled);
      vi.useRealTimers();
    }
  });
});
