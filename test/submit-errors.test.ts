import { describe, it, expect } from "vitest";
import { errToString } from "../src/submit/errors";

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
