import { describe, it, expect } from "vitest";
import { classifyHammingDistance } from "../src/proof/prover";
import { DEFAULT_THRESHOLD, DEFAULT_MIN_DISTANCE } from "../src/config";

// The pre-flight bounds check in pulse.ts mirrors entros_hamming.circom:54-66.
// The accept band is [minDistance, threshold): LessThan enforces a strict
// `distance < threshold`, GreaterEqThan enforces an inclusive
// `distance >= minDistance`. These tests lock those boundary semantics.
describe("classifyHammingDistance: circuit accept band [minDistance, threshold)", () => {
  const T = 96; // δ_max
  const M = 3; // δ_min

  it("returns in_bounds just below the threshold", () => {
    expect(classifyHammingDistance(95, T, M)).toBe("in_bounds");
  });

  it("returns drift_too_high at the threshold (LessThan is strict)", () => {
    expect(classifyHammingDistance(96, T, M)).toBe("drift_too_high");
  });

  it("returns drift_too_high above the threshold (the dist=111 incident)", () => {
    expect(classifyHammingDistance(111, T, M)).toBe("drift_too_high");
  });

  it("returns in_bounds at exactly minDistance (GreaterEqThan is inclusive)", () => {
    expect(classifyHammingDistance(3, T, M)).toBe("in_bounds");
  });

  it("returns below_min_distance just under minDistance", () => {
    expect(classifyHammingDistance(2, T, M)).toBe("below_min_distance");
  });

  it("returns below_min_distance for an exact replay (distance 0)", () => {
    expect(classifyHammingDistance(0, T, M)).toBe("below_min_distance");
  });

  it("agrees with the shipped circuit constants at both boundaries", () => {
    // Locks the helper against the published band (paper.md: δ_min=3, δ_max=96).
    expect(
      classifyHammingDistance(DEFAULT_MIN_DISTANCE, DEFAULT_THRESHOLD, DEFAULT_MIN_DISTANCE),
    ).toBe("in_bounds");
    expect(
      classifyHammingDistance(DEFAULT_THRESHOLD, DEFAULT_THRESHOLD, DEFAULT_MIN_DISTANCE),
    ).toBe("drift_too_high");
    expect(
      classifyHammingDistance(DEFAULT_MIN_DISTANCE - 1, DEFAULT_THRESHOLD, DEFAULT_MIN_DISTANCE),
    ).toBe("below_min_distance");
  });
});

describe("drift-too-high error copy: stable detection sentinel", () => {
  // entros.io step-views.tsx `categorizeFailure` and the embed
  // popup-content.tsx `categorizeError` both route the drift-too-high failure
  // by matching the substring "closely match your usual pattern". This guards
  // against silent copy drift in pulse.ts that would break that routing.
  it("pulse.ts emits the 'closely match your usual pattern' substring", async () => {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const pulseSource = fs.readFileSync(
      path.resolve(__dirname, "../src/pulse.ts"),
      "utf-8",
    );
    expect(pulseSource).toContain("closely match your usual pattern");
  });

  it("pulse.ts no longer returns raw proof diagnostics to the caller", async () => {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const pulseSource = fs.readFileSync(
      path.resolve(__dirname, "../src/pulse.ts"),
      "utf-8",
    );
    // The old user-facing leak embedded circuit internals + feature values.
    // Diagnostics now go to gated sdkWarn only; the returned error is clean.
    expect(pulseSource).toContain(
      "We couldn't generate the verification proof.",
    );
    expect(pulseSource).not.toContain("Check wasmUrl/zkeyUrl reachability. Diagnostics:");
  });
});
