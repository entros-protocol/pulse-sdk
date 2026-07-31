import { describe, it, expect } from "vitest";
import {
  isVerificationPhase,
  phaseChargesAttempt,
  phaseSpend,
  type VerificationPhase,
} from "../src/phases";
import { isClientOriginReason } from "../src/reasons";
import {
  CONFIRMATION_TIMEOUT_MS,
  MAX_VERIFICATION_MS,
  SIGNATURE_TIMEOUT_MS,
  VALIDATE_DEADLINE_MS,
} from "../src/config";

const ALL_PHASES: VerificationPhase[] = [
  "capture",
  "extraction",
  "validation",
  "baseline",
  "proving",
  "signing",
  "submission",
  "confirmation",
];

describe("isVerificationPhase", () => {
  it("accepts every phase", () => {
    for (const phase of ALL_PHASES) {
      expect(isVerificationPhase(phase)).toBe(true);
    }
  });

  it("rejects inherited property names", () => {
    // `"toString" in PHASE_POLICY` is true, so an `in` check would have
    // classified it and then read a function out of a table declared to hold
    // policies. Hosts construct phases from their own strings, so this is
    // reachable input rather than a curiosity.
    for (const inherited of ["toString", "constructor", "valueOf", "__proto__"]) {
      expect(isVerificationPhase(inherited)).toBe(false);
    }
  });

  it("rejects non-strings", () => {
    for (const value of [undefined, null, 3, {}, ["capture"]]) {
      expect(isVerificationPhase(value)).toBe(false);
    }
  });
});

describe("phaseChargesAttempt", () => {
  it("charges only for the phase that judged the capture", () => {
    // The defect this replaces: the attempt budget was charged on every
    // failure that carried no client-origin reason, so three declined wallet
    // prompts hard-failed a user whose capture had passed validation each
    // time. Nothing outside `validation` evaluates whether a person is there.
    for (const phase of ALL_PHASES) {
      expect(phaseChargesAttempt(phase)).toBe(phase === "validation");
    }
  });

  it("charges when the phase is unknown", () => {
    // Failing open would let a host that is a version behind hand out
    // unlimited attempts.
    expect(phaseChargesAttempt(undefined)).toBe(true);
    expect(phaseChargesAttempt("nonsense")).toBe(true);
    expect(phaseChargesAttempt("toString")).toBe(true);
  });

  it("composes with the reason taxonomy for a request that never arrived", () => {
    // `validation` is the only charging phase, and a validate request that
    // never produced a response reaches it. The reason is what says nothing
    // was judged, so a host has to consult both.
    expect(phaseChargesAttempt("validation")).toBe(true);
    expect(isClientOriginReason("validation_unavailable")).toBe(true);
    expect(isClientOriginReason("validation_timeout")).toBe(true);
    expect(isClientOriginReason("variance_floor")).toBe(false);
  });
});

describe("phaseSpend", () => {
  it("reports nothing spent before a transaction exists", () => {
    for (const phase of ["capture", "extraction", "validation", "baseline", "proving", "signing"] as const) {
      expect(phaseSpend(phase)).toBe("none");
    }
  });

  it("reports a broadcast transaction as unknown rather than free", () => {
    // `submission` is where every ambiguous outcome lands, including a
    // signature prompt answered after the SDK stopped waiting. Telling that
    // user nothing was charged would be a claim with no basis.
    expect(phaseSpend("submission")).toBe("possible");
  });

  it("reports a reverted transaction as certainly charged", () => {
    // Reset broadcast against a stale IDL for two months with preflight
    // skipped, so every attempt paid for a transaction that could not
    // deserialize. The interface said only that verification had failed.
    expect(phaseSpend("confirmation")).toBe("certain");
  });

  it("does not claim a refund for an unknown phase", () => {
    expect(phaseSpend(undefined)).toBe("possible");
    expect(phaseSpend("nonsense")).toBe("possible");
  });
});

describe("MAX_VERIFICATION_MS", () => {
  it("exceeds every clock it is meant to cover, together", () => {
    // A host backstop set below this pre-empts the SDK's own per-step clocks
    // and reports the failure against whatever step its message happens to
    // name. Three hosts each raced the whole of `complete()` against 120
    // seconds, which is less than the validate deadline on its own, and that
    // is how a pending wallet prompt was reported as a proving timeout.
    const clocks = VALIDATE_DEADLINE_MS + SIGNATURE_TIMEOUT_MS + CONFIRMATION_TIMEOUT_MS;
    expect(MAX_VERIFICATION_MS).toBeGreaterThan(clocks);
  });

  it("leaves room for the work no clock bounds", () => {
    // Extraction, proving, the challenge fetch and the SAS attestation. A
    // minute is the floor: proving alone runs tens of seconds on a low-end
    // phone.
    const clocks = VALIDATE_DEADLINE_MS + SIGNATURE_TIMEOUT_MS + CONFIRMATION_TIMEOUT_MS;
    expect(MAX_VERIFICATION_MS - clocks).toBeGreaterThanOrEqual(60_000);
  });

  it("stays inside the 32-bit setTimeout range", () => {
    // Past 2^31-1 ms a browser fires the timer immediately, which would turn
    // a backstop into an instant failure.
    expect(MAX_VERIFICATION_MS).toBeLessThan(2 ** 31 - 1);
  });
});
