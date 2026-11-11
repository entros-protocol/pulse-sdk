import { describe, expect, it } from "vitest";
import { isClientOriginReason, isVerificationReason, reasonDisposition } from "../src/reasons";

describe("baseline capture reasons", () => {
  it("makes quality retryable without classifying it as transport", () => {
    expect(isVerificationReason("anchor_retry")).toBe(true);
    expect(reasonDisposition("anchor_retry")).toBe("retry");
    expect(isClientOriginReason("anchor_retry")).toBe(false);
  });

  it("requires an updated paired client for a legacy baseline write", () => {
    expect(isVerificationReason("paired_required")).toBe(true);
    expect(reasonDisposition("paired_required")).toBe("fatal");
    expect(isClientOriginReason("paired_required")).toBe(false);
  });
});

describe("paired authorization reasons", () => {
  it("offers a new attempt after the wallet signs again, and charges nothing", () => {
    for (const reason of ["authorization_required", "authorization_invalid"]) {
      expect(isVerificationReason(reason), reason).toBe(true);
      expect(reasonDisposition(reason), reason).toBe("retry");
      // The relayer refuses the request before it judges anything.
      expect(isClientOriginReason(reason), reason).toBe(true);
    }
  });
});
