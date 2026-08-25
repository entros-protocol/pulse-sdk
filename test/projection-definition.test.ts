import { describe, expect, it } from "vitest";
import {
  getProjectionDefinition,
  HIGHEST_SUPPORTED_PROJECTION_VERSION,
} from "../src/projection";

describe("projection definitions", () => {
  it("pins every supported projection contract", () => {
    expect(HIGHEST_SUPPORTED_PROJECTION_VERSION).toBe(2);
    expect(getProjectionDefinition(0)).toEqual({
      featureSchemaVersion: 3,
      featurePipeline: "legacy",
      hyperplanes: { family: "legacy" },
      authenticatedTransitions: false,
    });
    expect(getProjectionDefinition(1)).toEqual({
      featureSchemaVersion: 4,
      featurePipeline: "corrected",
      hyperplanes: { family: "public", transcriptVersion: 1 },
      authenticatedTransitions: true,
    });
    expect(getProjectionDefinition(2)).toEqual({
      featureSchemaVersion: 5,
      featurePipeline: "normalized-touch",
      hyperplanes: { family: "public", transcriptVersion: 2 },
      authenticatedTransitions: true,
    });
  });

  it.each([-1, 0.5, 3, Number.NaN])(
    "rejects unsupported projection version %s",
    (projectionVersion) => {
      expect(() => getProjectionDefinition(projectionVersion)).toThrow(
        `Unsupported projection version ${projectionVersion}`,
      );
    },
  );
});
