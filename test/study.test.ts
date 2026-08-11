import { describe, expect, it } from "vitest";
import { featureSchemaVersionForProjection as exportedFeatureSchemaVersion } from "../src";
import { createStudyContext } from "../src/study";

describe("study context", () => {
  it("creates a fresh bounded record identifier", () => {
    const context = createStudyContext(
      {
        token: "A".repeat(43),
        feature_schema_version: 3,
        projection_version: 0,
      },
      "web-mobile",
    );

    expect(context.record_id).toMatch(/^[a-f0-9]{32}$/);
    expect(context.capture_class).toBe("web-mobile");
  });

  it("rejects malformed server grants", () => {
    expect(() =>
      createStudyContext(
        { token: "short", feature_schema_version: 3, projection_version: 0 },
        "web-mobile",
      ),
    ).toThrow("Study token is malformed");
    expect(() =>
      createStudyContext(
        { token: "A".repeat(43), feature_schema_version: -1, projection_version: 0 },
        "web-mobile",
      ),
    ).toThrow("Study feature schema version is malformed");
  });

  it("binds projection version 1 to feature schema version 4", () => {
    const context = createStudyContext(
      {
        token: "A".repeat(43),
        feature_schema_version: 4,
        projection_version: 1,
      },
      "web-mobile",
    );

    expect(context.feature_schema_version).toBe(4);
    expect(context.projection_version).toBe(1);
    expect(exportedFeatureSchemaVersion(0)).toBe(3);
    expect(exportedFeatureSchemaVersion(1)).toBe(4);
  });

  it("rejects a feature schema from another projection", () => {
    expect(() =>
      createStudyContext(
        {
          token: "A".repeat(43),
          feature_schema_version: 3,
          projection_version: 1,
        },
        "web-mobile",
      ),
    ).toThrow("does not match projection");
  });
});
