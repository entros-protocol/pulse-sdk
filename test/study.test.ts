import { describe, expect, it } from "vitest";
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
});
