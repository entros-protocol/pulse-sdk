import { test } from "vitest";
import assert from "node:assert/strict";
import { renderRelease } from "./prepare-release.mjs";

const history =
  "# Changelog\n\nHistorical context.\n\n## [1.0.0] - 2026-01-01\n\n- Original entry.\n";
const commits = [
  { hash: "1234567890", subject: "Add an optional configuration" },
];
test("generation preserves history and is deterministic", () => {
  const once = renderRelease(history, "1.1.0", "2026-09-06", commits);
  assert.equal(renderRelease(once, "1.1.0", "2026-09-06", commits), once);
  assert.ok(once.endsWith(history.slice(history.indexOf("## [1.0.0]"))));
  assert.ok(once.includes("Add an optional configuration (1234567)."));
});
test("regeneration updates only the generated release", () => {
  const once = renderRelease(history, "1.1.0", "2026-09-06", commits);
  const twice = renderRelease(once, "1.1.0", "2026-09-06", [
    ...commits,
    { hash: "abcdef123", subject: "Preserve baseline storage" },
  ]);
  assert.equal(twice.match(/## \[1.1.0\]/g)?.length, 1);
  assert.ok(twice.endsWith(history.slice(history.indexOf("## [1.0.0]"))));
});
test("generation rejects unknown existing entries, malformed versions, and empty history", () => {
  assert.throws(() => renderRelease(history, "1.0.0", "2026-09-06", commits));
  assert.throws(() =>
    renderRelease(history, "--unknown", "2026-09-06", commits),
  );
  assert.throws(() => renderRelease(history, "1.1.0", "2026-09-06", []));
});
