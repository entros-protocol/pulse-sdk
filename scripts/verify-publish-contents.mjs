#!/usr/bin/env node
/**
 * Pre-publish gate. Runs automatically via the `prepublishOnly` npm lifecycle.
 *
 * Guards against the class of leaks where an internal build target
 * (`dist-internal/`, harness artifacts, source trees) would be packed into
 * the npm tarball alongside the public `dist/` build.
 *
 * Enforced invariants on package.json:
 *   1. `files` field must exist and be an array (allowlist, not denylist)
 *   2. `files` must include `dist`
 *   3. `files` must not include any of the forbidden entries below, nor
 *      a catch-all like `*` or `.`
 *
 * If someone later removes the `files` field, widens it to `*`, or
 * explicitly adds `dist-internal`, this script fails before npm uploads
 * the tarball. The check runs in the npm publish lifecycle, so it cannot
 * be skipped by forgetting to run it.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const pkgPath = resolve(scriptDir, "..", "package.json");
const pkg = JSON.parse(readFileSync(pkgPath, "utf-8"));

const FORBIDDEN_PATHS = [
  "dist-internal",
  "src",
  "test",
  "tests",
  "scripts",
  ".npmrc",
];

const errors = [];

if (!Array.isArray(pkg.files)) {
  errors.push(
    "package.json must define a 'files' allowlist. Without it, npm " +
      "publish ships every non-gitignored path — including dist-internal/.",
  );
} else {
  if (!pkg.files.includes("dist")) {
    errors.push("'files' must include 'dist' (the public build output).");
  }

  for (const entry of pkg.files) {
    if (entry === "*" || entry === "." || entry === "**") {
      errors.push(`'files' contains catch-all '${entry}' — this defeats the allowlist.`);
      continue;
    }
    const normalized = entry.replace(/\/+$/, "");
    if (FORBIDDEN_PATHS.includes(normalized)) {
      errors.push(`'files' contains forbidden entry: '${entry}'. Never ship this to npm.`);
    }
    if (FORBIDDEN_PATHS.some((f) => normalized.startsWith(f + "/"))) {
      errors.push(`'files' contains forbidden subpath: '${entry}'.`);
    }
  }
}

if (errors.length > 0) {
  console.error("prepublish content gate FAILED:");
  for (const e of errors) console.error("  - " + e);
  console.error(
    "\nSee scripts/verify-publish-contents.mjs for the full invariant list.",
  );
  process.exit(1);
}

console.log("prepublish content gate OK: package.json files allowlist is safe.");
