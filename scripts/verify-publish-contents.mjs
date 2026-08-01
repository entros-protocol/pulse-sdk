#!/usr/bin/env node
/**
 * Pre-publish gate. Runs automatically via the `prepublishOnly` npm lifecycle.
 *
 * Guards against the class of leaks where an internal build target
 * (`dist-internal/`, harness artifacts, source trees) would be packed into
 * the npm tarball alongside the public `dist/` build.
 *
 * It also guards the class where the published artifact misdescribes itself:
 * a lockfile or a changelog left behind by a hand-edited version field.
 *
 * Enforced invariants:
 *   1. `files` must exist and be an array (allowlist, not denylist)
 *   2. `files` must include `dist`
 *   3. `files` must not include any of the forbidden entries below, nor
 *      a catch-all like `*` or `.`
 *   4. `package-lock.json` must agree with `package.json` on the version,
 *      in both places it records one
 *   5. `changelog.md` must have an entry for the version being published
 *
 * If someone later removes the `files` field, widens it to `*`, explicitly
 * adds `dist-internal`, or bumps the version by editing package.json alone,
 * this script fails before npm uploads the tarball. The check runs in the npm
 * publish lifecycle, so it cannot be skipped by forgetting to run it.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "..");
const pkgPath = resolve(repoRoot, "package.json");
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

// The lockfile records the version twice, and `npm version` writes both. A
// hand-edited `package.json` writes neither, which is how 4.0.0 shipped against
// a lockfile that still said 3.16.0. Nothing downstream reads that field, so
// the drift is silent until someone goes looking, and by then several releases
// have gone out describing themselves wrongly.
const LOCK_VERSION_PATHS = [
  ["version", (lock) => lock.version],
  ['packages[""].version', (lock) => lock.packages?.[""]?.version],
];

try {
  const lock = JSON.parse(readFileSync(resolve(repoRoot, "package-lock.json"), "utf-8"));
  for (const [label, read] of LOCK_VERSION_PATHS) {
    const found = read(lock);
    if (found !== pkg.version) {
      errors.push(
        `package-lock.json ${label} is '${found}', package.json is ` +
          `'${pkg.version}'. Bump with 'npm version ${pkg.version} ` +
          `--no-git-tag-version', which writes both, rather than editing ` +
          `package.json by hand.`,
      );
    }
  }
} catch (err) {
  errors.push(`package-lock.json could not be read or parsed: ${err.message}`);
}

// A release with no changelog entry is the same defect one layer up: the
// artifact ships and nothing records what changed. The heading format is the
// one every entry in the file already uses.
try {
  const changelog = readFileSync(resolve(repoRoot, "changelog.md"), "utf-8");
  if (!changelog.includes(`## [${pkg.version}]`)) {
    errors.push(
      `changelog.md has no '## [${pkg.version}]' entry. Write one before ` +
        `publishing.`,
    );
  }
} catch (err) {
  errors.push(`changelog.md could not be read: ${err.message}`);
}

if (errors.length > 0) {
  console.error("prepublish content gate FAILED:");
  for (const e of errors) console.error("  - " + e);
  console.error(
    "\nSee scripts/verify-publish-contents.mjs for the full invariant list.",
  );
  process.exit(1);
}

console.log(
  `prepublish content gate OK: files allowlist is safe, and package.json, ` +
    `package-lock.json and changelog.md all agree on ${pkg.version}.`,
);
