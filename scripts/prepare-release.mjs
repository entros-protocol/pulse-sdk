#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export function renderRelease(history, version, date, commits) {
  if (!/^\d+\.\d+\.\d+$/.test(version) || !/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    throw new Error("Expected a stable version and an ISO release date");
  }
  if (!commits.length)
    throw new Error("No commits exist after the previous release");
  const start = `<!-- generated-release:${version} -->`;
  const end = `<!-- /generated-release:${version} -->`;
  const section = `${start}\n## [${version}] - ${date}\n\n${commits.map(({ subject, hash }) => `- ${subject} (${hash.slice(0, 7)}).`).join("\n")}\n${end}\n\n`;
  const begin = history.indexOf(start);
  if (begin !== -1) {
    const finish = history.indexOf(end, begin);
    if (finish === -1)
      throw new Error("The generated release section is incomplete");
    const after = finish + end.length;
    return (
      history.slice(0, begin) +
      section +
      history.slice(after).replace(/^\n{1,2}/, "")
    );
  }
  if (history.includes(`## [${version}]`))
    throw new Error(
      "An existing release entry requires its original generator",
    );
  const first = history.indexOf("\n## [");
  if (first === -1)
    throw new Error("The changelog has no historical release boundary");
  return history.slice(0, first + 1) + section + history.slice(first + 1);
}

export function prepareRelease(
  version,
  since,
  check = false,
  through = "HEAD",
) {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  if (!/^\d+\.\d+\.\d+$/.test(version) || !/^v\d+\.\d+\.\d+$/.test(since))
    throw new Error(
      "Usage: prepare-release.mjs VERSION PREVIOUS_TAG [--check]",
    );
  if (through !== "HEAD" && !/^[0-9a-f]{40}$/.test(through))
    throw new Error("Use HEAD or a full source commit hash");
  const git = (...args) =>
    execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
  git("rev-parse", "--verify", `refs/tags/${since}^{commit}`);
  const sourceRevision = git("rev-parse", "--verify", `${through}^{commit}`);
  git("merge-base", "--is-ancestor", sourceRevision, "HEAD");
  const commits = git(
    "log",
    "--no-merges",
    "--reverse",
    "--format=%H%x09%s",
    `${since}..${sourceRevision}`,
  )
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const [hash, ...subject] = line.split("\t");
      return { hash, subject: subject.join("\t") };
    });
  const date = git("show", "-s", "--format=%cs", sourceRevision);
  const path = resolve(root, "changelog.md");
  const previous = readFileSync(path, "utf8");
  const next = renderRelease(previous, version, date, commits);
  if (check) {
    const pkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
    const lock = JSON.parse(
      readFileSync(resolve(root, "package-lock.json"), "utf8"),
    );
    if (
      previous !== next ||
      pkg.version !== version ||
      lock.version !== version ||
      lock.packages[""].version !== version
    )
      throw new Error("Release metadata does not match Git history");
  } else {
    const pkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
    if (pkg.version !== version)
      execFileSync(
        "npm",
        ["version", version, "--no-git-tag-version", "--ignore-scripts"],
        { cwd: root, stdio: "pipe" },
      );
    writeFileSync(path, next);
  }
  return {
    version,
    since,
    sourceRevision,
    date,
    commits: commits.length,
    check,
  };
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  console.log(
    JSON.stringify(
      prepareRelease(
        process.argv[2] ?? "",
        process.argv[3] ?? "",
        process.argv.slice(4).includes("--check"),
        process.argv
          .slice(4)
          .find((argument) => argument.startsWith("--through="))
          ?.slice("--through=".length) ?? "HEAD",
      ),
    ),
  );
}
