import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { beforeAll, describe, expect, it } from "vitest";

import {
  assertFingerprintArchitectureOutputAvailable,
  emitFingerprintArchitectureManifest,
  parseFingerprintArchitectureOutputPath,
  type FingerprintArchitectureTextSink,
} from "../scripts/support/fingerprint-architecture-output";

describe("fingerprint architecture manifest output", () => {
  let outputDirectory: string;

  beforeAll(async () => {
    outputDirectory = await mkdtemp(
      join(tmpdir(), "entros-pulse-fingerprint-output-"),
    );
  });

  it("uses stdout when no output path is configured", async () => {
    const chunks: string[] = [];
    const stdout: FingerprintArchitectureTextSink = {
      write(value) {
        chunks.push(value);
      },
    };

    await emitFingerprintArchitectureManifest(
      '{"schemaVersion":1}',
      undefined,
      stdout,
    );

    expect(chunks).toEqual(['{"schemaVersion":1}\n']);
  });

  it("requires an explicit absolute output path", () => {
    expect(parseFingerprintArchitectureOutputPath(undefined)).toBeUndefined();
    expect(() => parseFingerprintArchitectureOutputPath("")).toThrow(
      "must be an absolute path",
    );
    expect(() =>
      parseFingerprintArchitectureOutputPath("manifest.json"),
    ).toThrow("must be an absolute path");
  });

  it("writes a new file without using stdout", async () => {
    const outputPath = join(outputDirectory, "new-manifest.json");
    const stdout: FingerprintArchitectureTextSink = {
      write() {
        throw new Error("stdout must stay unused");
      },
    };

    await assertFingerprintArchitectureOutputAvailable(outputPath);
    await emitFingerprintArchitectureManifest(
      '{"schemaVersion":1}',
      outputPath,
      stdout,
    );

    expect(await readFile(outputPath, "utf8")).toBe('{"schemaVersion":1}\n');
  });

  it("refuses to overwrite an existing file", async () => {
    const outputPath = join(outputDirectory, "existing-manifest.json");
    const stdout: FingerprintArchitectureTextSink = { write: () => undefined };
    await emitFingerprintArchitectureManifest("first", outputPath, stdout);

    await expect(
      assertFingerprintArchitectureOutputAvailable(outputPath),
    ).rejects.toThrow("already exists");
    await expect(
      emitFingerprintArchitectureManifest("second", outputPath, stdout),
    ).rejects.toThrow("already exists");
    expect(await readFile(outputPath, "utf8")).toBe("first\n");
  });
});
