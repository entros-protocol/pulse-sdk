import { lstat, writeFile } from "node:fs/promises";
import { isAbsolute } from "node:path";

export interface FingerprintArchitectureTextSink {
  write(value: string): unknown;
}

function errorCode(error: unknown): string | undefined {
  if (typeof error !== "object" || error === null || !("code" in error)) {
    return undefined;
  }
  return String(error.code);
}

export function parseFingerprintArchitectureOutputPath(
  value: string | undefined,
): string | undefined {
  if (value === undefined) return undefined;
  if (value.length === 0 || !isAbsolute(value)) {
    throw new Error(
      "ENTROS_FINGERPRINT_ARCHITECTURE_MANIFEST_PATH must be an absolute path",
    );
  }
  return value;
}

/** Refuse an existing target before the full measurement starts. */
export async function assertFingerprintArchitectureOutputAvailable(
  outputPath: string,
): Promise<void> {
  try {
    await lstat(outputPath);
  } catch (error) {
    if (errorCode(error) === "ENOENT") return;
    throw error;
  }
  throw new Error(
    `Fingerprint architecture manifest already exists: ${outputPath}`,
  );
}

export async function emitFingerprintArchitectureManifest(
  serializedManifest: string,
  outputPath: string | undefined,
  stdout: FingerprintArchitectureTextSink,
): Promise<void> {
  const output = `${serializedManifest}\n`;
  if (outputPath === undefined) {
    stdout.write(output);
    return;
  }

  try {
    await writeFile(outputPath, output, { encoding: "utf8", flag: "wx" });
  } catch (error) {
    if (errorCode(error) === "EEXIST") {
      throw new Error(
        `Fingerprint architecture manifest already exists: ${outputPath}`,
      );
    }
    throw error;
  }
}
