import { buildFingerprintArchitectureManifest } from "../test/support/fingerprint-architecture-manifest";
import {
  assertFingerprintArchitectureOutputAvailable,
  emitFingerprintArchitectureManifest,
  parseFingerprintArchitectureOutputPath,
} from "./support/fingerprint-architecture-output";

async function main(): Promise<void> {
  const outputPath = parseFingerprintArchitectureOutputPath(
    process.env.ENTROS_FINGERPRINT_ARCHITECTURE_MANIFEST_PATH,
  );
  if (outputPath !== undefined) {
    await assertFingerprintArchitectureOutputAvailable(outputPath);
  }
  const manifest = await buildFingerprintArchitectureManifest();
  await emitFingerprintArchitectureManifest(
    JSON.stringify(manifest),
    outputPath,
    process.stdout,
  );
}

void main();
