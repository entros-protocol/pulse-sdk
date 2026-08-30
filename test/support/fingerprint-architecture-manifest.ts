import { endianness } from "node:os";

import packageMetadata from "../../package.json";
import { getProjectionDefinition } from "../../src/projection";
import { extractFeatures } from "../../src/pulse";
import { simhash, simhashDotProducts } from "../../src/hashing/simhash";
import { generateTBH } from "../../src/hashing/poseidon";
import { fingerprintToBytes } from "../../src/identity/baseline";
import {
  describeInputLevel,
  normalizeCaptureRMS,
} from "../../src/sensor/audio";
import {
  CANONICAL_SAMPLE_RATE,
  toCanonicalCapture,
} from "../../src/sensor/resample";
import type { SensorData } from "../../src/sensor/types";
import { bytesToHex } from "../../src/submit/receipt";
import {
  EXPECTED_FINGERPRINT_ARCHITECTURE_FIXTURE_SHA256,
  FINGERPRINT_ARCHITECTURE_FIXED_SALT,
  FINGERPRINT_ARCHITECTURE_FIXTURE_ID,
  FINGERPRINT_ARCHITECTURE_POLICY_CURRENT,
  FINGERPRINT_ARCHITECTURE_POLICY_MINIMUM,
  createFingerprintArchitectureFixture,
  fingerprintArchitectureFixtureDigest,
} from "./fingerprint-architecture-fixture";

const PROJECTION_VERSIONS = [0, 1, 2] as const;

export interface FingerprintArchitectureProjectionManifest {
  projectionVersion: number;
  featureSchemaVersion: number;
  rawFeaturesF64Hex: string[];
  normalizedFeaturesF64Hex: string[];
  /** Signed dot products. Their absolute values are the SimHash bit margins. */
  simhashDotProductsF64Hex: string[];
  fingerprintHex: string;
  commitmentHex: string;
}

export interface FingerprintArchitectureManifest {
  schemaVersion: 1;
  implementation: {
    name: string;
    version: string;
  };
  fixture: {
    id: string;
    sha256: string;
    sourceAudioSampleRateHz: number;
    sourceAudioSampleCount: number;
    canonicalAudioSampleRateHz: number;
    canonicalAudioSampleCount: number;
    motionSampleCount: number;
    touchSampleCount: number;
    inputLevel: {
      rmsF64Hex: string;
      peakF64Hex: string;
      gainF64Hex: string;
      gainClipped: boolean;
      voicedFrameRatioF64Hex: string;
    };
  };
  runtime: {
    engine: "node";
    engineVersion: string;
    v8Version: string;
    numericBackend: "javascript-number-float64";
    platform: NodeJS.Platform;
    arch: string;
    endianness: "BE" | "LE";
  };
  commitment: {
    saltDecimal: string;
    byteOrder: "big-endian";
  };
  projectionPolicy: {
    current: 1;
    minimum: 0;
  };
  projections: FingerprintArchitectureProjectionManifest[];
}

function float64ToHex(value: number): string {
  const bytes = new Uint8Array(8);
  new DataView(bytes.buffer).setFloat64(0, value, false);
  return bytesToHex(bytes);
}

function assertFiniteVector(name: string, values: number[]): void {
  const invalidIndex = values.findIndex((value) => !Number.isFinite(value));
  if (invalidIndex >= 0) {
    throw new Error(`${name} contains a non-finite value at ${invalidIndex}`);
  }
}

export async function buildFingerprintArchitectureManifest(): Promise<FingerprintArchitectureManifest> {
  const fixture = createFingerprintArchitectureFixture();
  const fixtureDigest = fingerprintArchitectureFixtureDigest(fixture);
  if (fixtureDigest !== EXPECTED_FINGERPRINT_ARCHITECTURE_FIXTURE_SHA256) {
    throw new Error(
      `Fingerprint architecture fixture digest mismatch: ${fixtureDigest}`,
    );
  }

  const canonicalAudio = await toCanonicalCapture(
    fixture.sourcePcm,
    fixture.sourceSampleRate,
  );
  if (canonicalAudio.sampleRate !== CANONICAL_SAMPLE_RATE) {
    throw new Error("Fingerprint architecture audio did not canonicalize");
  }

  const inputLevel = describeInputLevel(canonicalAudio.samples);
  const normalizedAudio = normalizeCaptureRMS(canonicalAudio.samples);

  const audioWindowMs =
    (canonicalAudio.samples.length * 1_000) / canonicalAudio.sampleRate;
  const sensorData: SensorData = {
    audio: {
      samples: normalizedAudio,
      sampleRate: canonicalAudio.sampleRate,
      duration: audioWindowMs / 1_000,
      windowStartMs: 0,
      windowEndMs: audioWindowMs,
      inputLevel,
      voiceIsolationApplied: null,
    },
    motion: fixture.motion,
    touch: fixture.touch,
    modalities: { audio: true, motion: true, touch: true },
  };

  const projections: FingerprintArchitectureProjectionManifest[] = [];
  for (const projectionVersion of PROJECTION_VERSIONS) {
    const features = await extractFeatures(sensorData, projectionVersion);
    assertFiniteVector("Raw feature vector", features.raw);
    assertFiniteVector("Normalized feature vector", features.normalized);
    if (features.raw.length !== 308 || features.normalized.length !== 308) {
      throw new Error(
        `Projection ${projectionVersion} did not produce 308 features`,
      );
    }

    const fingerprint = simhash(features.normalized, projectionVersion);
    const dotProducts = simhashDotProducts(
      features.normalized,
      projectionVersion,
    );
    assertFiniteVector("SimHash dot products", dotProducts);
    const tbh = await generateTBH(
      fingerprint,
      FINGERPRINT_ARCHITECTURE_FIXED_SALT,
    );

    projections.push({
      projectionVersion,
      featureSchemaVersion:
        getProjectionDefinition(projectionVersion).featureSchemaVersion,
      rawFeaturesF64Hex: features.raw.map(float64ToHex),
      normalizedFeaturesF64Hex: features.normalized.map(float64ToHex),
      simhashDotProductsF64Hex: dotProducts.map(float64ToHex),
      fingerprintHex: bytesToHex(fingerprintToBytes(fingerprint)),
      commitmentHex: bytesToHex(tbh.commitmentBytes),
    });
  }

  return {
    schemaVersion: 1,
    implementation: {
      name: packageMetadata.name,
      version: packageMetadata.version,
    },
    fixture: {
      id: FINGERPRINT_ARCHITECTURE_FIXTURE_ID,
      sha256: fixtureDigest,
      sourceAudioSampleRateHz: fixture.sourceSampleRate,
      sourceAudioSampleCount: fixture.sourcePcm.length,
      canonicalAudioSampleRateHz: canonicalAudio.sampleRate,
      canonicalAudioSampleCount: canonicalAudio.samples.length,
      motionSampleCount: fixture.motion.length,
      touchSampleCount: fixture.touch.length,
      inputLevel: {
        rmsF64Hex: float64ToHex(inputLevel.rms),
        peakF64Hex: float64ToHex(inputLevel.peak),
        gainF64Hex: float64ToHex(inputLevel.gain),
        gainClipped: inputLevel.gainClipped,
        voicedFrameRatioF64Hex: float64ToHex(inputLevel.voicedFrameRatio),
      },
    },
    runtime: {
      engine: "node",
      engineVersion: process.versions.node,
      v8Version: process.versions.v8,
      numericBackend: "javascript-number-float64",
      platform: process.platform,
      arch: process.arch,
      endianness: endianness(),
    },
    commitment: {
      saltDecimal: FINGERPRINT_ARCHITECTURE_FIXED_SALT.toString(10),
      byteOrder: "big-endian",
    },
    projectionPolicy: {
      current: FINGERPRINT_ARCHITECTURE_POLICY_CURRENT,
      minimum: FINGERPRINT_ARCHITECTURE_POLICY_MINIMUM,
    },
    projections,
  };
}
