import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import {
  bigintToBytes32,
  computeCommitment,
  generateTBH,
} from "../src/hashing/poseidon";
import { extractTouchFeatures } from "../src/extraction/kinematic";
import { canonicalizeTouchSamples } from "../src/sensor/touch";
import type { FingerprintArchitectureManifest } from "./support/fingerprint-architecture-manifest";
import { buildFingerprintArchitectureManifest } from "./support/fingerprint-architecture-manifest";
import {
  EXPECTED_FINGERPRINT_ARCHITECTURE_FIXTURE_SHA256,
  EXPECTED_FINGERPRINT_ARCHITECTURE_OUTPUTS,
  FINGERPRINT_ARCHITECTURE_FIXED_SALT,
  FINGERPRINT_ARCHITECTURE_DURATION_MS,
  FINGERPRINT_ARCHITECTURE_SENSOR_SAMPLE_COUNT,
  FINGERPRINT_ARCHITECTURE_SOURCE_SAMPLE_COUNT,
  FINGERPRINT_ARCHITECTURE_SOURCE_SAMPLE_RATE,
  createFingerprintArchitectureFixture,
  fingerprintArchitectureFixtureDigest,
} from "./support/fingerprint-architecture-fixture";

function float64FromHex(hex: string): number {
  const bytes = new Uint8Array(8);
  for (let index = 0; index < bytes.length; index++) {
    bytes[index] = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
  }
  return new DataView(bytes.buffer).getFloat64(0, false);
}

function fingerprintBit(hex: string, bitIndex: number): number {
  const byteIndex = bitIndex >> 3;
  const byte = Number.parseInt(hex.slice(byteIndex * 2, byteIndex * 2 + 2), 16);
  return (byte >> (bitIndex & 7)) & 1;
}

describe("fingerprint architecture fixture", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("pins the complete raw-input byte contract", () => {
    const fixture = createFingerprintArchitectureFixture();

    expect(fixture.sourceSampleRate).toBe(
      FINGERPRINT_ARCHITECTURE_SOURCE_SAMPLE_RATE,
    );
    expect(fixture.sourcePcm).toHaveLength(
      FINGERPRINT_ARCHITECTURE_SOURCE_SAMPLE_COUNT,
    );
    expect(fixture.motion).toHaveLength(
      FINGERPRINT_ARCHITECTURE_SENSOR_SAMPLE_COUNT,
    );
    expect(fixture.touch).toHaveLength(
      FINGERPRINT_ARCHITECTURE_SENSOR_SAMPLE_COUNT,
    );
    expect(fixture.motion.at(-1)?.timestamp).toBe(
      FINGERPRINT_ARCHITECTURE_DURATION_MS,
    );
    expect(fixture.touch.at(-1)?.timestamp).toBe(
      FINGERPRINT_ARCHITECTURE_DURATION_MS,
    );
    expect(fingerprintArchitectureFixtureDigest(fixture)).toBe(
      EXPECTED_FINGERPRINT_ARCHITECTURE_FIXTURE_SHA256,
    );
  });

  it("does not use transcendental or random functions", () => {
    const forbiddenCalls = [
      vi.spyOn(Math, "sin"),
      vi.spyOn(Math, "cos"),
      vi.spyOn(Math, "tan"),
      vi.spyOn(Math, "log"),
      vi.spyOn(Math, "log2"),
      vi.spyOn(Math, "log10"),
      vi.spyOn(Math, "pow"),
      vi.spyOn(Math, "sqrt"),
      vi.spyOn(Math, "random"),
    ];

    createFingerprintArchitectureFixture();

    for (const call of forbiddenCalls) {
      expect(call).not.toHaveBeenCalled();
    }
  });

  it("returns independent buffers and sample objects", () => {
    const first = createFingerprintArchitectureFixture();
    const second = createFingerprintArchitectureFixture();

    expect(second).not.toBe(first);
    expect(second.sourcePcm).not.toBe(first.sourcePcm);
    expect(second.motion).not.toBe(first.motion);
    expect(second.motion[0]).not.toBe(first.motion[0]);
    expect(second.touch).not.toBe(first.touch);
    expect(second.touch[0]).not.toBe(first.touch[0]);
    expect(fingerprintArchitectureFixtureDigest(second)).toBe(
      fingerprintArchitectureFixtureDigest(first),
    );
  });

  it.each([0, 1, 2])(
    "keeps coordinate jitter when contact channels are constant in projection %i",
    (projectionVersion) => {
      const fixture = createFingerprintArchitectureFixture();
      const samples = fixture.touch.map((sample) => ({
        ...sample,
        pressure: 1,
        width: 1,
        height: 1,
      }));
      const canonicalSamples = canonicalizeTouchSamples(
        samples,
        projectionVersion,
      );
      const features = extractTouchFeatures(
        canonicalSamples,
        projectionVersion,
      );

      expect(features[32]).toBeGreaterThan(0);
      expect(features[33]).toBeGreaterThan(0);
      expect(features[34]).toBe(0);
      expect(features[35]).toBe(0);
      expect(features[37]).toBe(0);
    },
  );
});

describe("fixed-salt TBH generation", () => {
  it("matches direct commitment and byte conversion", async () => {
    const fingerprint = Array.from(
      { length: 256 },
      (_, bitIndex) => (bitIndex * 13 + 7) & 1,
    );
    const commitment = await computeCommitment(
      fingerprint,
      FINGERPRINT_ARCHITECTURE_FIXED_SALT,
    );

    const tbh = await generateTBH(
      fingerprint,
      FINGERPRINT_ARCHITECTURE_FIXED_SALT,
    );

    expect(tbh.fingerprint).toBe(fingerprint);
    expect(tbh.salt).toBe(FINGERPRINT_ARCHITECTURE_FIXED_SALT);
    expect(tbh.commitment).toBe(commitment);
    expect(tbh.commitmentBytes).toEqual(bigintToBytes32(commitment));
  });
});

describe("fingerprint architecture manifest", () => {
  let manifest: FingerprintArchitectureManifest;
  let repeatedManifest: FingerprintArchitectureManifest;

  beforeAll(async () => {
    manifest = await buildFingerprintArchitectureManifest();
    repeatedManifest = await buildFingerprintArchitectureManifest();
  }, 600_000);

  it("is byte-value deterministic within one runtime", () => {
    expect(repeatedManifest).toEqual(manifest);
  });

  it("runs canonicalization and the complete pipeline for every projection", () => {
    expect(manifest.schemaVersion).toBe(1);
    expect(manifest.fixture).toEqual({
      id: "entros-fingerprint-architecture-fixture-v1",
      sha256: EXPECTED_FINGERPRINT_ARCHITECTURE_FIXTURE_SHA256,
      sourceAudioSampleRateHz: 48_000,
      sourceAudioSampleCount: 576_000,
      canonicalAudioSampleRateHz: 16_000,
      canonicalAudioSampleCount: 192_000,
      motionSampleCount: 769,
      touchSampleCount: 769,
      inputLevel: {
        rmsF64Hex: expect.stringMatching(/^[0-9a-f]{16}$/),
        peakF64Hex: expect.stringMatching(/^[0-9a-f]{16}$/),
        gainF64Hex: expect.stringMatching(/^[0-9a-f]{16}$/),
        gainClipped: false,
        voicedFrameRatioF64Hex: expect.stringMatching(/^[0-9a-f]{16}$/),
      },
    });
    expect(manifest.commitment).toEqual({
      saltDecimal: FINGERPRINT_ARCHITECTURE_FIXED_SALT.toString(10),
      byteOrder: "big-endian",
    });
    expect(manifest.runtime).toMatchObject({
      engine: "node",
      engineVersion: process.versions.node,
      v8Version: process.versions.v8,
      numericBackend: "javascript-number-float64",
      platform: process.platform,
      arch: process.arch,
    });
    expect(manifest.projectionPolicy).toEqual({ current: 1, minimum: 0 });
    expect(manifest.projections).toHaveLength(3);

    for (const [index, projection] of manifest.projections.entries()) {
      expect(projection).toMatchObject(
        EXPECTED_FINGERPRINT_ARCHITECTURE_OUTPUTS[index]!,
      );
      expect(projection.rawFeaturesF64Hex).toHaveLength(308);
      expect(projection.normalizedFeaturesF64Hex).toHaveLength(308);
      expect(projection.simhashDotProductsF64Hex).toHaveLength(256);
      for (const hex of projection.rawFeaturesF64Hex) {
        expect(hex).toMatch(/^[0-9a-f]{16}$/);
      }
      for (const hex of projection.normalizedFeaturesF64Hex) {
        expect(hex).toMatch(/^[0-9a-f]{16}$/);
      }
      for (const hex of projection.simhashDotProductsF64Hex) {
        expect(hex).toMatch(/^[0-9a-f]{16}$/);
      }
      expect(projection.fingerprintHex).toMatch(/^[0-9a-f]{64}$/);
      expect(projection.commitmentHex).toMatch(/^[0-9a-f]{64}$/);
    }
  });

  it("maps every diagnostic dot-product sign to its production bit", () => {
    for (const projection of manifest.projections) {
      projection.simhashDotProductsF64Hex.forEach((hex, bitIndex) => {
        const dotProduct = float64FromHex(hex);
        expect(Number.isFinite(dotProduct)).toBe(true);
        expect(fingerprintBit(projection.fingerprintHex, bitIndex)).toBe(
          dotProduct >= 0 ? 1 : 0,
        );
      });
    }
  });

  it("isolates projection 2 drift to normalized touch processing", () => {
    const projectionOne = manifest.projections[1]!;
    const projectionTwo = manifest.projections[2]!;

    expect(projectionTwo.rawFeaturesF64Hex.slice(0, 251)).toEqual(
      projectionOne.rawFeaturesF64Hex.slice(0, 251),
    );
    expect(projectionTwo.normalizedFeaturesF64Hex.slice(0, 251)).toEqual(
      projectionOne.normalizedFeaturesF64Hex.slice(0, 251),
    );
    expect(projectionTwo.rawFeaturesF64Hex.slice(251)).not.toEqual(
      projectionOne.rawFeaturesF64Hex.slice(251),
    );
  });
});
