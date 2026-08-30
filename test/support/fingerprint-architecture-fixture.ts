import { sha256 } from "@noble/hashes/sha2.js";

import type { MotionSample, TouchSample } from "../../src/sensor/types";
import { bytesToHex } from "../../src/submit/receipt";

export const FINGERPRINT_ARCHITECTURE_FIXTURE_ID =
  "entros-fingerprint-architecture-fixture-v1";
export const FINGERPRINT_ARCHITECTURE_SOURCE_SAMPLE_RATE = 48_000;
export const FINGERPRINT_ARCHITECTURE_DURATION_MS = 12_000;
export const FINGERPRINT_ARCHITECTURE_SOURCE_SAMPLE_COUNT = 576_000;
export const FINGERPRINT_ARCHITECTURE_SENSOR_SAMPLE_COUNT = 769;
export const FINGERPRINT_ARCHITECTURE_POLICY_CURRENT = 1;
export const FINGERPRINT_ARCHITECTURE_POLICY_MINIMUM = 0;
export const FINGERPRINT_ARCHITECTURE_FIXED_SALT =
  12_345_678_901_234_567_890_123_456_789_012_345_678_901_234_567_890n;

export const EXPECTED_FINGERPRINT_ARCHITECTURE_FIXTURE_SHA256 =
  "703845449ffde83e10925a66dba6225b41f7801d812a3f7b9c8305aa8e3ea790";

export const EXPECTED_FINGERPRINT_ARCHITECTURE_OUTPUTS = [
  {
    projectionVersion: 0,
    fingerprintHex:
      "fa8ef7bad393171a094feb86699765de35ce65ca59b8de15e76a900ab76de86c",
    commitmentHex:
      "0ce1b185d50e89cc0b5f2e4682cbf0e8020a0bff1b85302f9c394cf007ff069e",
  },
  {
    projectionVersion: 1,
    fingerprintHex:
      "234281cb32d28fb297e9ea54806a8e61a8be00f095d0df76a9f132cc1ff1105c",
    commitmentHex:
      "2582455281bbc908f1f5fa863fa42e961971127ac005187f3796c9dcd03892cb",
  },
  {
    projectionVersion: 2,
    fingerprintHex:
      "2d9c86198251b94e9285a3b4ae87667b52363261ed0eb523bc9877b4a1a63a15",
    commitmentHex:
      "0e1fe84296ee186cf1bf7ae8b1628d1fefa01b8ddb1a4f9eac43b69874889e9f",
  },
] as const;

export interface FingerprintArchitectureFixture {
  sourcePcm: Float32Array;
  sourceSampleRate: number;
  motion: MotionSample[];
  touch: TouchSample[];
}

function createSourcePcm(): Float32Array {
  const samples = new Float32Array(
    FINGERPRINT_ARCHITECTURE_SOURCE_SAMPLE_COUNT,
  );
  let state = 0x6d2b_79f5 | 0;

  for (let index = 0; index < samples.length; index++) {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) | 0;

    const phaseA = index & 0xff;
    const triangleA = phaseA < 128 ? phaseA : 256 - phaseA;
    const phaseB = Math.imul(index, 3) & 0x1ff;
    const triangleB = phaseB < 256 ? phaseB : 512 - phaseB;
    const noise = (state >>> 24) - 128;
    const integerSample =
      ((triangleA << 1) - 128) * 76 +
      ((triangleB << 1) - 256) * 18 +
      (noise << 2);

    // A power-of-two scale maps each integer to one exact binary fraction.
    samples[index] = integerSample / 32_768;
  }

  return samples;
}

function createMotion(): MotionSample[] {
  const samples = new Array<MotionSample>(
    FINGERPRINT_ARCHITECTURE_SENSOR_SAMPLE_COUNT,
  );
  for (let index = 0; index < samples.length; index++) {
    samples[index] = {
      timestamp: (index * 125) / 8,
      ax: (((index * 37) & 0x7f) - 64) / 64,
      ay: (((index * 53 + 17) & 0xff) - 128) / 128,
      az: 1 + (((index * 29 + 11) & 0x7f) - 64) / 256,
      gx: (((index * 19 + 7) & 0x7f) - 64) / 32,
      gy: (((index * 43 + 23) & 0xff) - 128) / 64,
      gz: (((index * 61 + 31) & 0xff) - 128) / 64,
    };
  }
  return samples;
}

function createTouch(): TouchSample[] {
  const samples = new Array<TouchSample>(
    FINGERPRINT_ARCHITECTURE_SENSOR_SAMPLE_COUNT,
  );
  for (let index = 0; index < samples.length; index++) {
    samples[index] = {
      timestamp: (index * 125) / 8,
      x: ((index * 47 + ((index * index) & 0x7f)) & 0x3ff) / 1_024,
      y: ((index * 71 + ((index * index * 3) & 0xff)) & 0x3ff) / 1_024,
      pressure: (192 + ((index * 29) & 0x1ff)) / 1_024,
      width: 1,
      height: 1,
    };
  }
  return samples;
}

export function createFingerprintArchitectureFixture(): FingerprintArchitectureFixture {
  return {
    sourcePcm: createSourcePcm(),
    sourceSampleRate: FINGERPRINT_ARCHITECTURE_SOURCE_SAMPLE_RATE,
    motion: createMotion(),
    touch: createTouch(),
  };
}

/** Serialize every source value into one versioned little-endian byte contract. */
export function encodeFingerprintArchitectureFixture(
  fixture: FingerprintArchitectureFixture,
): Uint8Array {
  const domain = new TextEncoder().encode(
    `${FINGERPRINT_ARCHITECTURE_FIXTURE_ID}\0`,
  );
  const byteLength =
    domain.length +
    4 +
    4 +
    fixture.sourcePcm.length * 4 +
    4 +
    fixture.motion.length * 7 * 8 +
    4 +
    fixture.touch.length * 6 * 8;
  const bytes = new Uint8Array(byteLength);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  bytes.set(domain, 0);
  let offset = domain.length;

  view.setUint32(offset, fixture.sourceSampleRate, true);
  offset += 4;
  view.setUint32(offset, fixture.sourcePcm.length, true);
  offset += 4;
  for (const sample of fixture.sourcePcm) {
    view.setFloat32(offset, sample, true);
    offset += 4;
  }

  view.setUint32(offset, fixture.motion.length, true);
  offset += 4;
  for (const sample of fixture.motion) {
    for (const value of [
      sample.timestamp,
      sample.ax,
      sample.ay,
      sample.az,
      sample.gx,
      sample.gy,
      sample.gz,
    ]) {
      view.setFloat64(offset, value, true);
      offset += 8;
    }
  }

  view.setUint32(offset, fixture.touch.length, true);
  offset += 4;
  for (const sample of fixture.touch) {
    for (const value of [
      sample.timestamp,
      sample.x,
      sample.y,
      sample.pressure,
      sample.width,
      sample.height,
    ]) {
      view.setFloat64(offset, value, true);
      offset += 8;
    }
  }

  if (offset !== bytes.length) {
    throw new Error("Fingerprint architecture fixture length mismatch");
  }
  return bytes;
}

export function fingerprintArchitectureFixtureDigest(
  fixture: FingerprintArchitectureFixture,
): string {
  return bytesToHex(sha256(encodeFingerprintArchitectureFixture(fixture)));
}
