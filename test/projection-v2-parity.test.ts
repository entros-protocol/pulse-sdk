import { describe, expect, it } from "vitest";
import { simhash } from "../src/hashing/simhash";
import { fuseFeaturesForProjection } from "../src/extraction/projection-scaling";
import {
  extractMouseDynamics,
  extractTouchFeatures,
} from "../src/extraction/kinematic";
import { fingerprintToBytes } from "../src/identity/baseline";
import {
  bigintToBytes32,
  computeCommitment,
  packBits,
} from "../src/hashing/poseidon";
import { canonicalizeTouchSamples } from "../src/sensor/touch";

const FEATURE_DIMENSION = 308;
const FIXED_SALT = BigInt("12345678901234567890123456789012345678901234567890");
const RAW_FEATURES = Array.from(
  { length: FEATURE_DIMENSION },
  (_, index) => ((index * 37) % 211 - 105) / 64,
);

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

describe("projection version 2 client parity", () => {
  it("pins normalized touch and pointer-motion extraction", () => {
    const source = Array.from({ length: 61 }, (_, index) => ({
      timestamp: index * 17,
      x: 0.5,
      y: 0.5,
      pressure: 0.5,
      width: 1,
      height: 1,
    }));
    const samples = canonicalizeTouchSamples(source, 2);
    const expectedTouch = new Array<number>(57).fill(0);
    const expectedMotion = new Array<number>(81).fill(0);

    expectedTouch[16] = 0.5;
    expectedTouch[20] = 1;
    expectedTouch[40] = 1;
    expectedTouch[50] = 34;
    expectedMotion[15] = 1;
    expectedMotion[38] = 0.5;
    expectedMotion[72] = 4.136029411764706;

    expect(samples).toHaveLength(31);
    expect(samples.slice(0, 3).map(({ timestamp }) => timestamp)).toEqual([
      0, 34, 68,
    ]);
    expect(extractTouchFeatures(samples, 2)).toEqual(expectedTouch);
    expect(extractMouseDynamics(samples, 2)).toEqual(expectedMotion);
  });

  it("pins the full SimHash and Poseidon pipeline", async () => {
    const fusedFeatures = fuseFeaturesForProjection(
      RAW_FEATURES.slice(0, 170),
      RAW_FEATURES.slice(170, 251),
      RAW_FEATURES.slice(251),
      2,
    );
    const fingerprint = simhash(fusedFeatures, 2);
    const packed = packBits(fingerprint);
    const commitment = await computeCommitment(fingerprint, FIXED_SALT);

    expect(bytesToHex(fingerprintToBytes(fingerprint))).toBe(
      "47931d625548b93b8dcfdca89301e99aca7ab2a1a755db8632f934b75b23f596",
    );
    expect(packed.lo).toBe(BigInt("205910948500125577978619551950382666567"));
    expect(packed.hi).toBe(BigInt("200657029248877128052315591129303251658"));
    expect(bytesToHex(bigintToBytes32(commitment))).toBe(
      "07c592a5a50730f610e8853b7e9597d3f6f7d66a1e4214c6b6c4762428759a5d",
    );
  });
});
