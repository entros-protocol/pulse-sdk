import { describe, expect, it } from "vitest";
import { simhash } from "../src/hashing/simhash";
import { fuseFeatures } from "../src/extraction/statistics";
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
    const fusedFeatures = fuseFeatures(
      RAW_FEATURES.slice(0, 170),
      RAW_FEATURES.slice(170, 251),
      RAW_FEATURES.slice(251),
    );
    const fingerprint = simhash(fusedFeatures, 2);
    const packed = packBits(fingerprint);
    const commitment = await computeCommitment(fingerprint, FIXED_SALT);

    expect(bytesToHex(fingerprintToBytes(fingerprint))).toBe(
      "171739965959892d2bfec818b1009923c870b6f105678bc5f0f931f49e6ae582",
    );
    expect(packed.lo).toBe(BigInt("47317415302883274880799270694016325399"));
    expect(packed.hi).toBe(BigInt("173990837961685963407535389837236793544"));
    expect(bytesToHex(bigintToBytes32(commitment))).toBe(
      "01a4d28e4832e357f7aac96ac793a97fec1cba126ce4720877a320e0c4ad957e",
    );
  });
});
