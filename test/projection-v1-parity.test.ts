import { describe, expect, it } from "vitest";
import { simhash } from "../src/hashing/simhash";
import { fuseFeatures } from "../src/extraction/statistics";
import { fingerprintToBytes } from "../src/identity/baseline";
import {
  bigintToBytes32,
  computeCommitment,
  packBits,
} from "../src/hashing/poseidon";

const FEATURE_DIMENSION = 308;
const FIXED_SALT = BigInt("12345678901234567890123456789012345678901234567890");

// Every value is an exact binary fraction, so all runtimes construct the same
// input bytes without depending on platform transcendental functions.
const RAW_FEATURES = Array.from(
  { length: FEATURE_DIMENSION },
  (_, index) => ((index * 37) % 211 - 105) / 64,
);

const EXPECTED_PACKED_FINGERPRINT_HEX =
  "ce300e44e1f1cc2cf30784af4bd1ba165ced3560eb8a217c58d23f250239b237";
const EXPECTED_DISPLAY_FINGERPRINT_HEX =
  "730c7022878f3334cfe021f5d28b5d683ab7ac06d751843e1a4bfca4409c4dec";
const EXPECTED_LO = BigInt("30213028142994471381011281194282528974");
const EXPECTED_HI = BigInt("74032924876321406520575610213660749148");
const EXPECTED_COMMITMENT_HEX =
  "09bf24c82ec449df9367d90e1c55b6b178b4df54b7230269ad3f5e1973f6d4a4";

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function bitsToDisplayHex(bits: number[]): string {
  let output = "";
  for (let index = 0; index < bits.length; index += 4) {
    output += Number.parseInt(bits.slice(index, index + 4).join(""), 2).toString(16);
  }
  return output;
}

describe("projection version 1 client parity", () => {
  it("pins the full SimHash and Poseidon pipeline", async () => {
    expect(RAW_FEATURES).toHaveLength(FEATURE_DIMENSION);
    expect(RAW_FEATURES.every(Number.isFinite)).toBe(true);

    const fusedFeatures = fuseFeatures(
      RAW_FEATURES.slice(0, 170),
      RAW_FEATURES.slice(170, 251),
      RAW_FEATURES.slice(251),
    );
    const fingerprint = simhash(fusedFeatures, 1);
    const packed = packBits(fingerprint);
    const commitment = await computeCommitment(fingerprint, FIXED_SALT);

    expect(bytesToHex(fingerprintToBytes(fingerprint))).toBe(
      EXPECTED_PACKED_FINGERPRINT_HEX,
    );
    // Display hex keeps bit 0 on the left. Witness bytes pack bit 0 as each
    // byte's least-significant bit, so their hex encodings are intentionally different.
    expect(bitsToDisplayHex(fingerprint)).toBe(EXPECTED_DISPLAY_FINGERPRINT_HEX);
    expect(packed.lo).toBe(EXPECTED_LO);
    expect(packed.hi).toBe(EXPECTED_HI);
    expect(bytesToHex(bigintToBytes32(commitment))).toBe(EXPECTED_COMMITMENT_HEX);
  });
});
