import { describe, it, expect } from "vitest";
import { simhash, hammingDistance } from "../src/hashing/simhash";
import { FINGERPRINT_BITS } from "../src/config";
import {
  generateProjectionWords,
  PROJECTION_PURPOSE,
} from "../src/hashing/hyperplanes";
import { sha256 } from "@noble/hashes/sha2.js";

const PUBLIC_SEED = Uint8Array.from([
  0x9e, 0xe9, 0xc0, 0x2f, 0x3f, 0xc6, 0xa2, 0xab,
  0xce, 0x70, 0x30, 0x10, 0xe6, 0x43, 0x78, 0xd4,
  0x53, 0x1f, 0x8b, 0xcb, 0x11, 0x0f, 0x0b, 0xc4,
  0xc1, 0x77, 0xc3, 0x6a, 0x60, 0xc7, 0x5b, 0xb5,
]);

function fingerprintHex(bits: number[]): string {
  let result = "";
  for (let index = 0; index < bits.length; index += 4) {
    result += Number.parseInt(bits.slice(index, index + 4).join(""), 2).toString(16);
  }
  return result;
}

function legacyStringSeed(value: string): number {
  let hash = 0;
  for (const character of value) {
    hash = ((hash << 5) - hash + character.charCodeAt(0)) | 0;
  }
  return hash;
}

describe("simhash", () => {
  const featureA = Array.from({ length: 100 }, (_, i) => Math.sin(i * 0.1));

  it("produces a 256-bit binary fingerprint", () => {
    const fp = simhash(featureA);
    expect(fp.length).toBe(FINGERPRINT_BITS);
    for (const bit of fp) {
      expect(bit === 0 || bit === 1).toBe(true);
    }
  });

  it("is deterministic", () => {
    const fp1 = simhash(featureA);
    const fp2 = simhash(featureA);
    expect(fp1).toEqual(fp2);
  });

  it("matches the frozen version 1 transcript", () => {
    expect(
      Array.from(
        generateProjectionWords(
          PUBLIC_SEED,
          PROJECTION_PURPOSE.public,
          1,
          308,
          16
        )
      )
    ).toEqual([
      3999861642, 2593092573, 4116727045, 2423131132,
      2704667368, 2600095892, 2308587662, 1382458421,
      177779353, 907165406, 8229536, 1299303692,
      2312639962, 2709577244, 1868880545, 2773743676,
    ]);
  });

  it("matches the frozen version 2 transcript", () => {
    expect(
      Array.from(
        generateProjectionWords(
          PUBLIC_SEED,
          PROJECTION_PURPOSE.public,
          2,
          308,
          16,
        ),
      ),
    ).toEqual([
      1816424877, 506286799, 3660086786, 1922004990,
      2305849189, 3430870315, 2837000082, 2235419823,
      2664381067, 3011143810, 321828308, 1387177461,
      4283390990, 3992665251, 2080844329, 3658837953,
    ]);
  });

  it("matches the independent version 1 fingerprint golden", () => {
    const features = Array.from(
      { length: 308 },
      (_, index) => (((index * 37) % 211) - 105) / 64
    );
    expect(fingerprintHex(simhash(features, 1))).toBe(
      "730c7022878f3334cfe021f5d28b5d6a3ab7ac06d751843e1a4bfca4409c4dec"
    );
  });

  it("pins legacy and normalized-touch fingerprint goldens", () => {
    const features = Array.from(
      { length: 308 },
      (_, index) => (((index * 37) % 211) - 105) / 64,
    );
    expect(fingerprintHex(simhash(features, 0))).toBe(
      "cb3a4b3db19d79f3fb1e22525fb909b180eae199b74654bdc6674f5e87c46534",
    );
    expect(fingerprintHex(simhash(features, 2))).toBe(
      "e8e89c699a9a91b4d47f13188d0099c4130e6dcfa0e6d1a30f9fac2f7d56a741",
    );
  });

  it("keeps direct callers on the legacy projection by default", () => {
    expect(simhash(featureA)).toEqual(simhash(featureA, 0));
    const versionedFeatures = Array.from(
      { length: 308 },
      (_, index) => (((index * 37) % 211) - 105) / 64
    );
    expect(simhash(versionedFeatures)).not.toEqual(simhash(versionedFeatures, 1));
  });

  it.each([1, 2])(
    "requires exactly 308 features under projection version %i",
    (projectionVersion) => {
      for (const dimension of [0, 307, 309]) {
        expect(() =>
          simhash(new Array(dimension).fill(0), projectionVersion),
        ).toThrow(
          `Projection version ${projectionVersion} requires exactly 308 features`,
        );
      }
    },
  );

  it("accepts exactly 308 features under projection version 1", () => {
    expect(simhash(new Array(308).fill(0), 1)).toHaveLength(FINGERPRINT_BITS);
    expect(simhash(new Array(308).fill(0), 2)).toHaveLength(FINGERPRINT_BITS);
  });

  it("bounds the exported projection word stream", () => {
    expect(() =>
      generateProjectionWords(PUBLIC_SEED, 255 as never, 1, 308, 1)
    ).toThrow("Projection purpose must be public or private");
    expect(() =>
      generateProjectionWords(
        PUBLIC_SEED,
        PROJECTION_PURPOSE.public,
        1,
        309,
        1
      )
    ).toThrow("Projection dimension must not exceed 308");
    expect(() =>
      generateProjectionWords(
        PUBLIC_SEED,
        PROJECTION_PURPOSE.public,
        1,
        308,
        308 * 256 + 1
      )
    ).toThrow("Projection word count must not exceed");
  });

  it("separates seed strings that collided under the legacy reducer", () => {
    expect(legacyStringSeed("Aa")).toBe(legacyStringSeed("BB"));

    const aaWords = generateProjectionWords(
      sha256(new TextEncoder().encode("Aa")),
      PROJECTION_PURPOSE.public,
      1,
      308,
      8
    );
    const bbWords = generateProjectionWords(
      sha256(new TextEncoder().encode("BB")),
      PROJECTION_PURPOSE.public,
      1,
      308,
      8
    );

    expect(Array.from(aaWords)).toEqual([
      1765681298, 1736910451, 3429764502, 1784947396,
      3474463449, 1478806281, 3051658660, 4255417318,
    ]);
    expect(Array.from(bbWords)).toEqual([
      2979806220, 1703678140, 3689915119, 1996824126,
      3453479279, 2376161568, 593243097, 308904270,
    ]);
    expect(aaWords).not.toEqual(bbWords);
  });

  it("similar vectors produce low Hamming distance", () => {
    // Slightly perturbed version of featureA
    const featureB = featureA.map((v) => v + (Math.random() - 0.5) * 0.01);
    const fpA = simhash(featureA);
    const fpB = simhash(featureB);
    const dist = hammingDistance(fpA, fpB);
    // Small perturbation should produce distance well below 128 (random chance)
    expect(dist).toBeLessThan(64);
  });

  it("dissimilar vectors produce high Hamming distance", () => {
    const featureC = Array.from({ length: 100 }, (_, i) => -Math.cos(i * 3.7));
    const fpA = simhash(featureA);
    const fpC = simhash(featureC);
    const dist = hammingDistance(fpA, fpC);
    // Different vectors should have distance closer to 128 (random)
    expect(dist).toBeGreaterThan(50);
  });

  it("empty feature vector returns all zeros", () => {
    const fp = simhash([]);
    expect(fp.length).toBe(FINGERPRINT_BITS);
    expect(fp.every((b) => b === 0)).toBe(true);
  });

  it("hamming distance is symmetric", () => {
    const fpA = simhash(featureA);
    const fpB = simhash(featureA.map((v) => v + 0.5));
    expect(hammingDistance(fpA, fpB)).toBe(hammingDistance(fpB, fpA));
  });

  it("hamming distance of identical fingerprints is zero", () => {
    const fp = simhash(featureA);
    expect(hammingDistance(fp, fp)).toBe(0);
  });
});
