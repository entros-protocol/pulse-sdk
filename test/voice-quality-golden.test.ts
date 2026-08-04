import { describe, it, expect } from "vitest";
import {
  extractVoiceQualityFeatures,
  cppBasis,
} from "../src/extraction/voice-quality";

/**
 * The portable half of the contract, and the one that actually guards the
 * optimisation: every cached coefficient must equal the inline expression it
 * replaced, with the original left-associative grouping intact.
 *
 * This is checked against `Math.cos` on the machine running the test rather
 * than against a hardcoded number, because `Math.cos` is not required by
 * IEEE-754 to be correctly rounded and V8 returns results differing by an ULP
 * across architectures. An exact vector pinned on one machine fails on another
 * for a reason that has nothing to do with the code, which is exactly what
 * happened here: `310.5943437277091` on macOS arm64, `...092` on Linux x64.
 *
 * Recomputing the reference in-process sidesteps that entirely and is a
 * stronger check than a tolerance. Reassociating the expression to
 * `piOverN * ((n + 0.5) * k)` shifts results by roughly 7e-15 relative, which
 * no usable tolerance separates from platform noise at 3e-16, but which fails
 * this test on the first coefficient that rounds differently.
 */
describe("cepstral DCT basis matches the expression it replaced", () => {
  it("reproduces the inline computation exactly at 16 kHz", () => {
    const N = 1024;
    const qMin = 40;
    const bandLen = 227;
    const basis = cppBasis(N, qMin, bandLen);
    const piOverN = Math.PI / N;

    expect(basis).toHaveLength(bandLen * N);
    let checked = 0;
    for (let bIdx = 0; bIdx < bandLen; bIdx++) {
      const k = qMin + bIdx;
      const row = bIdx * N;
      for (let n = 0; n < N; n++) {
        // Grouping is load-bearing: the original was left-associative.
        if (basis[row + n] !== Math.cos(piOverN * (n + 0.5) * k)) {
          throw new Error(`basis[${bIdx}][${n}] diverged from the inline form`);
        }
        checked++;
      }
    }
    expect(checked).toBe(bandLen * N);
  });

  it("returns the same table for a repeated shape", () => {
    expect(cppBasis(512, 20, 64)).toBe(cppBasis(512, 20, 64));
  });
});

/**
 * Golden vectors for the voice-quality feature block.
 *
 * IF ONE OF THESE FAILS, DO NOT UPDATE THE EXPECTED VALUE.
 *
 * These nine numbers land in the 170-feature audio block, which is z-scored
 * into the fused vector, which becomes the SimHash, which becomes the
 * commitment written on chain. Moving any of them is a projection change: it
 * invalidates every stored baseline, and users meet it as `drift-too-high`
 * with a reset as the only exit. That is the failure mode master-list #215
 * exists to prevent, and it would arrive here disguised as a performance
 * optimisation.
 *
 * The cepstral path is the one to watch. `cepstralPeakProminence` runs once
 * per frame, and at 16 kHz over a 12-second capture its band-limited DCT-II
 * evaluated `Math.cos` roughly 276 million times. That cost invites rewrites,
 * and the two rewrites that look most natural both change the numbers:
 *
 *   - Reassociating `(piOverN * (n + 0.5)) * k` to `piOverN * ((n + 0.5) * k)`
 *     rounds in a different place.
 *   - An FFT-based DCT is mathematically equal and numerically different.
 *
 * Either would pass a tolerance-based test. Only exact vectors catch them.
 *
 * Values produced by the implementation at commit `cf2bf5f`, before the DCT
 * basis was hoisted out of the per-frame loop, and verified unchanged after.
 *
 * **Compared with a relative tolerance, not exactly, and that is forced rather
 * than chosen.** The cepstral path runs through `Math.log` and `Math.cos`,
 * neither of which IEEE-754 requires to be correctly rounded, so V8 returns
 * results differing by an ULP between architectures. Pinned exactly, these
 * passed on macOS arm64 and failed on Linux x64 by one digit.
 *
 * The tolerance is therefore wide enough to survive platform noise, which
 * leaves it too wide to catch a reassociation. The exact guard against that
 * lives in the basis test above. These vectors catch the coarser thing: an
 * algorithm swapped for a different one, or a formant tracker that stopped
 * tracking formants.
 *
 * Worth knowing beyond this file: if `Math.cos` and `Math.log` differ across
 * architectures, feature values differ between a user's own devices, and those
 * feed the SimHash. Whether an ULP can flip a projection bit is untested.
 */
describe("voice-quality golden vectors", () => {
  const SAMPLE_RATE = 16000;
  const FRAME_SIZE = 2048;
  const HOP_SIZE = 160;
  const SESSION_LENGTH = SAMPLE_RATE * 12;

  function multiToneSamples(freqs: number[], amplitude = 0.3): Float32Array {
    const out = new Float32Array(SESSION_LENGTH);
    for (let i = 0; i < SESSION_LENGTH; i++) {
      let sum = 0;
      for (const f of freqs) {
        sum += Math.sin((2 * Math.PI * f * i) / SAMPLE_RATE);
      }
      out[i] = (amplitude / freqs.length) * sum;
    }
    return out;
  }

  const EXPECTED = [
    310.5943437277091,
    3.991878034646248e-11,
    -1.9382038914834339,
    1.3247805904250496e-15,
    -10.018040663313284,
    1.2445763520304567,
    0.33333333332415477,
    0.6666666666757534,
    9.273968098543825e-14,
  ];

  it("holds the pinned feature values", async () => {
    const samples = multiToneSamples([500, 1500, 2500]);
    const numFrames =
      Math.floor((SESSION_LENGTH - FRAME_SIZE) / HOP_SIZE) + 1;
    const f0PerFrame = Array.from(
      { length: numFrames },
      (_, i) => 110 + (i % 40)
    );

    const features = await extractVoiceQualityFeatures(
      samples,
      SAMPLE_RATE,
      FRAME_SIZE,
      HOP_SIZE,
      f0PerFrame
    );

    expect(features).toHaveLength(EXPECTED.length);
    for (let i = 0; i < EXPECTED.length; i++) {
      const want = EXPECTED[i]!;
      const got = features[i]!;
      // Relative where the value is large enough for that to mean anything,
      // absolute near zero, since three of these sit at 1e-11 or below.
      const tolerance = Math.max(Math.abs(want) * 1e-12, 1e-20);
      expect(Math.abs(got - want)).toBeLessThanOrEqual(tolerance);
    }
  }, 60_000);
});
