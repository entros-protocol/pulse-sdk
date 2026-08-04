import { describe, it, expect } from "vitest";
import { extractVoiceQualityFeatures } from "../src/extraction/voice-quality";

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
      expect(features[i]).toBe(EXPECTED[i]);
    }
  }, 60_000);
});
