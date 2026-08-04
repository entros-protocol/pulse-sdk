import { describe, it, expect } from "vitest";
import {
  extractFormantRatios,
  extractLpcAnalysis,
  hammingWindow,
  type LpcAnalysis,
} from "../src/extraction/lpc";

const SAMPLE_RATE = 16000;
const FRAME_SIZE = 2048;
const HOP_SIZE = 160;
const SESSION_LENGTH = SAMPLE_RATE * 12;

function sineSamples(length: number, freqHz: number, amplitude = 0.3): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * freqHz * i) / SAMPLE_RATE);
  }
  return out;
}

function multiToneSamples(length: number, freqs: number[], amplitude = 0.3): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    let sum = 0;
    for (const f of freqs) {
      sum += Math.sin((2 * Math.PI * f * i) / SAMPLE_RATE);
    }
    out[i] = (amplitude / freqs.length) * sum;
  }
  return out;
}

describe("extractLpcAnalysis", () => {
  it("returns the documented shape with LPC coefficient time series", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    expect(analysis.lpcCoefficients).toHaveLength(12); // default lpcOrder
    expect(analysis.numFramesAnalyzed).toBeGreaterThan(0);
    // Each per-coefficient track has one entry per analyzed frame.
    for (const coefTrack of analysis.lpcCoefficients) {
      expect(coefTrack.length).toBeLessThanOrEqual(analysis.numFramesAnalyzed);
    }
  });

  it("preserves backward compat with extractFormantRatios", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const legacy = extractFormantRatios(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    const fresh = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    // Same window, same frame iteration, same formant filter — should produce
    // identical ratio time series.
    expect(fresh.f1f2).toEqual(legacy.f1f2);
    expect(fresh.f2f3).toEqual(legacy.f2f3);
  });

  it("aligns formant absolute and bandwidth time series by index", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    expect(analysis.f1).toHaveLength(analysis.f2.length);
    expect(analysis.f2).toHaveLength(analysis.f3.length);
    expect(analysis.b1).toHaveLength(analysis.b2.length);
    expect(analysis.b2).toHaveLength(analysis.b3.length);
    expect(analysis.f1).toHaveLength(analysis.b1.length);
  });

  it("formants are sorted F1 < F2 < F3 within each frame", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    for (let t = 0; t < analysis.f1.length; t++) {
      expect(analysis.f1[t]!).toBeLessThan(analysis.f2[t]!);
      expect(analysis.f2[t]!).toBeLessThan(analysis.f3[t]!);
    }
  });

  it("bandwidths are positive and bounded", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    for (const arr of [analysis.b1, analysis.b2, analysis.b3]) {
      for (const bw of arr) {
        expect(bw).toBeGreaterThanOrEqual(0);
        expect(bw).toBeLessThan(500); // matches the formant filter
      }
    }
  });

  it("LPC coefficients are finite for non-degenerate input", () => {
    const samples = sineSamples(SESSION_LENGTH, 220);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);

    for (const coefTrack of analysis.lpcCoefficients) {
      for (const v of coefTrack) {
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  it("returns empty arrays on too-few-frames input", () => {
    const samples = sineSamples(1000, 220);
    const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    expect(analysis.numFramesAnalyzed).toBe(0);
    expect(analysis.f1).toHaveLength(0);
    for (const coefTrack of analysis.lpcCoefficients) {
      expect(coefTrack).toHaveLength(0);
    }
  });

  it("produces deterministic output", () => {
    const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
    const a = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    const b = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
    expect(a).toEqual(b);
  });

  it("type export is consumable", () => {
    const dummy: LpcAnalysis = {
      lpcCoefficients: [[]],
      f1: [],
      f2: [],
      f3: [],
      b1: [],
      b2: [],
      b3: [],
      f1f2: [],
      f2f3: [],
      numFramesAnalyzed: 0,
    };
    expect(dummy.numFramesAnalyzed).toBe(0);
  });
});

/**
 * Golden vectors for the LPC analysis path.
 *
 * IF ONE OF THESE FAILS, DO NOT UPDATE THE EXPECTED VALUE.
 *
 * A failure means the formant, bandwidth and LPC-coefficient series moved for
 * an input the pipeline has already been asked about in production. Those
 * series feed `speaker.ts`, which fills part of the 170-feature audio block,
 * which is z-scored into the fused vector, which becomes the SimHash, which
 * becomes the commitment written on chain. A changed value here therefore
 * shifts every fingerprint and strands every stored baseline behind
 * `drift-too-high`, with a reset as the only exit. That is the failure mode
 * master-list #215 exists to prevent, and it would arrive here disguised as a
 * performance optimisation.
 *
 * The values were produced by the implementation at commit `cf2bf5f`, before
 * the Hamming-window table was hoisted out of the per-frame loop, and verified
 * unchanged after it. Any future rewrite of the windowing, the autocorrelation,
 * the Levinson-Durbin recursion or the root finder has to reproduce them
 * exactly, or it is a projection change and must be sequenced behind #215.
 *
 * The other tests in this file check shape, alignment and determinism. Every
 * one of them passes against an implementation that computes different numbers.
 * Only these vectors catch that.
 */
describe("extractLpcAnalysis golden vectors", () => {
  const samples = multiToneSamples(SESSION_LENGTH, [500, 1500, 2500]);
  const analysis = extractLpcAnalysis(samples, SAMPLE_RATE, FRAME_SIZE, HOP_SIZE);
  const MID = 594;
  const LAST = 1187;

  it("analyses the pinned number of frames", () => {
    expect(analysis.numFramesAnalyzed).toBe(1188);
  });

  const series: Array<{ name: keyof typeof analysis; at: [number, number, number] }> = [
    { name: "f1", at: [500.01530584871745, 500.01530584879595, 500.01530584881914] },
    { name: "f2", at: [1500.09986324919, 1500.099863249147, 1500.0998632491408] },
    { name: "f3", at: [2500.072179315838, 2500.072179315844, 2500.072179315838] },
    { name: "b1", at: [0.2545482005833285, 0.25454820057654265, 0.25454820057710814] },
    { name: "b2", at: [0.14803473758696717, 0.1480347375844226, 0.14803473758413987] },
    { name: "b3", at: [0.038246138653724664, 0.03824613865400738, 0.038246138653724664] },
    { name: "f1f2", at: [0.3333213461973745, 0.3333213461974364, 0.33332134619745324] },
    { name: "f2f3", at: [0.6000226216107499, 0.6000226216107312, 0.6000226216107302] },
  ];

  // Relative, not exact. These run through `Math.cos`, which IEEE-754 does not
  // require to be correctly rounded, so V8 differs by an ULP across
  // architectures. The companion vectors in `voice-quality-golden.test.ts`
  // were pinned exactly and failed on Linux x64 having passed on macOS arm64.
  // The exact guard on the windowing itself is the hammingWindow test below,
  // which recomputes its reference in-process and so holds everywhere.
  const near = (got: number, want: number) => {
    expect(Math.abs(got - want)).toBeLessThanOrEqual(Math.abs(want) * 1e-12);
  };

  for (const { name, at } of series) {
    it(`holds the pinned ${String(name)} series`, () => {
      const track = analysis[name] as number[];
      expect(track).toHaveLength(1188);
      near(track[0]!, at[0]);
      near(track[MID]!, at[1]);
      near(track[LAST]!, at[2]);
    });
  }

  it("holds the pinned LPC coefficient tracks", () => {
    const first = analysis.lpcCoefficients[0]!;
    const twelfth = analysis.lpcCoefficients[11]!;
    near(first[0]!, -2.211867471627163);
    near(first[first.length - 1]!, -2.211867471722756);
    near(twelfth[0]!, 0.3522504587991376);
    near(twelfth[twelfth.length - 1]!, 0.35225045877833444);
  });

  it("builds the Hamming window exactly as the inline expression did", () => {
    // The portable, exact half of the contract. Checked against `Math.cos` on
    // this machine rather than a hardcoded number, so it holds on every
    // architecture while still failing on any reassociation of the grouping.
    for (const frameSize of [512, 1024, 2048]) {
      const window = hammingWindow(frameSize);
      expect(window).toHaveLength(frameSize);
      for (let j = 0; j < frameSize; j++) {
        if (
          window[j] !==
          0.54 - 0.46 * Math.cos((2 * Math.PI * j) / (frameSize - 1))
        ) {
          throw new Error(`window[${j}] at frameSize ${frameSize} diverged`);
        }
      }
    }
  });

  it("recovers the input tones, so the vectors pin a correct analysis", () => {
    // Guards against pinning a broken implementation. 500/1500/2500 Hz in, the
    // same frequencies out to within 0.15 Hz, which is under 0.01% error at
    // every one of the three. Without this the golden vectors above would
    // happily freeze a formant tracker that had stopped tracking formants.
    expect(Math.abs(analysis.f1[0]! - 500)).toBeLessThan(0.15);
    expect(Math.abs(analysis.f2[0]! - 1500)).toBeLessThan(0.15);
    expect(Math.abs(analysis.f3[0]! - 2500)).toBeLessThan(0.15);
  });
});
