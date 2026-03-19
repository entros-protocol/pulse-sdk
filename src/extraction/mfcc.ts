import type { AudioCapture } from "../sensor/types";
import { condense } from "./statistics";

// Frame parameters matching the research paper spec
const FRAME_SIZE = 400; // 25ms at 16kHz
const HOP_SIZE = 160; // 10ms hop
const NUM_MFCC = 13;

/**
 * Extract MFCC features from audio data.
 * Computes 13 MFCCs per frame, plus delta and delta-delta coefficients,
 * then condenses each coefficient's time series into 4 statistics.
 *
 * Returns: 13 coefficients × 3 (raw + delta + delta-delta) × 4 stats = 156 values
 */
export function extractMFCC(audio: AudioCapture): number[] {
  const { samples, sampleRate } = audio;

  // Lazy import of Meyda (browser/Node compatible)
  let Meyda: any;
  try {
    Meyda = require("meyda");
  } catch {
    // Meyda not available — return zeros (fallback for environments without it)
    return new Array(NUM_MFCC * 3 * 4).fill(0);
  }

  // Extract MFCCs per frame
  const numFrames = Math.floor((samples.length - FRAME_SIZE) / HOP_SIZE) + 1;
  if (numFrames < 3) return new Array(NUM_MFCC * 3 * 4).fill(0);

  const mfccFrames: number[][] = [];

  for (let i = 0; i < numFrames; i++) {
    const start = i * HOP_SIZE;
    const frame = samples.slice(start, start + FRAME_SIZE);

    // Pad if frame is shorter than expected
    const paddedFrame = new Float32Array(FRAME_SIZE);
    paddedFrame.set(frame);

    const features = Meyda.extract(["mfcc"], paddedFrame, {
      sampleRate,
      bufferSize: FRAME_SIZE,
      numberOfMFCCCoefficients: NUM_MFCC,
    });

    if (features?.mfcc) {
      mfccFrames.push(features.mfcc);
    }
  }

  if (mfccFrames.length < 3) return new Array(NUM_MFCC * 3 * 4).fill(0);

  // Compute delta (1st derivative) and delta-delta (2nd derivative)
  const deltaFrames = computeDeltas(mfccFrames);
  const deltaDeltaFrames = computeDeltas(deltaFrames);

  // Condense each coefficient across all frames into 4 statistics
  const features: number[] = [];

  for (let c = 0; c < NUM_MFCC; c++) {
    // Raw MFCC coefficient c across all frames
    const raw = mfccFrames.map((f) => f[c] ?? 0);
    const stats = condense(raw);
    features.push(stats.mean, stats.variance, stats.skewness, stats.kurtosis);
  }

  for (let c = 0; c < NUM_MFCC; c++) {
    const delta = deltaFrames.map((f) => f[c] ?? 0);
    const stats = condense(delta);
    features.push(stats.mean, stats.variance, stats.skewness, stats.kurtosis);
  }

  for (let c = 0; c < NUM_MFCC; c++) {
    const dd = deltaDeltaFrames.map((f) => f[c] ?? 0);
    const stats = condense(dd);
    features.push(stats.mean, stats.variance, stats.skewness, stats.kurtosis);
  }

  return features;
}

/** Compute delta coefficients (frame-to-frame differences) */
function computeDeltas(frames: number[][]): number[][] {
  const deltas: number[][] = [];
  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1]!;
    const curr = frames[i]!;
    deltas.push(curr.map((v, j) => v - (prev[j] ?? 0)));
  }
  return deltas;
}
