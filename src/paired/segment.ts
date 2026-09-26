/**
 * Round windows and the analysis signal.
 *
 * A round commits one window of the continuous canonical stream as signed
 * 16-bit PCM, with no levelling of its own. The analysis signal joins the
 * committed segments and levels them once, the way a single capture is
 * levelled, and the validator rebuilds the same signal from the same bytes.
 */

import { normalizeCaptureRMS } from "../sensor/audio";
import { decodePcm16 } from "../sensor/encode";
import { FRAME_SAMPLES, type VoicedRun } from "./tracker";

/** Twelve seconds at 16 kHz. The bound exists for resources, never for how long a person takes. */
export const MAX_ROUND_SAMPLES = 192_000;
/** Audio kept after the chosen voiced run when a long round is trimmed, 600 ms. */
export const TRIM_TAIL_SAMPLES = 9_600;

export interface SampleRange {
  start: number;
  end: number;
}

/** Converts the tracker's frame runs to sample ranges on the canonical stream. */
export function runsToSamples(roundStart: number, runs: readonly VoicedRun[]): SampleRange[] {
  return runs
    .filter((run) => run.qualifies)
    .map((run) => ({
      start: roundStart + run.startFrame * FRAME_SAMPLES,
      end: roundStart + run.endFrame * FRAME_SAMPLES,
    }));
}

/**
 * The canonical sample window a round commits. A round within the bound keeps
 * every sample. A longer round keeps the bounded window that covers the most
 * voiced audio, ending 600 ms after a voiced run, with ties going to the later
 * window. A round with no voiced run keeps its tail. The rule's one job is to
 * keep the spoken word.
 */
export function roundWindow(
  roundStart: number,
  roundEnd: number,
  voicedRuns: readonly SampleRange[],
): SampleRange {
  if (roundEnd - roundStart <= MAX_ROUND_SAMPLES) return { start: roundStart, end: roundEnd };
  let best: { covered: number; window: SampleRange } | null = null;
  for (const run of voicedRuns) {
    const anchor = Math.min(roundEnd, run.end + TRIM_TAIL_SAMPLES);
    const window =
      anchor - MAX_ROUND_SAMPLES < roundStart
        ? { start: roundStart, end: roundStart + MAX_ROUND_SAMPLES }
        : { start: anchor - MAX_ROUND_SAMPLES, end: anchor };
    const covered = voicedRuns.reduce(
      (sum, other) =>
        sum + Math.max(0, Math.min(window.end, other.end) - Math.max(window.start, other.start)),
      0,
    );
    if (best === null || covered >= best.covered) best = { covered, window };
  }
  return best?.window ?? { start: roundEnd - MAX_ROUND_SAMPLES, end: roundEnd };
}

/** The committed segments decoded and joined in round order, with no separators. */
export function joinSegments(segments: readonly Uint8Array[]): Float32Array {
  const decoded = segments.map(decodePcm16);
  const joined = new Float32Array(decoded.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of decoded) {
    joined.set(part, offset);
    offset += part.length;
  }
  return joined;
}

/**
 * The signal features are extracted from: the joined segments levelled once.
 * Built from the committed bytes, never from float buffers, so the validator
 * reproduces it exactly.
 */
export function analysisSignal(segments: readonly Uint8Array[]): Float32Array {
  return normalizeCaptureRMS(joinSegments(segments));
}
