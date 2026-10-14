/**
 * Decides when a paired round is finished, so the interface moves on without a
 * button.
 *
 * A round is finished once the trace has reached every waypoint, in the issued
 * order, and the person has spoken and then gone quiet. Either can happen
 * first. The tracker reads
 * one RMS level for each 800-sample frame of canonical audio, 50 ms, so every
 * constant counts frames and a browser and a phone reach the same decision
 * from the same audio.
 *
 * The tracker only paces the interaction. The one transcription at the end
 * judges the words.
 */

import type { GridPoint } from "./transcript";
import { captureNormalizationGain } from "../sensor/audio-normalization";

export const FRAME_SAMPLES = 800;
/** Grid distance within which a trace point reaches a waypoint. */
export const WAYPOINT_REACH = 100;
/** A voiced run needs this many voiced frames, 200 ms. */
export const MIN_VOICED_FRAMES = 4;
/** Quiet gaps of up to this many frames stay inside one run. */
export const MAX_GAP_FRAMES = 2;
/**
 * A run counts only when its voiced frames are at least this multiple of its
 * gap frames, so a train of clicks never adds up to a word.
 */
export const VOICED_TO_GAP_RATIO = 2;
/** Quiet after speech before the round ends, 600 ms. */
export const QUIET_FRAMES = 12;
/** How long a finished trace waits for speech before offering to continue, 4 s. */
export const STALL_FRAMES = 80;
/** How long an unfinished round runs before offering to continue, 15 s. */
export const OPEN_STALL_FRAMES = 300;
/** Speech has to stand this many times above the noise floor. */
export const FLOOR_RATIO = 4;
/** The noise floor is this low percentile of recent frames. */
export const FLOOR_PERCENTILE = 0.1;
/** Frames kept for the floor and for one round, 120 s. */
export const HISTORY_FRAMES = 2_400;
/**
 * Absolute speech floor in the common-gain domain. Committed samples stay raw.
 */
export const MIN_SPEECH_RMS = 0.004;
/** The lowest noise floor assumed, so digital silence sets no bar. */
export const MIN_FLOOR_RMS = 0.001;

/**
 * `complete` is reported once per round. After it the round reads as
 * `stalled`, so a round whose commit fails offers the manual continue instead
 * of completing again on every frame.
 */
export type RoundProgress = "open" | "complete" | "stalled";

export interface VoicedRun {
  startFrame: number;
  endFrame: number;
  voicedFrames: number;
  gapFrames: number;
  qualifies: boolean;
}

export interface RoundTracker {
  /** Records a frame heard before any round began. It only informs the noise floor. */
  observe(level: number): void;
  /** Starts a round. `traceRequired` is false for a speech-only round. */
  begin(waypoints: readonly GridPoint[], traceRequired: boolean): void;
  /** Records one trace point, in grid units. It counts toward the next waypoint only. */
  reach(point: GridPoint): void;
  /** Records one frame's level and reports where the round stands. */
  frame(level: number): RoundProgress;
  /** The round's voiced runs, in frame indices from the round's first frame. */
  runs(): VoicedRun[];
  /** Whether the latest round frame meets the current speech threshold. */
  speechActive(): boolean;
  /** Latched after speech and its quiet interval, independent of trace order. */
  speechReady(): boolean;
  /**
   * The first frame at the speech bar, preserving latched speech, or null.
   * A refined recording boundary may move forward up to this frame but not past it: everything
   * from here on is speech the person produced during the round.
   */
  firstVoicedFrame(): number | null;
  /** Removes frames excluded by a refined recording boundary. */
  discardPrefix(frames: number): void;
}

/** RMS of one frame of canonical samples. */
export function frameRms(samples: Float32Array): number {
  let sum = 0;
  for (let index = 0; index < samples.length; index++) {
    const value = samples[index]!;
    sum += value * value;
  }
  return samples.length === 0 ? 0 : Math.sqrt(sum / samples.length);
}

function pushBounded(values: number[], value: number): void {
  values.push(value);
  if (values.length > HISTORY_FRAMES) values.shift();
}

export function createRoundTracker(): RoundTracker {
  const history: number[] = [];
  let waypoints: readonly GridPoint[] = [];
  let traceRequired = true;
  let reached = 0;
  let round: number[] = [];
  let tracedAt: number | null = null;
  let reported = false;
  let speechReady = false;
  let readyFrom: number | null = null;
  let readyFirst: number | null = null;

  const discardPrefix = (frames: number): void => {
    const removed = Math.min(frames, round.length);
    round.splice(0, removed);
    if (readyFirst !== null) readyFirst = Math.max(0, readyFirst - removed);
    if (readyFrom !== null) {
      readyFrom -= removed;
      if (readyFrom < 0) {
        readyFrom = readyFirst = null;
        speechReady = false;
        reported = false;
      }
    }
    if (tracedAt !== null) tracedAt = Math.max(0, tracedAt - removed);
  };

  const bar = (): number => {
    const ordered = [...history].sort((left, right) => left - right);
    const low = ordered.length === 0 ? 0 : ordered[Math.floor(FLOOR_PERCENTILE * (ordered.length - 1))]!;
    let energy = 0;
    for (const level of history) energy += level * level;
    const rms = history.length === 0 ? 0 : Math.sqrt(energy / history.length);
    const gain = captureNormalizationGain(rms);
    // One gain for the whole history preserves speech/noise contrast. Dividing
    // the absolute floors keeps every level and recording boundary in raw units.
    return Math.max(MIN_SPEECH_RMS / gain, Math.max(MIN_FLOOR_RMS / gain, low) * FLOOR_RATIO);
  };

  const runsAgainst = (threshold: number): VoicedRun[] => {
    const voiced = round.map((level) => level >= threshold);
    const out: VoicedRun[] = [];
    let index = 0;
    while (index < voiced.length) {
      if (!voiced[index]) {
        index++;
        continue;
      }
      const start = index;
      let last = index;
      let voicedFrames = 0;
      let gapFrames = 0;
      let pending = 0;
      for (let cursor = index; cursor < voiced.length; cursor++) {
        if (voiced[cursor]) {
          voicedFrames++;
          gapFrames += pending;
          pending = 0;
          last = cursor;
        } else {
          pending++;
          if (pending > MAX_GAP_FRAMES) break;
        }
      }
      out.push({
        startFrame: start,
        endFrame: last + 1,
        voicedFrames,
        gapFrames,
        qualifies: voicedFrames >= MIN_VOICED_FRAMES && voicedFrames >= VOICED_TO_GAP_RATIO * gapFrames,
      });
      index = last + 1;
    }
    return out;
  };

  return {
    observe(level) {
      if (!Number.isFinite(level) || level < 0) throw new RangeError("A frame level must be finite and non-negative.");
      pushBounded(history, level);
    },

    begin(nextWaypoints, nextTraceRequired) {
      waypoints = nextWaypoints;
      traceRequired = nextTraceRequired;
      reached = 0;
      round = [];
      tracedAt = null;
      reported = false;
      speechReady = false;
      readyFrom = null;
      readyFirst = null;
    },

    reach(point) {
      for (let waypoint = waypoints[reached]; waypoint; waypoint = waypoints[reached]) {
        const dx = point.x - waypoint.x;
        const dy = point.y - waypoint.y;
        if (dx * dx + dy * dy > WAYPOINT_REACH * WAYPOINT_REACH) return;
        reached++;
      }
    },

    frame(level) {
      if (!Number.isFinite(level) || level < 0) throw new RangeError("A frame level must be finite and non-negative.");
      pushBounded(history, level);
      round.push(level);
      if (round.length > HISTORY_FRAMES) discardPrefix(1);
      const now = round.length - 1;
      const traced = !traceRequired || (waypoints.length > 0 && reached === waypoints.length);
      const threshold = bar();
      const spoken = runsAgainst(threshold).reverse().find((run) => run.qualifies);
      let lastVoiced: number | null = null;
      for (let index = round.length - 1; index >= 0; index--) {
        if (round[index]! >= threshold) {
          lastVoiced = index;
          break;
        }
      }
      if (spoken && lastVoiced !== null && now - lastVoiced >= QUIET_FRAMES) {
        speechReady = true;
        readyFrom = Math.max(readyFrom ?? 0, spoken.startFrame);
        readyFirst ??= round.findIndex((value) => value >= threshold);
      }
      if (!traced) return now >= OPEN_STALL_FRAMES ? "stalled" : "open";
      tracedAt ??= now;
      if (speechReady && !reported) {
        reported = true;
        return "complete";
      }
      return reported || now - tracedAt >= STALL_FRAMES ? "stalled" : "open";
    },

    runs() {
      return runsAgainst(bar());
    },

    speechReady: () => speechReady,

    firstVoicedFrame() {
      const threshold = bar();
      for (let index = 0; index < round.length; index++) {
        if (round[index]! >= threshold) return readyFirst === null ? index : Math.min(readyFirst, index);
      }
      return readyFirst;
    },

    discardPrefix,

    speechActive() {
      const level = round[round.length - 1];
      return level !== undefined && level >= bar();
    },
  };
}
