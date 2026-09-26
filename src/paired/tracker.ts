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
/** No frame below this counts as speech, however quiet the room. */
export const MIN_SPEECH_RMS = 0.01;
/** The lowest noise floor assumed, so digital silence sets no bar. */
export const MIN_FLOOR_RMS = 0.002;

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

  const bar = (): number => {
    const ordered = [...history].sort((left, right) => left - right);
    const low = ordered.length === 0 ? 0 : ordered[Math.floor(FLOOR_PERCENTILE * (ordered.length - 1))]!;
    return Math.max(MIN_SPEECH_RMS, Math.max(MIN_FLOOR_RMS, low) * FLOOR_RATIO);
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
      pushBounded(history, level);
    },

    begin(nextWaypoints, nextTraceRequired) {
      waypoints = nextWaypoints;
      traceRequired = nextTraceRequired;
      reached = 0;
      round = [];
      tracedAt = null;
      reported = false;
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
      pushBounded(history, level);
      pushBounded(round, level);
      const now = round.length - 1;
      const traced = !traceRequired || (waypoints.length > 0 && reached === waypoints.length);
      if (!traced) return now >= OPEN_STALL_FRAMES ? "stalled" : "open";
      tracedAt ??= now;

      const threshold = bar();
      const spoken = runsAgainst(threshold).some((run) => run.qualifies);
      let lastVoiced: number | null = null;
      for (let index = round.length - 1; index >= 0; index--) {
        if (round[index]! >= threshold) {
          lastVoiced = index;
          break;
        }
      }
      if (spoken && lastVoiced !== null && now - lastVoiced >= QUIET_FRAMES && !reported) {
        reported = true;
        return "complete";
      }
      return reported || now - tracedAt >= STALL_FRAMES ? "stalled" : "open";
    },

    runs() {
      return runsAgainst(bar());
    },
  };
}
