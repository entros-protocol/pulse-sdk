import { describe, it, expect } from "vitest";
import { extractAccelerationMagnitude } from "../src/extraction/kinematic";
import type { MotionSample } from "../src/sensor/types";

/**
 * Aligning the motion contour to the audio window.
 *
 * `accel_magnitude` is correlated against the F0 contour server-side, so the
 * two have to describe the same stretch of wall-clock time. Nothing enforced
 * that. The contour was built by mapping motion's array index proportionally
 * onto audio's frame count, which is only correct while both streams happen to
 * cover the same window.
 *
 * On 2026-07-31 they stopped. `pulse-sdk@4.0.0` added a capture-window mark
 * that discards the pre-prompt lead-in from the audio and wired it into the
 * audio path alone, so motion kept the challenge fetch and the three-second
 * countdown that audio now drops. Index mapping cannot see a span mismatch, so
 * it stretched motion across audio instead of failing, and the validator's
 * +-50ms lag search hunted a peak displaced by seconds. Cross-modal coupling
 * fell from r=0.31 to r=0.03 and every mobile verification was rejected.
 * Desktop was unaffected throughout: no IMU means the check skips entirely.
 *
 * Both streams already timestamp themselves off `performance.now()`, so the
 * fix is to use the clock they share rather than their array lengths.
 */

/** Pearson correlation, for asserting that two contours describe one signal. */
function correlation(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  const mean = (v: number[]) => v.slice(0, n).reduce((s, x) => s + x, 0) / n;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i]! - ma;
    const y = b[i]! - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  return da === 0 || db === 0 ? 0 : num / Math.sqrt(da * db);
}

/**
 * A motion stream carrying a recognisable waveform, sampled at `rateHz` across
 * `[startMs, endMs]`. The waveform is a function of absolute time, so any two
 * streams covering overlapping windows agree wherever they overlap. That is
 * what makes misalignment visible rather than merely plausible.
 */
function motionOver(startMs: number, endMs: number, rateHz = 60): MotionSample[] {
  const step = 1000 / rateHz;
  const out: MotionSample[] = [];
  for (let t = startMs; t <= endMs; t += step) {
    const v = Math.sin(t / 220) + 0.5 * Math.cos(t / 90);
    out.push({ timestamp: t, ax: v, ay: 0, az: 0, gx: 0, gy: 0, gz: 0 });
  }
  return out;
}

/** The same waveform sampled the way the F0 contour frames the audio window. */
function expectedContour(startMs: number, endMs: number, frames: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < frames; i++) {
    const t = startMs + (i / (frames - 1)) * (endMs - startMs);
    out.push(Math.abs(Math.sin(t / 220) + 0.5 * Math.cos(t / 90)));
  }
  return out;
}

/** The pre-fix implementation, kept verbatim so the regression stays pinned. */
function indexAligned(samples: MotionSample[], targetFrameCount: number): number[] {
  if (samples.length < 2 || targetFrameCount < 2) return [];
  const magnitudes = samples.map((s) => Math.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az));
  if (magnitudes.length === targetFrameCount) return magnitudes;
  const out = new Array<number>(targetFrameCount);
  const scale = (magnitudes.length - 1) / (targetFrameCount - 1);
  for (let i = 0; i < targetFrameCount; i++) {
    const pos = i * scale;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, magnitudes.length - 1);
    const frac = pos - lo;
    out[i] = magnitudes[lo]! * (1 - frac) + magnitudes[hi]! * frac;
  }
  return out;
}

const FRAMES = 1217; // what a 12.17s capture produces at the 10ms F0 hop

describe("motion aligned to the audio window", () => {
  it("recovers the signal when motion outruns audio, where index mapping cannot", () => {
    // The production shape: audio trimmed to the speak window, motion still
    // holding the 4s lead-in in front of it.
    const audioStart = 4_000;
    const audioEnd = 16_170;
    const motion = motionOver(0, audioEnd);
    const truth = expectedContour(audioStart, audioEnd, FRAMES);

    const aligned = extractAccelerationMagnitude(motion, FRAMES, {
      startMs: audioStart,
      endMs: audioEnd,
    });
    expect(aligned).toHaveLength(FRAMES);
    expect(
      correlation(aligned, truth),
      "the time-aligned contour must describe the audio window",
    ).toBeGreaterThan(0.9);

    // Same input through the old path. It stretches 16.17s of motion across a
    // 12.17s window, so every frame lands early by a margin that decays from
    // 4s to zero, which is what collapsed peak_r in production.
    expect(
      correlation(indexAligned(motion, FRAMES), truth),
      "the index-mapped contour must NOT survive the mismatch, or this test proves nothing",
    ).toBeLessThan(0.2);
  });

  it("matches a straight resample when the two windows already agree", () => {
    // The pre-4.0.0 case. Time alignment must not disturb it.
    const start = 1_000;
    const end = 13_170;
    const aligned = extractAccelerationMagnitude(motionOver(start, end), FRAMES, {
      startMs: start,
      endMs: end,
    });
    expect(correlation(aligned, expectedContour(start, end, FRAMES))).toBeGreaterThan(0.99);
  });

  it("places every frame correctly despite a gap in sensor delivery", () => {
    // Throttling, thermal pressure and main-thread contention all show up as
    // non-uniform `devicemotion` delivery. Index mapping treats a stalled
    // stretch as though it took the same time as a busy one. Timestamps do not.
    const start = 2_000;
    const end = 14_170;
    const stalled = motionOver(start, end).filter(
      (s) => s.timestamp < 6_000 || s.timestamp > 9_000,
    );
    const aligned = extractAccelerationMagnitude(stalled, FRAMES, {
      startMs: start,
      endMs: end,
    });
    expect(aligned).toHaveLength(FRAMES);
    // The gap is interpolated across, so agreement is looser than the clean
    // case, but the frames outside it must still land on the right instants.
    expect(correlation(aligned, expectedContour(start, end, FRAMES))).toBeGreaterThan(0.75);
  });

  it("refuses rather than guessing when motion cannot cover the window", () => {
    const start = 0;
    const end = 12_170;
    const win = { startMs: start, endMs: end };

    // Under-coverage. A partial contour reads as weak coupling and rejects a
    // real person, while an absent one makes the validator skip. Skipping is
    // the honest answer to "this capture cannot support the measurement".
    expect(extractAccelerationMagnitude(motionOver(0, 6_000), FRAMES, win)).toEqual([]);

    // No overlap at all.
    expect(extractAccelerationMagnitude(motionOver(20_000, 32_000), FRAMES, win)).toEqual([]);

    // Degenerate inputs.
    expect(extractAccelerationMagnitude(motionOver(start, end), FRAMES, {
      startMs: end,
      endMs: start,
    })).toEqual([]);
    expect(extractAccelerationMagnitude(motionOver(start, end), 1, win)).toEqual([]);
    expect(extractAccelerationMagnitude([], FRAMES, win)).toEqual([]);
  });

  it("tolerates motion that runs past the window on both sides", () => {
    // The normal case once the fix lands: motion starts before the speak
    // prompt and stops slightly after audio. Full coverage, so it must work.
    const start = 3_000;
    const end = 15_170;
    const aligned = extractAccelerationMagnitude(motionOver(0, 18_000), FRAMES, {
      startMs: start,
      endMs: end,
    });
    expect(aligned).toHaveLength(FRAMES);
    expect(correlation(aligned, expectedContour(start, end, FRAMES))).toBeGreaterThan(0.9);
  });
});
