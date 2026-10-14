import { describe, expect, it } from "vitest";
import { createRoundTracker, HISTORY_FRAMES } from "../src/paired/tracker";
import { captureNormalizationGain } from "../src/sensor/audio-normalization";

describe("common-gain progression", () => {
  it.each([0.002, 0.003, 0.0039, 0.004, 0.03])("finishes a word at raw RMS %s", (speech) => {
    const tracker = createRoundTracker();
    for (let index = 0; index < 20; index++) tracker.observe(0.0003);
    tracker.begin([], false);
    const levels = [...Array<number>(6).fill(0.0003), ...Array<number>(10).fill(speech), ...Array<number>(12).fill(0.0003)];
    const decisions = levels.map((level) => tracker.frame(level));
    expect(decisions.indexOf("complete")).toBe(27);
    expect(tracker.firstVoicedFrame()).toBe(6);
  });

  it("forgets old gain history with the noise history", () => {
    const tracker = createRoundTracker();
    for (let index = 0; index < HISTORY_FRAMES; index++) tracker.observe(0.9);
    for (let index = 0; index < HISTORY_FRAMES; index++) tracker.observe(0.0003);
    tracker.begin([], false);
    for (let index = 0; index < 10; index++) tracker.frame(0.0039);
    const decisions = Array.from({ length: 12 }, () => tracker.frame(0.0003));
    expect(decisions.at(-1)).toBe("complete");
  });

  it("keeps latched speech before a loud transient during boundary refinement", () => {
    const tracker = createRoundTracker();
    for (let index = 0; index < 20; index++) tracker.observe(0.0003);
    tracker.begin([{ x: 200, y: 200 }], true);
    for (const level of [
      ...Array<number>(6).fill(0.0003), ...Array<number>(10).fill(0.005),
      ...Array<number>(12).fill(0.0003),
    ]) tracker.frame(level);
    expect(tracker.speechReady()).toBe(true);
    tracker.frame(0.5);
    expect(tracker.firstVoicedFrame()).toBe(6);
    tracker.discardPrefix(Math.min(20, tracker.firstVoicedFrame()!));
    expect(tracker.firstVoicedFrame()).toBe(0);
    expect(tracker.speechReady()).toBe(true);
    tracker.reach({ x: 200, y: 200 });
    expect(tracker.frame(0.0003)).toBe("complete");
    tracker.begin([], false);
    expect(tracker.firstVoicedFrame()).toBeNull();
  });

  it("shares the silence, gain-cap and loud-capture contracts", () => {
    expect(captureNormalizationGain(0)).toBe(1);
    expect(captureNormalizationGain(0.000099)).toBe(1);
    expect(captureNormalizationGain(0.0001)).toBe(50);
    expect(captureNormalizationGain(0.001)).toBe(50);
    expect(captureNormalizationGain(0.05)).toBe(1);
    expect(captureNormalizationGain(0.5)).toBe(0.1);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, -0.1])("refuses invalid RMS %s", (level) => {
    const tracker = createRoundTracker();
    tracker.begin([], false);
    expect(() => tracker.observe(level)).toThrow(RangeError);
    expect(() => tracker.frame(level)).toThrow(RangeError);
  });
});
