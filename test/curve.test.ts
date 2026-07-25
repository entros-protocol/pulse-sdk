import { describe, it, expect } from "vitest";
import { resampleCurveTrace, CURVE_OUTLINE_POINTS } from "../src/sensor/curve";
import type { CurveTracePoint } from "../src/sensor/types";

/** n points along y=x, uniformly spaced `dt` ms apart. */
function line(n: number, dt: number): CurveTracePoint[] {
  return Array.from({ length: n }, (_, i) => ({ x: i, y: i, t: i * dt }));
}

describe("resampleCurveTrace", () => {
  it("returns undefined for fewer than two points", () => {
    expect(resampleCurveTrace([])).toBeUndefined();
    expect(resampleCurveTrace([{ x: 1, y: 2, t: 0 }])).toBeUndefined();
  });

  it("returns undefined for non-positive duration (all-equal timestamps)", () => {
    const same: CurveTracePoint[] = [
      { x: 0, y: 0, t: 5 },
      { x: 10, y: 10, t: 5 },
      { x: 20, y: 20, t: 5 },
    ];
    expect(resampleCurveTrace(same)).toBeUndefined();
  });

  it("produces exactly CURVE_OUTLINE_POINTS points and the true duration", () => {
    const out = resampleCurveTrace(line(10, 100));
    expect(out).toBeDefined();
    expect(out!.points.length).toBe(CURVE_OUTLINE_POINTS);
    expect(out!.duration_ms).toBeCloseTo(900, 5); // (10 - 1) * 100
  });

  it("preserves the first and last raw points at the endpoints", () => {
    const out = resampleCurveTrace(line(10, 100))!;
    expect(out.points[0]).toEqual([0, 0]);
    expect(out.points[CURVE_OUTLINE_POINTS - 1]).toEqual([9, 9]);
  });

  it("resamples by TIME, not index — a late-in-index, near-zero-time point does not skew the middle", () => {
    // Halfway in TIME (~500ms) sits at the geometric midpoint of segment 0→1,
    // regardless of the third point being dense in index but ~0 in time.
    const raw: CurveTracePoint[] = [
      { x: 0, y: 0, t: 0 },
      { x: 100, y: 0, t: 1000 },
      { x: 100, y: 100, t: 1000.001 },
    ];
    const out = resampleCurveTrace(raw, 3)!;
    expect(out.points[1]![0]).toBeCloseTo(50, 0);
    expect(out.points[1]![1]).toBeCloseTo(0, 0);
    expect(out.points[2]).toEqual([100, 100]);
  });

  it("handles a mid-trace pause (duplicate interior timestamps) without NaN", () => {
    const raw: CurveTracePoint[] = [
      { x: 0, y: 0, t: 0 },
      { x: 50, y: 50, t: 500 },
      { x: 50, y: 50, t: 500 }, // same clock tick
      { x: 100, y: 100, t: 1000 },
    ];
    const out = resampleCurveTrace(raw)!;
    expect(out.points.length).toBe(CURVE_OUTLINE_POINTS);
    expect(out.points.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y))).toBe(true);
  });

  it("clamps against non-monotonic timestamps without NaN", () => {
    const raw: CurveTracePoint[] = [
      { x: 0, y: 0, t: 0 },
      { x: 100, y: 0, t: 900 },
      { x: 50, y: 50, t: 400 }, // goes backwards
      { x: 200, y: 200, t: 1000 },
    ];
    const out = resampleCurveTrace(raw)!;
    expect(out.points.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y))).toBe(true);
  });

  it("rounds coordinates to one decimal", () => {
    const raw: CurveTracePoint[] = [
      { x: 0, y: 0, t: 0 },
      { x: 1 / 3, y: 2 / 3, t: 1000 },
    ];
    const out = resampleCurveTrace(raw)!;
    for (const [x, y] of out.points) {
      expect(x).toBe(Math.round(x * 10) / 10);
      expect(y).toBe(Math.round(y * 10) / 10);
    }
  });

  it("upsamples a sparse trace and downsamples a dense one, both to N", () => {
    expect(resampleCurveTrace(line(3, 100))!.points.length).toBe(CURVE_OUTLINE_POINTS);
    expect(resampleCurveTrace(line(500, 2))!.points.length).toBe(CURVE_OUTLINE_POINTS);
  });

  it("returns undefined for a strictly-decreasing (negative-duration) trace", () => {
    const decreasing: CurveTracePoint[] = [
      { x: 0, y: 0, t: 1000 },
      { x: 10, y: 10, t: 500 },
      { x: 20, y: 20, t: 0 },
    ];
    expect(resampleCurveTrace(decreasing)).toBeUndefined();
  });

  it("resamples a minimal 2-point trace with endpoints preserved", () => {
    const out = resampleCurveTrace([
      { x: 5, y: 7, t: 0 },
      { x: 9, y: 3, t: 500 },
    ])!;
    expect(out.points.length).toBe(CURVE_OUTLINE_POINTS);
    expect(out.points[0]).toEqual([5, 7]);
    expect(out.points[CURVE_OUTLINE_POINTS - 1]).toEqual([9, 3]);
  });

  it("rejects a non-integer, NaN, or Infinite n (no hang, no empty outline)", () => {
    const raw = line(10, 100);
    expect(resampleCurveTrace(raw, Number.NaN)).toBeUndefined();
    expect(resampleCurveTrace(raw, Number.POSITIVE_INFINITY)).toBeUndefined();
    expect(resampleCurveTrace(raw, 2.5)).toBeUndefined();
    expect(resampleCurveTrace(raw, 1)).toBeUndefined();
  });

  it("drops a non-finite endpoint timestamp and resamples the finite remainder", () => {
    const out = resampleCurveTrace([
      { x: 0, y: 0, t: 0 },
      { x: 10, y: 10, t: 100 },
      { x: 20, y: 20, t: Number.POSITIVE_INFINITY },
    ])!;
    expect(out.points.length).toBe(CURVE_OUTLINE_POINTS);
    expect(out.points[0]).toEqual([0, 0]);
    expect(out.points[CURVE_OUTLINE_POINTS - 1]).toEqual([10, 10]); // last finite point
    expect(out.points.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y))).toBe(true);
  });

  it("returns undefined when dropping non-finite points leaves fewer than two", () => {
    expect(
      resampleCurveTrace([
        { x: 0, y: 0, t: 0 },
        { x: 10, y: 10, t: Number.POSITIVE_INFINITY },
      ]),
    ).toBeUndefined();
  });

  it("drops non-finite / out-of-envelope coordinates (no null on the wire)", () => {
    const out = resampleCurveTrace([
      { x: 0, y: 0, t: 0 },
      { x: Number.NaN, y: 50, t: 250 }, // dropped
      { x: 1e9, y: 50, t: 500 }, // dropped (out of envelope)
      { x: 100, y: 100, t: 1000 },
    ])!;
    expect(out.points[0]).toEqual([0, 0]);
    expect(out.points[CURVE_OUTLINE_POINTS - 1]).toEqual([100, 100]);
    expect(out.points.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y))).toBe(true);
  });

  it("drops an interior NaN-timestamp point, preserving the last point", () => {
    const out = resampleCurveTrace([
      { x: 0, y: 0, t: 0 },
      { x: 50, y: 50, t: Number.NaN }, // dropped
      { x: 100, y: 100, t: 1000 },
    ])!;
    expect(out.points[CURVE_OUTLINE_POINTS - 1]).toEqual([100, 100]);
    expect(out.points.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y))).toBe(true);
  });
});
