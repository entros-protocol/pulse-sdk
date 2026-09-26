/**
 * The coarse outline of one round's trace.
 *
 * Only pressed points count. The outline resamples them evenly along the
 * trace's length and quantises each to the integer grid, so no pressure, no
 * raw coordinate and no timing leaves the device. Resampling by length keeps
 * the outline within half a spacing of the trace however its speed varied: a
 * pause at one dot adds no points there and takes none from the rest.
 */

import { WAYPOINT_REACH } from "./tracker";
import { COORDINATE_MAX, type GridPoint, MAX_PATH_POINTS, MIN_PATH_POINTS } from "./transcript";

/** Points in each committed outline. */
export const COARSE_PATH_POINTS = 64;
/**
 * The outline joins points spaced along the trace, so it can cut the corner
 * at a waypoint the trace itself passed through. Reach on the outline is
 * widened by this much.
 */
export const OUTLINE_MARGIN = 30;

export interface TracePoint {
  /** Pixels from the surface's left edge. */
  x: number;
  /** Pixels from the surface's top edge. */
  y: number;
  /** Milliseconds on any monotonic clock. Orders the points and nothing else. */
  t: number;
}

export interface SurfaceSize {
  width: number;
  height: number;
}

export type CoarsePathFailure = "too_few_points" | "zero_length" | "invalid_point";

export class CoarsePathError extends Error {
  constructor(readonly reason: CoarsePathFailure) {
    super(reason);
    this.name = "CoarsePathError";
  }
}

const clampToGrid = (value: number) => Math.min(COORDINATE_MAX, Math.max(0, Math.round(value)));

/** One surface point on the 0 to 1000 grid. */
export function toGridPoint(point: { x: number; y: number }, surface: SurfaceSize): GridPoint {
  return {
    x: clampToGrid((point.x / surface.width) * COORDINATE_MAX),
    y: clampToGrid((point.y / surface.height) * COORDINATE_MAX),
  };
}

/** Resamples pressed trace points evenly along their length and quantises them to the grid. */
export function toCoarsePath(
  points: readonly TracePoint[],
  surface: SurfaceSize,
  count: number = COARSE_PATH_POINTS,
): GridPoint[] {
  if (count < MIN_PATH_POINTS || count > MAX_PATH_POINTS) {
    throw new RangeError(`outline point count ${count} is out of range`);
  }
  if (!(surface.width > 0) || !(surface.height > 0)) throw new CoarsePathError("invalid_point");
  if (points.some((point) => ![point.x, point.y, point.t].every(Number.isFinite))) {
    throw new CoarsePathError("invalid_point");
  }
  if (points.length < 2) throw new CoarsePathError("too_few_points");
  // The trace in time order, on the unrounded grid.
  const ordered = [...points]
    .sort((left, right) => left.t - right.t)
    .map((point) => ({
      x: (point.x / surface.width) * COORDINATE_MAX,
      y: (point.y / surface.height) * COORDINATE_MAX,
    }));
  const along = [0];
  for (let index = 1; index < ordered.length; index++) {
    const dx = ordered[index]!.x - ordered[index - 1]!.x;
    const dy = ordered[index]!.y - ordered[index - 1]!.y;
    along.push(along[index - 1]! + Math.sqrt(dx * dx + dy * dy));
  }
  const total = along[along.length - 1]!;
  if (!(total > 0)) throw new CoarsePathError("zero_length");

  const out: GridPoint[] = [];
  let cursor = 0;
  for (let index = 0; index < count; index++) {
    const at = (total * index) / (count - 1);
    while (cursor < ordered.length - 2 && along[cursor + 1]! < at) cursor++;
    const left = ordered[cursor]!;
    const right = ordered[cursor + 1]!;
    const span = along[cursor + 1]! - along[cursor]!;
    const fraction = span > 0 ? Math.min(1, Math.max(0, (at - along[cursor]!) / span)) : 0;
    out.push({
      x: clampToGrid(left.x + (right.x - left.x) * fraction),
      y: clampToGrid(left.y + (right.y - left.y) * fraction),
    });
  }
  return out;
}

export interface PathScore {
  /** Waypoints reached in the issued order, counted from the first. */
  reached: number;
  /** Every waypoint was reached, in the issued order. */
  inOrder: boolean;
}

/**
 * Walks an outline from its start and counts the waypoints it reaches in the
 * issued order, by the rule the server applies to the committed outline. A
 * waypoint counts only after the one before it, and on the outline step that
 * reached the one before, only further along that step.
 */
export function scorePath(waypoints: readonly GridPoint[], outline: readonly GridPoint[]): PathScore {
  const reach = (WAYPOINT_REACH + OUTLINE_MARGIN) ** 2;
  let reached = 0;
  // Where the last waypoint was reached: its outline step and how far along it.
  let lastStep = -1;
  let lastAlong = 0;
  for (let step = 0; step < outline.length; step++) {
    const start = outline[step]!;
    const end = outline[Math.min(step + 1, outline.length - 1)]!;
    for (let waypoint = waypoints[reached]; waypoint; waypoint = waypoints[reached]) {
      const { distance, along } = closest(waypoint, start, end);
      if (distance > reach || (step === lastStep && along < lastAlong)) break;
      reached++;
      lastStep = step;
      lastAlong = along;
    }
  }
  return { reached, inOrder: waypoints.length > 0 && reached === waypoints.length };
}

/** Squared distance from `point` to the segment, and how far along it the closest point lies. */
function closest(point: GridPoint, start: GridPoint, end: GridPoint): { distance: number; along: number } {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  const along =
    lengthSquared === 0
      ? 0
      : Math.min(1, Math.max(0, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared));
  const cx = start.x + along * dx - point.x;
  const cy = start.y + along * dy - point.y;
  return { distance: cx * cx + cy * cy, along };
}
