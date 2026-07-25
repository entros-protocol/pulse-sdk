import type { CurveTracePoint, CurveTraceOutline } from "./types";

/**
 * Number of points in a transmitted curve outline. Calibration-coupled with the
 * executor's curve-trace scorer (its teleport threshold and test corpus assume
 * ~this many equal-time points) — keep the two in sync if ever tuned.
 */
export const CURVE_OUTLINE_POINTS = 64;

/**
 * Coordinates beyond this magnitude are malformed for a 200-unit viewBox (a
 * pointer leaving the box exceeds 200 only slightly). Dropping them keeps
 * NaN / Infinity / absurd values off the wire; the executor enforces the same
 * envelope server-side.
 */
const COORD_ENVELOPE = 1e4;

function round1(v: number): number {
  return Math.round(v * 10) / 10;
}

/**
 * Resample a raw curve trace to a coarse, EQUAL-TIME outline for transmission.
 *
 * The "trace the curve" gesture is captured as `{x, y, t}` samples at irregular
 * pointer-event intervals. The validation service's kinematic checks (speed
 * variation, teleport detection) are only meaningful at a CONSTANT timestep, so
 * we resample against the real time axis — not by index or arc-length — to `n`
 * points and drop the timestamps. Only the coarse `{x, y}` outline (viewBox-200
 * coords) plus the total `duration_ms` ever leaves the device; the raw per-point
 * timing stays local.
 *
 * Hardened against malformed / hostile input: `n` must be an integer ≥ 2, and
 * points with a non-finite timestamp or a non-finite / out-of-envelope
 * coordinate are dropped up front so nothing corrupts the outline (`NaN`/`null`
 * on the wire) or defeats the duration guard (e.g. an `Infinity` endpoint
 * timestamp). Returns `undefined` when fewer than two valid points remain or the
 * valid span is non-positive; the caller then omits the field.
 */
export function resampleCurveTrace(
  raw: CurveTracePoint[],
  n: number = CURVE_OUTLINE_POINTS
): CurveTraceOutline | undefined {
  if (!Number.isInteger(n) || n < 2) return undefined;

  // Drop non-finite timestamps and non-finite / out-of-envelope coordinates so a
  // malformed trace can neither corrupt the outline nor defeat the duration
  // guard. `Math.abs(NaN) <= COORD_ENVELOPE` is `false`, so this also removes
  // NaN / Infinity coordinates.
  const pts = raw.filter(
    (p) =>
      Number.isFinite(p.t) &&
      Math.abs(p.x) <= COORD_ENVELOPE &&
      Math.abs(p.y) <= COORD_ENVELOPE,
  );
  if (pts.length < 2) return undefined;

  const t0 = pts[0]!.t;
  const duration = pts[pts.length - 1]!.t - t0;
  if (!Number.isFinite(duration) || duration <= 0) return undefined;

  const points: [number, number][] = [];
  let cursor = 0; // left index of the source segment bracketing the target time

  for (let i = 0; i < n; i++) {
    const targetT = t0 + (i / (n - 1)) * duration;

    // Advance to the segment [cursor, cursor+1] whose right end reaches targetT.
    while (cursor < pts.length - 2 && pts[cursor + 1]!.t < targetT) {
      cursor++;
    }

    const a = pts[cursor]!;
    const b = pts[cursor + 1]!;
    const span = b.t - a.t;
    // Zero-width segment (duplicate timestamps) → left endpoint; else the
    // clamped fraction along the segment (clamp guards any non-monotonic t).
    const frac = span > 0 ? Math.min(1, Math.max(0, (targetT - a.t) / span)) : 0;

    points.push([round1(a.x + (b.x - a.x) * frac), round1(a.y + (b.y - a.y) * frac)]);
  }

  return { points, duration_ms: round1(duration) };
}
