import type { MotionSample, TouchSample } from "../sensor/types";
import { condense, mean, variance, entropy, autocorrelation } from "./statistics";
import { realFFT, bandEnergy, peakInBand, nextPow2 } from "./fft";

// v2 motion block widens 54 → 81: 54 legacy (jerk + jounce stats × 6 axes,
// jitter variance × 6) followed by 27 new features. Order is fixed by
// `MOTION_FEATURE_COUNT` and asserted in tests/extraction.test.ts.
export const MOTION_LEGACY_COUNT = 54;
export const MOTION_V2_ADDITIONS = 27;
export const MOTION_FEATURE_COUNT = MOTION_LEGACY_COUNT + MOTION_V2_ADDITIONS;

// v2 touch block widens 36 → 57: 36 legacy followed by 21 new features.
export const TOUCH_LEGACY_COUNT = 36;
export const TOUCH_V2_ADDITIONS = 21;
export const TOUCH_FEATURE_COUNT = TOUCH_LEGACY_COUNT + TOUCH_V2_ADDITIONS;

// Mouse-dynamics keeps width parity with the motion block so that desktop
// captures fuse cleanly into the same fingerprint slot as mobile IMU
// captures. The first 54 entries are the legacy mouse-dynamics features;
// the remaining 27 are zero (no IMU on desktop), keeping the bit-influence
// share per modality identical across device classes.
export const MOUSE_DYNAMICS_FEATURE_COUNT = MOTION_FEATURE_COUNT;

/**
 * Compute per-sample acceleration magnitude |a| = √(ax² + ay² + az²) and
 * linearly resample to a target frame count. Surfaced for server-side
 * analysis paired against the F0 contour; the two time-series must share
 * the same frame count when consumed downstream.
 *
 * Returns an empty array if motion data is absent or too short.
 */
export function extractAccelerationMagnitude(
  samples: MotionSample[],
  targetFrameCount: number,
): number[] {
  if (samples.length < 2 || targetFrameCount < 2) return [];

  const magnitudes = samples.map((s) => Math.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az));

  if (magnitudes.length === targetFrameCount) return magnitudes;

  // Linear resample: map target index i to source position (i / (target-1)) * (source-1)
  const out = new Array<number>(targetFrameCount);
  const srcLen = magnitudes.length;
  const scale = (srcLen - 1) / (targetFrameCount - 1);
  for (let i = 0; i < targetFrameCount; i++) {
    const pos = i * scale;
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, srcLen - 1);
    const t = pos - lo;
    out[i] = magnitudes[lo]! * (1 - t) + magnitudes[hi]! * t;
  }
  return out;
}

/**
 * Extract kinematic features from motion (IMU) data.
 *
 * Layout (`MOTION_FEATURE_COUNT = 81`):
 *   `[0..48)`  legacy: 6 axes × (jerk stats 4 + jounce stats 4)
 *   `[48..54)` legacy: jitter variance per axis (6)
 *   `[54..60)` v2:    cross-axis covariance (6 selected pairs)
 *   `[60..72)` v2:    FFT band energy in {0-2, 2-6, 6-12, 12-30} Hz × {ax, ay, az}
 *   `[72..74)` v2:    physiological tremor peak frequency + amplitude (4-12 Hz)
 *   `[74..76)` v2:    direction-reversal rate per axis: mean, variance across {ax, ay, az}
 *   `[76]`     v2:    mean angular velocity (|gyro| over the capture)
 *   `[77..81)` v2:    motion-magnitude autocorrelation at lags {1, 5, 10, 25}
 *
 * @privacyGuarantee Operates on already-on-device IMU samples and emits
 * statistical / spectral aggregates (variances, covariances, band sums,
 * autocorrelation scalars). The full sample stream is never transmitted.
 */
export function extractMotionFeatures(samples: MotionSample[]): number[] {
  if (samples.length < 5) return new Array(MOTION_FEATURE_COUNT).fill(0);

  // Extract acceleration and rotation time series
  const axes = {
    ax: samples.map((s) => s.ax),
    ay: samples.map((s) => s.ay),
    az: samples.map((s) => s.az),
    gx: samples.map((s) => s.gx),
    gy: samples.map((s) => s.gy),
    gz: samples.map((s) => s.gz),
  };

  const features: number[] = [];

  for (const values of Object.values(axes)) {
    // Jerk = 3rd derivative of position = 1st derivative of acceleration
    const jerk = derivative(values);
    // Jounce = 4th derivative of position = 2nd derivative of acceleration
    const jounce = derivative(jerk);

    const jerkStats = condense(jerk);
    const jounceStats = condense(jounce);

    features.push(
      jerkStats.mean,
      jerkStats.variance,
      jerkStats.skewness,
      jerkStats.kurtosis,
      jounceStats.mean,
      jounceStats.variance,
      jounceStats.skewness,
      jounceStats.kurtosis
    );
  }

  // Jitter variance per axis: variance of windowed jerk variance.
  // Captures temporal fluctuation in the motion signal.
  for (const values of Object.values(axes)) {
    const jerk = derivative(values);
    const windowSize = Math.max(5, Math.floor(jerk.length / 4));
    const windowVariances: number[] = [];
    for (let i = 0; i <= jerk.length - windowSize; i += windowSize) {
      windowVariances.push(variance(jerk.slice(i, i + windowSize)));
    }
    features.push(windowVariances.length >= 2 ? variance(windowVariances) : 0);
  }

  // ---- v2 additions ----
  features.push(...computeMotionV2(axes, samples));

  return features;
}

/**
 * v2 motion additions (27 features). Pulled into a dedicated helper so the
 * legacy 54-feature block stays isolated and visually identifiable in the
 * git history of `extractMotionFeatures`.
 */
function computeMotionV2(
  axes: Record<"ax" | "ay" | "az" | "gx" | "gy" | "gz", number[]>,
  samples: MotionSample[]
): number[] {
  const out: number[] = [];

  // 1. Cross-axis covariance — 6 selected pairs (per blueprint §2.2). The
  // pairs target identity-bearing motor coordinations: accel-gyro coupling
  // (ax-gy, ay-gx, az-gz) for natural hand sway, accel-accel coupling
  // (ax-az, ay-az) for axis-of-grip leakage, and gyro-gyro coupling
  // (gx-gy) for wrist-rotation patterns.
  const covPairs: Array<[number[], number[]]> = [
    [axes.ax, axes.gy],
    [axes.ay, axes.gx],
    [axes.az, axes.gz],
    [axes.ax, axes.az],
    [axes.ay, axes.az],
    [axes.gx, axes.gy],
  ];
  for (const [a, b] of covPairs) out.push(covariance(a, b));

  // 2. FFT band energy on the 3 accelerometer axes.
  // Sample rate is recovered from timestamps so we report energy in
  // physical Hz rather than bin units (IMU rates vary 50-200 Hz across
  // devices).
  const sampleRate = sampleRateFromTimestamps(samples.map((s) => s.timestamp));
  const fftSize = nextPow2(Math.max(64, axes.ax.length));
  const bands: Array<[number, number]> = [
    [0, 2],
    [2, 6],
    [6, 12],
    [12, 30],
  ];

  // Pre-FFT each accel axis once; reuse the spectra for both band-energy
  // and the magnitude path below.
  const accelSpectra = [axes.ax, axes.ay, axes.az].map((axis) =>
    realFFT(meanCenter(axis), fftSize)
  );
  for (const spectrum of accelSpectra) {
    for (const [lo, hi] of bands) {
      out.push(bandEnergy(spectrum.real, spectrum.imag, sampleRate, lo, hi));
    }
  }

  // 3. Physiological-tremor peak (4-12 Hz) on motion magnitude.
  const magnitude = samples.map((s) =>
    Math.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az)
  );
  const magSpectrum = realFFT(meanCenter(magnitude), fftSize);
  const tremor = peakInBand(
    magSpectrum.real,
    magSpectrum.imag,
    sampleRate,
    4,
    12
  );
  out.push(tremor.freq, tremor.amplitude);

  // 4. Direction-reversal rate per second per accel axis (mean, variance).
  // A reversal is a sign change of velocity (= sign change of d/dt of
  // acceleration). Rate is normalized by capture duration so it's
  // dimension-stable across IMU sample-rates.
  const duration = captureDurationSec(samples);
  const reversalRates = [axes.ax, axes.ay, axes.az].map((axis) =>
    duration > 0 ? signChangeCount(derivative(axis)) / duration : 0
  );
  out.push(mean(reversalRates), variance(reversalRates));

  // 5. Mean angular velocity (|gyro| over the capture).
  let gyroSum = 0;
  for (let i = 0; i < samples.length; i++) {
    const gx = samples[i]!.gx;
    const gy = samples[i]!.gy;
    const gz = samples[i]!.gz;
    gyroSum += Math.sqrt(gx * gx + gy * gy + gz * gz);
  }
  out.push(samples.length > 0 ? gyroSum / samples.length : 0);

  // 6. Motion-magnitude autocorrelation at lags 1, 5, 10, 25 — captures
  // periodic structure (gait, tremor harmonics) that escapes the
  // moment-based features.
  for (const lag of [1, 5, 10, 25]) {
    out.push(autocorrelation(magnitude, lag));
  }

  return out;
}

/**
 * Extract kinematic features from touch data.
 *
 * Layout (`TOUCH_FEATURE_COUNT = 57`):
 *   `[0..32)`  legacy: velocity / accel / pressure / area / jerk stats (32)
 *   `[32..36)` legacy: jitter variance for {vx, vy, pressure, area} (4)
 *   `[36..40)` v2:    pressure first-derivative stats (mean, var, skew, kurt)
 *   `[40..42)` v2:    contact aspect-ratio stats (mean, var)
 *   `[42..44)` v2:    contact-area first-derivative stats (mean, var)
 *   `[44..47)` v2:    trajectory curvature stats (mean, var, skew)
 *   `[47..50)` v2:    velocity autocorrelation at lags {1, 3, 5}
 *   `[50..54)` v2:    inter-touch gap duration stats (mean, var, skew, kurt)
 *   `[54]`     v2:    path efficiency (straight-line / total path length)
 *   `[55..57)` v2:    per-stroke total path length: mean, variance
 *
 * @privacyGuarantee Operates on already-on-device touch samples and emits
 * statistical aggregates only. The full coordinate stream is never
 * transmitted; downstream phase-content (e.g. typed text) is not
 * recoverable from the per-stroke summaries.
 */
export function extractTouchFeatures(samples: TouchSample[]): number[] {
  if (samples.length < 5) return new Array(TOUCH_FEATURE_COUNT).fill(0);

  const x = samples.map((s) => s.x);
  const y = samples.map((s) => s.y);
  const pressure = samples.map((s) => s.pressure);
  const area = samples.map((s) => s.width * s.height);

  const features: number[] = [];

  // X velocity and acceleration
  const vx = derivative(x);
  const accX = derivative(vx);
  features.push(...Object.values(condense(vx)));
  features.push(...Object.values(condense(accX)));

  // Y velocity and acceleration
  const vy = derivative(y);
  const accY = derivative(vy);
  features.push(...Object.values(condense(vy)));
  features.push(...Object.values(condense(accY)));

  // Pressure statistics
  features.push(...Object.values(condense(pressure)));

  // Contact area statistics
  features.push(...Object.values(condense(area)));

  // Jerk of touch path
  const jerkX = derivative(accX);
  const jerkY = derivative(accY);
  features.push(...Object.values(condense(jerkX)));
  features.push(...Object.values(condense(jerkY)));

  // Jitter variance for touch signals: detects synthetic smoothness
  for (const values of [vx, vy, pressure, area]) {
    const windowSize = Math.max(5, Math.floor(values.length / 4));
    const windowVariances: number[] = [];
    for (let i = 0; i <= values.length - windowSize; i += windowSize) {
      windowVariances.push(variance(values.slice(i, i + windowSize)));
    }
    features.push(windowVariances.length >= 2 ? variance(windowVariances) : 0);
  }

  // ---- v2 additions ----
  features.push(...computeTouchV2(samples, vx, vy));

  return features;
}

/**
 * v2 touch additions (21 features). Pulled into a helper so the legacy
 * 36-feature block stays a visually identifiable unit.
 */
function computeTouchV2(
  samples: TouchSample[],
  vx: number[],
  vy: number[]
): number[] {
  const out: number[] = [];

  // 1. Pressure first-derivative stats (4) — temporal RATE of pressure
  // variation, complementing the existing pressure mean/var/skew/kurt.
  const pressure = samples.map((s) => s.pressure);
  const dPressure = derivative(pressure);
  out.push(...Object.values(condense(dPressure)));

  // 2. Contact aspect ratio stats (mean, variance). width/height captures
  // finger-vs-thumb-vs-stylus identity even when raw area drifts.
  const aspect = samples.map((s) => {
    const h = s.height;
    return h > 0 ? s.width / h : 0;
  });
  out.push(mean(aspect), variance(aspect));

  // 3. Contact-area first-derivative stats (mean, variance) — rate of
  // pressure-spread change, a finer-grained signal than raw area moments.
  const area = samples.map((s) => s.width * s.height);
  const dArea = derivative(area);
  out.push(mean(dArea), variance(dArea));

  // 4. Trajectory curvature stats (mean, var, skew). Curvature is the
  // absolute angle change between successive velocity vectors —
  // identity-bearing motor coordination.
  const curvatures: number[] = [];
  for (let i = 1; i < vx.length; i++) {
    const a1 = Math.atan2(vy[i - 1] ?? 0, vx[i - 1] ?? 0);
    const a2 = Math.atan2(vy[i] ?? 0, vx[i] ?? 0);
    let d = a2 - a1;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    curvatures.push(Math.abs(d));
  }
  const curvStats = condense(curvatures);
  out.push(curvStats.mean, curvStats.variance, curvStats.skewness);

  // 5. Velocity-magnitude autocorrelation at short lags — captures rhythm
  // in touch motion below the resolution of moment statistics.
  const speed = vx.map((dx, i) => {
    const dy = vy[i] ?? 0;
    return Math.sqrt(dx * dx + dy * dy);
  });
  for (const lag of [1, 3, 5]) out.push(autocorrelation(speed, lag));

  // 6. Inter-touch gap duration stats (mean, var, skew, kurt). Gaps are
  // the millisecond intervals between successive touch events — touch
  // rhythm is highly individual (think tap cadence vs swipe cadence).
  const gaps: number[] = [];
  for (let i = 1; i < samples.length; i++) {
    gaps.push((samples[i]?.timestamp ?? 0) - (samples[i - 1]?.timestamp ?? 0));
  }
  out.push(...Object.values(condense(gaps)));

  // 7. Path efficiency = straight-line displacement / total path length.
  // 1.0 = perfectly straight movement, near-0 = highly tortuous.
  const totalPath = speed.reduce((a, b) => a + b, 0);
  const dx = (samples[samples.length - 1]?.x ?? 0) - (samples[0]?.x ?? 0);
  const dy = (samples[samples.length - 1]?.y ?? 0) - (samples[0]?.y ?? 0);
  const straight = Math.sqrt(dx * dx + dy * dy);
  out.push(totalPath > 0 ? straight / totalPath : 0);

  // 8. Per-stroke total path length: split on speed troughs (≤ 0.5 px/sample
  // from rest, matching the mouse-dynamics pause threshold), then take
  // mean and variance. Captures motor-planning style — burst-then-pause
  // vs continuous-glide users.
  const strokeLengths = perStrokePathLengths(speed);
  out.push(mean(strokeLengths), variance(strokeLengths));

  return out;
}

/** Split a speed series into stroke segments at rest-points and return
 *  the cumulative speed (≈ path length in pixels) of each stroke.
 *  A "stroke" is a contiguous run of speed ≥ threshold. */
function perStrokePathLengths(speed: number[]): number[] {
  const PAUSE_THRESHOLD = 0.5;
  const lengths: number[] = [];
  let acc = 0;
  let inStroke = false;
  for (const s of speed) {
    if (s >= PAUSE_THRESHOLD) {
      acc += s;
      inStroke = true;
    } else if (inStroke) {
      lengths.push(acc);
      acc = 0;
      inStroke = false;
    }
  }
  if (inStroke && acc > 0) lengths.push(acc);
  return lengths;
}

/** Compute discrete derivative (differences between consecutive values) */
function derivative(values: number[]): number[] {
  const d: number[] = [];
  for (let i = 1; i < values.length; i++) {
    d.push((values[i] ?? 0) - (values[i - 1] ?? 0));
  }
  return d;
}

/** Subtract the arithmetic mean from a series; returns a new array. */
function meanCenter(values: number[]): number[] {
  if (values.length === 0) return [];
  let sum = 0;
  for (const v of values) sum += v;
  const m = sum / values.length;
  return values.map((v) => v - m);
}

/** Sample covariance Cov(a, b) = mean((a-mean(a))(b-mean(b))). */
function covariance(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 0;
  let sumA = 0;
  let sumB = 0;
  for (let i = 0; i < n; i++) {
    sumA += a[i] ?? 0;
    sumB += b[i] ?? 0;
  }
  const meanA = sumA / n;
  const meanB = sumB / n;
  let cov = 0;
  for (let i = 0; i < n; i++) {
    cov += ((a[i] ?? 0) - meanA) * ((b[i] ?? 0) - meanB);
  }
  return cov / (n - 1);
}

/** Count strict sign changes (zero-crossings excluding zero-runs). */
function signChangeCount(values: number[]): number {
  let count = 0;
  let last = 0;
  for (const v of values) {
    if (v > 0 && last < 0) count++;
    else if (v < 0 && last > 0) count++;
    if (v !== 0) last = v;
  }
  return count;
}

/**
 * Recover the sample rate (Hz) from a millisecond-timestamped sensor
 * stream. Returns 0 when the input is too short to estimate or contains
 * non-monotone timestamps (defensive — pulse.ts caps this with a default
 * downstream so 0 propagates as "no spectral feature available").
 */
function sampleRateFromTimestamps(timestampsMs: number[]): number {
  if (timestampsMs.length < 2) return 0;
  const span = (timestampsMs[timestampsMs.length - 1] ?? 0) - (timestampsMs[0] ?? 0);
  if (!Number.isFinite(span) || span <= 0) return 0;
  return ((timestampsMs.length - 1) * 1000) / span;
}

/** Capture duration in seconds from a millisecond-timestamped sample set. */
function captureDurationSec(
  samples: Array<{ timestamp: number }>
): number {
  if (samples.length < 2) return 0;
  const span =
    (samples[samples.length - 1]?.timestamp ?? 0) -
    (samples[0]?.timestamp ?? 0);
  return Number.isFinite(span) && span > 0 ? span / 1000 : 0;
}

/**
 * Extract mouse dynamics features as a desktop replacement for motion sensor data.
 * Captures behavioral patterns from mouse/pointer movement that are user-specific:
 * path curvature, speed patterns, micro-corrections, pause behavior.
 *
 * Returns: `MOUSE_DYNAMICS_FEATURE_COUNT` (= `MOTION_FEATURE_COUNT`) values.
 * The first 54 entries are the legacy mouse-dynamics signal; the trailing
 * v2-block slots stay zero on desktop so the per-modality bit-influence
 * share matches a mobile IMU capture under the new pipeline.
 */
export function extractMouseDynamics(samples: TouchSample[]): number[] {
  if (samples.length < 10) return new Array(MOUSE_DYNAMICS_FEATURE_COUNT).fill(0);

  const x = samples.map((s) => s.x);
  const y = samples.map((s) => s.y);
  const pressure = samples.map((s) => s.pressure);
  const area = samples.map((s) => s.width * s.height);

  // Velocity
  const vx = derivative(x);
  const vy = derivative(y);
  const speed = vx.map((dx, i) => Math.sqrt(dx * dx + (vy[i] ?? 0) * (vy[i] ?? 0)));

  // Acceleration
  const accX = derivative(vx);
  const accY = derivative(vy);
  const acc = accX.map((ax, i) => Math.sqrt(ax * ax + (accY[i] ?? 0) * (accY[i] ?? 0)));

  // Jerk (derivative of acceleration)
  const jerkX = derivative(accX);
  const jerkY = derivative(accY);
  const jerk = jerkX.map((jx, i) => Math.sqrt(jx * jx + (jerkY[i] ?? 0) * (jerkY[i] ?? 0)));

  // Path curvature: angle change between consecutive movement vectors
  const curvatures: number[] = [];
  for (let i = 1; i < vx.length; i++) {
    const angle1 = Math.atan2(vy[i - 1] ?? 0, vx[i - 1] ?? 0);
    const angle2 = Math.atan2(vy[i] ?? 0, vx[i] ?? 0);
    let diff = angle2 - angle1;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    curvatures.push(Math.abs(diff));
  }

  // Movement directions for directional entropy
  const directions = vx.map((dx, i) => Math.atan2(vy[i] ?? 0, dx));

  // Micro-corrections: direction reversals
  let reversals = 0;
  for (let i = 2; i < directions.length; i++) {
    const d1 = directions[i - 1]! - directions[i - 2]!;
    const d2 = directions[i]! - directions[i - 1]!;
    if (d1 * d2 < 0) reversals++;
  }
  const reversalRate = directions.length > 2 ? reversals / (directions.length - 2) : 0;
  const reversalMagnitude = curvatures.length > 0
    ? curvatures.reduce((a, b) => a + b, 0) / curvatures.length
    : 0;

  // Pause detection: frames where speed is near zero
  const speedThreshold = 0.5;
  const pauseFrames = speed.filter((s) => s < speedThreshold).length;
  const pauseRatio = speed.length > 0 ? pauseFrames / speed.length : 0;

  // Path efficiency: straight-line distance / total path length
  const totalPathLength = speed.reduce((a, b) => a + b, 0);
  const straightLine = Math.sqrt(
    (x[x.length - 1]! - x[0]!) ** 2 + (y[y.length - 1]! - y[0]!) ** 2
  );
  const pathEfficiency = totalPathLength > 0 ? straightLine / totalPathLength : 0;

  // Movement durations between pauses
  const movementDurations: number[] = [];
  let currentDuration = 0;
  for (const s of speed) {
    if (s >= speedThreshold) {
      currentDuration++;
    } else if (currentDuration > 0) {
      movementDurations.push(currentDuration);
      currentDuration = 0;
    }
  }
  if (currentDuration > 0) movementDurations.push(currentDuration);

  // Segment lengths between direction changes
  const segmentLengths: number[] = [];
  let segLen = 0;
  for (let i = 1; i < directions.length; i++) {
    segLen += speed[i] ?? 0;
    const angleDiff = Math.abs(directions[i]! - directions[i - 1]!);
    if (angleDiff > Math.PI / 4) {
      segmentLengths.push(segLen);
      segLen = 0;
    }
  }
  if (segLen > 0) segmentLengths.push(segLen);

  // Windowed jitter variance of speed
  const windowSize = Math.max(5, Math.floor(speed.length / 4));
  const windowVariances: number[] = [];
  for (let i = 0; i + windowSize <= speed.length; i += windowSize) {
    const window = speed.slice(i, i + windowSize);
    windowVariances.push(variance(window));
  }
  const speedJitter = windowVariances.length > 1 ? variance(windowVariances) : 0;

  // Path length normalized by capture duration
  const duration = samples.length > 1
    ? (samples[samples.length - 1]!.timestamp - samples[0]!.timestamp) / 1000
    : 1;
  const normalizedPathLength = totalPathLength / Math.max(duration, 0.001);

  // Angle autocorrelation at lags 1, 2, 3
  const angleAutoCorr: number[] = [];
  for (let lag = 1; lag <= 3; lag++) {
    if (directions.length <= lag) {
      angleAutoCorr.push(0);
      continue;
    }
    const n = directions.length - lag;
    const meanDir = directions.reduce((a, b) => a + b, 0) / directions.length;
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
      num += (directions[i]! - meanDir) * (directions[i + lag]! - meanDir);
      den += (directions[i]! - meanDir) ** 2;
    }
    angleAutoCorr.push(den > 0 ? num / den : 0);
  }

  // Assemble 54 features
  const curvatureStats = condense(curvatures);               // 4
  const dirEntropy = entropy(directions, 16);                 // 1
  const speedStats = condense(speed);                         // 4
  const accStats = condense(acc);                             // 4
  // micro-corrections: reversalRate + reversalMagnitude       // 2
  // pauseRatio                                                // 1
  // pathEfficiency                                            // 1
  // speedJitter                                               // 1
  const jerkStats = condense(jerk);                           // 4
  const vxStats = condense(vx);                               // 4
  const vyStats = condense(vy);                               // 4
  const accXStats = condense(accX);                           // 4
  const accYStats = condense(accY);                           // 4
  const pressureStats = condense(pressure);                   // 4
  const moveDurStats = condense(movementDurations);           // 4
  const segLenStats = condense(segmentLengths);               // 4
  // angleAutoCorr[0..2]                                       // 3
  // normalizedPathLength                                      // 1
  // Total: 4+1+4+4+2+1+1+1+4+4+4+4+4+4+4+4+3+1 = 54

  const legacyMouseDynamics = [
    curvatureStats.mean, curvatureStats.variance, curvatureStats.skewness, curvatureStats.kurtosis,
    dirEntropy,
    speedStats.mean, speedStats.variance, speedStats.skewness, speedStats.kurtosis,
    accStats.mean, accStats.variance, accStats.skewness, accStats.kurtosis,
    reversalRate, reversalMagnitude,
    pauseRatio,
    pathEfficiency,
    speedJitter,
    jerkStats.mean, jerkStats.variance, jerkStats.skewness, jerkStats.kurtosis,
    vxStats.mean, vxStats.variance, vxStats.skewness, vxStats.kurtosis,
    vyStats.mean, vyStats.variance, vyStats.skewness, vyStats.kurtosis,
    accXStats.mean, accXStats.variance, accXStats.skewness, accXStats.kurtosis,
    accYStats.mean, accYStats.variance, accYStats.skewness, accYStats.kurtosis,
    pressureStats.mean, pressureStats.variance, pressureStats.skewness, pressureStats.kurtosis,
    moveDurStats.mean, moveDurStats.variance, moveDurStats.skewness, moveDurStats.kurtosis,
    segLenStats.mean, segLenStats.variance, segLenStats.skewness, segLenStats.kurtosis,
    angleAutoCorr[0] ?? 0, angleAutoCorr[1] ?? 0, angleAutoCorr[2] ?? 0,
    normalizedPathLength,
  ];

  // Pad to MOUSE_DYNAMICS_FEATURE_COUNT so the desktop fingerprint slot
  // has the same width as a mobile IMU capture. The padding zeros are
  // consistent across all desktop sessions (no IMU available), so they
  // don't introduce per-session noise — they just keep the bit-influence
  // share per modality identical across device classes.
  const padding = MOUSE_DYNAMICS_FEATURE_COUNT - legacyMouseDynamics.length;
  return padding > 0
    ? [...legacyMouseDynamics, ...new Array(padding).fill(0)]
    : legacyMouseDynamics;
}
