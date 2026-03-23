import type { MotionSample, TouchSample } from "../sensor/types";
import { condense, variance, entropy } from "./statistics";

/**
 * Extract kinematic features from motion (IMU) data.
 * Computes jerk (3rd derivative) and jounce (4th derivative) of acceleration,
 * then condenses each axis into statistics.
 *
 * Returns: ~54 values (6 axes × 2 derivatives × 4 stats + 6 jitter variance values)
 */
export function extractMotionFeatures(samples: MotionSample[]): number[] {
  if (samples.length < 5) return new Array(54).fill(0);

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
  // Real human tremor fluctuates over time (high jitter variance).
  // Synthetic/replay data has constant jitter (low jitter variance).
  for (const values of Object.values(axes)) {
    const jerk = derivative(values);
    const windowSize = Math.max(5, Math.floor(jerk.length / 4));
    const windowVariances: number[] = [];
    for (let i = 0; i <= jerk.length - windowSize; i += windowSize) {
      windowVariances.push(variance(jerk.slice(i, i + windowSize)));
    }
    features.push(windowVariances.length >= 2 ? variance(windowVariances) : 0);
  }

  return features;
}

/**
 * Extract kinematic features from touch data.
 * Computes velocity and acceleration of touch coordinates,
 * plus pressure and area statistics.
 *
 * Returns: ~36 values (32 base + 4 jitter variance for x, y, pressure, area)
 */
export function extractTouchFeatures(samples: TouchSample[]): number[] {
  if (samples.length < 5) return new Array(36).fill(0);

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

  return features;
}

/** Compute discrete derivative (differences between consecutive values) */
function derivative(values: number[]): number[] {
  const d: number[] = [];
  for (let i = 1; i < values.length; i++) {
    d.push((values[i] ?? 0) - (values[i - 1] ?? 0));
  }
  return d;
}

/**
 * Extract mouse dynamics features as a desktop replacement for motion sensor data.
 * Captures behavioral patterns from mouse/pointer movement that are user-specific:
 * path curvature, speed patterns, micro-corrections, pause behavior.
 *
 * Returns: 54 values (matches motion feature dimension for consistent SimHash input)
 */
export function extractMouseDynamics(samples: TouchSample[]): number[] {
  if (samples.length < 10) return new Array(54).fill(0);

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

  return [
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
}
