import type { MotionSample, TouchSample } from "../sensor/types";
import { condense } from "./statistics";

/**
 * Extract kinematic features from motion (IMU) data.
 * Computes jerk (3rd derivative) and jounce (4th derivative) of acceleration,
 * then condenses each axis into statistics.
 *
 * Returns: ~48 values (6 axes × 2 derivatives × 4 stats)
 */
export function extractMotionFeatures(samples: MotionSample[]): number[] {
  if (samples.length < 5) return new Array(48).fill(0);

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

  return features;
}

/**
 * Extract kinematic features from touch data.
 * Computes velocity and acceleration of touch coordinates,
 * plus pressure and area statistics.
 *
 * Returns: ~32 values
 */
export function extractTouchFeatures(samples: TouchSample[]): number[] {
  if (samples.length < 5) return new Array(32).fill(0);

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
