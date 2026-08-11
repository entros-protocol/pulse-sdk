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
// the remaining 27 are zero (no IMU on desktop).
//
// Why zero-padding doesn't break SimHash bit-influence parity (verified
// algebraically): `normalizeGroup` z-scores the 81-element block. With 27
// zeros, the padded variance equals (54/81) × pure_variance, so the
// padded std equals √(2/3) × pure_std ≈ 0.816 × pure_std. After dividing
// by the padded std, real features scale by 1/0.816 ≈ 1.225, and zero
// padding stays at -mean/std (≈ 0 when the real-feature mean is small).
// The collective variance of the 81-element normalized block is exactly
// 1 by construction (z-score guarantee), so the modality contributes the
// same expected variance to the SimHash random-projection dot product
// as the mobile case (also a unit-variance 81-element block). The
// per-feature magnitude of real features inflates by 22% on desktop;
// the per-modality bit-influence share stays equal.
export const MOUSE_DYNAMICS_FEATURE_COUNT = MOTION_FEATURE_COUNT;

/**
 * How much of the audio window motion must span before a contour is worth
 * building. Below this the frames outside the covered stretch would be filled
 * by edge-clamping, and a flat run reads to the validator's cross-correlation
 * as weak coupling rather than as missing data.
 */
const MIN_WINDOW_COVERAGE = 0.9;

/**
 * Compute per-sample acceleration magnitude |a| = √(ax² + ay² + az²) and
 * resample it onto `window`, the wall-clock stretch the transmitted audio
 * covers, at `targetFrameCount` equally spaced instants.
 *
 * The result is correlated against the F0 contour server-side, so the two have
 * to describe the same stretch of time. `window` is what makes that true, and
 * it is required rather than optional on purpose: this used to map motion's
 * array index proportionally onto audio's frame count, which silently
 * time-warped one stream against the other whenever their spans diverged.
 * `pulse-sdk@4.0.0` diverged them by trimming the pre-prompt lead-in out of the
 * audio alone, and cross-modal coupling fell from r=0.31 to r=0.03. A required
 * parameter turns the next such divergence into a compile error.
 *
 * `window` and {@link MotionSample.timestamp} are both in the
 * `performance.now()` domain, so they compare directly.
 *
 * Returns an empty array when the capture cannot support an honest contour:
 * too few samples, a degenerate window, or motion spanning less than
 * {@link MIN_WINDOW_COVERAGE} of it. The validator treats an absent contour as
 * "skip", which is the fail-safe direction. A misaligned one reads as weak
 * coupling and rejects a real person.
 */
export function extractAccelerationMagnitude(
  samples: MotionSample[],
  targetFrameCount: number,
  window: { startMs: number; endMs: number },
): number[] {
  if (samples.length < 2 || targetFrameCount < 2) return [];

  const { startMs, endMs } = window;
  const span = endMs - startMs;
  if (!Number.isFinite(span) || span <= 0) return [];

  const firstAt = samples[0]!.timestamp;
  const lastAt = samples[samples.length - 1]!.timestamp;
  // Clamped at zero so a stream sitting entirely outside the window reports no
  // coverage rather than a negative one.
  const overlap = Math.max(0, Math.min(endMs, lastAt) - Math.max(startMs, firstAt));
  if (overlap / span < MIN_WINDOW_COVERAGE) return [];

  const magnitudes = samples.map((s) => Math.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az));

  const out = new Array<number>(targetFrameCount);
  // `t` increases every iteration and sample timestamps are monotonic, so the
  // cursor only ever moves forward. One pass over both series, not a search
  // per frame.
  let cursor = 0;
  for (let i = 0; i < targetFrameCount; i++) {
    const t = startMs + (i / (targetFrameCount - 1)) * span;
    while (cursor + 1 < samples.length && samples[cursor + 1]!.timestamp <= t) cursor++;

    const at = samples[cursor]!.timestamp;
    if (t <= at || cursor + 1 >= samples.length) {
      // Before the first sample or past the last. Hold the edge value rather
      // than extrapolating a trend the sensor never reported.
      out[i] = magnitudes[cursor]!;
      continue;
    }
    const nextAt = samples[cursor + 1]!.timestamp;
    const step = nextAt - at;
    // Two readings sharing a timestamp carry no gradient to interpolate along.
    const frac = step > 0 ? (t - at) / step : 0;
    out[i] = magnitudes[cursor]! * (1 - frac) + magnitudes[cursor + 1]! * frac;
  }
  return out;
}

/**
 * Observe-only description of how the motion stream sat against the audio
 * window. Snake_case to match the wire, versioned like `ClientSignals`.
 *
 * @see describeCaptureTiming for what each field is for and why it exists.
 */
export interface CaptureTiming {
  /** Envelope schema version. */
  v: number;
  /** Motion readings collected across the whole recorder run. */
  motion_samples: number;
  /** First to last motion timestamp. */
  motion_span_ms: number;
  /** Achieved delivery rate. Nominal is 60-100Hz. Well below means throttling. */
  motion_rate_hz: number;
  /** Coefficient of variation of inter-sample gaps. Near 0 is even delivery. */
  motion_rate_cv: number;
  /** Duration of the audio actually transmitted. */
  audio_window_ms: number;
  /** First motion sample minus the audio window start. Near 0 once aligned. */
  window_offset_ms: number;
  /** Fraction of the audio window the motion stream spans, 0 to 1. */
  window_coverage: number;
  /** Microphone RMS over the transmitted window, before normalisation. */
  audio_input_rms: number;
  /** Largest absolute sample. Separates uniformly quiet from quiet-with-peaks. */
  audio_input_peak: number;
  /** Gain applied to reach the target RMS. 1.0 means none was needed. */
  audio_normalization_gain: number;
  /** True when the input fell below what the gain ceiling can recover. */
  audio_gain_clipped: boolean;
  /** Fraction of 10ms frames above the speech-presence threshold. */
  audio_voiced_frame_ratio: number;
}

/**
 * Summarise how the motion stream lined up with the audio window.
 *
 * Purely observational. It exists because the 4.0.0 misalignment was invisible
 * for a day: the only symptom was a correlation the validator could not explain,
 * and every number that would have named the cause was computed on-device and
 * discarded. `window_offset_ms` is that number.
 *
 * **This must never feed a decision.** It is client-supplied, so a forged value
 * has to be capable of corrupting our own calibration data and nothing else.
 * Read it from logs, never from a code path that can reject or accept.
 */
export function describeCaptureTiming(
  samples: MotionSample[],
  window: { startMs: number; endMs: number },
  inputLevel: {
    rms: number;
    peak: number;
    gain: number;
    gainClipped: boolean;
    voicedFrameRatio: number;
  },
): CaptureTiming {
  const span = Math.max(0, window.endMs - window.startMs);
  const timestamps = samples.map((s) => s.timestamp);
  const motionSpan =
    samples.length >= 2 ? Math.max(0, timestamps[timestamps.length - 1]! - timestamps[0]!) : 0;

  // Spread of the inter-sample gaps, normalised by their mean so it compares
  // across devices with different nominal rates.
  let rateCv = 0;
  if (samples.length >= 3) {
    const gaps: number[] = [];
    for (let i = 1; i < timestamps.length; i++) gaps.push(timestamps[i]! - timestamps[i - 1]!);
    const gapMean = mean(gaps);
    rateCv = gapMean > 0 ? Math.sqrt(variance(gaps, gapMean)) / gapMean : 0;
  }

  const overlap =
    samples.length >= 2 && span > 0
      ? Math.max(
          0,
          Math.min(window.endMs, timestamps[timestamps.length - 1]!) -
            Math.max(window.startMs, timestamps[0]!),
        )
      : 0;

  const round = (v: number, dp = 2) =>
    Number.isFinite(v) ? Number(v.toFixed(dp)) : 0;

  return {
    v: 1,
    motion_samples: samples.length,
    motion_span_ms: round(motionSpan),
    motion_rate_hz: round(sampleRateFromTimestamps(timestamps)),
    motion_rate_cv: round(rateCv, 4),
    audio_window_ms: round(span),
    window_offset_ms: samples.length > 0 ? round(timestamps[0]! - window.startMs) : 0,
    window_coverage: span > 0 ? round(overlap / span, 4) : 0,
    audio_input_rms: round(inputLevel.rms, 6),
    audio_input_peak: round(inputLevel.peak, 6),
    audio_normalization_gain: round(inputLevel.gain, 3),
    audio_gain_clipped: inputLevel.gainClipped,
    audio_voiced_frame_ratio: round(inputLevel.voicedFrameRatio, 4),
  };
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
export function extractMotionFeatures(
  samples: MotionSample[],
  projectionVersion = 0,
): number[] {
  if (samples.length < 5) return new Array(MOTION_FEATURE_COUNT).fill(0);

  const corrected = usesCorrectedExtraction(projectionVersion);
  const timestamps = samples.map((sample) => sample.timestamp);

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
    const jerk = derivativeForProjection(values, timestamps, corrected);
    // Jounce = 4th derivative of position = 2nd derivative of acceleration
    const jounce = derivativeForProjection(jerk.values, jerk.timestamps, corrected);

    const jerkStats = condense(jerk.values);
    const jounceStats = condense(jounce.values);

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
    const jerk = derivativeForProjection(values, timestamps, corrected).values;
    const windowSize = Math.max(5, Math.floor(jerk.length / 4));
    const windowVariances: number[] = [];
    for (let i = 0; i <= jerk.length - windowSize; i += windowSize) {
      windowVariances.push(variance(jerk.slice(i, i + windowSize)));
    }
    features.push(windowVariances.length >= 2 ? variance(windowVariances) : 0);
  }

  // ---- v2 additions ----
  features.push(...computeMotionV2(axes, samples, corrected));

  return features;
}

/**
 * v2 motion additions (27 features). Pulled into a dedicated helper so the
 * legacy 54-feature block stays isolated and visually identifiable in the
 * git history of `extractMotionFeatures`.
 */
function computeMotionV2(
  axes: Record<"ax" | "ay" | "az" | "gx" | "gy" | "gz", number[]>,
  samples: MotionSample[],
  corrected: boolean,
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
  const normalizationCount = corrected ? samples.length : fftSize;
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
      out.push(
        bandEnergy(
          spectrum.real,
          spectrum.imag,
          sampleRate,
          lo,
          hi,
          normalizationCount
        )
      );
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
    12,
    normalizationCount
  );
  out.push(tremor.freq, tremor.amplitude);

  // 4. Direction-reversal rate per second per accel axis (mean, variance).
  // A "reversal" here is a sign change of jerk (= the derivative of
  // acceleration). Counting on jerk rather than raw acceleration removes
  // the gravity DC bias on the vertical axis (raw az hovers around -9.8
  // and rarely crosses zero) and captures the rate of micro-correction
  // events on each axis. Rate is normalized by capture duration so it's
  // dimension-stable across IMU sample-rates. Mean + variance over the
  // 3 accel axes captures both how busy the user's motion is and which
  // axis dominates — a per-axis dominance pattern that's identity-bearing.
  const duration = captureDurationSec(samples);
  const timestamps = samples.map((sample) => sample.timestamp);
  const reversalRates = [axes.ax, axes.ay, axes.az].map((axis) =>
    duration > 0
      ? signChangeCount(
          derivativeForProjection(axis, timestamps, corrected).values,
        ) / duration
      : 0
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
  // moment-based features. Lags chosen to span the physiological-tremor
  // band: at a typical 60 Hz IMU rate, lag 5 ≈ 83 ms (12 Hz cycle), lag
  // 10 ≈ 167 ms (6 Hz), lag 25 ≈ 417 ms (sub-tremor, gait-rate signal).
  // The asymmetry vs touch's autocorrelation lags (1, 3, 5) is intentional
  // — touch captures finer rhythms (50-100 ms inter-event coherence)
  // while motion captures slower oscillatory patterns.
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
export function extractTouchFeatures(
  samples: TouchSample[],
  projectionVersion = 0,
): number[] {
  if (samples.length < 5) return new Array(TOUCH_FEATURE_COUNT).fill(0);

  const corrected = usesCorrectedExtraction(projectionVersion);
  const timestamps = samples.map((sample) => sample.timestamp);
  const x = samples.map((s) => s.x);
  const y = samples.map((s) => s.y);
  const pressure = samples.map((s) => s.pressure);
  const area = samples.map((s) => s.width * s.height);

  const features: number[] = [];

  // X velocity and acceleration
  const vx = derivativeForProjection(x, timestamps, corrected);
  const accX = derivativeForProjection(vx.values, vx.timestamps, corrected);
  features.push(...Object.values(condense(vx.values)));
  features.push(...Object.values(condense(accX.values)));

  // Y velocity and acceleration
  const vy = derivativeForProjection(y, timestamps, corrected);
  const accY = derivativeForProjection(vy.values, vy.timestamps, corrected);
  features.push(...Object.values(condense(vy.values)));
  features.push(...Object.values(condense(accY.values)));

  // Pressure statistics
  features.push(...Object.values(condense(pressure)));

  // Contact area statistics
  features.push(...Object.values(condense(area)));

  // Jerk of touch path
  const jerkX = derivativeForProjection(accX.values, accX.timestamps, corrected);
  const jerkY = derivativeForProjection(accY.values, accY.timestamps, corrected);
  features.push(...Object.values(condense(jerkX.values)));
  features.push(...Object.values(condense(jerkY.values)));

  // Jitter variance for touch signals: detects synthetic smoothness
  for (const values of [vx.values, vy.values, pressure, area]) {
    const windowSize = Math.max(5, Math.floor(values.length / 4));
    const windowVariances: number[] = [];
    for (let i = 0; i <= values.length - windowSize; i += windowSize) {
      windowVariances.push(variance(values.slice(i, i + windowSize)));
    }
    features.push(windowVariances.length >= 2 ? variance(windowVariances) : 0);
  }

  // ---- v2 additions ----
  features.push(...computeTouchV2(samples, vx.values, vy.values, corrected));

  return features;
}

/**
 * v2 touch additions (21 features). Pulled into a helper so the legacy
 * 36-feature block stays a visually identifiable unit.
 */
function computeTouchV2(
  samples: TouchSample[],
  vx: number[],
  vy: number[],
  corrected: boolean,
): number[] {
  const out: number[] = [];
  const timestamps = samples.map((sample) => sample.timestamp);

  // 1. Pressure first-derivative stats (4) — temporal RATE of pressure
  // variation, complementing the existing pressure mean/var/skew/kurt.
  const pressure = samples.map((s) => s.pressure);
  const dPressure = derivativeForProjection(
    pressure,
    timestamps,
    corrected,
  ).values;
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
  const dArea = derivativeForProjection(area, timestamps, corrected).values;
  out.push(mean(dArea), variance(dArea));

  // 4. Trajectory curvature stats (mean, var, skew). Curvature is the
  // absolute angle change between successive velocity vectors —
  // identity-bearing motor coordination. Skip rest-frames where either
  // velocity vector is below `CURVATURE_REST_EPS` because `atan2(0, 0)`
  // returns 0 silently, which would inject a spurious large curvature
  // spike whenever motion resumes from a pause.
  const CURVATURE_REST_EPS = 1e-3;
  const curvatures: number[] = [];
  for (let i = 1; i < vx.length; i++) {
    const v1x = vx[i - 1] ?? 0;
    const v1y = vy[i - 1] ?? 0;
    const v2x = vx[i] ?? 0;
    const v2y = vy[i] ?? 0;
    if (
      Math.hypot(v1x, v1y) < CURVATURE_REST_EPS ||
      Math.hypot(v2x, v2y) < CURVATURE_REST_EPS
    ) {
      continue;
    }
    const a1 = Math.atan2(v1y, v1x);
    const a2 = Math.atan2(v2y, v2x);
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
  const stepDistances = coordinateStepDistances(samples);
  const pathSeries = corrected
    ? stepDistances
    : vx.map((dx, index) => Math.hypot(dx, vy[index] ?? 0));
  const totalPath = pathSeries.reduce((a, b) => a + b, 0);
  const dx = (samples[samples.length - 1]?.x ?? 0) - (samples[0]?.x ?? 0);
  const dy = (samples[samples.length - 1]?.y ?? 0) - (samples[0]?.y ?? 0);
  const straight = Math.sqrt(dx * dx + dy * dy);
  out.push(totalPath > 0 ? straight / totalPath : 0);

  // 8. Per-stroke total path length: split on movement troughs
  // (at most 0.5 px per sample), then take
  // mean and variance. Captures motor-planning style — burst-then-pause
  // vs continuous-glide users.
  const strokeLengths = perStrokePathLengths(pathSeries);
  out.push(mean(strokeLengths), variance(strokeLengths));

  return out;
}

/** Split step distances at rest points and return each stroke's path length. */
function perStrokePathLengths(stepDistances: number[]): number[] {
  const PAUSE_THRESHOLD = 0.5;
  const lengths: number[] = [];
  let acc = 0;
  let inStroke = false;
  for (const distance of stepDistances) {
    if (distance >= PAUSE_THRESHOLD) {
      acc += distance;
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

interface SampledSeries {
  values: number[];
  timestamps: number[];
}

function usesCorrectedExtraction(projectionVersion: number): boolean {
  if (projectionVersion !== 0 && projectionVersion !== 1) {
    throw new Error(`Unsupported projection version ${projectionVersion}`);
  }
  return projectionVersion === 1;
}

function legacyDerivative(values: number[]): number[] {
  const derivatives: number[] = [];
  for (let index = 1; index < values.length; index++) {
    derivatives.push((values[index] ?? 0) - (values[index - 1] ?? 0));
  }
  return derivatives;
}

function derivativeForProjection(
  values: number[],
  timestamps: number[],
  corrected: boolean,
): SampledSeries {
  return corrected
    ? differentiate(values, timestamps)
    : {
        values: legacyDerivative(values),
        timestamps: timestamps.slice(1),
      };
}

/** Differentiate a series using its measured sample intervals. */
function differentiate(values: number[], timestamps: number[]): SampledSeries {
  const derivatives: number[] = [];
  const midpointTimestamps: number[] = [];
  const count = Math.min(values.length, timestamps.length);
  for (let i = 1; i < count; i++) {
    const previousTimestamp = timestamps[i - 1] ?? 0;
    const timestamp = timestamps[i] ?? previousTimestamp;
    const intervalSeconds = (timestamp - previousTimestamp) / 1000;
    const difference = (values[i] ?? 0) - (values[i - 1] ?? 0);
    derivatives.push(
      Number.isFinite(intervalSeconds) && intervalSeconds > 0
        ? difference / intervalSeconds
        : 0
    );
    midpointTimestamps.push((previousTimestamp + timestamp) / 2);
  }
  return { values: derivatives, timestamps: midpointTimestamps };
}

function coordinateStepDistances(samples: TouchSample[]): number[] {
  const distances: number[] = [];
  for (let i = 1; i < samples.length; i++) {
    distances.push(
      Math.hypot(
        (samples[i]?.x ?? 0) - (samples[i - 1]?.x ?? 0),
        (samples[i]?.y ?? 0) - (samples[i - 1]?.y ?? 0)
      )
    );
  }
  return distances;
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
export function extractMouseDynamics(
  samples: TouchSample[],
  projectionVersion = 0,
): number[] {
  if (samples.length < 10) return new Array(MOUSE_DYNAMICS_FEATURE_COUNT).fill(0);

  const corrected = usesCorrectedExtraction(projectionVersion);
  const x = samples.map((s) => s.x);
  const y = samples.map((s) => s.y);
  const pressure = samples.map((s) => s.pressure);
  const area = samples.map((s) => s.width * s.height);
  const timestamps = samples.map((sample) => sample.timestamp);
  const stepDistances = coordinateStepDistances(samples);

  // Velocity
  const vxSeries = derivativeForProjection(x, timestamps, corrected);
  const vySeries = derivativeForProjection(y, timestamps, corrected);
  const vx = vxSeries.values;
  const vy = vySeries.values;
  const speed = vx.map((dx, i) => Math.sqrt(dx * dx + (vy[i] ?? 0) * (vy[i] ?? 0)));

  // Acceleration
  const accXSeries = derivativeForProjection(vx, vxSeries.timestamps, corrected);
  const accYSeries = derivativeForProjection(vy, vySeries.timestamps, corrected);
  const accX = accXSeries.values;
  const accY = accYSeries.values;
  const acc = accX.map((ax, i) => Math.sqrt(ax * ax + (accY[i] ?? 0) * (accY[i] ?? 0)));

  // Jerk (derivative of acceleration)
  const jerkXSeries = derivativeForProjection(
    accX,
    accXSeries.timestamps,
    corrected,
  );
  const jerkYSeries = derivativeForProjection(
    accY,
    accYSeries.timestamps,
    corrected,
  );
  const jerkX = jerkXSeries.values;
  const jerkY = jerkYSeries.values;
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
  const pathSeries = corrected ? stepDistances : speed;
  const pauseFrames = pathSeries.filter((distance) => distance < speedThreshold).length;
  const pauseRatio = pathSeries.length > 0 ? pauseFrames / pathSeries.length : 0;

  // Path efficiency: straight-line distance / total path length
  const totalPathLength = pathSeries.reduce((a, b) => a + b, 0);
  const straightLine = Math.sqrt(
    (x[x.length - 1]! - x[0]!) ** 2 + (y[y.length - 1]! - y[0]!) ** 2
  );
  const pathEfficiency = totalPathLength > 0 ? straightLine / totalPathLength : 0;

  // Movement durations between pauses
  const movementDurations: number[] = [];
  let currentDuration = 0;
  for (const distance of pathSeries) {
    if (distance >= speedThreshold) {
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
    segLen += pathSeries[i] ?? 0;
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

  // Mouse V2 additions — 27 features mirroring `computeMotionV2`'s layout
  // exactly so desktop and mobile fingerprints share parallel structure
  // in the same indices. Replaces the original zero-padding scheme that
  // contributed ~85 deterministic bits across all desktop users (the
  // May-2026 cross-person Hamming collision contributor for the motion
  // block). Real signals fill every slot from the same mouse data the
  // legacy 54 features already consume, so no new sensor access required.
  const v2 = computeMouseV2(
    samples,
    vx,
    vy,
    accX,
    accY,
    speed,
    acc,
    jerk,
    directions,
    vxSeries.timestamps,
    accXSeries.timestamps,
    jerkXSeries.timestamps,
    corrected,
  );
  return [...legacyMouseDynamics, ...v2];
}

/**
 * v2 mouse-dynamics additions (27 features). Pulled into a dedicated
 * helper that mirrors `computeMotionV2` index-for-index so desktop and
 * mobile fingerprints have parallel semantic structure. All inputs are
 * already computed in the calling `extractMouseDynamics` scope; passing
 * them through avoids a second pass over the touch sample stream.
 *
 * Layout (relative to mouse block start):
 *   `[54..60)` cross-axis covariance: 6 pairs from {vx, vy, accX, accY}
 *   `[60..72)` FFT band energy {0-2, 2-6, 6-12, 12-30} Hz × {speed, acc, jerk}
 *   `[72..74)` physiological tremor peak (4-12 Hz) on speed: freq + amplitude
 *   `[74..76)` reversal-rate-per-second per channel {vx, vy, speed}: mean + variance
 *   `[76]`     mean angular speed (mean |Δdirection|)
 *   `[77..81)` speed-magnitude autocorrelation at lags {1, 5, 10, 25}
 */
function computeMouseV2(
  samples: TouchSample[],
  vx: number[],
  vy: number[],
  accX: number[],
  accY: number[],
  speed: number[],
  acc: number[],
  jerk: number[],
  directions: number[],
  velocityTimestamps: number[],
  accelerationTimestamps: number[],
  jerkTimestamps: number[],
  corrected: boolean,
): number[] {
  const out: number[] = [];

  // 1. Cross-axis covariance — 6 unique pairs from the 4-channel
  // {vx, vy, accX, accY} basis. Captures motor-coordination signature:
  // vx-vy coupling (handedness), velocity-acceleration coupling per axis
  // (motor-control style), and X/Y acceleration coupling (cursor-gesture
  // diagonality preference). The mouse equivalent of `computeMotionV2`'s
  // accel-gyro-and-axes pairings.
  const covPairs: Array<[number[], number[]]> = [
    [vx, vy],
    [vx, accX],
    [vx, accY],
    [vy, accX],
    [vy, accY],
    [accX, accY],
  ];
  for (const [a, b] of covPairs) out.push(covariance(a, b));

  // 2. FFT band energy on 3 channels: speed, acc, jerk magnitudes. Sample
  // rate is recovered from timestamps so band boundaries are reported in
  // physical Hz across mouse-event rates that vary 60-125 Hz across
  // browsers and OSs. Pre-FFT each channel once; reuse the speed
  // spectrum for the tremor peak below.
  const sampleRate = sampleRateFromTimestamps(
    corrected ? velocityTimestamps : samples.map((sample) => sample.timestamp),
  );
  const fftSize = nextPow2(Math.max(64, speed.length));
  const bands: Array<[number, number]> = [
    [0, 2],
    [2, 6],
    [6, 12],
    [12, 30],
  ];
  const speedSpectrum = realFFT(meanCenter(speed), fftSize);
  const accSpectrum = realFFT(meanCenter(acc), fftSize);
  const jerkSpectrum = realFFT(meanCenter(jerk), fftSize);
  for (const [spectrum, realSampleCount, spectrumSampleRate] of [
    [speedSpectrum, corrected ? speed.length : fftSize, sampleRate],
    [
      accSpectrum,
      corrected ? acc.length : fftSize,
      corrected ? sampleRateFromTimestamps(accelerationTimestamps) : sampleRate,
    ],
    [
      jerkSpectrum,
      corrected ? jerk.length : fftSize,
      corrected ? sampleRateFromTimestamps(jerkTimestamps) : sampleRate,
    ],
  ] as const) {
    for (const [lo, hi] of bands) {
      out.push(
        bandEnergy(
          spectrum.real,
          spectrum.imag,
          spectrumSampleRate,
          lo,
          hi,
          realSampleCount
        )
      );
    }
  }

  // 3. Physiological-tremor peak (4-12 Hz) on speed magnitude. Mouse-using
  // hands carry the same 4-12 Hz physiological tremor as IMU-tracked hands;
  // it surfaces in cursor-speed envelope as a small periodic component
  // riding on top of intentional motion.
  const tremor = peakInBand(
    speedSpectrum.real,
    speedSpectrum.imag,
    sampleRate,
    4,
    12,
    corrected ? speed.length : fftSize,
  );
  out.push(tremor.freq, tremor.amplitude);

  // 4. Reversal rate per second per channel (mean, variance across
  // {vx, vy, speed}). Sign change of the channel's first derivative
  // counts micro-corrections — wrist-style movements differ across
  // people in both rate and per-axis distribution.
  const duration = captureDurationSec(samples);
  const reversalChannels: Array<[number[], number[]]> = [
    [vx, velocityTimestamps],
    [vy, velocityTimestamps],
    [speed, velocityTimestamps],
  ];
  const reversalRates = reversalChannels.map(([channel, channelTimestamps]) =>
    duration > 0
      ? signChangeCount(
          derivativeForProjection(channel, channelTimestamps, corrected).values,
        ) / duration
      : 0,
  );
  out.push(mean(reversalRates), variance(reversalRates));

  // 5. Mean angular speed: mean of unwrapped |Δdirection|. Captures
  // overall steering activity — looser-grip hands change direction more
  // often than precision-grip hands. Equivalent to motion's mean |gyro|.
  let dirAccum = 0;
  for (let i = 1; i < directions.length; i++) {
    let diff = directions[i]! - directions[i - 1]!;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;
    dirAccum += Math.abs(diff);
  }
  out.push(directions.length > 1 ? dirAccum / (directions.length - 1) : 0);

  // 6. Speed-magnitude autocorrelation at lags 1, 5, 10, 25 — captures
  // periodic structure (drag rhythms, repeated sub-gestures) that escapes
  // moment-based features. Lag choices match motion v2 so the fingerprint
  // band-by-band layout stays parallel across device classes.
  for (const lag of [1, 5, 10, 25]) {
    out.push(autocorrelation(speed, lag));
  }

  // Defensive finite-cast: any helper that hits an edge case (zero-length
  // spectrum, constant channel, sub-1-sample direction series) should
  // return 0 rather than NaN/Infinity, matching the SDK's "feature vector
  // is always finite" contract enforced by the validator's NonFinite check.
  return out.map((v) => (Number.isFinite(v) ? v : 0));
}
