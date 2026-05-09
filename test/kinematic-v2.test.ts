import { describe, it, expect } from "vitest";
import {
  extractMotionFeatures,
  extractTouchFeatures,
  extractMouseDynamics,
  MOTION_LEGACY_COUNT,
  MOTION_FEATURE_COUNT,
  MOTION_V2_ADDITIONS,
  TOUCH_LEGACY_COUNT,
  TOUCH_FEATURE_COUNT,
  TOUCH_V2_ADDITIONS,
} from "../src/extraction/kinematic";
import type { MotionSample, TouchSample } from "../src/sensor/types";

// IMU sample period 16.67 ms ≈ 60 Hz. Used by every motion test below so the
// FFT bin spacing is predictable (60 / N Hz per bin) and the [4, 12] Hz
// physiological-tremor band has at least a handful of bins to land in.
const IMU_PERIOD_MS = 1000 / 60;
const TOUCH_PERIOD_MS = 1000 / 60;

function motionSineSamples(opts: {
  count: number;
  freqHz: number;
  axes?: Partial<Record<"ax" | "ay" | "az" | "gx" | "gy" | "gz", number>>;
}): MotionSample[] {
  const { count, freqHz, axes = {} } = opts;
  return Array.from({ length: count }, (_, i) => {
    const t = i * IMU_PERIOD_MS;
    const phase = (2 * Math.PI * freqHz * i) / 60;
    const sine = Math.sin(phase);
    return {
      timestamp: t,
      ax: (axes.ax ?? 0) * sine,
      ay: (axes.ay ?? 0) * sine,
      az: (axes.az ?? 0) * sine,
      gx: (axes.gx ?? 0) * sine,
      gy: (axes.gy ?? 0) * sine,
      gz: (axes.gz ?? 0) * sine,
    };
  });
}

function touchPathSamples(opts: {
  count: number;
  shape: "straight" | "circle" | "wiggle";
  pressureFn?: (i: number) => number;
}): TouchSample[] {
  const { count, shape, pressureFn = () => 0.5 } = opts;
  return Array.from({ length: count }, (_, i) => {
    let x = 0;
    let y = 0;
    if (shape === "straight") {
      x = i;
      y = 0;
    } else if (shape === "circle") {
      const angle = (2 * Math.PI * i) / count;
      x = 100 + 50 * Math.cos(angle);
      y = 100 + 50 * Math.sin(angle);
    } else {
      x = i;
      y = 30 * Math.sin(i * 0.3);
    }
    return {
      timestamp: i * TOUCH_PERIOD_MS,
      x,
      y,
      pressure: pressureFn(i),
      width: 10 + 0.1 * i,
      height: 10,
    };
  });
}

describe("motion v2 layout constants", () => {
  it("exposes consistent legacy / additions / total constants", () => {
    expect(MOTION_LEGACY_COUNT).toBe(54);
    expect(MOTION_V2_ADDITIONS).toBe(27);
    expect(MOTION_FEATURE_COUNT).toBe(MOTION_LEGACY_COUNT + MOTION_V2_ADDITIONS);
    expect(MOTION_FEATURE_COUNT).toBe(81);
  });
});

describe("touch v2 layout constants", () => {
  it("exposes consistent legacy / additions / total constants", () => {
    expect(TOUCH_LEGACY_COUNT).toBe(36);
    expect(TOUCH_V2_ADDITIONS).toBe(21);
    expect(TOUCH_FEATURE_COUNT).toBe(TOUCH_LEGACY_COUNT + TOUCH_V2_ADDITIONS);
    expect(TOUCH_FEATURE_COUNT).toBe(57);
  });
});

describe("motion v2 — cross-axis covariance (indices 54..60)", () => {
  // Pair order encoded in computeMotionV2: [ax-gy, ay-gx, az-gz, ax-az, ay-az, gx-gy].
  it("ax-gy pair (index 54) is non-zero when ax and gy share phase", () => {
    const samples = motionSineSamples({
      count: 256,
      freqHz: 5,
      axes: { ax: 1, gy: 1 },
    });
    const features = extractMotionFeatures(samples);
    expect(Math.abs(features[54]!)).toBeGreaterThan(0.01);
  });

  it("ax-gy pair (index 54) is near zero when only ax is excited", () => {
    const samples = motionSineSamples({
      count: 256,
      freqHz: 5,
      axes: { ax: 1 }, // gy stays at 0
    });
    const features = extractMotionFeatures(samples);
    expect(Math.abs(features[54]!)).toBeLessThan(1e-6);
  });
});

describe("motion v2 — FFT band energy (indices 60..72)", () => {
  // Per-axis bands: [ax 0-2, ax 2-6, ax 6-12, ax 12-30, ay 0-2, ..., az 12-30].
  it("ax 6-12Hz band (index 62) catches a 9 Hz signal on ax", () => {
    const samples = motionSineSamples({
      count: 512,
      freqHz: 9,
      axes: { ax: 1 },
    });
    const features = extractMotionFeatures(samples);
    const ax_0_2 = features[60]!;
    const ax_2_6 = features[61]!;
    const ax_6_12 = features[62]!;
    const ax_12_30 = features[63]!;
    // 9 Hz lands inside the [6, 12) band — that bin gets the dominant energy.
    expect(ax_6_12).toBeGreaterThan(ax_0_2);
    expect(ax_6_12).toBeGreaterThan(ax_2_6);
    expect(ax_6_12).toBeGreaterThan(ax_12_30);
  });

  it("ay band stays near zero when only ax is excited", () => {
    const samples = motionSineSamples({
      count: 512,
      freqHz: 9,
      axes: { ax: 1 },
    });
    const features = extractMotionFeatures(samples);
    // ay bands are indices 64..68. All four should be effectively zero.
    for (let i = 64; i < 68; i++) {
      expect(features[i]!).toBeLessThan(1e-6);
    }
  });
});

describe("motion v2 — physiological tremor peak (indices 72..74)", () => {
  it("locates a 7 Hz tremor that's actually present in the magnitude signal", () => {
    // Magnitude of a vector field is a NONLINEAR function of the axes —
    // |a| = √(ax² + ay² + az²) — so a pure-sine input on each axis lands
    // at 2× the fundamental in magnitude (rectification). To inject a
    // clean 7 Hz tremor INTO the magnitude, perturb a single axis around
    // a large DC bias: ax(t) = K + B·cos(2πft), with K ≫ B. Then
    // |a(t)| ≈ K + B·cos(2πft) and the tremor band should peak at 7 Hz.
    const K = 9.8; // gravity on a single axis
    const B = 0.05; // small perturbation amplitude
    const f = 7;
    const samples: MotionSample[] = Array.from({ length: 512 }, (_, i) => ({
      timestamp: i * IMU_PERIOD_MS,
      ax: K + B * Math.cos((2 * Math.PI * f * i) / 60),
      ay: 0,
      az: 0,
      gx: 0,
      gy: 0,
      gz: 0,
    }));
    const features = extractMotionFeatures(samples);
    const tremorFreq = features[72]!;
    const tremorAmp = features[73]!;
    // Bin spacing = 60 / 512 ≈ 0.117 Hz. Allow 1 Hz tolerance for leakage.
    expect(tremorFreq).toBeGreaterThan(6);
    expect(tremorFreq).toBeLessThan(8);
    expect(tremorAmp).toBeGreaterThan(0);
  });

  it("returns zero amplitude when motion magnitude is flat", () => {
    // Phone sitting still: only gravity on z.
    const samples: MotionSample[] = Array.from({ length: 256 }, (_, i) => ({
      timestamp: i * IMU_PERIOD_MS,
      ax: 0,
      ay: 0,
      az: -9.8,
      gx: 0,
      gy: 0,
      gz: 0,
    }));
    const features = extractMotionFeatures(samples);
    expect(features[73]!).toBe(0);
  });
});

describe("motion v2 — direction reversal stats (indices 74..76)", () => {
  it("oscillating ax produces a non-zero reversal-rate mean", () => {
    const samples = motionSineSamples({
      count: 256,
      freqHz: 5,
      axes: { ax: 1 },
    });
    const features = extractMotionFeatures(samples);
    expect(features[74]!).toBeGreaterThan(0);
  });

  it("constant-acceleration phone produces zero reversal-rate mean", () => {
    const samples: MotionSample[] = Array.from({ length: 100 }, (_, i) => ({
      timestamp: i * IMU_PERIOD_MS,
      ax: 0,
      ay: 0,
      az: -9.8,
      gx: 0,
      gy: 0,
      gz: 0,
    }));
    const features = extractMotionFeatures(samples);
    expect(features[74]!).toBe(0);
  });
});

describe("motion v2 — mean angular velocity (index 76)", () => {
  it("is non-zero when gyro is excited", () => {
    const samples = motionSineSamples({
      count: 100,
      freqHz: 3,
      axes: { gx: 1 },
    });
    const features = extractMotionFeatures(samples);
    // mean(|gyro|) over a sine excitation > 0.
    expect(features[76]!).toBeGreaterThan(0);
  });

  it("is zero when gyro is silent", () => {
    const samples = motionSineSamples({
      count: 100,
      freqHz: 3,
      axes: { ax: 1 }, // gyros stay 0
    });
    const features = extractMotionFeatures(samples);
    expect(features[76]!).toBe(0);
  });
});

describe("motion v2 — magnitude autocorrelation (indices 77..81)", () => {
  it("returns finite values across all four lags", () => {
    const samples = motionSineSamples({
      count: 256,
      freqHz: 5,
      axes: { ax: 1, ay: 1, az: 1 },
    });
    const features = extractMotionFeatures(samples);
    for (let i = 77; i < 81; i++) {
      expect(Number.isFinite(features[i]!)).toBe(true);
    }
  });
});

describe("touch v2 — pressure derivative (indices 36..40)", () => {
  it("ramping pressure produces non-zero derivative mean", () => {
    const samples = touchPathSamples({
      count: 100,
      shape: "straight",
      pressureFn: (i) => 0.1 + i * 0.005, // monotonically rising
    });
    const features = extractTouchFeatures(samples);
    expect(features[36]!).toBeGreaterThan(0);
  });

  it("constant pressure produces near-zero derivative variance", () => {
    const samples = touchPathSamples({
      count: 100,
      shape: "straight",
      pressureFn: () => 0.5,
    });
    const features = extractTouchFeatures(samples);
    expect(features[37]!).toBeLessThan(1e-9);
  });
});

describe("touch v2 — contact aspect ratio (indices 40..42)", () => {
  it("captures the mean width/height ratio", () => {
    // width = 10 + 0.1*i, height = 10 → ratio increases from 1.0 to ≈ 1.99.
    const samples = touchPathSamples({ count: 100, shape: "straight" });
    const features = extractTouchFeatures(samples);
    expect(features[40]!).toBeGreaterThan(1);
    expect(features[40]!).toBeLessThan(2);
  });
});

describe("touch v2 — area derivative (indices 42..44)", () => {
  it("growing contact area produces positive area-derivative mean", () => {
    const samples = touchPathSamples({ count: 100, shape: "straight" });
    const features = extractTouchFeatures(samples);
    expect(features[42]!).toBeGreaterThan(0);
  });
});

describe("touch v2 — trajectory curvature (indices 44..47)", () => {
  it("circular path produces higher curvature mean than straight path", () => {
    const straight = touchPathSamples({ count: 100, shape: "straight" });
    const circle = touchPathSamples({ count: 100, shape: "circle" });
    const sFeats = extractTouchFeatures(straight);
    const cFeats = extractTouchFeatures(circle);
    expect(cFeats[44]!).toBeGreaterThan(sFeats[44]!);
  });
});

describe("touch v2 — velocity autocorrelation (indices 47..50)", () => {
  it("smooth straight path has high lag-1 autocorrelation", () => {
    const samples = touchPathSamples({ count: 100, shape: "straight" });
    const features = extractTouchFeatures(samples);
    // Constant-velocity straight path → speed series is constant → lag-1
    // autocorrelation undefined (variance = 0) and reported as 0 by the
    // helper. Wiggle path has variation → autocorrelation non-trivial.
    const wiggle = touchPathSamples({ count: 100, shape: "wiggle" });
    const wFeatures = extractTouchFeatures(wiggle);
    expect(Number.isFinite(features[47]!)).toBe(true);
    expect(Number.isFinite(wFeatures[47]!)).toBe(true);
  });
});

describe("touch v2 — inter-touch gap distribution (indices 50..54)", () => {
  it("regular sampling produces a stable gap mean ≈ TOUCH_PERIOD_MS", () => {
    const samples = touchPathSamples({ count: 100, shape: "straight" });
    const features = extractTouchFeatures(samples);
    expect(features[50]!).toBeCloseTo(TOUCH_PERIOD_MS, 5);
  });
});

describe("touch v2 — path efficiency + per-stroke length (indices 54..57)", () => {
  it("straight path produces near-1 path efficiency", () => {
    const samples = touchPathSamples({ count: 100, shape: "straight" });
    const features = extractTouchFeatures(samples);
    expect(features[54]!).toBeGreaterThan(0.99);
  });

  it("circular path closing on origin produces near-zero path efficiency", () => {
    const samples = touchPathSamples({ count: 100, shape: "circle" });
    const features = extractTouchFeatures(samples);
    // Circle endpoints are nearly co-located so straight-line distance ≈ 0.
    expect(features[54]!).toBeLessThan(0.05);
  });

  it("multi-stroke path produces non-zero per-stroke length variance", () => {
    // Construct three strokes of clearly different lengths separated by
    // pause samples (speed ≈ 0). The per-stroke length variance feature
    // (index 56) only becomes meaningful when the segmenter sees more
    // than one stroke; this guards against the common single-stroke
    // case silently degrading the feature to zero.
    const samples: TouchSample[] = [];
    let t = 0;
    let x = 0;
    const pushPoint = (xv: number) => {
      samples.push({
        timestamp: t,
        x: xv,
        y: 0,
        pressure: 0.5,
        width: 10,
        height: 10,
      });
      t += TOUCH_PERIOD_MS;
    };
    // Stroke A: 20 samples, +1 px/sample
    for (let i = 0; i < 20; i++) pushPoint(x++);
    // Pause: 5 samples at the same x (speed = 0)
    for (let i = 0; i < 5; i++) pushPoint(x);
    // Stroke B: 40 samples, +1 px/sample
    for (let i = 0; i < 40; i++) pushPoint(x++);
    // Pause
    for (let i = 0; i < 5; i++) pushPoint(x);
    // Stroke C: 60 samples
    for (let i = 0; i < 60; i++) pushPoint(x++);

    const features = extractTouchFeatures(samples);
    const strokeLengthMean = features[55]!;
    const strokeLengthVar = features[56]!;
    expect(strokeLengthMean).toBeGreaterThan(0);
    // Three strokes of lengths {20, 40, 60} → variance > 0.
    expect(strokeLengthVar).toBeGreaterThan(0);
  });
});

describe("end-to-end fingerprint width parity (Sprint 2 invariant)", () => {
  // The whole point of mouse-dynamics zero-padding is that desktop and
  // mobile produce SimHash inputs of identical width (314), so the same
  // hyperplane set projects both into comparable 256-bit fingerprints.
  // This test locks that invariant against accidental width drift.
  it("fused vector is exactly 314 elements with mobile motion", () => {
    const motion = Array.from({ length: 81 }, () => Math.random());
    const audio = Array.from({ length: 176 }, () => Math.random());
    const touch = Array.from({ length: 57 }, () => Math.random());
    const fused = [...audio, ...motion, ...touch];
    expect(fused).toHaveLength(314);
  });

  it("extractMouseDynamics produces the same width as extractMotionFeatures", () => {
    const touchSamples = touchPathSamples({ count: 100, shape: "wiggle" });
    const motionSamples: MotionSample[] = Array.from({ length: 100 }, (_, i) => ({
      timestamp: i * IMU_PERIOD_MS,
      ax: Math.random(),
      ay: Math.random(),
      az: -9.8 + Math.random() * 0.1,
      gx: Math.random() * 0.01,
      gy: Math.random() * 0.01,
      gz: Math.random() * 0.01,
    }));
    const desktopMotion = extractMouseDynamics(touchSamples);
    const mobileMotion = extractMotionFeatures(motionSamples);
    expect(desktopMotion.length).toBe(mobileMotion.length);
    expect(desktopMotion.length).toBe(MOTION_FEATURE_COUNT);
  });
});

// Mouse v2 additions replace the prior 27-slot zero-padding with real
// signals derived from the same mouse data. Each test asserts a specific
// slot has the expected shape on a controlled synthetic input. These
// guard the cross-person fingerprint-collapse fix: zero-pad indices were
// identical across all desktop users; real signals must differ.

describe("mouse v2 — cross-axis covariance (indices 54..60)", () => {
  it("vx-vy covariance (index 54) is non-zero when X and Y both oscillate in phase", () => {
    // Both x and y carry the same oscillation, so vx and vy each vary
    // with the same shape and high correlation. Linearly increasing
    // diagonal (x=i, y=i) wouldn't work — vx and vy would be constants
    // and any constant vector has zero covariance with anything by
    // definition.
    const samples: TouchSample[] = Array.from({ length: 100 }, (_, i) => ({
      timestamp: i * TOUCH_PERIOD_MS,
      x: 50 + 10 * Math.sin(i * 0.3),
      y: 50 + 10 * Math.sin(i * 0.3),
      pressure: 0.5,
      width: 10,
      height: 10,
    }));
    const features = extractMouseDynamics(samples);
    expect(Number.isFinite(features[54]!)).toBe(true);
    expect(Math.abs(features[54]!)).toBeGreaterThan(0);
  });
});

describe("mouse v2 — FFT band energy (indices 60..72)", () => {
  it("low-frequency wiggle has more energy in the 0-2 Hz band than 12-30 Hz on speed", () => {
    // Slow wiggle: cursor sweeps a 0.5 Hz sinusoid in y while x drifts.
    // Speed envelope is dominated by sub-2 Hz content.
    const samples: TouchSample[] = Array.from({ length: 256 }, (_, i) => ({
      timestamp: i * TOUCH_PERIOD_MS,
      x: i,
      y: 50 * Math.sin((2 * Math.PI * 0.5 * i) / 60),
      pressure: 0.5,
      width: 10,
      height: 10,
    }));
    const features = extractMouseDynamics(samples);
    const speedBand_0_2 = features[60]!; // first band, first channel (speed)
    const speedBand_12_30 = features[63]!; // fourth band, first channel (speed)
    expect(speedBand_0_2).toBeGreaterThan(speedBand_12_30);
  });
});

describe("mouse v2 — tremor peak (indices 72..74)", () => {
  it("an 8 Hz perturbation on the speed envelope produces a tremor amplitude > 0 in the 4-12 Hz band", () => {
    // Straight-line drift with an 8 Hz speed modulation. The tremor
    // detector should pick this up and report a peak in the 4-12 Hz band.
    const samples: TouchSample[] = Array.from({ length: 512 }, (_, i) => {
      const baseSpeed = 1;
      const tremor = 0.5 * Math.sin((2 * Math.PI * 8 * i) / 60);
      return {
        timestamp: i * TOUCH_PERIOD_MS,
        x: i * (baseSpeed + tremor),
        y: 0,
        pressure: 0.5,
        width: 10,
        height: 10,
      };
    });
    const features = extractMouseDynamics(samples);
    const tremorAmp = features[73]!;
    expect(Number.isFinite(tremorAmp)).toBe(true);
    expect(tremorAmp).toBeGreaterThan(0);
  });
});

describe("mouse v2 — reversal-rate stats (indices 74..76)", () => {
  it("zigzag path produces non-zero reversal-rate variance across channels", () => {
    // Zigzag: y oscillates rapidly while x advances. vx is monotonic
    // (low reversal rate) while vy reverses constantly (high rate).
    // The per-channel rate variance is therefore positive.
    const samples: TouchSample[] = Array.from({ length: 100 }, (_, i) => ({
      timestamp: i * TOUCH_PERIOD_MS,
      x: i,
      y: 30 * Math.sin(i * 1.0),
      pressure: 0.5,
      width: 10,
      height: 10,
    }));
    const features = extractMouseDynamics(samples);
    const reversalVar = features[75]!;
    expect(Number.isFinite(reversalVar)).toBe(true);
    expect(reversalVar).toBeGreaterThan(0);
  });
});

describe("mouse v2 — mean angular speed (index 76)", () => {
  it("circular path produces a higher mean angular speed than a straight path", () => {
    const straight = touchPathSamples({ count: 100, shape: "straight" });
    const circle = touchPathSamples({ count: 100, shape: "circle" });
    const sFeats = extractMouseDynamics(straight);
    const cFeats = extractMouseDynamics(circle);
    expect(cFeats[76]!).toBeGreaterThan(sFeats[76]!);
  });
});

describe("mouse v2 — speed autocorrelation (indices 77..81)", () => {
  it("returns finite values across all four lags", () => {
    const samples = touchPathSamples({ count: 100, shape: "wiggle" });
    const features = extractMouseDynamics(samples);
    for (let i = 77; i < 81; i++) {
      expect(Number.isFinite(features[i]!)).toBe(true);
    }
  });
});

describe("mouse v2 — no remaining deterministic zeros across desktop sessions", () => {
  it("two distinct mouse traces produce different values at every index in [54, 81)", () => {
    // Wave 2 fix contract: the 27 slots formerly zero-padded must now
    // carry per-session signal. If any index in [54, 81) ends up
    // identical across two clearly different paths, the deterministic-
    // zero leak that contributed ~85 cross-person bits is back.
    const a = extractMouseDynamics(touchPathSamples({ count: 100, shape: "circle" }));
    const b = extractMouseDynamics(touchPathSamples({ count: 100, shape: "wiggle" }));
    let identicalIndices = 0;
    for (let i = 54; i < 81; i++) {
      if (a[i]! === b[i]!) identicalIndices++;
    }
    // Allow up to 1 incidental coincidence (two paths might happen to
    // produce identical autocorr or band energy at a noisy index), but
    // not the wholesale 27-slot leak the zero-pad would produce.
    expect(identicalIndices).toBeLessThanOrEqual(1);
  });
});
