import {
  extractMouseDynamics,
  extractTouchFeatures,
} from "../../src/extraction/kinematic";
import { fuseFeatures } from "../../src/extraction/statistics";
import { hammingDistance, simhash } from "../../src/hashing/simhash";
import type { TouchSample } from "../../src/sensor/types";

export type CoordinatePolicy = "viewport" | "unit";
export type ContactPolicy = "css-pixels" | "surface-relative" | "neutral";
export type PathProfile = "smooth" | "paused" | "threshold";

export interface SurfaceFrame {
  left: number;
  top: number;
  width: number;
  height: number;
}

interface UnitTouchPoint {
  timestamp: number;
  x: number;
  y: number;
  pressure: number;
  widthCss: number;
  heightCss: number;
}

interface ProjectionOneResult {
  touch: number[];
  motion: number[];
  fingerprint: number[];
}

export interface MeasurementDelta {
  left: string;
  right: string;
  changedTouchFeatures: number;
  changedMotionFeatures: number;
  maximumTouchDelta: number;
  maximumMotionDelta: number;
  hammingDistance: number;
}

export interface LoadMeasurement {
  sampleCount: number;
  iterations: number;
  touchFeatureCount: number;
  motionFeatureCount: number;
  fingerprintBits: number;
  finite: boolean;
  totalElapsedMs: number;
  meanElapsedMs: number;
}

export interface ResamplingMeasurement {
  targetSampleCount: number;
  ratePairs: MeasurementDelta[];
}

export interface BranchObservation {
  profile: PathProfile;
  rateHz: number;
  pauseRatio: number;
  curvatureMean: number;
  curvatureVariance: number;
  strokeLengthMean: number;
  strokeLengthVariance: number;
}

export interface PathProfileMeasurement {
  profile: PathProfile;
  extremeScale: MeasurementDelta;
  translation: MeasurementDelta;
  inputRates: MeasurementDelta[];
  resamplingTargets: ResamplingMeasurement[];
  branches: BranchObservation[];
}

export interface TouchScaleMeasurementReport {
  projectionVersion: 1;
  sampleCount: number;
  surfaceSizes: number[];
  rawScalePairs: MeasurementDelta[];
  translatedSurface: MeasurementDelta;
  nonSquareSurface: MeasurementDelta;
  contactPolicies: Record<ContactPolicy, MeasurementDelta>;
  nativeParity: {
    cssContactToNative: MeasurementDelta;
    neutralContract: MeasurementDelta;
  };
  profiles: PathProfileMeasurement[];
  load: LoadMeasurement[];
}

const DEFAULT_SAMPLE_COUNT = 240;
const DEFAULT_DURATION_MS = 4_000;
const FEATURE_DELTA_TOLERANCE = 1e-9;

const deterministicAudio = Array.from(
  { length: 170 },
  (_, index) => Math.sin(index * 0.37) + 0.25 * Math.cos(index * 0.11),
);

function unitPath(
  sampleCount = DEFAULT_SAMPLE_COUNT,
  durationMs = DEFAULT_DURATION_MS,
  profile: PathProfile = "smooth",
): UnitTouchPoint[] {
  if (!Number.isInteger(sampleCount) || sampleCount < 2) {
    throw new Error("sampleCount must be an integer greater than one");
  }
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    throw new Error("durationMs must be positive");
  }

  return Array.from({ length: sampleCount }, (_, index) => {
    const progress = index / (sampleCount - 1);
    const movementProgress =
      profile === "paused"
        ? pausedProgress(progress)
        : progress;
    const x =
      profile === "threshold"
        ? 0.42 + 0.16 * movementProgress
        : 0.1 + 0.8 * movementProgress;
    const y =
      profile === "threshold"
        ? 0.5 + 0.04 * Math.sin(movementProgress * Math.PI * 4)
        : 0.5 +
          0.28 * Math.sin(movementProgress * Math.PI * 4) +
          0.04 * Math.sin(movementProgress * Math.PI * 23);
    return {
      timestamp: progress * durationMs,
      x,
      y,
      pressure: 0.35 + 0.25 * Math.sin(progress * Math.PI),
      widthCss: 10 + 1.5 * Math.sin(progress * Math.PI * 3),
      heightCss: 9 + Math.cos(progress * Math.PI * 2),
    };
  });
}

function pausedProgress(progress: number): number {
  if (progress < 0.25) return progress * 1.2;
  if (progress < 0.4) return 0.3;
  if (progress < 0.7) return 0.3 + (progress - 0.4) * (4 / 3);
  if (progress < 0.82) return 0.7;
  return 0.7 + (progress - 0.82) * (5 / 3);
}

export function renderTouchPath(
  frame: SurfaceFrame,
  coordinatePolicy: CoordinatePolicy,
  contactPolicy: ContactPolicy,
  sampleCount = DEFAULT_SAMPLE_COUNT,
  durationMs = DEFAULT_DURATION_MS,
  profile: PathProfile = "smooth",
): TouchSample[] {
  if (
    !Number.isFinite(frame.left) ||
    !Number.isFinite(frame.top) ||
    !Number.isFinite(frame.width) ||
    !Number.isFinite(frame.height) ||
    frame.width <= 0 ||
    frame.height <= 0
  ) {
    throw new Error("surface frame must contain finite positive dimensions");
  }

  return unitPath(sampleCount, durationMs, profile).map((point) => {
    const width =
      contactPolicy === "surface-relative"
        ? point.widthCss / frame.width
        : contactPolicy === "neutral"
          ? 1
          : point.widthCss;
    const height =
      contactPolicy === "surface-relative"
        ? point.heightCss / frame.height
        : contactPolicy === "neutral"
          ? 1
          : point.heightCss;

    return {
      timestamp: point.timestamp,
      x:
        coordinatePolicy === "unit"
          ? point.x
          : frame.left + point.x * frame.width,
      y:
        coordinatePolicy === "unit"
          ? point.y
          : frame.top + point.y * frame.height,
      pressure: point.pressure,
      width,
      height,
    };
  });
}

export function resampleTouchPath(
  samples: TouchSample[],
  sampleCount = DEFAULT_SAMPLE_COUNT,
): TouchSample[] {
  if (!Number.isInteger(sampleCount) || sampleCount < 2) {
    throw new Error("sampleCount must be an integer greater than one");
  }
  if (samples.length < 2) {
    throw new Error("at least two touch samples are required");
  }

  for (let index = 1; index < samples.length; index += 1) {
    if ((samples[index]?.timestamp ?? 0) <= (samples[index - 1]?.timestamp ?? 0)) {
      throw new Error("touch timestamps must increase");
    }
  }

  const start = samples[0]!.timestamp;
  const end = samples[samples.length - 1]!.timestamp;
  const duration = end - start;
  let sourceIndex = 1;

  return Array.from({ length: sampleCount }, (_, index) => {
    const progress = index / (sampleCount - 1);
    const timestamp = start + progress * duration;
    while (
      sourceIndex < samples.length - 1 &&
      (samples[sourceIndex]?.timestamp ?? end) < timestamp
    ) {
      sourceIndex += 1;
    }

    const right = samples[sourceIndex]!;
    const left = samples[sourceIndex - 1]!;
    const span = right.timestamp - left.timestamp;
    const weight = span > 0 ? (timestamp - left.timestamp) / span : 0;
    const interpolate = (leftValue: number, rightValue: number) =>
      leftValue + (rightValue - leftValue) * weight;

    return {
      timestamp,
      x: interpolate(left.x, right.x),
      y: interpolate(left.y, right.y),
      pressure: interpolate(left.pressure, right.pressure),
      width: interpolate(left.width, right.width),
      height: interpolate(left.height, right.height),
    };
  });
}

function evaluateProjectionOne(samples: TouchSample[]): ProjectionOneResult {
  const touch = extractTouchFeatures(samples, 1);
  const motion = extractMouseDynamics(samples, 1);
  const fused = fuseFeatures(deterministicAudio, motion, touch);
  return { touch, motion, fingerprint: simhash(fused, 1) };
}

function differs(left: number, right: number): boolean {
  const scale = Math.max(1, Math.abs(left), Math.abs(right));
  return Math.abs(left - right) > FEATURE_DELTA_TOLERANCE * scale;
}

function maximumDelta(left: number[], right: number[]): number {
  return left.reduce(
    (maximum, value, index) =>
      Math.max(maximum, Math.abs(value - (right[index] ?? 0))),
    0,
  );
}

export function compareResults(
  leftName: string,
  left: ProjectionOneResult,
  rightName: string,
  right: ProjectionOneResult,
): MeasurementDelta {
  return {
    left: leftName,
    right: rightName,
    changedTouchFeatures: left.touch.filter((value, index) =>
      differs(value, right.touch[index] ?? 0),
    ).length,
    changedMotionFeatures: left.motion.filter((value, index) =>
      differs(value, right.motion[index] ?? 0),
    ).length,
    maximumTouchDelta: maximumDelta(left.touch, right.touch),
    maximumMotionDelta: maximumDelta(left.motion, right.motion),
    hammingDistance: hammingDistance(left.fingerprint, right.fingerprint),
  };
}

function surface(size: number, left = 40, top = 80): SurfaceFrame {
  return { left, top, width: size, height: size };
}

function allFinite(result: ProjectionOneResult): boolean {
  return [...result.touch, ...result.motion, ...result.fingerprint].every(
    Number.isFinite,
  );
}

function measureLoad(sampleCount: number, iterations = 100): LoadMeasurement {
  const samples = renderTouchPath(
    surface(800),
    "viewport",
    "css-pixels",
    sampleCount,
  );
  const startedAt = performance.now();
  let result = evaluateProjectionOne(samples);
  let finite = allFinite(result);
  for (let iteration = 1; iteration < iterations; iteration += 1) {
    result = evaluateProjectionOne(samples);
    finite = finite && allFinite(result);
  }
  const totalElapsedMs = performance.now() - startedAt;
  return {
    sampleCount,
    iterations,
    touchFeatureCount: result.touch.length,
    motionFeatureCount: result.motion.length,
    fingerprintBits: result.fingerprint.length,
    finite,
    totalElapsedMs,
    meanElapsedMs: totalElapsedMs / iterations,
  };
}

function compareAdjacentRates(
  results: Map<number, ProjectionOneResult>,
): MeasurementDelta[] {
  return [
    compareResults("30Hz", results.get(30)!, "60Hz", results.get(60)!),
    compareResults("60Hz", results.get(60)!, "120Hz", results.get(120)!),
  ];
}

function observeBranches(
  profile: PathProfile,
  rateHz: number,
  result: ProjectionOneResult,
): BranchObservation {
  return {
    profile,
    rateHz,
    pauseRatio: result.motion[15] ?? 0,
    curvatureMean: result.touch[44] ?? 0,
    curvatureVariance: result.touch[45] ?? 0,
    strokeLengthMean: result.touch[55] ?? 0,
    strokeLengthVariance: result.touch[56] ?? 0,
  };
}

function measureProfile(profile: PathProfile): PathProfileMeasurement {
  const rates = [30, 60, 120];
  const rateResults = new Map(
    rates.map((rate) => {
      const sampleCount = rate * 4 + 1;
      return [
        rate,
        evaluateProjectionOne(
          renderTouchPath(
            surface(800),
            "viewport",
            "css-pixels",
            sampleCount,
            DEFAULT_DURATION_MS,
            profile,
          ),
        ),
      ] as const;
    }),
  );
  const resamplingTargets = [121, 240, 481].map((targetSampleCount) => {
    const results = new Map(
      rates.map((rate) => {
        const sampleCount = rate * 4 + 1;
        const samples = renderTouchPath(
          surface(800),
          "unit",
          "css-pixels",
          sampleCount,
          DEFAULT_DURATION_MS,
          profile,
        );
        return [
          rate,
          evaluateProjectionOne(resampleTouchPath(samples, targetSampleCount)),
        ] as const;
      }),
    );
    return {
      targetSampleCount,
      ratePairs: compareAdjacentRates(results),
    };
  });
  const small = evaluateProjectionOne(
    renderTouchPath(
      surface(200),
      "viewport",
      "css-pixels",
      DEFAULT_SAMPLE_COUNT,
      DEFAULT_DURATION_MS,
      profile,
    ),
  );
  const large = evaluateProjectionOne(
    renderTouchPath(
      surface(1_600),
      "viewport",
      "css-pixels",
      DEFAULT_SAMPLE_COUNT,
      DEFAULT_DURATION_MS,
      profile,
    ),
  );
  const untranslated = evaluateProjectionOne(
    renderTouchPath(
      surface(400, 0, 0),
      "viewport",
      "css-pixels",
      DEFAULT_SAMPLE_COUNT,
      DEFAULT_DURATION_MS,
      profile,
    ),
  );
  const translated = evaluateProjectionOne(
    renderTouchPath(
      surface(400, 640, 360),
      "viewport",
      "css-pixels",
      DEFAULT_SAMPLE_COUNT,
      DEFAULT_DURATION_MS,
      profile,
    ),
  );

  return {
    profile,
    extremeScale: compareResults("200px", small, "1600px", large),
    translation: compareResults(
      "400px at 0,0",
      untranslated,
      "400px at 640,360",
      translated,
    ),
    inputRates: compareAdjacentRates(rateResults),
    resamplingTargets,
    branches: rates.map((rate) =>
      observeBranches(profile, rate, rateResults.get(rate)!),
    ),
  };
}

export function runTouchScaleMeasurement(): TouchScaleMeasurementReport {
  const sizes = [200, 400, 800, 1_600];
  const rawResults = new Map(
    sizes.map((size) => [
      size,
      evaluateProjectionOne(
        renderTouchPath(surface(size), "viewport", "css-pixels"),
      ),
    ]),
  );
  const rawScalePairs: MeasurementDelta[] = [];
  for (let leftIndex = 0; leftIndex < sizes.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < sizes.length; rightIndex += 1) {
      const leftSize = sizes[leftIndex] ?? 0;
      const rightSize = sizes[rightIndex] ?? 0;
      const left = rawResults.get(leftSize);
      const right = rawResults.get(rightSize);
      if (left && right) {
        rawScalePairs.push(
          compareResults(`${leftSize}px`, left, `${rightSize}px`, right),
        );
      }
    }
  }

  const untranslated = evaluateProjectionOne(
    renderTouchPath(surface(400, 0, 0), "viewport", "css-pixels"),
  );
  const translated = evaluateProjectionOne(
    renderTouchPath(surface(400, 640, 360), "viewport", "css-pixels"),
  );
  const square = evaluateProjectionOne(
    renderTouchPath(surface(400), "viewport", "css-pixels"),
  );
  const nonSquare = evaluateProjectionOne(
    renderTouchPath(
      { left: 40, top: 80, width: 800, height: 200 },
      "viewport",
      "css-pixels",
    ),
  );

  const contactPolicies = Object.fromEntries(
    (["css-pixels", "surface-relative", "neutral"] as const).map((policy) => {
      const small = evaluateProjectionOne(
        renderTouchPath(surface(200), "unit", policy),
      );
      const large = evaluateProjectionOne(
        renderTouchPath(surface(1_600), "unit", policy),
      );
      return [policy, compareResults("200px", small, "1600px", large)];
    }),
  ) as Record<ContactPolicy, MeasurementDelta>;
  const browserCssContact = evaluateProjectionOne(
    renderTouchPath(surface(800), "unit", "css-pixels"),
  );
  const browserNeutral = evaluateProjectionOne(
    renderTouchPath(surface(800), "unit", "neutral"),
  );
  const nativeUnit = evaluateProjectionOne(
    renderTouchPath(surface(1), "unit", "neutral"),
  );

  return {
    projectionVersion: 1,
    sampleCount: DEFAULT_SAMPLE_COUNT,
    surfaceSizes: sizes,
    rawScalePairs,
    translatedSurface: compareResults(
      "400px at 0,0",
      untranslated,
      "400px at 640,360",
      translated,
    ),
    nonSquareSurface: compareResults(
      "400x400px",
      square,
      "800x200px",
      nonSquare,
    ),
    contactPolicies,
    nativeParity: {
      cssContactToNative: compareResults(
        "browser unit coordinates with CSS contact",
        browserCssContact,
        "native unit coordinates with neutral contact",
        nativeUnit,
      ),
      neutralContract: compareResults(
        "browser unit coordinates with neutral contact",
        browserNeutral,
        "native unit coordinates with neutral contact",
        nativeUnit,
      ),
    },
    profiles: (["smooth", "paused", "threshold"] as const).map(
      measureProfile,
    ),
    load: [measureLoad(720), measureLoad(1_500)],
  };
}
