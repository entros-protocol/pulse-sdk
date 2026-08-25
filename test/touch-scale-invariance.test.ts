import { describe, expect, it } from "vitest";
import {
  renderTouchPath,
  resampleTouchPath,
  runTouchScaleMeasurement,
} from "./support/touch-scale-measurement";
import {
  extractMouseDynamics,
  extractTouchFeatures,
} from "../src/extraction/kinematic";
import { fuseFeatures } from "../src/extraction/statistics";
import { hammingDistance, simhash } from "../src/hashing/simhash";
import { canonicalizeTouchSamples } from "../src/sensor/touch";

const projectionTwoAudio = Array.from(
  { length: 170 },
  (_, index) => Math.sin(index * 0.37) + 0.25 * Math.cos(index * 0.11),
);

function projectionTwoResult(sampleCount: number, profile: "smooth" | "paused") {
  const source = renderTouchPath(
    { left: 0, top: 0, width: 200, height: 200 },
    "unit",
    "neutral",
    sampleCount,
    4_000,
    profile,
  );
  const samples = canonicalizeTouchSamples(source, 2);
  const touch = extractTouchFeatures(samples, 2);
  const motion = extractMouseDynamics(samples, 2);
  return {
    touch,
    motion,
    fingerprint: simhash(fuseFeatures(projectionTwoAudio, motion, touch), 2),
  };
}

describe("projection 1 touch scale measurement", () => {
  const report = runTouchScaleMeasurement();

  it("reproduces scale-dependent feature and fingerprint changes", () => {
    expect(report.rawScalePairs).toHaveLength(6);
    expect(
      report.rawScalePairs.every(
        (pair) =>
          pair.changedTouchFeatures > 0 && pair.changedMotionFeatures > 0,
      ),
    ).toBe(true);
    expect(
      report.rawScalePairs.some((pair) => pair.hammingDistance > 0),
    ).toBe(true);
  });

  it("isolates aspect-ratio distortion from uniform scaling", () => {
    expect(report.nonSquareSurface.changedTouchFeatures).toBeGreaterThan(0);
    expect(report.nonSquareSurface.changedMotionFeatures).toBeGreaterThan(0);
    expect(report.nonSquareSurface.hammingDistance).toBeGreaterThan(0);
  });

  it("measures translation cancellation without a fingerprint change", () => {
    expect(report.translatedSurface.hammingDistance).toBe(0);
    expect(report.translatedSurface.changedTouchFeatures).toBeGreaterThan(0);
    expect(report.translatedSurface.changedMotionFeatures).toBeGreaterThan(0);
  });

  it("separates coordinate normalization from contact-geometry policy", () => {
    expect(report.contactPolicies["css-pixels"].hammingDistance).toBe(0);
    expect(report.contactPolicies.neutral.hammingDistance).toBe(0);
    expect(
      report.contactPolicies["surface-relative"].changedTouchFeatures,
    ).toBeGreaterThan(0);
  });

  it("measures event-rate sensitivity without non-finite values", () => {
    expect(report.profiles).toHaveLength(3);
    expect(
      report.profiles.every(
        (profile) =>
          profile.inputRates.length === 2 &&
          profile.resamplingTargets.length === 3,
      ),
    ).toBe(true);
    expect(report.load.every((entry) => entry.finite)).toBe(true);
  });

  it("measures smooth, paused, and threshold-straddling paths", () => {
    expect(report.profiles.map((profile) => profile.profile)).toEqual([
      "smooth",
      "paused",
      "threshold",
    ]);
    expect(
      report.profiles.every(
        (profile) => profile.extremeScale.changedTouchFeatures > 0,
      ),
    ).toBe(true);
  });

  it("measures translation for every path profile", () => {
    expect(
      report.profiles.every(
        (profile) =>
          Number.isFinite(profile.translation.maximumTouchDelta) &&
          Number.isFinite(profile.translation.maximumMotionDelta),
      ),
    ).toBe(true);
  });

  it("isolates native parity from browser contact geometry", () => {
    expect(report.nativeParity.neutralContract).toMatchObject({
      changedTouchFeatures: 0,
      changedMotionFeatures: 0,
      hammingDistance: 0,
    });
    expect(
      report.nativeParity.cssContactToNative.changedTouchFeatures,
    ).toBeGreaterThan(0);
  });

  it("exercises pause, curvature, and stroke branches at every rate", () => {
    expect(
      report.profiles.every(
        (profile) =>
          profile.branches.length === 3 &&
          profile.branches.every((branch) =>
            [
              branch.pauseRatio,
              branch.curvatureMean,
              branch.curvatureVariance,
              branch.strokeLengthMean,
              branch.strokeLengthVariance,
            ].every(Number.isFinite),
          ),
      ),
    ).toBe(true);
    const paused = report.profiles.find(
      (profile) => profile.profile === "paused",
    );
    const smooth = report.profiles.find(
      (profile) => profile.profile === "smooth",
    );
    expect(paused?.branches[1]?.pauseRatio).toBeGreaterThan(
      smooth?.branches[1]?.pauseRatio ?? 0,
    );
  });

  it("shows when a common low-rate timeline removes fixture rate drift", () => {
    for (const profile of report.profiles) {
      const lowRate = profile.resamplingTargets.find(
        (entry) => entry.targetSampleCount === 121,
      );
      expect(
        lowRate?.ratePairs.every((pair) => pair.hammingDistance === 0),
        profile.profile,
      ).toBe(true);
    }
  });

  it.each([720, 1_500])(
    "keeps derived output bounded for %i pointer samples",
    (sampleCount) => {
      const load = report.load.find((entry) => entry.sampleCount === sampleCount);
      expect(load).toMatchObject({
        sampleCount,
        iterations: 100,
        touchFeatureCount: 57,
        motionFeatureCount: 81,
        fingerprintBits: 256,
        finite: true,
      });
    },
  );

  it("rejects invalid surface geometry", () => {
    expect(() =>
      renderTouchPath(
        { left: 0, top: 0, width: 0, height: 200 },
        "unit",
        "css-pixels",
      ),
    ).toThrow("surface frame must contain finite positive dimensions");
  });

  it("preserves resampling endpoints", () => {
    const source = renderTouchPath(
      { left: 0, top: 0, width: 200, height: 200 },
      "unit",
      "neutral",
      31,
    );
    const resampled = resampleTouchPath(source, 121);
    expect(resampled[0]).toEqual(source[0]);
    expect(resampled.at(-1)).toEqual(source.at(-1));
  });

  it("rejects non-increasing resampling timestamps", () => {
    const source = renderTouchPath(
      { left: 0, top: 0, width: 200, height: 200 },
      "unit",
      "neutral",
      5,
    );
    source[2] = { ...source[2]!, timestamp: source[1]!.timestamp };
    expect(() => resampleTouchPath(source)).toThrow(
      "touch timestamps must increase",
    );
  });
});

describe("projection 2 touch invariance", () => {
  it.each(["smooth", "paused"] as const)(
    "keeps the %s fingerprint stable across 30, 60, and 120 Hz sources",
    (profile) => {
      const results = [121, 241, 481].map((sampleCount) =>
        projectionTwoResult(sampleCount, profile),
      );
      for (const candidate of results.slice(1)) {
        expect(hammingDistance(results[0]!.fingerprint, candidate.fingerprint)).toBe(0);
        expect(candidate.touch.every(Number.isFinite)).toBe(true);
        expect(candidate.motion.every(Number.isFinite)).toBe(true);
      }
    },
  );

  it("uses the normalized movement threshold in rate units", () => {
    const samplesAt = (speed: number) =>
      Array.from({ length: 31 }, (_, index) => ({
        timestamp: (index * 1_000) / 30,
        x: (index * speed) / 30,
        y: 0.5,
        pressure: 0.5,
        width: 1,
        height: 1,
      }));

    expect(extractMouseDynamics(samplesAt(0), 2)[15]).toBe(1);
    expect(extractMouseDynamics(samplesAt(0.149), 2)[15]).toBe(1);
    expect(extractMouseDynamics(samplesAt(0.15), 2)[15]).toBe(0);
    expect(extractMouseDynamics(samplesAt(0.151), 2)[15]).toBe(0);
  });
});
