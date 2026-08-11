import { describe, expect, it } from "vitest";
import {
  extractMotionFeatures,
  extractMouseDynamics,
  extractSpeakerFeatures,
  extractTouchFeatures,
  fuseFeatures,
  simhash,
} from "../src";
import type { AudioCapture, MotionSample, TouchSample } from "../src";

function deterministicMotion(): MotionSample[] {
  return Array.from({ length: 129 }, (_, index) => ({
    timestamp: index * 16.75,
    ax: (((index * 17) % 31) - 15) / 32,
    ay: (((index * 19) % 37) - 18) / 32,
    az: 1 + (((index * 23) % 29) - 14) / 64,
    gx: (((index * 7) % 23) - 11) / 16,
    gy: (((index * 11) % 27) - 13) / 16,
    gz: (((index * 13) % 33) - 16) / 16,
  }));
}

function deterministicTouch(): TouchSample[] {
  return Array.from({ length: 129 }, (_, index) => ({
    timestamp: index * 15.25,
    x: (index * 37) % 401,
    y: (index * 53) % 307,
    pressure: (((index * 7) % 19) + 1) / 20,
    width: 8 + (index % 7),
    height: 9 + (index % 5),
  }));
}

function deterministicAudio(): AudioCapture {
  return {
    samples: Float32Array.from(
      { length: 32_000 },
      (_, index) => (((index * 17) % 257) - 128) / 256,
    ),
    sampleRate: 16_000,
    duration: 2,
    windowStartMs: 0,
    windowEndMs: 2_000,
    inputLevel: {
      rms: 0.2,
      peak: 0.5,
      gain: 1,
      gainClipped: false,
      voicedFrameRatio: 1,
    },
    voiceIsolationApplied: null,
  };
}

describe("projection version 0 compatibility", () => {
  it("keeps the default path identical to explicit version 0", async () => {
    const motionDefault = extractMotionFeatures(deterministicMotion());
    const motionVersioned = extractMotionFeatures(deterministicMotion(), 0);
    const touchDefault = extractTouchFeatures(deterministicTouch());
    const touchVersioned = extractTouchFeatures(deterministicTouch(), 0);
    const mouseDefault = extractMouseDynamics(deterministicTouch());
    const mouseVersioned = extractMouseDynamics(deterministicTouch(), 0);
    const speakerDefault = await extractSpeakerFeatures(deterministicAudio());
    const speakerVersioned = await extractSpeakerFeatures(deterministicAudio(), 0);

    expect(motionDefault).toEqual(motionVersioned);
    expect(touchDefault).toEqual(touchVersioned);
    expect(mouseDefault).toEqual(mouseVersioned);
    expect(speakerDefault).toEqual(speakerVersioned);

    const defaultFused = fuseFeatures(speakerDefault, motionDefault, touchDefault);
    const versionedFused = fuseFeatures(
      speakerVersioned,
      motionVersioned,
      touchVersioned,
    );
    expect(defaultFused).toEqual(versionedFused);
    expect(simhash(defaultFused)).toEqual(simhash(versionedFused, 0));
  });

  it("activates the corrected speaker path only in version 1", async () => {
    const audio = deterministicAudio();
    expect(await extractSpeakerFeatures(audio, 1)).not.toEqual(
      await extractSpeakerFeatures(audio, 0),
    );
  });
});
