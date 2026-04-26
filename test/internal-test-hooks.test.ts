import { describe, it, expect } from "vitest";
import { PulseSDK } from "../src/pulse";
import type {
  AudioCapture,
  MotionSample,
  TouchSample,
} from "../src/sensor/types";

// `IAM_INTERNAL_TEST=1 npm run test:internal` flips vitest's `define` constant
// so __injectSensorData accepts injection. Default `npm test` keeps it off,
// verifying the production throw path. The test file runs under both modes,
// with assertions gated by this flag.
const isInternalTestBuild = process.env.IAM_INTERNAL_TEST === "1";

function validAudio(): AudioCapture {
  const samples = new Float32Array(20000);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = Math.sin(i * 0.01) * 0.1;
  }
  return { samples, sampleRate: 16000, duration: 1.25 };
}

function validMotion(count = 20): MotionSample[] {
  return Array.from({ length: count }, (_, i) => ({
    timestamp: i * 50,
    ax: Math.sin(i * 0.1) * 0.5,
    ay: Math.cos(i * 0.1) * 0.5,
    az: 9.8 + Math.sin(i * 0.3) * 0.2,
    gx: Math.sin(i * 0.2) * 0.1,
    gy: Math.cos(i * 0.2) * 0.1,
    gz: Math.sin(i * 0.15) * 0.05,
  }));
}

function validTouch(count = 20): TouchSample[] {
  return Array.from({ length: count }, (_, i) => ({
    timestamp: i * 50,
    x: 100 + i * 5,
    y: 200 + Math.sin(i * 0.2) * 20,
    pressure: 0.5,
    width: 20,
    height: 20,
  }));
}

function newSession() {
  const sdk = new PulseSDK({
    relayerUrl: "http://127.0.0.1:1",
    relayerApiKey: "test",
  });
  return sdk.createSession();
}

describe("PulseSession.__injectSensorData — production-build behavior", () => {
  it.skipIf(isInternalTestBuild)(
    "throws when called in default builds (npm publish path)",
    () => {
      const session = newSession();
      expect(() =>
        session.__injectSensorData({
          audio: validAudio(),
          motion: validMotion(),
          touch: validTouch(),
        }),
      ).toThrow(/internal test builds/i);
    },
  );

  it.skipIf(isInternalTestBuild)(
    "throw message instructs how to enable the hook",
    () => {
      const session = newSession();
      try {
        session.__injectSensorData({
          audio: validAudio(),
          motion: validMotion(),
          touch: validTouch(),
        });
        throw new Error("expected __injectSensorData to throw");
      } catch (err) {
        expect((err as Error).message).toMatch(/IAM_INTERNAL_TEST=1/);
      }
    },
  );
});

describe("PulseSession.__injectSensorData — internal-build behavior", () => {
  it.skipIf(!isInternalTestBuild)(
    "accepts valid sensor data and primes all three stages",
    () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validTouch(),
      });
      const s = session as unknown as {
        audioStageState: string;
        motionStageState: string;
        touchStageState: string;
        audioData: AudioCapture | null;
        motionData: MotionSample[];
        touchData: TouchSample[];
      };
      expect(s.audioStageState).toBe("captured");
      expect(s.motionStageState).toBe("captured");
      expect(s.touchStageState).toBe("captured");
      expect(s.audioData).not.toBeNull();
      expect(s.audioData!.samples.length).toBeGreaterThanOrEqual(16000);
      expect(s.motionData.length).toBeGreaterThanOrEqual(10);
      expect(s.touchData.length).toBeGreaterThanOrEqual(10);
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "rejects audio shorter than the capture-path minimum",
    () => {
      const session = newSession();
      expect(() =>
        session.__injectSensorData({
          audio: {
            samples: new Float32Array(100),
            sampleRate: 16000,
            duration: 0.01,
          },
          motion: validMotion(),
          touch: validTouch(),
        }),
      ).toThrow(/audio required, minimum/);
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "rejects motion below MIN_MOTION_SAMPLES",
    () => {
      const session = newSession();
      expect(() =>
        session.__injectSensorData({
          audio: validAudio(),
          motion: validMotion(3),
          touch: validTouch(),
        }),
      ).toThrow(/motion required, minimum/);
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "rejects touch below MIN_TOUCH_SAMPLES",
    () => {
      const session = newSession();
      expect(() =>
        session.__injectSensorData({
          audio: validAudio(),
          motion: validMotion(),
          touch: [],
        }),
      ).toThrow(/touch required, minimum/);
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "refuses injection while a stage is actively capturing",
    () => {
      const session = newSession();
      // Force a stage into "capturing" without invoking browser APIs by
      // mutating private state directly (test-only narrow cast).
      const s = session as unknown as { audioStageState: string };
      s.audioStageState = "capturing";
      expect(() =>
        session.__injectSensorData({
          audio: validAudio(),
          motion: validMotion(),
          touch: validTouch(),
        }),
      ).toThrow(/cannot inject while stages are capturing: audio/);
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "complete() after injection passes the capture-state pre-flight check",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validTouch(),
      });
      // complete() will fail downstream (unreachable relayer, no wallet) but must
      // NOT throw the "stages still capturing" synchronous error — injection
      // should have transitioned every stage to "captured".
      const result = await session.complete();
      expect(result).toBeDefined();
      expect(typeof result.success).toBe("boolean");
      if (result.error) {
        expect(result.error).not.toMatch(/stages still capturing/i);
        expect(result.error).not.toMatch(/Insufficient behavioral data/i);
        expect(result.error).not.toMatch(/No voice data detected/i);
      }
    },
  );
});
