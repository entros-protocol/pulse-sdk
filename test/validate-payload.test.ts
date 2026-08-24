import { describe, it, expect, afterEach, vi } from "vitest";
import { PublicKey } from "@solana/web3.js";
import { PulseSDK } from "../src/pulse";
import { PROGRAM_IDS } from "../src/config";
import { resampleCurveTrace } from "../src/sensor/curve";
import { extractMotionFeatures } from "../src/extraction/kinematic";
import type { AudioCapture, MotionSample, TouchSample, CurveTracePoint } from "../src/sensor/types";
import type { StudyContext } from "../src/study";

// Body-shape assertions need the internal injection hook (see
// internal-test-hooks.test.ts). Under default `npm test` these are skipped.
const isInternalTestBuild = process.env.IAM_INTERNAL_TEST === "1";

function validAudio(): AudioCapture {
  const samples = new Float32Array(20000);
  for (let i = 0; i < samples.length; i++) samples[i] = Math.sin(i * 0.01) * 0.1;
  const duration = samples.length / 16000;
  // The window and `validMotion` share a zero-based clock, and the motion span
  // covers the whole window. A fixture whose two streams disagree yields no
  // contour, so the cross-modal fields of the payload would go untested by the
  // suite whose subject is the payload. `tsconfig.json` excludes `test/`, so
  // no type error says an `AudioCapture` here is incomplete.
  return {
    samples,
    sampleRate: 16000,
    duration,
    windowStartMs: 0,
    windowEndMs: duration * 1000,
    inputLevel: { rms: 0.07, peak: 0.1, gain: 1, gainClipped: false, voicedFrameRatio: 1 },
    voiceIsolationApplied: null,
  };
}
// 26 samples at 50ms spans 1250ms, exactly `validAudio`'s duration.
function validMotion(count = 26): MotionSample[] {
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
function rawOutline(): CurveTracePoint[] {
  return Array.from({ length: 30 }, (_, i) => ({ x: i * 3, y: 100 + i, t: i * 40 }));
}

const fakeWallet = { publicKey: new PublicKey("So11111111111111111111111111111111111111112") };
const registryProgramId = new PublicKey(PROGRAM_IDS.entrosRegistry);
const [protocolConfigPda] = PublicKey.findProgramAddressSync(
  [new TextEncoder().encode("protocol_config")],
  registryProgramId,
);
const fakeConnection = {
  getAccountInfo: async (address: PublicKey) =>
    address.equals(protocolConfigPda)
      ? { data: Buffer.alloc(109), owner: registryProgramId }
      : null,
};
const versionOneConnection = {
  getAccountInfo: async (address: PublicKey) => {
    if (!address.equals(protocolConfigPda)) return null;
    const data = Buffer.alloc(113);
    data.writeUInt16LE(1, 109);
    data.writeUInt16LE(0, 111);
    return { data, owner: registryProgramId };
  },
};
const versionTwoConnection = {
  getAccountInfo: async (address: PublicKey) => {
    if (!address.equals(protocolConfigPda)) return null;
    const data = Buffer.alloc(113);
    data.writeUInt16LE(2, 109);
    data.writeUInt16LE(1, 111);
    return { data, owner: registryProgramId };
  },
};

function validNormalizedTouch(count = 61): TouchSample[] {
  return Array.from({ length: count }, (_, index) => ({
    timestamp: index * 50,
    x: 0.1 + (index / (count - 1)) * 0.8,
    y: 0.5 + Math.sin(index * 0.1) * 0.2,
    pressure: 0.5,
    width: 1,
    height: 1,
  }));
}

function newSession(studyContext?: StudyContext) {
  const sdk = new PulseSDK({
    cluster: "devnet",
    relayerUrl: "https://executor.test",
    relayerApiKey: "test",
  });
  return sdk.createSession(undefined, studyContext);
}

function primeSessionWithoutTouch(session: ReturnType<typeof newSession>): void {
  const internal = session as unknown as {
    audioData: AudioCapture;
    motionData: MotionSample[];
    touchData: TouchSample[];
    audioStageState: "captured";
    motionStageState: "captured";
    touchStageState: "captured";
  };
  internal.audioData = validAudio();
  internal.motionData = validMotion();
  internal.touchData = [];
  internal.audioStageState = "captured";
  internal.motionStageState = "captured";
  internal.touchStageState = "captured";
}

/**
 * Stub global fetch (returning a non-OK response so the SDK short-circuits after
 * the validate call without running proof/submit) and return a getter for the
 * parsed `/validate-features` request body.
 */
function stubFetchCapturing(): () => Record<string, unknown> | undefined {
  const mockFetch = vi.fn().mockResolvedValue({
    ok: false,
    status: 500,
    json: async () => ({ error: "stubbed" }),
  } as Response);
  vi.stubGlobal("fetch", mockFetch);
  return () => {
    const call = mockFetch.mock.calls.find(
      (c) => typeof c[0] === "string" && (c[0] as string).includes("/validate-features"),
    );
    if (!call) return undefined;
    return JSON.parse((call[1] as RequestInit).body as string) as Record<string, unknown>;
  };
}

afterEach(() => vi.restoreAllMocks());

describe("/validate-features body — curve_trace", () => {
  it.skipIf(!isInternalTestBuild)("verify carries the resampled curve_trace outline", async () => {
    const session = newSession();
    session.__injectSensorData({ audio: validAudio(), motion: validMotion(), touch: validTouch() });
    const getBody = stubFetchCapturing();
    const outline = rawOutline();

    await session.complete(fakeWallet, fakeConnection, undefined, outline);

    const body = getBody();
    expect(body).toBeDefined();
    expect(body!.curve_trace).toEqual(resampleCurveTrace(outline));
    expect(body!.baseline_reset).toBe(false);
  });

  it.skipIf(!isInternalTestBuild)("verify without an outline omits curve_trace", async () => {
    const session = newSession();
    session.__injectSensorData({ audio: validAudio(), motion: validMotion(), touch: validTouch() });
    const getBody = stubFetchCapturing();

    await session.complete(fakeWallet, fakeConnection);

    const body = getBody();
    expect(body).toBeDefined();
    expect("curve_trace" in body!).toBe(false);
  });

  it.skipIf(!isInternalTestBuild)("reset never carries curve_trace", async () => {
    const session = newSession();
    session.__injectSensorData({ audio: validAudio(), motion: validMotion(), touch: validTouch() });
    const getBody = stubFetchCapturing();

    await session.completeReset(fakeWallet, fakeConnection);

    const body = getBody();
    expect(body).toBeDefined();
    expect("curve_trace" in body!).toBe(false);
  });

  it.skipIf(!isInternalTestBuild)("requests a purpose-3 receipt for a version-one reset", async () => {
    const session = newSession();
    session.__injectSensorData({ audio: validAudio(), motion: validMotion(), touch: validTouch() });
    const getBody = stubFetchCapturing();

    await session.completeReset(fakeWallet, versionOneConnection);

    expect(getBody()).toMatchObject({
      projection_version: 1,
      request_receipt: true,
      receipt_purpose: "reset",
      baseline_reset: true,
    });
  });
});

describe("/validate-features body - study context", () => {
  it.skipIf(!isInternalTestBuild)("omits study for every normal request", async () => {
    const session = newSession();
    session.__injectSensorData({ audio: validAudio(), motion: validMotion(), touch: validTouch() });
    const getBody = stubFetchCapturing();

    await session.complete(fakeWallet, fakeConnection);

    const body = getBody();
    expect(body).toBeDefined();
    expect("study" in body!).toBe(false);
  });

  it.skipIf(!isInternalTestBuild)("forwards only the typed active study context", async () => {
    const context: StudyContext = {
      token: "opaque-study-token",
      record_id: "00112233445566778899aabbccddeeff",
      capture_class: "web-mobile",
      feature_schema_version: 3,
      projection_version: 0,
    };
    const session = newSession(context);
    session.__injectSensorData({ audio: validAudio(), motion: validMotion(), touch: validTouch() });
    const getBody = stubFetchCapturing();

    await session.complete(fakeWallet, fakeConnection);

    expect(getBody()?.study).toEqual(context);
  });

  it.skipIf(!isInternalTestBuild)("binds projection 2 capture and study metadata", async () => {
    const context: StudyContext = {
      token: "opaque-study-token",
      record_id: "00112233445566778899aabbccddeeff",
      capture_class: "web-desktop",
      feature_schema_version: 5,
      projection_version: 2,
    };
    const session = newSession(context);
    session.__injectSensorData({
      audio: validAudio(),
      motion: validMotion(),
      touch: validNormalizedTouch(),
    });
    const getBody = stubFetchCapturing();

    await session.complete(fakeWallet, versionTwoConnection);

    const body = getBody();
    expect(body).toMatchObject({
      projection_version: 2,
      study: context,
    });
    expect(body?.features).toHaveLength(308);
    expect(body).not.toHaveProperty("touch_samples");
  });

  it.skipIf(!isInternalTestBuild)("does not reinterpret capture after a policy cutover", async () => {
    const session = newSession();
    const internal = session as unknown as {
      pinProjectionPolicy(connection?: unknown): Promise<{
        current: number;
        minimum: number;
      }>;
    };
    await internal.pinProjectionPolicy(versionOneConnection);
    session.__injectSensorData({
      audio: validAudio(),
      motion: validMotion(),
      touch: validTouch(),
    });
    const getBody = stubFetchCapturing();

    const result = await session.complete(fakeWallet, versionTwoConnection);

    expect(result).toMatchObject({
      success: false,
      failedAt: "capture",
    });
    expect(result.error).toMatch(/projection policy changed/i);
    expect(getBody()).toBeUndefined();
  });

  it("rejects projection 2 completion without touch evidence", async () => {
    const session = newSession();
    primeSessionWithoutTouch(session);
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    const result = await session.complete(fakeWallet, versionTwoConnection);

    expect(result).toMatchObject({
      success: false,
      failedAt: "capture",
    });
    expect(result.error).toMatch(/no touch trace/i);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("rejects projection 2 reset without touch evidence", async () => {
    const session = newSession();
    primeSessionWithoutTouch(session);
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    const result = await session.completeReset(fakeWallet, versionTwoConnection);

    expect(result).toMatchObject({
      success: false,
      failedAt: "capture",
    });
    expect(result.error).toMatch(/no touch trace/i);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

describe("/validate-features body - mobile modality selection", () => {
  it.skipIf(!isInternalTestBuild)(
    "uses accelerometer features when a complete capture also has touch",
    async () => {
      const motion = validMotion();
      const session = newSession();
      session.__injectSensorData({ audio: validAudio(), motion, touch: validTouch() });
      const getBody = stubFetchCapturing();

      await session.complete(fakeWallet, versionOneConnection);

      const features = getBody()?.features as number[];
      expect(features.slice(170, 251)).toEqual(extractMotionFeatures(motion, 1));
    },
  );

  it.skipIf(!isInternalTestBuild)("reports the bounded voice-isolation state", async () => {
    const audio = validAudio();
    audio.voiceIsolationApplied = true;
    const session = newSession();
    session.__injectSensorData({ audio, motion: validMotion(), touch: validTouch() });
    const getBody = stubFetchCapturing();

    await session.complete(fakeWallet, fakeConnection);

    const clientSignals = getBody()?.client_signals as {
      capture?: { voice_isolation_applied?: boolean | null };
    };
    expect(clientSignals.capture?.voice_isolation_applied).toBe(true);
  });
});
