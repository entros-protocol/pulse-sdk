import { describe, it, expect, afterEach, vi } from "vitest";
import { PulseSDK } from "../src/pulse";
import { resampleCurveTrace } from "../src/sensor/curve";
import type { AudioCapture, MotionSample, TouchSample, CurveTracePoint } from "../src/sensor/types";

// Body-shape assertions need the internal injection hook (see
// internal-test-hooks.test.ts). Under default `npm test` these are skipped.
const isInternalTestBuild = process.env.IAM_INTERNAL_TEST === "1";

function validAudio(): AudioCapture {
  const samples = new Float32Array(20000);
  for (let i = 0; i < samples.length; i++) samples[i] = Math.sin(i * 0.01) * 0.1;
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
function rawOutline(): CurveTracePoint[] {
  return Array.from({ length: 30 }, (_, i) => ({ x: i * 3, y: 100 + i, t: i * 40 }));
}

// walletAddress derivation only needs publicKey.toBase58() (pulse.ts:510-511).
const fakeWallet = { publicKey: { toBase58: () => "So11111111111111111111111111111111111111112" } };
const fakeConnection = { getAccountInfo: async () => null };

function newSession() {
  const sdk = new PulseSDK({ relayerUrl: "https://executor.test", relayerApiKey: "test" });
  return sdk.createSession();
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

    await session.complete(fakeWallet, undefined, undefined, outline);

    const body = getBody();
    expect(body).toBeDefined();
    expect(body!.curve_trace).toEqual(resampleCurveTrace(outline));
  });

  it.skipIf(!isInternalTestBuild)("verify without an outline omits curve_trace", async () => {
    const session = newSession();
    session.__injectSensorData({ audio: validAudio(), motion: validMotion(), touch: validTouch() });
    const getBody = stubFetchCapturing();

    await session.complete(fakeWallet);

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
});
