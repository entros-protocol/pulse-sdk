import { describe, it, expect, afterEach, vi } from "vitest";
import { ed25519 } from "@noble/curves/ed25519";
import { Keypair, PublicKey } from "@solana/web3.js";
import { createHash } from "node:crypto";
import { PulseSDK } from "../src/pulse";
import { PROGRAM_IDS } from "../src/config";
import { resampleCurveTrace } from "../src/sensor/curve";
import {
  extractMotionFeatures,
  extractMouseDynamics,
} from "../src/extraction/kinematic";
import type { AudioCapture, MotionSample, TouchSample, CurveTracePoint } from "../src/sensor/types";
import type { StudyContext } from "../src/study";
import {
  buildValidationAuthorizationMessage,
  buildValidationRequestDigest,
  type ValidationDigestRequest,
} from "../src/validation/authorization";

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
const projectionTwoWalletSeed = new Uint8Array(32).fill(7);
const projectionTwoKeypair = Keypair.fromSeed(projectionTwoWalletSeed);
const projectionTwoWallet = {
  publicKey: projectionTwoKeypair.publicKey,
  signMessage: async (message: Uint8Array) =>
    ed25519.sign(message, projectionTwoWalletSeed),
};
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
const versionOneExistingConnection = {
  getAccountInfo: async (address: PublicKey) => {
    if (address.equals(protocolConfigPda)) {
      const data = Buffer.alloc(113);
      data.writeUInt16LE(1, 109);
      data.writeUInt16LE(0, 111);
      return { data, owner: registryProgramId };
    }
    return { data: identityAccount(1) };
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
const identityDiscriminator = createHash("sha256")
  .update("account:IdentityState")
  .digest()
  .subarray(0, 8);

function identityAccount(projectionVersion: number): Buffer {
  const data = Buffer.alloc(593);
  identityDiscriminator.copy(data, 0);
  data.fill(0x11, 8, 40);
  data.fill(0x22, 62, 94);
  data.fill(0x33, 94, 126);
  data.writeUInt16LE(projectionVersion, 583);
  return data;
}

function versionTwoConnectionWithIdentity(identityProjection: number) {
  return {
    getAccountInfo: async (address: PublicKey) => {
      if (address.equals(protocolConfigPda)) {
        const data = Buffer.alloc(113);
        data.writeUInt16LE(2, 109);
        data.writeUInt16LE(1, 111);
        return { data, owner: registryProgramId };
      }
      return { data: identityAccount(identityProjection) };
    },
  };
}

const versionTwoExistingConnection = versionTwoConnectionWithIdentity(2);
const versionTwoMigrationConnection = versionTwoConnectionWithIdentity(1);

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

function primeProjectionTwoDesktopSession(
  session: ReturnType<typeof newSession>,
  compatibilityTouch: TouchSample[],
): void {
  const internal = session as unknown as {
    audioData: AudioCapture;
    motionData: MotionSample[];
    touchData: TouchSample[];
    compatibilityTouchData: TouchSample[];
    audioStageState: "captured";
    motionStageState: "skipped";
    touchStageState: "captured";
  };
  internal.audioData = validAudio();
  internal.motionData = [];
  internal.touchData = validNormalizedTouch();
  internal.compatibilityTouchData = compatibilityTouch;
  internal.audioStageState = "captured";
  internal.motionStageState = "skipped";
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

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

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
      compatibilityTouch: validTouch(),
    });
    session.bindValidationChallenge(
      new Uint8Array(32).fill(0x41),
      performance.now() + 180_000,
    );
    const getBody = stubFetchCapturing();

    await session.complete(projectionTwoWallet, versionTwoConnection);

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

describe("baseline recovery result", () => {
  it.skipIf(!isInternalTestBuild)(
    "returns a structured reason when an existing identity has no local baseline",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validTouch(),
      });
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(
          new Response(JSON.stringify({ valid: true }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        ),
      );

      const result = await session.complete(
        fakeWallet,
        versionOneExistingConnection,
      );

      expect(result).toMatchObject({
        success: false,
        failedAt: "baseline",
        baselineRecovery: "signing-unavailable",
      });
      expect(result.error).toMatch(/baseline is missing/i);
    },
  );
});

describe("/validate-features body - projection 2 authorization", () => {
  it.skipIf(!isInternalTestBuild).each([
    ["projection 0", fakeConnection],
    ["projection 1", versionOneConnection],
  ])("preserves the legacy request field sequence for %s", async (_, connection) => {
    const session = newSession();
    session.__injectSensorData({
      audio: validAudio(),
      motion: validMotion(),
      touch: validTouch(),
    });
    const getBody = stubFetchCapturing();

    await session.complete(fakeWallet, connection);

    expect(Object.keys(getBody()!)).toEqual([
      "features",
      "projection_version",
      "f0_contour",
      "accel_magnitude",
      "wallet_id",
      "audio_samples_b64",
      "audio_sample_rate_hz",
      "commitment_new_hex",
      "request_receipt",
      "receipt_purpose",
      "baseline_reset",
      "client_signals",
      "capture_timing",
    ]);
  });

  it.skipIf(!isInternalTestBuild)(
    "binds a projection 2 mint to the challenge and same-capture projection 1 evidence",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validNormalizedTouch(),
        compatibilityTouch: validTouch(),
      });
      session.bindValidationChallenge(
        new Uint8Array(32).fill(0x5a),
        performance.now() + 180_000,
      );
      const getBody = stubFetchCapturing();

      await session.complete(projectionTwoWallet, versionTwoConnection);

      const body = getBody();
      expect(body?.compatibility_evidence).toMatchObject({
        projection_version: 1,
        feature_schema_version: 4,
      });
      expect(
        (body?.compatibility_evidence as { features: number[] }).features,
      ).toHaveLength(308);
      expect(
        (body?.compatibility_evidence as { features: number[] }).features.slice(
          0,
          170,
        ),
      ).toEqual((body?.features as number[]).slice(0, 170));
      expect(body?.wallet_authorization).toMatchObject({
        nonce: Array.from(new Uint8Array(32).fill(0x5a)),
      });
      const authorization = body?.wallet_authorization as {
        nonce: number[];
        signature_hex: string;
      };
      expect(authorization.signature_hex).toMatch(/^[0-9a-f]{128}$/);

      const signature = Uint8Array.from(
        authorization.signature_hex.match(/.{2}/g)!.map((byte) => parseInt(byte, 16)),
      );
      const digest = buildValidationRequestDigest(
        body as unknown as ValidationDigestRequest,
      );
      const message = new TextEncoder().encode(
        buildValidationAuthorizationMessage(
          projectionTwoWallet.publicKey.toBase58(),
          Uint8Array.from(authorization.nonce),
          2,
          digest,
        ),
      );
      expect(
        ed25519.verify(signature, message, projectionTwoWallet.publicKey.toBytes()),
      ).toBe(true);
      expect(
        (
          session as unknown as {
            compatibilityTouchData: TouchSample[];
          }
        ).compatibilityTouchData,
      ).toEqual([]);
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "omits compatibility evidence for a projection 2 update",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validNormalizedTouch(),
        compatibilityTouch: validTouch(),
      });
      session.bindValidationChallenge(
        new Uint8Array(32).fill(0x42),
        performance.now() + 180_000,
      );
      const getBody = stubFetchCapturing();

      await session.complete(projectionTwoWallet, versionTwoExistingConnection);

      const body = getBody();
      expect(body?.projection_version).toBe(2);
      expect(body).not.toHaveProperty("compatibility_evidence");
      expect(body).toHaveProperty("wallet_authorization");
      expect(body?.request_receipt).toBe(false);
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "derives desktop compatibility motion from raw-scale pointer samples",
    async () => {
      const compatibilityTouch = validTouch();
      const session = newSession();
      primeProjectionTwoDesktopSession(session, compatibilityTouch);
      session.bindValidationChallenge(
        new Uint8Array(32).fill(0x46),
        performance.now() + 180_000,
      );
      const getBody = stubFetchCapturing();

      await session.complete(projectionTwoWallet, versionTwoConnection);

      const body = getBody();
      const compatibilityFeatures = (
        body?.compatibility_evidence as { features: number[] }
      ).features;
      expect(compatibilityFeatures.slice(170, 251)).toEqual(
        extractMouseDynamics(compatibilityTouch, 1),
      );
      expect(body).not.toHaveProperty("touch_samples");
      expect(body).not.toHaveProperty("compatibility_touch");
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "includes compatibility evidence for a projection 2 rebaseline",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validNormalizedTouch(),
        compatibilityTouch: validTouch(),
      });
      session.bindValidationChallenge(
        new Uint8Array(32).fill(0x44),
        performance.now() + 180_000,
      );
      const getBody = stubFetchCapturing();

      await session.complete(projectionTwoWallet, versionTwoMigrationConnection);

      expect(getBody()).toMatchObject({
        projection_version: 2,
        receipt_purpose: "rebaseline",
        request_receipt: true,
        compatibility_evidence: {
          projection_version: 1,
          feature_schema_version: 4,
        },
      });
      expect(getBody()).toHaveProperty("wallet_authorization");
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "includes compatibility evidence for a projection 2 reset",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validNormalizedTouch(),
        compatibilityTouch: validTouch(),
      });
      session.bindValidationChallenge(
        new Uint8Array(32).fill(0x45),
        performance.now() + 180_000,
      );
      const getBody = stubFetchCapturing();

      await session.completeReset(
        projectionTwoWallet,
        versionTwoExistingConnection,
      );

      expect(getBody()).toMatchObject({
        projection_version: 2,
        receipt_purpose: "reset",
        request_receipt: true,
        baseline_reset: true,
        compatibility_evidence: {
          projection_version: 1,
          feature_schema_version: 4,
        },
      });
      expect(getBody()).toHaveProperty("wallet_authorization");
      expect(
        (
          session as unknown as {
            compatibilityTouchData: TouchSample[];
          }
        ).compatibilityTouchData,
      ).toEqual([]);
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "fails before transport when a projection 2 baseline capture lacks compatibility data",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validNormalizedTouch(),
      });
      session.bindValidationChallenge(
        new Uint8Array(32).fill(0x47),
        performance.now() + 180_000,
      );
      const fetchSpy = vi.spyOn(globalThis, "fetch");

      const result = await session.complete(
        projectionTwoWallet,
        versionTwoConnection,
      );

      expect(result).toMatchObject({ success: false, failedAt: "extraction" });
      expect(fetchSpy).not.toHaveBeenCalled();
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "fails before transport when the projection 2 wallet cannot sign messages",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validNormalizedTouch(),
        compatibilityTouch: validTouch(),
      });
      session.bindValidationChallenge(
        new Uint8Array(32).fill(0x48),
        performance.now() + 180_000,
      );
      const fetchSpy = vi.spyOn(globalThis, "fetch");

      const result = await session.complete(fakeWallet, versionTwoConnection);

      expect(result).toMatchObject({ success: false, failedAt: "signing" });
      expect(result.error).toMatch(/message signing/i);
      expect(fetchSpy).not.toHaveBeenCalled();
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "fails before wallet authorization when the projection 2 challenge expired",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validNormalizedTouch(),
        compatibilityTouch: validTouch(),
      });
      const now = vi.spyOn(performance, "now").mockReturnValue(1_000);
      session.bindValidationChallenge(new Uint8Array(32).fill(0x49), 3_000);
      now.mockReturnValue(4_001);
      const signMessage = vi.fn(projectionTwoWallet.signMessage);
      const fetchSpy = vi.spyOn(globalThis, "fetch");

      const result = await session.complete(
        { publicKey: projectionTwoWallet.publicKey, signMessage },
        versionTwoConnection,
      );

      expect(result).toMatchObject({ success: false, failedAt: "signing" });
      expect(result.error).toMatch(/challenge expired/i);
      expect(signMessage).not.toHaveBeenCalled();
      expect(fetchSpy).not.toHaveBeenCalled();
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "reports a projection 2 challenge that expires after wallet signing",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validNormalizedTouch(),
        compatibilityTouch: validTouch(),
      });
      const now = vi.spyOn(performance, "now").mockReturnValue(1_000);
      session.bindValidationChallenge(new Uint8Array(32).fill(0x4a), 5_000);
      const signMessage = vi.fn(async (message: Uint8Array) => {
        const signature = await projectionTwoWallet.signMessage(message);
        now.mockReturnValue(4_001);
        return signature;
      });
      const fetchSpy = vi.spyOn(globalThis, "fetch");

      const result = await session.complete(
        { publicKey: projectionTwoWallet.publicKey, signMessage },
        versionTwoConnection,
      );

      expect(result).toMatchObject({
        success: false,
        failedAt: "validation",
      });
      expect(result.error).toMatch(/challenge expired/i);
      expect(result).not.toHaveProperty("reason", "validation_unavailable");
      expect(signMessage).toHaveBeenCalledOnce();
      expect(fetchSpy).not.toHaveBeenCalled();
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "bounds validation transport by the remaining projection 2 challenge lifetime",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validNormalizedTouch(),
        compatibilityTouch: validTouch(),
      });
      const now = vi.spyOn(performance, "now").mockReturnValue(1_000);
      session.bindValidationChallenge(new Uint8Array(32).fill(0x4a), 31_000);
      const wallet = {
        publicKey: projectionTwoWallet.publicKey,
        signMessage: async (message: Uint8Array) => {
          now.mockReturnValue(29_500);
          return projectionTwoWallet.signMessage(message);
        },
      };
      vi.stubGlobal(
        "fetch",
        vi.fn().mockImplementation(
          (_url: string, init: RequestInit) =>
            new Promise((_resolve, reject) => {
              init.signal?.addEventListener(
                "abort",
                () => {
                  const error = new Error("aborted");
                  error.name = "AbortError";
                  reject(error);
                },
                { once: true },
              );
            }),
        ),
      );

      const result = await session.complete(wallet, versionTwoConnection);

      expect(result).toMatchObject({
        success: false,
        failedAt: "validation",
      });
      expect(result.error).toMatch(/challenge expired/i);
    },
  );

  it.skipIf(!isInternalTestBuild)(
    "consumes the challenge before transport so one session cannot replay it",
    async () => {
      const session = newSession();
      session.__injectSensorData({
        audio: validAudio(),
        motion: validMotion(),
        touch: validNormalizedTouch(),
        compatibilityTouch: validTouch(),
      });
      session.bindValidationChallenge(
        new Uint8Array(32).fill(0x43),
        performance.now() + 180_000,
      );
      stubFetchCapturing();

      await session.complete(
        projectionTwoWallet,
        versionTwoExistingConnection,
      );
      const replay = await session.complete(
        projectionTwoWallet,
        versionTwoExistingConnection,
      );

      const validateCalls = vi.mocked(globalThis.fetch).mock.calls.filter(
        (call) =>
          typeof call[0] === "string" &&
          call[0].includes("/validate-features"),
      );
      expect(validateCalls).toHaveLength(1);
      expect(replay).toMatchObject({ success: false, failedAt: "signing" });
      expect(replay.error).toMatch(/challenge nonce/i);
    },
  );
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
