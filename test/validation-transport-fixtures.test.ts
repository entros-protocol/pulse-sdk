import { Buffer } from "node:buffer";

import { describe, expect, it } from "vitest";

import {
  buildSyntheticValidationTransportRequest,
  buildSyntheticValidationAuthorizationDigest,
  decodeSyntheticValidationTransportEnvelope,
  encodeSyntheticValidationTransportEnvelope,
  EXECUTOR_REQUEST_BODY_LIMIT_BYTES,
  inspectSyntheticValidationTransportRequest,
  SYNTHETIC_AUDIO_SAMPLE_RATE_HZ,
  SYNTHETIC_CURVE_POINT_COUNT,
  SYNTHETIC_FEATURE_COUNT,
  syntheticPcmSample,
  SYNTHETIC_WALLET_ID,
  VALIDATION_TRANSPORT_FIXTURE_VERSION,
  VALIDATION_TRANSPORT_PROFILES,
} from "../scripts/support/validation-transport-fixtures";

const ENVELOPE_HEADER_BYTES = 16;

const COMMON_FIELDS = [
  "features",
  "projection_version",
  "wallet_id",
  "f0_contour",
  "accel_magnitude",
  "audio_samples_b64",
  "audio_sample_rate_hz",
  "commitment_new_hex",
  "request_receipt",
  "receipt_purpose",
  "baseline_reset",
  "client_signals",
  "curve_trace",
];

const EXPECTED_STABLE_METRICS = {
  "projection-1-12s": {
    jsonBytes: 531_933,
    binaryBytes: 403_926,
    binarySha256:
      "50e4485e854d0811ae940b8426ab0ead42843f083f96f3eb83b678f7b7de26b3",
    base64Bytes: 512_000,
    decodedPcmBytes: 384_000,
    pcmSha256:
      "1d48b0b5fbfba855850def32635aa57da79de251f0922c32cc6fac4da0b63358",
    authorizationDigest:
      "11f299282f68352e2a289117f7eee1a2fe63e610ba13419e2162831c44ccf460",
    fixtureSha256:
      "969fbce2cd41b5f5ca5a47a4fd55aafb2c4d24cfd340ba9d4a890db11cf5ee4d",
  },
  "projection-2-12s": {
    jsonBytes: 534_612,
    binaryBytes: 406_605,
    binarySha256:
      "94a4d1f8a2473cc6e8655c55b655561db7ef9c985fdb643c7dab382ec9d3275c",
    base64Bytes: 512_000,
    decodedPcmBytes: 384_000,
    pcmSha256:
      "1d48b0b5fbfba855850def32635aa57da79de251f0922c32cc6fac4da0b63358",
    authorizationDigest:
      "8b86127ac85da7e3c81fa0571bc3d249effa555579ed3d40710fa02e48f417e4",
    fixtureSha256:
      "6c5e983939dbef111e387b643e53b822a9d093f6b7c8333cb019b4a7c358962e",
  },
  "projection-1-20s": {
    jsonBytes: 883_227,
    binaryBytes: 669_884,
    binarySha256:
      "a6aab11fcb19555f9cf959cf4256402c577edece9c2b12e193368b715e61a4ef",
    base64Bytes: 853_336,
    decodedPcmBytes: 640_000,
    pcmSha256:
      "76613da053ac8dbe3d1d42b1f34128c350ea2d80f94f2bef5893c67a8f18f071",
    authorizationDigest:
      "3b7d7acda8f4b3972d4b8fd064c54f7ab0cf8e383092929acb2f725032a3614e",
    fixtureSha256:
      "394f5cae1e54dda03ac68a01ff485f13063eb673f60bbe316458305fa803718b",
  },
  "projection-2-20s": {
    jsonBytes: 885_906,
    binaryBytes: 672_563,
    binarySha256:
      "531cc0f7b323903c646646cd40d6e3ad75be501404003765c1cfa5d0a694b717",
    base64Bytes: 853_336,
    decodedPcmBytes: 640_000,
    pcmSha256:
      "76613da053ac8dbe3d1d42b1f34128c350ea2d80f94f2bef5893c67a8f18f071",
    authorizationDigest:
      "698b79352706fb7f47d8080ad03a73ec6ca37cb8bbdfa9f50def963a64c2f3cb",
    fixtureSha256:
      "6bf7a57f5dbeb14aaf8b92bcdac7f354db97bcf09a54d964323db7e24a09f52f",
  },
} as const;

function collectObjectKeys(
  value: unknown,
  keys = new Set<string>(),
): Set<string> {
  if (Array.isArray(value)) {
    for (const item of value) collectObjectKeys(item, keys);
    return keys;
  }
  if (!value || typeof value !== "object") return keys;

  for (const [key, nested] of Object.entries(value)) {
    keys.add(key);
    collectObjectKeys(nested, keys);
  }
  return keys;
}

function replaceEnvelopeMetadata(
  envelope: Buffer,
  metadata: Uint8Array,
): Buffer {
  const originalMetadataLength = envelope.readUInt32LE(8);
  const pcm = envelope.subarray(ENVELOPE_HEADER_BYTES + originalMetadataLength);
  const header = Buffer.from(envelope.subarray(0, ENVELOPE_HEADER_BYTES));
  header.writeUInt32LE(metadata.byteLength, 8);
  return Buffer.concat([header, metadata, pcm]);
}

describe("synthetic validation transport fixtures", () => {
  it("freezes the four projection and duration profiles", () => {
    expect(VALIDATION_TRANSPORT_PROFILES).toEqual([
      {
        name: "projection-1-12s",
        projectionVersion: 1,
        durationMs: 12_000,
      },
      {
        name: "projection-2-12s",
        projectionVersion: 2,
        durationMs: 12_000,
      },
      {
        name: "projection-1-20s",
        projectionVersion: 1,
        durationMs: 20_000,
      },
      {
        name: "projection-2-20s",
        projectionVersion: 2,
        durationMs: 20_000,
      },
    ]);
    expect(Object.isFrozen(VALIDATION_TRANSPORT_PROFILES)).toBe(true);
    for (const profile of VALIDATION_TRANSPORT_PROFILES) {
      expect(Object.isFrozen(profile)).toBe(true);
    }
  });

  it.each(VALIDATION_TRANSPORT_PROFILES)(
    "builds deterministic derived fields for $name",
    (profile) => {
      const request = buildSyntheticValidationTransportRequest(profile);
      expect(request.features).toHaveLength(SYNTHETIC_FEATURE_COUNT);
      expect(request.features[0]).toBe(-154 / 16);
      expect(request.features[154]).toBe(0);
      expect(request.features[307]).toBe(153 / 16);
      expect(request.f0_contour).toHaveLength(profile.durationMs / 10);
      expect(request.accel_magnitude).toHaveLength(profile.durationMs / 10);
      expect(request.curve_trace.points).toHaveLength(
        SYNTHETIC_CURVE_POINT_COUNT,
      );
      expect(request.curve_trace.duration_ms).toBe(profile.durationMs);
      expect(request.audio_sample_rate_hz).toBe(SYNTHETIC_AUDIO_SAMPLE_RATE_HZ);

      const decoded = Buffer.from(request.audio_samples_b64, "base64");
      const sampleCount =
        (profile.durationMs * SYNTHETIC_AUDIO_SAMPLE_RATE_HZ) / 1_000;
      expect(decoded.byteLength).toBe(sampleCount * 2);
      for (const index of [
        0,
        1,
        Math.floor(sampleCount / 2),
        sampleCount - 1,
      ]) {
        expect(decoded.readInt16LE(index * 2)).toBe(syntheticPcmSample(index));
      }
    },
  );

  it.each(VALIDATION_TRANSPORT_PROFILES)(
    "keeps $name within field and privacy boundaries",
    (profile) => {
      const request = buildSyntheticValidationTransportRequest(profile);
      const expectedFields =
        profile.projectionVersion === 2
          ? [...COMMON_FIELDS, "compatibility_evidence", "wallet_authorization"]
          : COMMON_FIELDS;
      expect(Object.keys(request)).toEqual(expectedFields);
      expect(request.wallet_id).toBe(SYNTHETIC_WALLET_ID);
      expect(request.client_signals).toEqual({
        v: 1,
        env: "non-browser",
        automation: { webdriver: false, tells: [] },
        capture: {
          virtual_device: false,
          voice_isolation_applied: null,
          flatness: 0.125,
          centroid: 2_400,
        },
      });
      const keys = collectObjectKeys(request);
      for (const forbidden of [
        "study",
        "token",
        "record_id",
        "participant",
        "participant_id",
        "email",
        "user_agent",
        "ip_address",
        "transcript",
        "phrase",
        "raw_motion",
        "motion_data",
        "touch_samples",
        "raw_touch",
        "compatibility_touch",
        "audio_samples",
        "fingerprint",
      ]) {
        expect(keys.has(forbidden)).toBe(false);
      }
      for (const point of request.curve_trace.points) {
        expect(point).toHaveLength(2);
      }
    },
  );

  it.each(VALIDATION_TRANSPORT_PROFILES)(
    "includes projection-specific evidence for $name",
    (profile) => {
      const request = buildSyntheticValidationTransportRequest(profile);
      if (profile.projectionVersion === 1) {
        expect(request.compatibility_evidence).toBeUndefined();
        expect(request.wallet_authorization).toBeUndefined();
        return;
      }

      expect(request.compatibility_evidence).toMatchObject({
        projection_version: 1,
        feature_schema_version: 4,
      });
      expect(request.compatibility_evidence?.features).toHaveLength(
        SYNTHETIC_FEATURE_COUNT,
      );
      expect(request.compatibility_evidence?.features[0]).toBe(-154 / 32);
      expect(request.compatibility_evidence?.features[154]).toBe(0);
      expect(request.compatibility_evidence?.features[307]).toBe(153 / 32);
      expect(request.wallet_authorization?.nonce).toHaveLength(32);
      expect(request.wallet_authorization?.signature_hex).toMatch(
        /^[0-9a-f]{128}$/,
      );
    },
  );

  it.each(VALIDATION_TRANSPORT_PROFILES)(
    "stays below the executor body limit for $name",
    (profile) => {
      const request = buildSyntheticValidationTransportRequest(profile);
      const metrics = inspectSyntheticValidationTransportRequest(
        profile,
        request,
      );
      expect({
        jsonBytes: metrics.jsonBytes,
        binaryBytes: metrics.binaryBytes,
        binarySha256: metrics.binarySha256,
        base64Bytes: metrics.base64Bytes,
        decodedPcmBytes: metrics.decodedPcmBytes,
        pcmSha256: metrics.pcmSha256,
        authorizationDigest: metrics.authorizationDigest,
        fixtureSha256: metrics.fixtureSha256,
      }).toEqual(EXPECTED_STABLE_METRICS[profile.name]);
      expect(metrics.jsonBytes).toBeLessThan(EXECUTOR_REQUEST_BODY_LIMIT_BYTES);
      expect(metrics.binaryBytes).toBeLessThan(metrics.jsonBytes);
      expect(metrics.base64Bytes).toBe(request.audio_samples_b64.length);
      expect(metrics.decodedPcmBytes).toBe(
        (profile.durationMs * SYNTHETIC_AUDIO_SAMPLE_RATE_HZ * 2) / 1_000,
      );
    },
  );

  it.each(VALIDATION_TRANSPORT_PROFILES)(
    "preserves $name through the versioned binary envelope",
    (profile) => {
      const request = buildSyntheticValidationTransportRequest(profile);
      const envelope = encodeSyntheticValidationTransportEnvelope(request);
      const decoded = decodeSyntheticValidationTransportEnvelope(envelope);

      expect(envelope.subarray(0, 4).toString("ascii")).toBe("ENTV");
      expect(envelope.readUInt16LE(4)).toBe(
        VALIDATION_TRANSPORT_FIXTURE_VERSION,
      );
      expect(envelope.readUInt16LE(6)).toBe(1);
      expect(
        ENVELOPE_HEADER_BYTES +
          envelope.readUInt32LE(8) +
          envelope.readUInt32LE(12),
      ).toBe(envelope.byteLength);
      const metadata = JSON.parse(
        envelope
          .subarray(
            ENVELOPE_HEADER_BYTES,
            ENVELOPE_HEADER_BYTES + envelope.readUInt32LE(8),
          )
          .toString("utf8"),
      ) as Record<string, unknown>;
      expect(metadata).not.toHaveProperty("audio_samples_b64");
      const { audio_samples_b64: decodedAudio, ...decodedMetadata } = decoded;
      expect(decodedMetadata).toEqual(metadata);
      expect(decodedAudio).toBe(request.audio_samples_b64);
      expect(buildSyntheticValidationAuthorizationDigest(decoded)).toBe(
        buildSyntheticValidationAuthorizationDigest(request),
      );

      const metrics = inspectSyntheticValidationTransportRequest(
        profile,
        request,
      );
      expect(metrics.binaryBytes).toBe(
        EXPECTED_STABLE_METRICS[profile.name].binaryBytes,
      );
      expect(metrics.binarySha256).toBe(
        EXPECTED_STABLE_METRICS[profile.name].binarySha256,
      );
    },
  );

  it.each(VALIDATION_TRANSPORT_PROFILES)(
    "keeps serialization and authorization stable for $name",
    (profile) => {
      const first = buildSyntheticValidationTransportRequest(profile);
      const second = buildSyntheticValidationTransportRequest(profile);
      const firstMetrics = inspectSyntheticValidationTransportRequest(
        profile,
        first,
      );
      const secondMetrics = inspectSyntheticValidationTransportRequest(
        profile,
        second,
      );
      expect(secondMetrics).toEqual(firstMetrics);
      expect(buildSyntheticValidationAuthorizationDigest(second)).toBe(
        buildSyntheticValidationAuthorizationDigest(first),
      );
    },
  );
});

describe("versioned binary envelope rejection", () => {
  const request = buildSyntheticValidationTransportRequest(
    VALIDATION_TRANSPORT_PROFILES[1],
  );
  const envelope = encodeSyntheticValidationTransportEnvelope(request);

  it("rejects truncated headers and bodies", () => {
    expect(() =>
      decodeSyntheticValidationTransportEnvelope(envelope.subarray(0, 15)),
    ).toThrow(/truncated envelope header/i);
    expect(() =>
      decodeSyntheticValidationTransportEnvelope(envelope.subarray(0, -1)),
    ).toThrow(/truncated or trailing envelope data/i);
  });

  it("rejects bad magic, unknown versions, and unknown flags", () => {
    const badMagic = Buffer.from(envelope);
    badMagic[0] ^= 0xff;
    expect(() => decodeSyntheticValidationTransportEnvelope(badMagic)).toThrow(
      /invalid envelope magic/i,
    );

    const unknownVersion = Buffer.from(envelope);
    unknownVersion.writeUInt16LE(2, 4);
    expect(() =>
      decodeSyntheticValidationTransportEnvelope(unknownVersion),
    ).toThrow(/unsupported envelope version 2/i);

    const unknownFlags = Buffer.from(envelope);
    unknownFlags.writeUInt16LE(3, 6);
    expect(() =>
      decodeSyntheticValidationTransportEnvelope(unknownFlags),
    ).toThrow(/invalid envelope flags/i);
  });

  it("rejects malformed and duplicate-audio metadata", () => {
    const malformed = replaceEnvelopeMetadata(envelope, Buffer.from([0xff]));
    expect(() => decodeSyntheticValidationTransportEnvelope(malformed)).toThrow(
      /invalid envelope metadata/i,
    );

    for (const invalidMetadata of [null, []]) {
      const invalid = replaceEnvelopeMetadata(
        envelope,
        Buffer.from(JSON.stringify(invalidMetadata), "utf8"),
      );
      expect(() => decodeSyntheticValidationTransportEnvelope(invalid)).toThrow(
        /invalid envelope metadata/i,
      );
    }

    const metadataLength = envelope.readUInt32LE(8);
    const metadata = JSON.parse(
      envelope
        .subarray(ENVELOPE_HEADER_BYTES, ENVELOPE_HEADER_BYTES + metadataLength)
        .toString("utf8"),
    ) as Record<string, unknown>;
    metadata.audio_samples_b64 = "AA==";
    const duplicateAudio = replaceEnvelopeMetadata(
      envelope,
      Buffer.from(JSON.stringify(metadata), "utf8"),
    );
    expect(() =>
      decodeSyntheticValidationTransportEnvelope(duplicateAudio),
    ).toThrow(/metadata duplicates envelope PCM/i);

    const missingRequiredFields = replaceEnvelopeMetadata(
      envelope,
      Buffer.from("{}", "utf8"),
    );
    expect(() =>
      decodeSyntheticValidationTransportEnvelope(missingRequiredFields),
    ).toThrow(/invalid envelope metadata/i);

    const invalidUtf8 = Buffer.from(JSON.stringify(metadata), "utf8");
    const walletOffset = invalidUtf8.indexOf(
      Buffer.from(request.wallet_id, "utf8"),
    );
    expect(walletOffset).toBeGreaterThanOrEqual(0);
    invalidUtf8[walletOffset] = 0xff;
    expect(() =>
      decodeSyntheticValidationTransportEnvelope(
        replaceEnvelopeMetadata(envelope, invalidUtf8),
      ),
    ).toThrow(/invalid envelope metadata/i);
  });

  it("rejects zero-length and odd-length PCM16", () => {
    for (const audio_samples_b64 of ["", "AA=="]) {
      expect(() =>
        encodeSyntheticValidationTransportEnvelope({
          ...request,
          audio_samples_b64,
        }),
      ).toThrow(/requires non-empty PCM16/i);
    }

    const metadataLength = envelope.readUInt32LE(8);
    const empty = Buffer.from(
      envelope.subarray(0, ENVELOPE_HEADER_BYTES + metadataLength),
    );
    empty.writeUInt32LE(0, 12);
    expect(() => decodeSyntheticValidationTransportEnvelope(empty)).toThrow(
      /invalid envelope PCM length/i,
    );

    const odd = Buffer.from(envelope.subarray(0, -1));
    odd.writeUInt32LE(envelope.readUInt32LE(12) - 1, 12);
    expect(() => decodeSyntheticValidationTransportEnvelope(odd)).toThrow(
      /invalid envelope PCM length/i,
    );
  });

  it("rejects trailing bytes", () => {
    const trailing = Buffer.concat([envelope, Buffer.from([0])]);
    expect(() => decodeSyntheticValidationTransportEnvelope(trailing)).toThrow(
      /truncated or trailing envelope data/i,
    );
  });

  it("rejects non-canonical base64 and oversized envelopes", () => {
    for (const audio_samples_b64 of ["Y W I =", "-_8="]) {
      expect(() =>
        encodeSyntheticValidationTransportEnvelope({
          ...request,
          audio_samples_b64,
        }),
      ).toThrow(/requires canonical base64 PCM/i);
    }

    expect(() =>
      decodeSyntheticValidationTransportEnvelope(
        Buffer.alloc(EXECUTOR_REQUEST_BODY_LIMIT_BYTES + 1),
      ),
    ).toThrow(/exceeds executor body limit/i);
  });
});
