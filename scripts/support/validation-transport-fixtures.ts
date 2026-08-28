import { Buffer } from "node:buffer";

import { sha256 } from "@noble/hashes/sha2.js";

import type { ClientSignals } from "../../src/client-signals/automation";
import type { CurveTraceOutline } from "../../src/sensor/types";
import { bytesToHex } from "../../src/submit/receipt";
import {
  buildValidationRequestDigest,
  type ValidationDigestRequest,
  type WalletAuthorization,
} from "../../src/validation/authorization";

export const EXECUTOR_REQUEST_BODY_LIMIT_BYTES = 1_048_576;
export const VALIDATION_TRANSPORT_FIXTURE_VERSION = 1;
export const VALIDATION_TRANSPORT_CONTENT_TYPE =
  "application/vnd.entros.validation+binary;v=1";
export const SYNTHETIC_AUDIO_SAMPLE_RATE_HZ = 16_000;
export const SYNTHETIC_FEATURE_COUNT = 308;
export const SYNTHETIC_CURVE_POINT_COUNT = 64;
export const SYNTHETIC_WALLET_ID = "11111111111111111111111111111111";

const VALIDATION_TRANSPORT_ENVELOPE_MAGIC = Buffer.from("ENTV", "ascii");
const VALIDATION_TRANSPORT_ENVELOPE_HEADER_BYTES = 16;
const VALIDATION_TRANSPORT_ENVELOPE_PCM16_LE_FLAG = 1;
const STANDARD_BASE64 =
  /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/;

export interface ValidationTransportProfile {
  name: string;
  projectionVersion: 1 | 2;
  durationMs: 12_000 | 20_000;
}

export const VALIDATION_TRANSPORT_PROFILES = Object.freeze([
  Object.freeze({
    name: "projection-1-12s",
    projectionVersion: 1,
    durationMs: 12_000,
  }),
  Object.freeze({
    name: "projection-2-12s",
    projectionVersion: 2,
    durationMs: 12_000,
  }),
  Object.freeze({
    name: "projection-1-20s",
    projectionVersion: 1,
    durationMs: 20_000,
  }),
  Object.freeze({
    name: "projection-2-20s",
    projectionVersion: 2,
    durationMs: 20_000,
  }),
] as const satisfies readonly ValidationTransportProfile[]);

export interface SyntheticValidationTransportRequest extends ValidationDigestRequest {
  features: number[];
  projection_version: 1 | 2;
  f0_contour: number[];
  accel_magnitude: number[];
  wallet_id: string;
  audio_samples_b64: string;
  audio_sample_rate_hz: number;
  commitment_new_hex: string;
  request_receipt: true;
  receipt_purpose: "mint";
  baseline_reset: false;
  client_signals: ClientSignals;
  curve_trace: CurveTraceOutline;
  wallet_authorization?: WalletAuthorization;
}

export interface SyntheticValidationTransportMetrics {
  profile: string;
  projectionVersion: 1 | 2;
  durationMs: number;
  jsonBytes: number;
  binaryBytes: number;
  binarySha256: string;
  base64Bytes: number;
  decodedPcmBytes: number;
  pcmSha256: string;
  authorizationDigest: string;
  fixtureSha256: string;
}

export function syntheticPcmSample(index: number): number {
  return ((index * 1_103 + 7_919) & 0xffff) - 32_768;
}

export function buildSyntheticValidationTransportRequest(
  profile: ValidationTransportProfile,
): SyntheticValidationTransportRequest {
  const contourLength = profile.durationMs / 10;
  const pcmBytes = buildPcmBytes(profile.durationMs);
  const request: SyntheticValidationTransportRequest = {
    features: Array.from(
      { length: SYNTHETIC_FEATURE_COUNT },
      (_, index) => (index - 154) / 16,
    ),
    projection_version: profile.projectionVersion,
    wallet_id: SYNTHETIC_WALLET_ID,
    f0_contour: Array.from(
      { length: contourLength },
      (_, index) => 80 + (index % 37) / 2,
    ),
    accel_magnitude: Array.from(
      { length: contourLength },
      (_, index) => ((index % 29) - 14) / 64,
    ),
    audio_samples_b64: Buffer.from(pcmBytes).toString("base64"),
    audio_sample_rate_hz: SYNTHETIC_AUDIO_SAMPLE_RATE_HZ,
    commitment_new_hex: "11".repeat(32),
    request_receipt: true,
    receipt_purpose: "mint",
    baseline_reset: false,
    client_signals: buildClientSignals(),
    curve_trace: buildCurveTrace(profile.durationMs),
  };

  if (profile.projectionVersion === 2) {
    request.compatibility_evidence = {
      projection_version: 1,
      feature_schema_version: 4,
      features: Array.from(
        { length: SYNTHETIC_FEATURE_COUNT },
        (_, index) => (index - 154) / 32,
      ),
    };
    request.wallet_authorization = buildDummyWalletAuthorization();
  }

  return request;
}

export function inspectSyntheticValidationTransportRequest(
  profile: ValidationTransportProfile,
  request: SyntheticValidationTransportRequest,
): SyntheticValidationTransportMetrics {
  const json = JSON.stringify(request);
  const jsonBytes = new TextEncoder().encode(json);
  const binary = encodeSyntheticValidationTransportEnvelope(request);
  const decodedPcm = decodeCanonicalBase64(request.audio_samples_b64);
  return {
    profile: profile.name,
    projectionVersion: profile.projectionVersion,
    durationMs: profile.durationMs,
    jsonBytes: jsonBytes.byteLength,
    binaryBytes: binary.byteLength,
    binarySha256: bytesToHex(sha256(binary)),
    base64Bytes: Buffer.byteLength(request.audio_samples_b64, "utf8"),
    decodedPcmBytes: decodedPcm.byteLength,
    pcmSha256: bytesToHex(sha256(decodedPcm)),
    authorizationDigest: buildSyntheticValidationAuthorizationDigest(request),
    fixtureSha256: bytesToHex(sha256(jsonBytes)),
  };
}

export function buildSyntheticValidationAuthorizationDigest(
  request: SyntheticValidationTransportRequest,
): string {
  return bytesToHex(buildValidationRequestDigest(request));
}

export function encodeSyntheticValidationTransportEnvelope(
  request: SyntheticValidationTransportRequest,
): Buffer {
  const { audio_samples_b64: audioBase64, ...metadata } = request;
  if (typeof audioBase64 !== "string") {
    throw new Error("Binary envelope requires PCM");
  }

  const pcmBytes = decodeCanonicalBase64(audioBase64);
  if (
    pcmBytes.byteLength === 0 ||
    pcmBytes.byteLength % Int16Array.BYTES_PER_ELEMENT !== 0
  ) {
    throw new Error("Binary envelope requires non-empty PCM16");
  }
  const metadataBytes = Buffer.from(JSON.stringify(metadata), "utf8");
  if (
    VALIDATION_TRANSPORT_ENVELOPE_HEADER_BYTES +
      metadataBytes.byteLength +
      pcmBytes.byteLength >
    EXECUTOR_REQUEST_BODY_LIMIT_BYTES
  ) {
    throw new Error("Binary envelope exceeds executor body limit");
  }
  const header = Buffer.alloc(VALIDATION_TRANSPORT_ENVELOPE_HEADER_BYTES);
  VALIDATION_TRANSPORT_ENVELOPE_MAGIC.copy(header, 0);
  header.writeUInt16LE(VALIDATION_TRANSPORT_FIXTURE_VERSION, 4);
  header.writeUInt16LE(VALIDATION_TRANSPORT_ENVELOPE_PCM16_LE_FLAG, 6);
  header.writeUInt32LE(metadataBytes.byteLength, 8);
  header.writeUInt32LE(pcmBytes.byteLength, 12);
  return Buffer.concat([header, metadataBytes, pcmBytes]);
}

export function decodeSyntheticValidationTransportEnvelope(
  envelope: Uint8Array,
): SyntheticValidationTransportRequest {
  const bytes = Buffer.from(envelope);
  if (bytes.byteLength > EXECUTOR_REQUEST_BODY_LIMIT_BYTES) {
    throw new Error("Binary envelope exceeds executor body limit");
  }
  if (bytes.byteLength < VALIDATION_TRANSPORT_ENVELOPE_HEADER_BYTES) {
    throw new Error("Truncated envelope header");
  }
  if (!bytes.subarray(0, 4).equals(VALIDATION_TRANSPORT_ENVELOPE_MAGIC)) {
    throw new Error("Invalid envelope magic");
  }

  const version = bytes.readUInt16LE(4);
  if (version !== VALIDATION_TRANSPORT_FIXTURE_VERSION) {
    throw new Error(`Unsupported envelope version ${version}`);
  }
  const flags = bytes.readUInt16LE(6);
  if (flags !== VALIDATION_TRANSPORT_ENVELOPE_PCM16_LE_FLAG) {
    throw new Error("Invalid envelope flags");
  }

  const metadataLength = bytes.readUInt32LE(8);
  const pcmLength = bytes.readUInt32LE(12);
  if (pcmLength === 0 || pcmLength % Int16Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Invalid envelope PCM length");
  }
  const expectedLength =
    VALIDATION_TRANSPORT_ENVELOPE_HEADER_BYTES + metadataLength + pcmLength;
  if (bytes.byteLength !== expectedLength) {
    throw new Error("Truncated or trailing envelope data");
  }

  const metadataEnd =
    VALIDATION_TRANSPORT_ENVELOPE_HEADER_BYTES + metadataLength;
  let metadata: unknown;
  try {
    const metadataJson = new TextDecoder("utf-8", { fatal: true }).decode(
      bytes.subarray(VALIDATION_TRANSPORT_ENVELOPE_HEADER_BYTES, metadataEnd),
    );
    metadata = JSON.parse(metadataJson);
  } catch {
    throw new Error("Invalid envelope metadata");
  }
  if (
    metadata === null ||
    typeof metadata !== "object" ||
    Array.isArray(metadata)
  ) {
    throw new Error("Invalid envelope metadata");
  }
  const record = metadata as Record<string, unknown>;
  if (
    !Array.isArray(record.features) ||
    record.features.some(
      (value) => typeof value !== "number" || !Number.isFinite(value),
    ) ||
    typeof record.wallet_id !== "string"
  ) {
    throw new Error("Invalid envelope metadata");
  }
  if (Object.prototype.hasOwnProperty.call(metadata, "audio_samples_b64")) {
    throw new Error("Metadata duplicates envelope PCM");
  }

  return {
    ...metadata,
    audio_samples_b64: bytes.subarray(metadataEnd).toString("base64"),
  } as SyntheticValidationTransportRequest;
}

function decodeCanonicalBase64(value: string): Buffer {
  if (!STANDARD_BASE64.test(value)) {
    throw new Error("Binary envelope requires canonical base64 PCM");
  }
  const decoded = Buffer.from(value, "base64");
  if (decoded.toString("base64") !== value) {
    throw new Error("Binary envelope requires canonical base64 PCM");
  }
  return decoded;
}

function buildPcmBytes(durationMs: number): Uint8Array {
  const sampleCount = (durationMs * SYNTHETIC_AUDIO_SAMPLE_RATE_HZ) / 1_000;
  if (!Number.isInteger(sampleCount)) {
    throw new Error("Synthetic PCM duration must produce a whole sample count");
  }

  const bytes = new Uint8Array(sampleCount * 2);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < sampleCount; index += 1) {
    view.setInt16(index * 2, syntheticPcmSample(index), true);
  }
  return bytes;
}

function buildClientSignals(): ClientSignals {
  return {
    v: 1,
    env: "non-browser",
    automation: { webdriver: false, tells: [] },
    capture: {
      virtual_device: false,
      voice_isolation_applied: null,
      flatness: 0.125,
      centroid: 2_400,
    },
  };
}

function buildCurveTrace(durationMs: number): CurveTraceOutline {
  return {
    points: Array.from(
      { length: SYNTHETIC_CURVE_POINT_COUNT },
      (_, index) =>
        [(index * 100) / 63, (((index * 17) % 64) * 100) / 63] as [
          number,
          number,
        ],
    ),
    duration_ms: durationMs,
  };
}

function buildDummyWalletAuthorization(): WalletAuthorization {
  return {
    nonce: new Array<number>(32).fill(0x5a),
    signature_hex: "ab".repeat(64),
  };
}
