import { sha256 } from "@noble/hashes/sha2.js";

import type { ClientSignals } from "../client-signals/automation";
import { bytesToHex } from "../submit/receipt";
import type { StudyContext } from "../study";

const DIGEST_DOMAIN = new TextEncoder().encode(
  "entros-validate-request-digest-v1\0",
);
const MESSAGE_DOMAIN = "Entros-Validate-v1";

export interface ProjectionCompatibilityEvidence {
  projection_version: number;
  feature_schema_version: number;
  features: number[];
}

export interface ValidationDigestRequest {
  baseline_reset: boolean;
  features: number[];
  compatibility_evidence?: ProjectionCompatibilityEvidence;
  f0_contour?: number[];
  accel_magnitude?: number[];
  audio_samples_b64?: string;
  audio_sample_rate_hz?: number;
  client_signals?: ClientSignals;
  study?: StudyContext;
}

export interface WalletAuthorization {
  nonce: number[];
  signature_hex: string;
}

class DigestEncoder {
  private readonly digest = sha256.create().update(DIGEST_DOMAIN);
  private readonly numberBuffer = new ArrayBuffer(8);
  private readonly numberView = new DataView(this.numberBuffer);

  boolean(value: boolean): void {
    this.digest.update(Uint8Array.of(value ? 1 : 0));
  }

  u8(value: number): void {
    this.assertUnsignedInteger(value, 0xff, "u8");
    this.digest.update(Uint8Array.of(value));
  }

  u16(value: number): void {
    this.assertUnsignedInteger(value, 0xffff, "u16");
    this.numberView.setUint16(0, value, true);
    this.appendNumberBytes(2);
  }

  u32(value: number): void {
    this.assertUnsignedInteger(value, 0xffffffff, "u32");
    this.numberView.setUint32(0, value, true);
    this.appendNumberBytes(4);
  }

  length(value: number): void {
    this.u32(value);
  }

  byteVector(value: Uint8Array): void {
    this.length(value.length);
    this.digest.update(value);
  }

  string(value: string): void {
    this.byteVector(new TextEncoder().encode(value));
  }

  optionString(value: string | undefined): void {
    this.boolean(value !== undefined);
    if (value !== undefined) this.string(value);
  }

  finiteF64(value: number): void {
    if (!Number.isFinite(value)) {
      throw new Error("Signed validation request contains a non-finite number");
    }
    this.numberView.setFloat64(0, value === 0 ? 0 : value, true);
    this.appendNumberBytes(8);
  }

  optionF64(value: number | undefined): void {
    this.boolean(value !== undefined);
    if (value !== undefined) this.finiteF64(value);
  }

  f64Vector(values: number[]): void {
    this.length(values.length);
    for (const value of values) this.finiteF64(value);
  }

  optionF64Vector(values: number[] | undefined): void {
    this.boolean(values !== undefined);
    if (values !== undefined) this.f64Vector(values);
  }

  stringVector(values: string[]): void {
    this.length(values.length);
    for (const value of values) this.string(value);
  }

  clientSignals(signals: ClientSignals | undefined): void {
    this.boolean(signals !== undefined);
    if (!signals) return;

    this.u32(signals.v);
    this.optionString(signals.env);
    this.boolean(signals.automation !== undefined);
    if (signals.automation) {
      this.boolean(signals.automation.webdriver);
      this.stringVector(signals.automation.tells);
    }
    this.boolean(signals.capture !== undefined);
    if (signals.capture) {
      this.boolean(signals.capture.virtual_device);
      this.optionF64(signals.capture.flatness);
      this.optionF64(signals.capture.centroid);
    }
  }

  studyContext(study: StudyContext | undefined): void {
    this.boolean(study !== undefined);
    if (!study) return;

    this.string(study.token);
    this.string(study.record_id);
    this.u8(studyCaptureClassCode(study.capture_class));
    this.u16(study.feature_schema_version);
    this.u16(study.projection_version);
  }

  finish(): Uint8Array {
    return this.digest.digest();
  }

  private appendNumberBytes(length: number): void {
    const encoded = new Uint8Array(this.numberBuffer, 0, length);
    this.digest.update(encoded);
  }

  private assertUnsignedInteger(
    value: number,
    maximum: number,
    label: string,
  ): void {
    if (!Number.isInteger(value) || value < 0 || value > maximum) {
      throw new Error(`Signed validation request ${label} is out of range`);
    }
  }
}

export function buildValidationRequestDigest(
  request: ValidationDigestRequest,
): Uint8Array {
  const encoder = new DigestEncoder();
  encoder.boolean(request.baseline_reset);
  encoder.f64Vector(request.features);

  encoder.boolean(request.compatibility_evidence !== undefined);
  if (request.compatibility_evidence) {
    encoder.u16(request.compatibility_evidence.projection_version);
    encoder.u16(request.compatibility_evidence.feature_schema_version);
    encoder.f64Vector(request.compatibility_evidence.features);
  }

  encoder.optionF64Vector(request.f0_contour);
  encoder.optionF64Vector(request.accel_magnitude);
  encoder.optionString(request.audio_samples_b64);
  encoder.boolean(request.audio_sample_rate_hz !== undefined);
  if (request.audio_sample_rate_hz !== undefined) {
    encoder.u32(request.audio_sample_rate_hz);
  }
  encoder.clientSignals(request.client_signals);
  encoder.studyContext(request.study);
  return encoder.finish();
}

export function buildValidationAuthorizationMessage(
  walletAddress: string,
  nonce: Uint8Array,
  projectionVersion: number,
  digest: Uint8Array,
): string {
  if (nonce.length !== 32) {
    throw new Error("Validation challenge nonce must be 32 bytes");
  }
  if (digest.length !== 32) {
    throw new Error("Validation request digest must be 32 bytes");
  }
  if (
    !Number.isInteger(projectionVersion) ||
    projectionVersion < 0 ||
    projectionVersion > 0xffff
  ) {
    throw new Error("Validation projection version is out of range");
  }
  return [
    MESSAGE_DOMAIN,
    `wallet:${walletAddress}`,
    `nonce:${bytesToHex(nonce)}`,
    `projection:${projectionVersion}`,
    `request_sha256:${bytesToHex(digest)}`,
  ].join("\n");
}

function studyCaptureClassCode(
  captureClass: StudyContext["capture_class"],
): number {
  switch (captureClass) {
    case "web-mobile":
      return 0;
    case "web-desktop":
      return 1;
    case "native-ios":
      return 2;
    case "native-android":
      return 3;
  }
}
