export type StudyCaptureClass =
  | "web-mobile"
  | "web-desktop"
  | "native-ios"
  | "native-android";

export type StudyRecordStatus =
  | "recorded"
  | "replayed"
  | "technical_failure"
  | "invalid_token"
  | "disabled";

export interface StudyContext {
  token: string;
  record_id: string;
  capture_class: StudyCaptureClass;
  feature_schema_version: number;
  projection_version: number;
}

export interface StudyGrant {
  token: string;
  feature_schema_version: number;
  projection_version: number;
}

export function featureSchemaVersionForProjection(
  projectionVersion: number,
): number {
  if (projectionVersion === 0) return 3;
  if (projectionVersion === 1) return 4;
  throw new Error(`Unsupported projection version ${projectionVersion}`);
}

export function createStudyContext(
  grant: StudyGrant,
  captureClass: StudyCaptureClass,
): StudyContext {
  if (!/^[A-Za-z0-9_-]{43}$/.test(grant.token)) {
    throw new Error("Study token is malformed");
  }
  for (const [name, value] of [
    ["feature schema version", grant.feature_schema_version],
    ["projection version", grant.projection_version],
  ] as const) {
    if (!Number.isInteger(value) || value < 0 || value > 65_535) {
      throw new Error(`Study ${name} is malformed`);
    }
  }
  const expectedFeatureSchemaVersion = featureSchemaVersionForProjection(
    grant.projection_version,
  );
  if (grant.feature_schema_version !== expectedFeatureSchemaVersion) {
    throw new Error(
      `Study feature schema ${grant.feature_schema_version} does not match projection ${grant.projection_version}`,
    );
  }
  if (!globalThis.crypto?.getRandomValues) {
    throw new Error("Secure randomness is unavailable for this study capture");
  }
  const recordId = new Uint8Array(16);
  globalThis.crypto.getRandomValues(recordId);
  return {
    token: grant.token,
    record_id: Array.from(recordId, (byte) => byte.toString(16).padStart(2, "0")).join(""),
    capture_class: captureClass,
    feature_schema_version: grant.feature_schema_version,
    projection_version: grant.projection_version,
  };
}
