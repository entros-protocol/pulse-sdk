import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";

import {
  buildSyntheticValidationTransportRequest,
  inspectSyntheticValidationTransportRequest,
  VALIDATION_TRANSPORT_PROFILES,
} from "./support/validation-transport-fixtures";

const SERIALIZATION_RUNS = 200;
const WARMUP_RUNS = 20;

interface PackageJson {
  version: string;
}

function percentile(samples: number[], quantile: number): number {
  const ordered = [...samples].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(quantile * ordered.length) - 1);
  return ordered[index]!;
}

function measureSerialization(request: object): {
  p50Ms: number;
  p95Ms: number;
} {
  for (let index = 0; index < WARMUP_RUNS; index += 1) {
    JSON.stringify(request);
  }

  const samples: number[] = [];
  for (let index = 0; index < SERIALIZATION_RUNS; index += 1) {
    const startedAt = performance.now();
    JSON.stringify(request);
    samples.push(performance.now() - startedAt);
  }

  return {
    p50Ms: Number(percentile(samples, 0.5).toFixed(6)),
    p95Ms: Number(percentile(samples, 0.95).toFixed(6)),
  };
}

const packageJson = JSON.parse(
  readFileSync(new URL("../package.json", import.meta.url), "utf8"),
) as PackageJson;

const profiles = VALIDATION_TRANSPORT_PROFILES.map((profile) => {
  const request = buildSyntheticValidationTransportRequest(profile);
  const metrics = inspectSyntheticValidationTransportRequest(profile, request);
  const serialization = measureSerialization(request);
  return {
    profile: metrics.profile,
    projection_version: metrics.projectionVersion,
    duration_ms: metrics.durationMs,
    json_bytes: metrics.jsonBytes,
    binary_bytes: metrics.binaryBytes,
    binary_savings_bytes: metrics.jsonBytes - metrics.binaryBytes,
    binary_savings_percent:
      ((metrics.jsonBytes - metrics.binaryBytes) / metrics.jsonBytes) * 100,
    binary_sha256: metrics.binarySha256,
    base64_bytes: metrics.base64Bytes,
    decoded_pcm_bytes: metrics.decodedPcmBytes,
    pcm_sha256: metrics.pcmSha256,
    authorization_digest: metrics.authorizationDigest,
    fixture_sha256: metrics.fixtureSha256,
    serialization_p50_ms: serialization.p50Ms,
    serialization_p95_ms: serialization.p95Ms,
  };
});

console.log(
  JSON.stringify(
    {
      package_version: packageJson.version,
      node_version: process.version,
      serialization_runs: SERIALIZATION_RUNS,
      profiles,
    },
    null,
    2,
  ),
);
