import { sha256 } from "@noble/hashes/sha256";

export const SCALAR_MODULUS =
  21888242871839275222246405745257275088548364400416034343698204186575808495617n;
export interface RequestBoundDeployment {
  readonly generation: "request-bound-v1";
  readonly deploymentDomain: string;
  readonly genesisHash: string;
  readonly verifierProgram: string;
  readonly consumerProgram: string;
}
export interface RequestBoundManifest extends RequestBoundDeployment {
  readonly wasm: Readonly<{ url: string; sha256: string }>;
  readonly zkey: Readonly<{ url: string; sha256: string }>;
}
export interface AnchorUpdateAction {
  readonly identity: string;
  readonly mint: string;
  readonly counter: bigint;
  readonly projectionVersion: number;
  readonly commitmentNew: string;
  readonly commitmentPrevious: string;
  readonly threshold: number;
  readonly minDistance: number;
  readonly validUntil: bigint;
}
export interface ProofRequestContext {
  readonly deploymentDomain: string;
  readonly verifier: string;
  readonly consumer: string;
  readonly wallet: string;
  readonly nonce: string;
  readonly actionKind: 1;
  readonly action: AnchorUpdateAction;
}
export interface PreparedProofRequest extends ProofRequestContext {
  readonly generation: "request-bound-v1";
  readonly nonceSource: "client" | "executor";
  readonly digest: string;
  readonly digestHi: string;
  readonly digestLo: string;
}
export function bytesHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}
export function hexBytes(value: string, length = 32): Uint8Array {
  if (
    typeof value !== "string" ||
    !new RegExp(`^[0-9a-f]{${length * 2}}$`).test(value)
  )
    throw new Error("Expected canonical fixed-length lowercase hex");
  return Uint8Array.from(value.match(/../g)!, (byte) =>
    Number.parseInt(byte, 16),
  );
}
export function canonicalScalar(value: string): Uint8Array {
  const bytes = hexBytes(value);
  if (BigInt(`0x${value}`) >= SCALAR_MODULUS)
    throw new Error("Scalar exceeds the field modulus");
  return bytes;
}
function uint(value: bigint | number, length: number): Uint8Array {
  if (
    (typeof value !== "bigint" && typeof value !== "number") ||
    (typeof value === "number" && !Number.isSafeInteger(value))
  )
    throw new Error("Expected an unsigned integer");
  let integer = BigInt(value);
  if (integer < 0n || integer >= 1n << BigInt(length * 8))
    throw new Error("Unsigned integer exceeds its encoding");
  const result = new Uint8Array(length);
  for (let i = length - 1; i >= 0; i--) {
    result[i] = Number(integer & 255n);
    integer >>= 8n;
  }
  return result;
}
function domain(label: string): Uint8Array {
  const result = new Uint8Array(32);
  result.set(new TextEncoder().encode(label));
  return result;
}
function concat(parts: Uint8Array[]): Uint8Array {
  const bytes = new Uint8Array(
    parts.reduce((length, part) => length + part.length, 0),
  );
  let offset = 0;
  for (const part of parts) {
    bytes.set(part, offset);
    offset += part.length;
  }
  return bytes;
}
export function encodeAnchorUpdateAction(
  action: AnchorUpdateAction,
): Uint8Array {
  if (
    typeof action.counter !== "bigint" ||
    typeof action.validUntil !== "bigint"
  )
    throw new Error("Counter and expiry must be bigint values");
  if (
    action.validUntil <= 0n ||
    action.commitmentNew === "0".repeat(64) ||
    action.commitmentPrevious === "0".repeat(64)
  )
    throw new Error("Action requires nonzero commitments and expiry");
  if (
    !Number.isInteger(action.threshold) ||
    action.threshold > 256 ||
    action.minDistance < 1 ||
    action.minDistance >= action.threshold
  )
    throw new Error("Invalid distance bounds");
  return concat([
    domain("ENTROS_ANCHOR_UPDATE_V1"),
    hexBytes(action.identity),
    hexBytes(action.mint),
    uint(action.counter, 8),
    uint(action.projectionVersion, 2),
    canonicalScalar(action.commitmentNew),
    canonicalScalar(action.commitmentPrevious),
    uint(action.threshold, 2),
    uint(action.minDistance, 2),
    uint(action.validUntil, 8),
  ]);
}
export function encodeProofRequest(context: ProofRequestContext): Uint8Array {
  if (context.actionKind !== 1) throw new Error("Unsupported proof action");
  if (context.nonce === "0".repeat(64))
    throw new Error("Proof nonce must not be zero");
  return concat([
    domain("ENTROS_PROOF_REQUEST_V1"),
    hexBytes(context.deploymentDomain),
    hexBytes(context.verifier),
    hexBytes(context.consumer),
    hexBytes(context.wallet),
    hexBytes(context.nonce),
    uint(context.actionKind, 1),
    sha256(encodeAnchorUpdateAction(context.action)),
  ]);
}
export function prepareProofRequest(
  context: ProofRequestContext,
  nonceSource: "client" | "executor" = "client",
): PreparedProofRequest {
  if (nonceSource !== "client" && nonceSource !== "executor")
    throw new Error("Unsupported nonce source");
  const digest = bytesHex(sha256(encodeProofRequest(context)));
  return Object.freeze({
    ...context,
    action: Object.freeze({ ...context.action }),
    generation: "request-bound-v1",
    nonceSource,
    digest,
    digestHi: BigInt(`0x${digest.slice(0, 32)}`).toString(),
    digestLo: BigInt(`0x${digest.slice(32)}`).toString(),
  });
}
export function assertPreparedProofRequest(
  request: PreparedProofRequest,
): void {
  if (request.generation !== "request-bound-v1")
    throw new Error("Unsupported proof generation");
  if (request.nonceSource !== "client" && request.nonceSource !== "executor")
    throw new Error("Unsupported nonce source");
  const expected = prepareProofRequest(request, request.nonceSource);
  if (
    expected.digest !== request.digest ||
    expected.digestHi !== request.digestHi ||
    expected.digestLo !== request.digestLo
  )
    throw new Error("Prepared proof request changed");
}
export function assertBoundPublicInputs(
  request: PreparedProofRequest,
  inputs: readonly Uint8Array[],
): void {
  assertPreparedProofRequest(request);
  const action = request.action;
  const expected = [
    action.commitmentNew,
    action.commitmentPrevious,
    bytesHex(uint(action.threshold, 32)),
    bytesHex(uint(action.minDistance, 32)),
    bytesHex(uint(BigInt(request.digestHi), 32)),
    bytesHex(uint(BigInt(request.digestLo), 32)),
  ];
  if (
    inputs.length !== 6 ||
    inputs.some(
      (input, i) =>
        !(input instanceof Uint8Array) || bytesHex(input) !== expected[i],
    )
  )
    throw new Error("Proof public inputs do not match the prepared request");
}
export function validateRequestBoundManifest(
  manifest: RequestBoundManifest,
): void {
  if (manifest.generation !== "request-bound-v1")
    throw new Error("Unsupported proof generation");
  hexBytes(manifest.deploymentDomain);
  for (const artifact of [manifest.wasm, manifest.zkey]) {
    hexBytes(artifact.sha256);
    if (typeof artifact.url !== "string" || !artifact.url.length)
      throw new Error("Missing artifact URL");
  }
  if (typeof manifest.genesisHash !== "string" || !manifest.genesisHash)
    throw new Error("Missing genesis hash");
}
