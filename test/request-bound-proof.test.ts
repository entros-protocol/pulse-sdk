import { describe, expect, it, vi, afterEach } from "vitest";
import { sha256 } from "@noble/hashes/sha256";
import fixture from "./fixtures/request-bound-schema.json";
import {
  assertBoundPublicInputs,
  bytesHex,
  canonicalScalar,
  encodeAnchorUpdateAction,
  encodeProofRequest,
  hexBytes,
  prepareProofRequest,
  SCALAR_MODULUS,
} from "../src/proof/request";
import type {
  AnchorUpdateAction,
  ProofRequestContext,
  RequestBoundManifest,
} from "../src/proof/request";
import { serializeProof, toBigEndian32 } from "../src/proof/serializer";
import { loadRequestBoundArtifacts } from "../src/proof/prover";

export function requestContext(): ProofRequestContext {
  return {
    deploymentDomain: fixture.fields.deployment,
    verifier: fixture.fields.verifier,
    consumer: fixture.fields.consumer,
    wallet: fixture.fields.wallet,
    nonce: fixture.fields.nonce,
    actionKind: 1,
    action: {
      identity: fixture.fields.identity,
      mint: fixture.fields.mint,
      counter: 7n,
      projectionVersion: 1,
      commitmentNew: fixture.commitmentNew,
      commitmentPrevious: fixture.commitmentPrevious,
      threshold: 30,
      minDistance: 3,
      validUntil: 1800000000n,
    },
  };
}
describe("canonical request binding", () => {
  it("matches the independently encoded fixed-width vector", () => {
    const context = requestContext();
    expect(bytesHex(encodeAnchorUpdateAction(context.action))).toBe(
      fixture.actionHex,
    );
    expect(bytesHex(encodeProofRequest(context))).toBe(fixture.requestHex);
    const request = prepareProofRequest(context);
    expect(request.digest).toBe(fixture.requestDigest);
    expect(request.digestHi).toBe(fixture.digestHi);
    expect(request.digestLo).toBe(fixture.digestLo);
    expect(Object.isFrozen(request)).toBe(true);
    expect(Object.isFrozen(request.action)).toBe(true);
  });
  it.each([
    "wallet",
    "nonce",
    "deploymentDomain",
    "consumer",
    "verifier",
  ] as const)("changes the statement when %s changes", (field) => {
    expect(
      prepareProofRequest({ ...requestContext(), [field]: "22".repeat(32) })
        .digest,
    ).not.toBe(fixture.requestDigest);
  });
  it.each([
    ["counter", 8n],
    ["projectionVersion", 2],
    ["validUntil", 1800000001n],
    ["identity", "22".repeat(32)],
    ["mint", "22".repeat(32)],
    ["threshold", 31],
    ["minDistance", 4],
    ["commitmentNew", "00".repeat(31) + "03"],
    ["commitmentPrevious", "00".repeat(31) + "03"],
  ] as const)(
    "changes the statement when action %s changes",
    (field, value) => {
      const context = requestContext();
      expect(
        prepareProofRequest({
          ...context,
          action: { ...context.action, [field]: value },
        }).digest,
      ).not.toBe(fixture.requestDigest);
    },
  );
  it.each(["-1", "01", "1.1", "0x01", (1n << 256n).toString()])(
    "rejects noncanonical integer %s",
    (value) => expect(() => toBigEndian32(value)).toThrow(),
  );
  it("rejects scalar aliases and malformed bytes", () => {
    expect(() =>
      canonicalScalar(SCALAR_MODULUS.toString(16).padStart(64, "0")),
    ).toThrow();
    expect(() => hexBytes("AA".repeat(32))).toThrow();
    expect(() => hexBytes("00")).toThrow();
  });
  it.each([
    [-1n, 1n],
    [1n << 64n, 1n],
    [1n, 0n],
    [1n, 1n << 64n],
  ])("rejects counter or expiry overflow", (counter, validUntil) =>
    expect(() =>
      encodeAnchorUpdateAction({
        ...requestContext().action,
        counter,
        validUntil,
      }),
    ).toThrow(),
  );
  it("rejects wrong runtime types and unsupported actions", () => {
    expect(() =>
      prepareProofRequest({
        ...requestContext(),
        actionKind: 2,
      } as unknown as ProofRequestContext),
    ).toThrow();
    expect(() =>
      encodeAnchorUpdateAction({
        ...requestContext().action,
        counter: "7",
      } as unknown as AnchorUpdateAction),
    ).toThrow();
    expect(() =>
      prepareProofRequest({ ...requestContext(), nonce: "00".repeat(32) }),
    ).toThrow();
  });
  it("rejects changed or missing public inputs", () => {
    const request = prepareProofRequest(requestContext());
    const inputs = [
      "1",
      "2",
      "30",
      "3",
      fixture.digestHi,
      fixture.digestLo,
    ].map(toBigEndian32);
    expect(() => assertBoundPublicInputs(request, inputs)).not.toThrow();
    expect(() =>
      assertBoundPublicInputs(request, inputs.slice(0, 4)),
    ).toThrow();
    expect(() =>
      assertBoundPublicInputs(request, [
        ...inputs.slice(0, 5),
        toBigEndian32("1"),
      ]),
    ).toThrow();
  });
  it("serializes six inputs only under the explicit generation", () => {
    const proof = {
      pi_a: ["1", "2"],
      pi_b: [
        ["1", "2"],
        ["3", "4"],
      ],
      pi_c: ["1", "2"],
      protocol: "groth16",
      curve: "bn128",
    };
    const signals = ["1", "2", "30", "3", fixture.digestHi, fixture.digestLo];
    expect(() => serializeProof(proof, signals)).toThrow();
    expect(
      serializeProof(proof, signals, "request-bound-v1").publicInputs,
    ).toHaveLength(6);
    expect(() =>
      serializeProof(
        proof,
        [...signals.slice(0, 5), (1n << 128n).toString()],
        "request-bound-v1",
      ),
    ).toThrow();
    expect(() =>
      serializeProof(proof, [SCALAR_MODULUS.toString(), "2", "30", "3"]),
    ).toThrow();
  });
});

describe("artifact integrity", () => {
  afterEach(() => vi.unstubAllGlobals());
  function manifest(tag: string, bytes: Uint8Array): RequestBoundManifest {
    const hash = bytesHex(sha256(bytes));
    return {
      generation: "request-bound-v1",
      deploymentDomain: "11".repeat(32),
      genesisHash: "synthetic",
      verifierProgram: "synthetic",
      consumerProgram: "synthetic",
      wasm: { url: `https://example.invalid/${tag}.wasm`, sha256: hash },
      zkey: { url: `https://example.invalid/${tag}.zkey`, sha256: hash },
    };
  }
  it("checks both hashes and protects cached bytes from caller mutation", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const fetch = vi.fn(async () => new Response(bytes));
    vi.stubGlobal("fetch", fetch);
    const config = manifest("valid", bytes);
    const first = await loadRequestBoundArtifacts(config);
    first.wasm[0] = 99;
    expect((await loadRequestBoundArtifacts(config)).wasm[0]).toBe(1);
    expect(fetch).toHaveBeenCalledTimes(2);
  });
  it("does not fall back to unchecked URLs after a hash failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(new Uint8Array([4]))),
    );
    await expect(
      loadRequestBoundArtifacts(manifest("invalid", new Uint8Array([1]))),
    ).rejects.toThrow("hash mismatch");
  });
  it("invalidates cache identity when the hash changes at the same URL", async () => {
    let bytes = new Uint8Array([1]);
    const fetch = vi.fn(async () => new Response(bytes));
    vi.stubGlobal("fetch", fetch);
    await loadRequestBoundArtifacts(manifest("replaced", bytes));
    bytes = new Uint8Array([2]);
    expect(
      (await loadRequestBoundArtifacts(manifest("replaced", bytes))).wasm[0],
    ).toBe(2);
    expect(fetch).toHaveBeenCalledTimes(4);
  });
  it("pins artifact URLs and hashes before yielding to callers", async () => {
    const original = new Uint8Array([1, 2, 3]);
    const replacement = new Uint8Array([7, 8, 9]);
    const config = manifest("synchronous-snapshot", original);
    const mutable = {
      ...config,
      wasm: { ...config.wasm },
      zkey: { ...config.zkey },
    };
    const fetch = vi.fn(
      async (_url: string | URL | Request) => new Response(original),
    );
    vi.stubGlobal("fetch", fetch);
    const pending = loadRequestBoundArtifacts(mutable);
    mutable.wasm.url = "https://example.invalid/substituted.wasm";
    mutable.zkey.url = "https://example.invalid/substituted.zkey";
    mutable.wasm.sha256 = bytesHex(sha256(replacement));
    mutable.zkey.sha256 = bytesHex(sha256(replacement));
    expect((await pending).wasm).toEqual(original);
    expect(fetch.mock.calls.map(([url]) => url)).toEqual([
      config.wasm.url,
      config.zkey.url,
    ]);
  });
  it("rejects concurrent hash substitution and retries without poisoning the cache", async () => {
    const original = new Uint8Array([1, 2, 3]);
    const replacement = new Uint8Array([7, 8, 9]);
    const config = manifest("concurrent-snapshot", original);
    const mutable = {
      ...config,
      wasm: { ...config.wasm },
      zkey: { ...config.zkey },
    };
    let finish: () => void = () => {};
    let started: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      finish = resolve;
    });
    const downloads = new Promise<void>((resolve) => {
      started = resolve;
    });
    let count = 0;
    let served = replacement;
    const fetch = vi.fn(async () => {
      if (++count === 2) started();
      await gate;
      return new Response(served);
    });
    vi.stubGlobal("fetch", fetch);
    const pending = Promise.allSettled(
      Array.from({ length: 64 }, () => loadRequestBoundArtifacts(mutable)),
    );
    await downloads;
    mutable.wasm.sha256 = bytesHex(sha256(replacement));
    mutable.zkey.sha256 = bytesHex(sha256(replacement));
    finish();
    const rejected = await pending;
    expect(rejected.every((result) => result.status === "rejected")).toBe(true);
    expect(fetch).toHaveBeenCalledTimes(2);
    served = original;
    const retried = await Promise.all(
      Array.from({ length: 64 }, () => loadRequestBoundArtifacts(config)),
    );
    expect(
      retried.every(
        (artifacts) => bytesHex(artifacts.wasm) === bytesHex(original),
      ),
    ).toBe(true);
    expect(fetch).toHaveBeenCalledTimes(4);
    retried[0]!.wasm[0] = 99;
    expect(retried[1]!.wasm).toEqual(original);
    expect((await loadRequestBoundArtifacts(config)).wasm).toEqual(original);
  });
  it("retains the two most recently used artifact pairs", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const fetch = vi.fn(async () => new Response(bytes));
    vi.stubGlobal("fetch", fetch);
    const first = manifest("eviction-first", bytes);
    const second = manifest("eviction-second", bytes);
    const third = manifest("eviction-third", bytes);
    await loadRequestBoundArtifacts(first);
    await loadRequestBoundArtifacts(second);
    await loadRequestBoundArtifacts(first);
    await loadRequestBoundArtifacts(third);
    await loadRequestBoundArtifacts(first);
    expect(fetch).toHaveBeenCalledTimes(6);
    await loadRequestBoundArtifacts(second);
    expect(fetch).toHaveBeenCalledTimes(8);
  });
  it("keeps a replacement cache entry when its evicted request later fails", async () => {
    const bytes = new Uint8Array([1, 2, 3]);
    const first = manifest("pending-first", bytes);
    let finish: () => void = () => {};
    const gate = new Promise<void>((resolve) => {
      finish = resolve;
    });
    let firstRequests = 0;
    const fetch = vi.fn(async (url: string | URL | Request) => {
      if (
        (url === first.wasm.url || url === first.zkey.url) &&
        ++firstRequests <= 2
      ) {
        await gate;
        return new Response(null, { status: 503 });
      }
      return new Response(bytes);
    });
    vi.stubGlobal("fetch", fetch);
    const old = Promise.allSettled([loadRequestBoundArtifacts(first)]);
    await loadRequestBoundArtifacts(manifest("pending-second", bytes));
    await loadRequestBoundArtifacts(manifest("pending-third", bytes));
    await loadRequestBoundArtifacts(first);
    expect(fetch).toHaveBeenCalledTimes(8);
    finish();
    expect((await old)[0]?.status).toBe("rejected");
    expect((await loadRequestBoundArtifacts(first)).wasm).toEqual(bytes);
    expect(fetch).toHaveBeenCalledTimes(8);
  });
});
