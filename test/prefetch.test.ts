import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import * as path from "path";
import * as fs from "fs";
import {
  prefetchCircuitArtifacts,
  takeCircuitArtifacts,
  clearPrefetchedArtifacts,
  prepareCircuitInput,
  generateProof,
} from "../src/proof/prover";
import { generateTBH } from "../src/hashing/poseidon";
import {
  FINGERPRINT_BITS,
  DEFAULT_THRESHOLD,
  DEFAULT_MIN_DISTANCE,
} from "../src/config";

const WASM = "https://example.test/entros_hamming.wasm";
const ZKEY = "https://example.test/entros_hamming_final.zkey";

/**
 * The prefetch exists to move a 2.64 MiB download off the critical path. Its
 * whole safety argument is that it can fail in any way without changing what
 * the user gets, so most of what is worth testing is the failure behaviour
 * rather than the happy path.
 */
describe("circuit artifact prefetch", () => {
  beforeEach(() => {
    clearPrefetchedArtifacts();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    clearPrefetchedArtifacts();
  });

  function stubFetch(impl: (url: string, init?: RequestInit) => Promise<Response>) {
    const spy = vi.fn(impl);
    vi.stubGlobal("fetch", spy);
    return spy;
  }

  function okResponse(bytes: Uint8Array): Response {
    return {
      ok: true,
      status: 200,
      arrayBuffer: async () =>
        bytes.buffer.slice(
          bytes.byteOffset,
          bytes.byteOffset + bytes.byteLength
        ),
    } as unknown as Response;
  }

  it("hands back exactly the bytes it downloaded", async () => {
    const wasmBytes = new Uint8Array([1, 2, 3, 4]);
    const zkeyBytes = new Uint8Array([9, 8, 7]);
    stubFetch(async (url) =>
      okResponse(url === WASM ? wasmBytes : zkeyBytes)
    );

    prefetchCircuitArtifacts(WASM, ZKEY);
    const got = await takeCircuitArtifacts(WASM, ZKEY);

    expect(got).not.toBeNull();
    expect(Array.from(got!.wasm)).toEqual([1, 2, 3, 4]);
    expect(Array.from(got!.zkey)).toEqual([9, 8, 7]);
  });

  it("requests both artifacts at low priority", async () => {
    const spy = stubFetch(async () => okResponse(new Uint8Array([0])));
    prefetchCircuitArtifacts(WASM, ZKEY);
    await takeCircuitArtifacts(WASM, ZKEY);

    expect(spy).toHaveBeenCalledTimes(2);
    for (const call of spy.mock.calls) {
      // Keeps the transfer below the microphone and motion sampler, which are
      // running while it downloads. Engines without Priority Hints ignore it.
      expect((call[1] as { priority?: string }).priority).toBe("low");
    }
  });

  it("resolves null when the download fails, without rejecting", async () => {
    stubFetch(async () => {
      throw new Error("offline");
    });

    // The rejection is swallowed inside the prefetch. If it were not, this
    // would surface as an unhandled rejection rather than a null.
    prefetchCircuitArtifacts(WASM, ZKEY);
    await expect(takeCircuitArtifacts(WASM, ZKEY)).resolves.toBeNull();
  });

  it("resolves null on a non-ok response", async () => {
    stubFetch(async () => ({ ok: false, status: 404 }) as unknown as Response);
    prefetchCircuitArtifacts(WASM, ZKEY);
    await expect(takeCircuitArtifacts(WASM, ZKEY)).resolves.toBeNull();
  });

  it("never serves a prefetch started for different URLs", async () => {
    stubFetch(async () => okResponse(new Uint8Array([1])));
    prefetchCircuitArtifacts(WASM, ZKEY);

    // A host that reconfigures its artifact URLs mid-session must not be handed
    // the bytes fetched for the previous pair.
    await expect(
      takeCircuitArtifacts("https://other.test/a.wasm", ZKEY)
    ).resolves.toBeNull();
  });

  it("releases the buffers so 2.64 MiB is not held for the page lifetime", async () => {
    stubFetch(async () => okResponse(new Uint8Array([1])));
    prefetchCircuitArtifacts(WASM, ZKEY);

    expect(await takeCircuitArtifacts(WASM, ZKEY)).not.toBeNull();
    // Second call finds nothing, so the proof path falls back to URLs. That is
    // the behaviour that ships today, which is why dropping them is safe.
    expect(await takeCircuitArtifacts(WASM, ZKEY)).toBeNull();
  });

  it("does not restart a download already in flight for the same URLs", async () => {
    const spy = stubFetch(async () => okResponse(new Uint8Array([1])));
    prefetchCircuitArtifacts(WASM, ZKEY);
    prefetchCircuitArtifacts(WASM, ZKEY);
    await takeCircuitArtifacts(WASM, ZKEY);

    expect(spy).toHaveBeenCalledTimes(2); // one per artifact, not four
  });

  it("aborts the transfer when the caller aborts", async () => {
    const seen: AbortSignal[] = [];
    stubFetch(async (_url, init) => {
      if (init?.signal) seen.push(init.signal);
      return okResponse(new Uint8Array([1]));
    });
    const controller = new AbortController();
    prefetchCircuitArtifacts(WASM, ZKEY, controller.signal);
    await takeCircuitArtifacts(WASM, ZKEY);

    expect(seen).toHaveLength(2);
    expect(seen.every((s) => s.aborted)).toBe(false);
    controller.abort();
    // Identity is not asserted: the caller's signal is composed with a timeout,
    // so what matters is that the caller still controls cancellation.
    expect(seen.every((s) => s.aborted)).toBe(true);
  });

  it("bounds the transfer even when the caller gives no signal", async () => {
    // `PulseSession` passes none, because every controller in its scope aborts
    // on normal completion. The timeout is what stops a hung connection making
    // proving worse than it is today, since `takeCircuitArtifacts` awaits.
    const seen: AbortSignal[] = [];
    stubFetch(async (_url, init) => {
      if (init?.signal) seen.push(init.signal);
      return okResponse(new Uint8Array([1]));
    });
    prefetchCircuitArtifacts(WASM, ZKEY);
    await takeCircuitArtifacts(WASM, ZKEY);

    expect(seen).toHaveLength(2);
  });

  it("does nothing when fetch is unavailable", async () => {
    vi.stubGlobal("fetch", undefined);
    expect(() => prefetchCircuitArtifacts(WASM, ZKEY)).not.toThrow();
    await expect(takeCircuitArtifacts(WASM, ZKEY)).resolves.toBeNull();
  });
});

/**
 * The property the chain depends on: proving from prefetched bytes must produce
 * the same public signals as proving from a URL. Groth16 proving is randomised,
 * so the proof bytes differ between any two runs and cannot be compared, but
 * the public signals are deterministic and are what `entros-verifier` checks.
 *
 * Artifacts are resolved from the circuits repo when it has been built, and
 * from the site's public directory otherwise, so this runs in an ordinary
 * checkout rather than skipping the way `integration.test.ts` currently does.
 */
const CANDIDATES = [
  [
    path.resolve(__dirname, "../../circuits/build/entros_hamming_js/entros_hamming.wasm"),
    path.resolve(__dirname, "../../circuits/build/entros_hamming_final.zkey"),
  ],
  [
    path.resolve(__dirname, "../../entros.io/public/circuits/entros_hamming.wasm"),
    path.resolve(__dirname, "../../entros.io/public/circuits/entros_hamming_final.zkey"),
  ],
];
const found = CANDIDATES.find(
  ([w, z]) => fs.existsSync(w!) && fs.existsSync(z!)
);

describe.skipIf(!found)("prefetched bytes prove identically to a URL", () => {
  it("produces the same public signals from either artifact form", async () => {
    const [wasmPath, zkeyPath] = found!;

    const prev = Array.from(
      { length: FINGERPRINT_BITS },
      (_, i) => (i * 7919) % 2
    );
    const prevTBH = await generateTBH(prev);

    // Land the Hamming distance inside the circuit's accept band so the witness
    // is satisfiable; out of band throws a circuit assert by design.
    const flips = Math.floor((DEFAULT_THRESHOLD + DEFAULT_MIN_DISTANCE) / 2);
    const next = [...prev];
    for (let i = 0; i < flips; i++) next[i] = next[i] === 1 ? 0 : 1;
    const nextTBH = await generateTBH(next);

    const input = prepareCircuitInput(nextTBH, prevTBH);

    const fromBytes = await generateProof(
      input,
      new Uint8Array(fs.readFileSync(wasmPath!)),
      new Uint8Array(fs.readFileSync(zkeyPath!))
    );
    const fromPath = await generateProof(input, wasmPath!, zkeyPath!);

    expect(fromBytes.publicSignals).toEqual(fromPath.publicSignals);
  }, 120_000);
});
