import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { PublicKey } from "@solana/web3.js";
import { sha256 } from "@noble/hashes/sha256";
import {
  generateRequestBoundProof,
  verifyProofLocally,
} from "../src/proof/prover";
import { bytesHex, hexBytes, prepareProofRequest } from "../src/proof/request";
import { BN254_BASE_FIELD } from "../src/config";
import type { CircuitInput, RawProof } from "../src/proof/types";

const directory = process.env.ENTROS_BOUND_ARTIFACT_DIR;
describe.skipIf(!directory)("isolated request-bound prover integration", () => {
  it("generates a verifiable proof through the hash-pinned client pipeline", async () => {
    const artifact = JSON.parse(
      readFileSync(join(directory!, "proof-fixture.json"), "utf8"),
    ) as {
      input: CircuitInput;
      context: {
        fields: Record<string, string>;
        counter: string;
        projection: number;
        commitmentNew: string;
        commitmentPrevious: string;
        threshold: number;
        minDistance: number;
        validUntil: string;
      };
      public_signals_decimal: string[];
      requestDigest: string;
    };
    const context = artifact.context;
    const request = prepareProofRequest({
      deploymentDomain: context.fields.deployment!,
      verifier: context.fields.verifier!,
      consumer: context.fields.consumer!,
      wallet: context.fields.wallet!,
      nonce: context.fields.nonce!,
      actionKind: 1,
      action: {
        identity: context.fields.identity!,
        mint: context.fields.mint!,
        counter: BigInt(context.counter),
        projectionVersion: context.projection,
        commitmentNew: context.commitmentNew,
        commitmentPrevious: context.commitmentPrevious,
        threshold: context.threshold,
        minDistance: context.minDistance,
        validUntil: BigInt(context.validUntil),
      },
    });
    expect(request.digest).toBe(artifact.requestDigest);
    const wasm = new Uint8Array(
      readFileSync(join(directory!, "entrosrequestboundv1.wasm")),
    );
    const zkey = new Uint8Array(
      readFileSync(join(directory!, "entros_request_bound_v1_final.zkey")),
    );
    const fetch = vi.fn(async (url: string | URL | Request) => {
      if (url === "https://example.invalid/bound.wasm")
        return new Response(wasm);
      if (url === "https://example.invalid/bound.zkey")
        return new Response(zkey);
      throw new Error("Unexpected network request");
    });
    vi.stubGlobal("fetch", fetch);
    try {
      const proof = await generateRequestBoundProof(artifact.input, request, {
        generation: "request-bound-v1",
        deploymentDomain: context.fields.deployment!,
        genesisHash: "synthetic",
        verifierProgram: new PublicKey(
          hexBytes(context.fields.verifier!),
        ).toBase58(),
        consumerProgram: new PublicKey(
          hexBytes(context.fields.consumer!),
        ).toBase58(),
        wasm: {
          url: "https://example.invalid/bound.wasm",
          sha256: bytesHex(sha256(wasm)),
        },
        zkey: {
          url: "https://example.invalid/bound.zkey",
          sha256: bytesHex(sha256(zkey)),
        },
      });
      const signals = proof.publicInputs.map((bytes) =>
        BigInt(`0x${bytesHex(bytes)}`).toString(),
      );
      expect(signals).toEqual(artifact.public_signals_decimal);
      const coordinate = (offset: number) =>
        BigInt(
          `0x${bytesHex(proof.proofBytes.subarray(offset, offset + 32))}`,
        ).toString();
      const raw: RawProof = {
        protocol: "groth16",
        curve: "bn128",
        pi_a: [
          coordinate(0),
          (
            (BN254_BASE_FIELD - BigInt(coordinate(32))) %
            BN254_BASE_FIELD
          ).toString(),
          "1",
        ],
        pi_b: [
          [coordinate(96), coordinate(64)],
          [coordinate(160), coordinate(128)],
          ["1", "0"],
        ],
        pi_c: [coordinate(192), coordinate(224), "1"],
      };
      const vk = JSON.parse(
        readFileSync(join(directory!, "verification_key.json"), "utf8"),
      ) as Record<string, unknown>;
      expect(await verifyProofLocally(raw, signals, vk)).toBe(true);
      expect(
        await verifyProofLocally(
          raw,
          [...signals.slice(0, 5), (BigInt(signals[5]!) + 1n).toString()],
          vk,
        ),
      ).toBe(false);
      expect(fetch).toHaveBeenCalledTimes(2);
    } finally {
      vi.unstubAllGlobals();
    }
  }, 120_000);
});
