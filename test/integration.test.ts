import { describe, it, expect } from "vitest";
import * as path from "path";
import * as fs from "fs";
import { simhash, hammingDistance } from "../src/hashing/simhash";
import { generateTBH, computeCommitment } from "../src/hashing/poseidon";
import { prepareCircuitInput, generateProof } from "../src/proof/prover";
import { serializeProof } from "../src/proof/serializer";
import { TOTAL_PROOF_SIZE, NUM_PUBLIC_INPUTS } from "../src/config";

// Circuit artifacts from adjacent circuits repo
const WASM_PATH = path.resolve(
  __dirname,
  "../../circuits/build/iam_hamming_js/iam_hamming.wasm"
);
const ZKEY_PATH = path.resolve(
  __dirname,
  "../../circuits/build/iam_hamming_final.zkey"
);
const VK_PATH = path.resolve(
  __dirname,
  "../../circuits/keys/verification_key.json"
);

// Skip tests if circuit artifacts aren't built
const circuitArtifactsExist =
  fs.existsSync(WASM_PATH) && fs.existsSync(ZKEY_PATH) && fs.existsSync(VK_PATH);

describe.skipIf(!circuitArtifactsExist)(
  "integration: full crypto pipeline",
  () => {
    it("generates a valid proof from mock features end-to-end", async () => {
      // 1. Create mock feature vector (~236 random values)
      const features = Array.from({ length: 236 }, (_, i) =>
        Math.sin(i * 0.3) * Math.cos(i * 0.7)
      );

      // 2. SimHash → 256-bit fingerprint
      const fpNew = simhash(features);
      expect(fpNew.length).toBe(256);

      // 3. Create "previous" fingerprint by flipping 10 bits
      const fpPrev = [...fpNew];
      for (let i = 0; i < 10; i++) {
        fpPrev[i * 25] = fpPrev[i * 25] === 1 ? 0 : 1;
      }
      expect(hammingDistance(fpNew, fpPrev)).toBe(10);

      // 4. Generate TBHs with Poseidon commitments
      const tbhNew = await generateTBH(fpNew);
      const tbhPrev = await generateTBH(fpPrev);

      expect(tbhNew.commitment).toBeGreaterThan(BigInt(0));
      expect(tbhPrev.commitment).toBeGreaterThan(BigInt(0));

      // 5. Prepare circuit input
      const input = prepareCircuitInput(tbhNew, tbhPrev, 30);
      expect(input.ft_new.length).toBe(256);
      expect(input.threshold).toBe("30");

      // 6. Generate Groth16 proof
      const { proof, publicSignals } = await generateProof(
        input,
        WASM_PATH,
        ZKEY_PATH
      );
      expect(publicSignals.length).toBe(NUM_PUBLIC_INPUTS);

      // 7. Serialize for Solana
      const { proofBytes, publicInputs } = serializeProof(
        proof,
        publicSignals
      );
      expect(proofBytes.length).toBe(TOTAL_PROOF_SIZE);
      expect(publicInputs.length).toBe(NUM_PUBLIC_INPUTS);
      for (const input of publicInputs) {
        expect(input.length).toBe(32);
      }

      // 8. Verify locally
      const snarkjs = await import("snarkjs");
      const vk = JSON.parse(fs.readFileSync(VK_PATH, "utf-8"));
      const valid = await snarkjs.groth16.verify(vk, publicSignals, proof);
      expect(valid).toBe(true);
    });

    it("fails proof generation when distance exceeds threshold", async () => {
      // Create fingerprints with ~200 bits different
      const fpNew = Array.from({ length: 256 }, () =>
        Math.random() > 0.5 ? 1 : 0
      );
      const fpPrev = fpNew.map((b) => (Math.random() > 0.2 ? 1 - b : b));

      const tbhNew = await generateTBH(fpNew);
      const tbhPrev = await generateTBH(fpPrev);
      const input = prepareCircuitInput(tbhNew, tbhPrev, 30);

      await expect(
        generateProof(input, WASM_PATH, ZKEY_PATH)
      ).rejects.toThrow();
    });
  }
);
