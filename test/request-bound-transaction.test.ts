import { describe, expect, it, vi } from "vitest";
import {
  Keypair,
  Ed25519Program,
  PublicKey,
  Transaction,
  type ParsedTransactionWithMeta,
} from "@solana/web3.js";
import { utils, BorshInstructionCoder, type Idl } from "@coral-xyz/anchor";
import { PROGRAM_IDS } from "../src/config";
import { prepareProofRequest, bytesHex } from "../src/proof/request";
import type { RequestBoundManifest } from "../src/proof/request";
import { toBigEndian32 } from "../src/proof/serializer";
import {
  submitViaWallet,
  submitResetViaWallet,
  submitRebaselineViaWallet,
} from "../src/submit/wallet";
import type { SignedReceiptDto } from "../src/submit/types";
import { qualifyingTransaction } from "../src/identity/integrator-transaction";
import { entrosVerifierIdl } from "../src/protocol/idl";

const wallet = Keypair.fromSeed(new Uint8Array(32).fill(19)).publicKey;
function deploymentFixture(isolated = false) {
  const anchor = new PublicKey(isolated ? new Uint8Array(32).fill(61) : PROGRAM_IDS.entrosAnchor);
  const verifier = new PublicKey(isolated ? new Uint8Array(32).fill(62) : PROGRAM_IDS.entrosVerifier);
  const [identity] = PublicKey.findProgramAddressSync(
    [Buffer.from("identity"), wallet.toBytes()],
    anchor,
  );
  const [mint] = PublicKey.findProgramAddressSync(
    [Buffer.from("mint"), wallet.toBytes()],
    anchor,
  );
  const manifest: RequestBoundManifest = {
    generation: "request-bound-v1",
    deploymentDomain: "11".repeat(32),
    genesisHash: isolated ? "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG" : "synthetic",
    verifierProgram: verifier.toBase58(),
    consumerProgram: anchor.toBase58(),
    wasm: { url: "synthetic.wasm", sha256: "12".repeat(32) },
    zkey: { url: "synthetic.zkey", sha256: "13".repeat(32) },
  };
  const request = prepareProofRequest({
    deploymentDomain: manifest.deploymentDomain,
    verifier: bytesHex(verifier.toBytes()),
    consumer: bytesHex(anchor.toBytes()),
    wallet: bytesHex(wallet.toBytes()),
    nonce: "09".repeat(32),
    actionKind: 1,
    action: {
      identity: bytesHex(identity.toBytes()),
      mint: bytesHex(mint.toBytes()),
      counter: 7n,
      projectionVersion: 1,
      commitmentNew: "00".repeat(31) + "01",
      commitmentPrevious: "00".repeat(31) + "02",
      threshold: 30,
      minDistance: 3,
      validUntil: 1800000180n,
    },
  });
  const proof = {
    proofBytes: new Uint8Array(256),
    publicInputs: ["1", "2", "30", "3", request.digestHi, request.digestLo].map(
      toBigEndian32,
    ),
  };
  const connection = {
    rpcEndpoint: "http://localhost:8899",
    commitment: "confirmed",
    getGenesisHash: async () => manifest.genesisHash,
    getLatestBlockhash: async () => ({
      blockhash: "11111111111111111111111111111111",
      lastValidBlockHeight: 1,
    }),
    confirmTransaction: async () => ({ value: { err: null } }),
    getAccountInfo: async () => null,
  };
  let transaction: Transaction | undefined;
  const signer = {
    publicKey: wallet,
    signTransaction: async (value: Transaction) => value,
    signAllTransactions: async (values: Transaction[]) => values,
    sendTransaction: vi.fn(async (value: Transaction) => {
      transaction = value;
      return "bound-signature";
    }),
  };
  return {
    identity,
    manifest,
    request,
    proof,
    connection,
    signer,
    transaction: () => transaction!,
  };
}
function parsed(transaction: Transaction): ParsedTransactionWithMeta {
  const message = transaction.compileMessage();
  return {
    slot: 100,
    blockTime: 1800000000,
    meta: { err: null, fee: 0, preBalances: [], postBalances: [] },
    transaction: {
      signatures: ["bound-signature"],
      message: {
        accountKeys: message.accountKeys.map((pubkey, index) => ({
          pubkey,
          signer: message.isAccountSigner(index),
          writable: message.isAccountWritable(index),
          source: "transaction" as const,
        })),
        recentBlockhash: message.recentBlockhash,
        instructions: transaction.instructions.map((ix) => ({
          programId: ix.programId,
          accounts: ix.keys.map((key) => key.pubkey),
          data: utils.bytes.bs58.encode(ix.data),
        })),
      },
    },
  };
}
describe.each([false, true])("prepared bound transaction isolated=%s", (isolated) => {
  const fixture = () => deploymentFixture(isolated);
  it("preserves the nonce and encodes an atomic bounded transaction", async () => {
    const f = fixture();
    const fetch = vi.spyOn(globalThis, "fetch");
    try {
      const result = await submitViaWallet(f.proof, f.proof.publicInputs[0]!, {
        wallet: f.signer,
        connection: f.connection,
        isFirstVerification: false,
        preparedRequest: f.request,
        requestBoundManifest: f.manifest,
        encryptedBaselineBlob: new Uint8Array(96),
        ...(isolated ? { relayerUrl: "https://executor.invalid/verify" } : {}),
      });
      expect(result.success, result.error).toBe(true);
      expect(fetch).not.toHaveBeenCalled();
      const transaction = f.transaction();
      if (isolated) {
        const addresses = transaction.instructions.flatMap(ix => [ix.programId.toBase58(), ...ix.keys.map(key => key.pubkey.toBase58())]);
        expect(addresses).not.toContain(PROGRAM_IDS.entrosAnchor);
        expect(addresses).not.toContain(PROGRAM_IDS.entrosVerifier);
        const [baseline] = PublicKey.findProgramAddressSync([Buffer.from("encrypted_baseline"), wallet.toBytes()], new PublicKey(f.manifest.consumerProgram));
        expect(addresses).toContain(baseline.toBase58());
      }
      expect(transaction.instructions.map((ix) => ix.data.length)).toEqual([
        5, 8, 40, 372, 40, 104,
      ]);
      const coder = new BorshInstructionCoder(entrosVerifierIdl as Idl);
      const decoded = coder.decode(transaction.instructions[3]!.data)!;
      expect(decoded.name).toBe("verify_proof_bound");
      expect((decoded.data as { nonce: number[] }).nonce).toEqual(
        Array<number>(32).fill(9),
      );
      const length = transaction.serialize({
        requireAllSignatures: false,
        verifySignatures: false,
      }).length;
      expect(length).toBe(1097);
      expect(length).toBeLessThanOrEqual(1232);
      if (!isolated) expect(
        await qualifyingTransaction(
          parsed(transaction),
          "bound-signature",
          wallet,
          f.identity,
        ),
      ).toMatchObject({
        kind: "update",
        commitment: f.request.action.commitmentNew,
      });
    } finally {
      fetch.mockRestore();
    }
  });
  it.each(["mint", "reset", "rebaseline"] as const)(
    "includes durable state in the %s receipt transaction",
    async (kind) => {
      const f = fixture();
      const commitment = f.proof.publicInputs[0]!;
      const message = Buffer.alloc(103);
      message.set(Buffer.from("entros-validator-receipt-v2\0", "ascii"));
      message[28] = kind === "mint" ? 1 : kind === "reset" ? 3 : 2;
      message.writeUInt16LE(1, 29);
      message.set(wallet.toBytes(), 31);
      message.set(commitment, 63);
      message.writeBigInt64LE(1800000000n, 95);
      const signedReceipt: SignedReceiptDto = {
        validator_pubkey_hex: "11".repeat(32),
        signature_hex: "22".repeat(64),
        message_hex: message.toString("hex"),
      };
      const common = {
        wallet: f.signer,
        connection: f.connection,
        requestBoundManifest: f.manifest,
        signedReceipt,
        encryptedBaselineBlob: new Uint8Array(96),
        ...(isolated ? { relayerUrl: "https://executor.invalid/verify" } : {}),
      };
      const result =
        kind === "mint"
          ? await submitViaWallet(
              { proofBytes: new Uint8Array(), publicInputs: [] },
              commitment,
              { ...common, isFirstVerification: true },
            )
          : kind === "reset"
            ? await submitResetViaWallet(commitment, {
                ...common,
                projectionVersion: 1,
              })
            : await submitRebaselineViaWallet(commitment, 1, common);
      expect(result.success, result.error).toBe(true);
      const transaction = f.transaction();
      if (isolated) {
        const addresses = transaction.instructions.flatMap(ix => [ix.programId.toBase58(), ...ix.keys.map(key => key.pubkey.toBase58())]);
        expect(addresses).not.toContain(PROGRAM_IDS.entrosAnchor);
        expect(addresses).not.toContain(PROGRAM_IDS.entrosVerifier);
        const [baseline] = PublicKey.findProgramAddressSync([Buffer.from("encrypted_baseline"), wallet.toBytes()], new PublicKey(f.manifest.consumerProgram));
        expect(addresses).toContain(baseline.toBase58());
      }
      expect(transaction.instructions[1]!.programId.toBase58()).toBe(
        Ed25519Program.programId.toBase58(),
      );
      const action = transaction.instructions[2]!;
      const [state] = PublicKey.findProgramAddressSync(
        [Buffer.from("proof_request_state"), wallet.toBytes()],
        new PublicKey(f.manifest.consumerProgram),
      );
      expect(action.keys[action.keys.length - 1]?.pubkey.toBase58()).toBe(
        state.toBase58(),
      );
      expect(action.keys[action.keys.length - 1]?.isWritable).toBe(true);
      if (!isolated && kind !== "reset")
        expect(
          await qualifyingTransaction(
            parsed(transaction),
            "bound-signature",
            wallet,
            f.identity,
          ),
        ).toMatchObject({ kind, commitment: bytesHex(commitment) });
    },
  );

  it.each(["wallet", "digest", "counter", "nonce", "genesis", "manifest"])(
    "rejects changed %s before the wallet prompt",
    async (field) => {
      const f = fixture();
      let request = f.request;
      if (field === "wallet")
        request = prepareProofRequest({ ...request, wallet: "22".repeat(32) });
      if (field === "digest") request = { ...request, digest: "00".repeat(32) };
      if (field === "counter")
        request = prepareProofRequest({
          ...request,
          action: { ...request.action, counter: 8n },
        });
      if (field === "nonce")
        request = prepareProofRequest({ ...request, nonce: "22".repeat(32) });
      if (field === "genesis")
        f.connection.getGenesisHash = async () => "different";
      const result = await submitViaWallet(f.proof, f.proof.publicInputs[0]!, {
        wallet: f.signer,
        connection: f.connection,
        isFirstVerification: false,
        preparedRequest: request,
        requestBoundManifest: field === "manifest" ? undefined : f.manifest,
      });
      expect(result.success).toBe(false);
      expect(f.signer.sendTransaction).not.toHaveBeenCalled();
    },
  );
  it("requires the matching bound proof instruction and account context in the strict reader", async () => {
    const f = fixture();
    await submitViaWallet(f.proof, f.proof.publicInputs[0]!, {
      wallet: f.signer,
      connection: f.connection,
      isFirstVerification: false,
      preparedRequest: f.request,
      requestBoundManifest: f.manifest,
    });
    const valid = parsed(f.transaction());
    const missing = parsed(f.transaction());
    missing.transaction.message.instructions =
      valid.transaction.message.instructions.filter(
        (_ix, index) => index !== 3,
      );
    expect(
      await qualifyingTransaction(
        missing,
        "bound-signature",
        wallet,
        f.identity,
      ),
    ).toBeNull();
    const mismatched = parsed(f.transaction());
    const ix = mismatched.transaction.message.instructions[3]!;
    if (!("accounts" in ix))
      throw new Error("Expected partially decoded instruction");
    ix.accounts[4] = PublicKey.default;
    expect(
      await qualifyingTransaction(
        mismatched,
        "bound-signature",
        wallet,
        f.identity,
      ),
    ).toBeNull();
  });
});
