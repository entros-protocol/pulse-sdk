import {
  BN,
  BorshAccountsCoder,
  BorshInstructionCoder,
  utils,
  type Idl,
} from "@coral-xyz/anchor";
import {
  PublicKey,
  type AccountInfo,
  type ParsedTransactionWithMeta,
} from "@solana/web3.js";
import { PROGRAM_IDS, SAS_CONFIG } from "../../src/config";
import { entrosAnchorIdl, entrosVerifierIdl } from "../../src/protocol/idl";
import { INTEGRATOR_DEVNET_GENESIS_HASH } from "../../src/identity/integrator";
import type { IntegratorEvidenceConnection } from "../../src/identity/integrator-types";

export interface EvidenceFixtureOptions {
  nowSeconds?: number;
  kind?: "mint" | "update" | "compact" | "rebaseline";
  trustScore?: number;
  verificationCount?: number;
  creationTimestamp?: number;
  lastVerificationTimestamp?: number;
  lastResetTimestamp?: number;
  lastRebaselineTimestamp?: number;
  blockTime?: number;
  attestation?: "present" | "missing" | "unavailable";
  identity?: "present" | "missing" | "unavailable";
  expiry?: number;
}

export async function createEvidenceFixture(
  options: EvidenceFixtureOptions = {},
) {
  const nowSeconds = options.nowSeconds ?? 1_800_000_000;
  const blockTime = options.blockTime ?? nowSeconds - 10;
  const wallet = new PublicKey(new Uint8Array(32).fill(7));
  const program = new PublicKey(PROGRAM_IDS.entrosAnchor);
  const [identityPda, bump] = PublicKey.findProgramAddressSync(
    [Buffer.from("identity"), wallet.toBuffer()],
    program,
  );
  const [mint] = PublicKey.findProgramAddressSync(
    [Buffer.from("mint"), wallet.toBuffer()],
    program,
  );
  const [attestationPda] = PublicKey.findProgramAddressSync(
    [
      Buffer.from("attestation"),
      new PublicKey(SAS_CONFIG.entrosCredentialPda).toBuffer(),
      new PublicKey(SAS_CONFIG.entrosSchemaPda).toBuffer(),
      wallet.toBuffer(),
    ],
    new PublicKey(SAS_CONFIG.programId),
  );
  const signature = utils.bytes.bs58.encode(Buffer.alloc(64, 8));
  const commitment = Array<number>(32).fill(3);
  const nonce = Array<number>(32).fill(4);
  const kind = options.kind ?? "update";
  const accountCoder = new BorshAccountsCoder(entrosAnchorIdl as Idl);
  const identityData = await accountCoder.encode("IdentityState", {
    owner: wallet,
    creation_timestamp: new BN(
      options.creationTimestamp ??
        (kind === "mint" ? blockTime : nowSeconds - 1000),
    ),
    last_verification_timestamp: new BN(
      options.lastVerificationTimestamp ?? blockTime,
    ),
    verification_count: options.verificationCount ?? (kind === "mint" ? 0 : 7),
    trust_score: options.trustScore ?? (kind === "mint" ? 0 : 300),
    current_commitment: commitment,
    mint,
    bump,
    recent_timestamps: Array.from({ length: 52 }, () => new BN(0)),
    last_reset_timestamp: new BN(options.lastResetTimestamp ?? 0),
    new_wallet: PublicKey.default,
    projection_version: 1,
    last_rebaseline_timestamp: new BN(
      options.lastRebaselineTimestamp ??
        (kind === "rebaseline" ? blockTime : 0),
    ),
  });
  const identityAccount: AccountInfo<Buffer> = {
    data: identityData,
    executable: false,
    lamports: 1,
    owner: program,
    rentEpoch: 0,
  };
  const sasData = Buffer.alloc(204);
  sasData[0] = 2;
  wallet.toBuffer().copy(sasData, 1);
  new PublicKey(SAS_CONFIG.entrosCredentialPda).toBuffer().copy(sasData, 33);
  new PublicKey(SAS_CONFIG.entrosSchemaPda).toBuffer().copy(sasData, 65);
  sasData.writeUInt32LE(31, 97);
  sasData[101] = 1;
  sasData.writeUInt16LE(300, 102);
  sasData.writeBigInt64LE(BigInt(blockTime), 104);
  sasData.writeUInt32LE(16, 112);
  sasData.write("wallet-connected", 116);
  sasData.fill(9, 132, 164);
  sasData.writeBigInt64LE(BigInt(options.expiry ?? nowSeconds + 1000), 164);
  const sasAccount: AccountInfo<Buffer> = {
    data: sasData,
    executable: false,
    lamports: 1,
    owner: new PublicKey(SAS_CONFIG.programId),
    rentEpoch: 0,
  };
  const anchorCoder = new BorshInstructionCoder(entrosAnchorIdl as Idl);
  const name =
    kind === "compact"
      ? "update_anchor_compact"
      : kind === "mint"
        ? "mint_anchor"
        : kind === "rebaseline"
          ? "rebaseline_anchor"
          : "update_anchor";
  const definition = entrosAnchorIdl.instructions.find(
    (ix) => ix.name === name,
  );
  if (!definition) throw new Error("Fixture instruction is unavailable");
  const [verificationPda] = PublicKey.findProgramAddressSync(
    [Buffer.from("verification"), wallet.toBuffer(), Buffer.from(nonce)],
    new PublicKey(PROGRAM_IDS.entrosVerifier),
  );
  const accounts = definition.accounts.map((entry, index) =>
    index === 0
      ? wallet
      : index === 1
        ? identityPda
        : entry.name === "verification_result"
          ? verificationPda
          : PublicKey.default,
  );
  const data = anchorCoder.encode(
    name,
    kind === "compact"
      ? { verification_nonce: nonce }
      : kind === "mint"
        ? { initial_commitment: commitment }
        : kind === "rebaseline"
          ? { new_commitment: commitment, projection_version: 1 }
          : { new_commitment: commitment, verification_nonce: nonce },
  );
  const transaction: ParsedTransactionWithMeta = {
    slot: 100,
    blockTime,
    version: "legacy",
    meta: { err: null, fee: 0, preBalances: [], postBalances: [] },
    transaction: {
      signatures: [signature],
      message: {
        accountKeys: [{ pubkey: wallet, signer: true, writable: true }],
        recentBlockhash: PublicKey.default.toBase58(),
        instructions: [
          { programId: program, accounts, data: utils.bytes.bs58.encode(data) },
        ],
      },
    },
  };
  if (kind === "compact") {
    const verifierCoder = new BorshInstructionCoder(entrosVerifierIdl as Idl);
    transaction.transaction.message.instructions.unshift({
      programId: new PublicKey(PROGRAM_IDS.entrosVerifier),
      accounts: [wallet, PublicKey.default, verificationPda, PublicKey.default],
      data: utils.bytes.bs58.encode(
        verifierCoder.encode("verify_proof_compact", {
          nonce,
          proof_bytes: Array<number>(256).fill(1),
          commitment_new: commitment,
          commitment_prev: Array<number>(32).fill(2),
          threshold: 96,
          min_distance: 3,
        }),
      ),
    });
  }
  const status = {
    slot: 100,
    confirmations: 1,
    err: null,
    confirmationStatus: "confirmed" as const,
  };
  const calls: string[] = [];
  const connection: IntegratorEvidenceConnection = {
    async getGenesisHash() {
      calls.push("genesis");
      return INTEGRATOR_DEVNET_GENESIS_HASH;
    },
    async getSignatureStatuses() {
      calls.push("status");
      return { context: { slot: 101 }, value: [status] };
    },
    async getParsedTransaction() {
      calls.push("transaction");
      return transaction;
    },
    async getAccountInfoAndContext(address) {
      const isIdentity = address.equals(identityPda);
      calls.push(isIdentity ? "identity" : "attestation");
      const mode = isIdentity ? options.identity : options.attestation;
      if (mode === "unavailable") throw new Error("Synthetic RPC failure");
      return {
        context: { slot: 102 },
        value:
          mode === "missing" ? null : isIdentity ? identityAccount : sasAccount,
      };
    },
  };
  return {
    input: {
      walletPubkey: wallet.toBase58(),
      transactionSignature: signature,
      connection,
      nowSeconds,
    },
    connection,
    identityAccount,
    sasAccount,
    transaction,
    status,
    wallet,
    identityPda,
    attestationPda,
    calls,
  };
}
