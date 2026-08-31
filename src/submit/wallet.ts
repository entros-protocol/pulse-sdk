/* eslint-disable @typescript-eslint/no-explicit-any */
// Anchor program interactions are typed dynamically because the SDK's
// peer dep on @coral-xyz/anchor + @solana/web3.js is loaded via dynamic
// import (avoiding a hard dep for tree-shaking).
import type { Idl, Wallet } from "@coral-xyz/anchor";
import type {
  Commitment,
  PublicKey,
  SendOptions,
  Transaction,
  TransactionInstruction,
  TransactionSignature,
} from "@solana/web3.js";
import type { SolanaProof } from "../proof/types";
import type { SignedReceiptDto, SubmissionResult } from "./types";
import type { VerificationPhase } from "../phases";
import {
  ATTESTATION_SIGNATURE_TIMEOUT_MS,
  CONFIRMATION_TIMEOUT_MS,
  MAX_THRESHOLD,
  MIN_DISTANCE_FLOOR,
  NUM_PUBLIC_INPUTS,
  PROGRAM_IDS,
  SIGNATURE_TIMEOUT_MS,
  TOTAL_PROOF_SIZE,
} from "../config";
import { sdkLog, sdkWarn } from "../log";
import { entrosAnchorIdl, entrosVerifierIdl } from "../protocol/idl";
import { buildEd25519ReceiptIx, receiptMatchesBinding } from "./receipt";
import { ENCRYPTED_BASELINE_BLOB_BYTES } from "../identity/baseline";
import { getProjectionDefinition } from "../projection";
import {
  chainRevertError,
  errToString,
  isChainRevertError,
  isUserRejection,
  withTimeout,
} from "./errors";
import {
  ASSOCIATED_TOKEN_PROGRAM_ADDRESS,
  TOKEN_2022_PROGRAM_ADDRESS,
  deriveToken2022AssociatedAddress,
} from "./associated-token";

interface SubmissionConnection {
  getLatestBlockhash(commitment: Commitment): Promise<{
    blockhash: string;
    lastValidBlockHeight: number;
  }>;
  confirmTransaction(
    signature: TransactionSignature,
    commitment: Commitment,
  ): Promise<{ value: { err: unknown | null } }>;
}

interface SubmissionWallet {
  publicKey: PublicKey;
  signTransaction(transaction: Transaction): Promise<Transaction>;
  signAllTransactions(transactions: Transaction[]): Promise<Transaction[]>;
  sendTransaction(
    transaction: Transaction,
    connection: SubmissionConnection,
    options?: SendOptions,
  ): Promise<TransactionSignature>;
}

interface RebaselineInstructionBuilder {
  accounts(accounts: Record<string, unknown>): RebaselineInstructionBuilder;
  instruction(): Promise<TransactionInstruction>;
}

interface RebaselineProgram {
  methods: {
    rebaselineAnchor(
      commitment: number[],
      projectionVersion: number,
    ): RebaselineInstructionBuilder;
  };
}

interface CompactProofArguments {
  proofBytes: number[];
  commitmentNew: number[];
  commitmentPrev: number[];
  threshold: number;
  minDistance: number;
}

function fixedBytes(value: unknown, length: number, name: string): number[] {
  if (!Array.isArray(value) && !(value instanceof Uint8Array)) {
    throw new Error(`${name} must be a byte array`);
  }
  const bytes = Array.from(value);
  if (bytes.length !== length) {
    throw new Error(`${name} must contain ${length} bytes, got ${bytes.length}`);
  }
  if (
    bytes.some(
      (byte) => !Number.isInteger(byte) || byte < 0 || byte > 255,
    )
  ) {
    throw new Error(`${name} must contain only bytes`);
  }
  return bytes;
}

function nonZeroFixedBytes(
  value: unknown,
  length: number,
  name: string,
): number[] {
  const bytes = fixedBytes(value, length, name);
  if (!bytes.some((byte) => byte !== 0)) {
    throw new Error(`${name} must not be zero`);
  }
  return bytes;
}

function decodeU16FieldElement(value: Uint8Array, name: string): number {
  const bytes = fixedBytes(value, 32, name);
  for (let index = 0; index < 30; index += 1) {
    if (bytes[index] !== 0) {
      throw new Error(`${name} does not fit in u16`);
    }
  }
  return (bytes[30]! << 8) | bytes[31]!;
}

function compactProofArguments(proof: SolanaProof): CompactProofArguments {
  if (proof.publicInputs.length !== NUM_PUBLIC_INPUTS) {
    throw new Error(
      `proof must contain ${NUM_PUBLIC_INPUTS} public inputs, got ${proof.publicInputs.length}`,
    );
  }

  const threshold = decodeU16FieldElement(proof.publicInputs[2]!, "threshold");
  const minDistance = decodeU16FieldElement(
    proof.publicInputs[3]!,
    "min_distance",
  );
  if (threshold > MAX_THRESHOLD) {
    throw new Error(`threshold must be at most ${MAX_THRESHOLD}`);
  }
  if (minDistance < MIN_DISTANCE_FLOOR) {
    throw new Error(`min_distance must be at least ${MIN_DISTANCE_FLOOR}`);
  }
  if (minDistance >= threshold) {
    throw new Error("min_distance must be less than threshold");
  }

  return {
    proofBytes: fixedBytes(proof.proofBytes, TOTAL_PROOF_SIZE, "proof_bytes"),
    commitmentNew: nonZeroFixedBytes(
      proof.publicInputs[0]!,
      32,
      "commitment_new",
    ),
    commitmentPrev: nonZeroFixedBytes(
      proof.publicInputs[1]!,
      32,
      "commitment_prev",
    ),
    threshold,
    minDistance,
  };
}

function randomNonce(): number[] {
  const nonce = crypto.getRandomValues(new Uint8Array(32));
  if (!nonce.some((byte) => byte !== 0)) {
    nonce[31] = 1;
  }
  return Array.from(nonce);
}

function validatedServerNonce(value: unknown): number[] | null {
  try {
    return nonZeroFixedBytes(value, 32, "server nonce");
  } catch {
    return null;
  }
}

/**
 * Build a `set_encrypted_baseline` instruction for the given anchor program
 * + wallet pubkey + 96-byte encrypted blob. Callers pass a pre-built blob
 * (from `encryptBaselineBlob`) and the helper derives both the IdentityState
 * (UncheckedAccount) PDA and the EncryptedBaseline PDA from the wallet pubkey.
 *
 * The on-chain handler uses `init_if_needed`, so the first call for a wallet
 * creates the PDA and subsequent calls overwrite the existing blob.
 */
async function buildSetEncryptedBaselineIx(
  anchorProgram: any,
  walletPubkey: any,
  blob: Uint8Array,
): Promise<any> {
  if (blob.length !== ENCRYPTED_BASELINE_BLOB_BYTES) {
    throw new Error(
      `encrypted baseline blob must be ${ENCRYPTED_BASELINE_BLOB_BYTES} bytes, got ${blob.length}`,
    );
  }
  const { PublicKey, SystemProgram } = await import("@solana/web3.js");
  const programId = new PublicKey(PROGRAM_IDS.entrosAnchor);
  const [identityPda] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("identity"), walletPubkey.toBuffer()],
    programId,
  );
  const [encryptedBaselinePda] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("encrypted_baseline"), walletPubkey.toBuffer()],
    programId,
  );
  return anchorProgram.methods
    .setEncryptedBaseline(Array.from(blob))
    .accounts({
      authority: walletPubkey,
      identityState: identityPda,
      encryptedBaseline: encryptedBaselinePda,
      systemProgram: SystemProgram.programId,
    })
    .instruction();
}

/**
 * Wait for a tx to confirm AND throw if the chain-side execution errored.
 * web3.js 1.x's `connection.confirmTransaction` resolves successfully even
 * when the tx reverted on chain (it only checks signature inclusion); the
 * caller MUST inspect `value.err`. Without this, on-chain Anchor errors
 * (CommitmentMismatch, MissingValidatorReceipt, ResetCooldownActive,
 * InsufficientFunds, etc.) are silently swallowed and `submitViaWallet`
 * returns a "successful" txSignature for
 * a tx that never mutated state — a credibility hit. The thrown message
 * preserves the JSON `InstructionError` shape so downstream regex parsing
 * can extract the `Custom` code.
 */
async function confirmAndCheck(
  connection: any,
  signature: string | undefined,
): Promise<void> {
  if (!signature) {
    throw new Error("confirmAndCheck called without a transaction signature");
  }
  const confirmation = await connection.confirmTransaction(signature, "confirmed");
  if (confirmation?.value?.err != null) {
    // Marked so `sendAndConfirm` can tell a reported on-chain failure, where
    // the fee was definitely taken, from an RPC that stopped answering, where
    // the transaction may still be in flight. Only the former is the
    // `confirmation` phase.
    throw chainRevertError(
      `Transaction failed on chain: ${JSON.stringify(confirmation.value.err)} (sig=${signature})`,
    );
  }
}

/**
 * Sign, broadcast and confirm, attributing any failure to the phase it
 * belongs to.
 *
 * Wallet adapters merge signing and sending into one `sendTransaction` call.
 * The host needs separate signing, submission, and confirmation outcomes.
 *
 * Two rules, both deliberately conservative:
 *
 *   - Only a declined prompt is `signing`. It is the single outcome that is
 *     certainly not on the wire. Everything else out of `sendTransaction`,
 *     including this function's own timeout, is `submission`, whose spend is
 *     `possible` rather than a claim in either direction.
 *   - Only a cluster-reported execution failure is `confirmation`. A
 *     confirmation timeout is `submission` for the same reason.
 *
 * Neither timeout cancels the work it bounds, because nothing in a wallet
 * adapter or in web3.js can be cancelled. A prompt approved after the clock
 * expires still broadcasts, which is why `SIGNATURE_TIMEOUT_MS` is set where it
 * fires for a hung wallet and effectively never for a slow user.
 */
async function sendAndConfirm(
  wallet: any,
  connection: any,
  tx: any,
  sendOptions?: { skipPreflight: boolean },
): Promise<
  | { ok: true; txSig: string }
  | { ok: false; error: string; failedAt: VerificationPhase }
> {
  let txSig: string;
  try {
    // The third argument is omitted rather than passed as `undefined`. The
    // standard adapter defaults it, but wrappers in the wild read
    // `options.skipPreflight` without one, and the reset path is the caller
    // that relies on preflight staying on.
    txSig = await withTimeout(
      sendOptions
        ? wallet.sendTransaction(tx, connection, sendOptions)
        : wallet.sendTransaction(tx, connection),
      SIGNATURE_TIMEOUT_MS,
      "Your wallet did not respond to the signature request. Open your wallet, check whether a request is still pending, then try again.",
    );
  } catch (err) {
    return {
      ok: false,
      error: errToString(err),
      failedAt: isUserRejection(err) ? "signing" : "submission",
    };
  }

  try {
    await withTimeout(
      confirmAndCheck(connection, txSig),
      CONFIRMATION_TIMEOUT_MS,
      "The network did not confirm your transaction in time. It may still land. Check your wallet's recent activity before trying again.",
    );
  } catch (err) {
    return {
      ok: false,
      error: errToString(err),
      failedAt: isChainRevertError(err) ? "confirmation" : "submission",
    };
  }

  return { ok: true, txSig };
}

/**
 * Best-effort SAS attestation request. POSTs to the executor's `/attest`
 * endpoint with the wallet's public key, a server-issued challenge nonce,
 * and an `Entros-ATTEST:{wallet}:{timestamp}` ownership signature.
 *
 * Returns the attestation tx signature on success, `undefined` on any
 * failure (attestation is non-fatal — the on-chain tx has already confirmed
 * by the time this is called).
 *
 * Wallet-only path: the executor's `/attest` endpoint requires nonce +
 * signature + message on every request (walletless tier no longer writes
 * to SAS). If any of those is unavailable on the client side — wallet
 * adapter has no `signMessage`, signing throws, or no server nonce was
 * issued during this verification — we skip the request entirely instead
 * of sending a doomed-to-400 call.
 */
async function requestSasAttestation(
  wallet: any,
  walletAddress: string,
  relayerUrl: string,
  relayerApiKey: string | undefined,
  serverNonce: number[] | undefined,
): Promise<string | undefined> {
  if (!serverNonce) {
    sdkLog("[Entros SDK] Skipping SAS attestation: no server-issued nonce");
    return undefined;
  }
  if (!wallet?.signMessage) {
    sdkLog("[Entros SDK] Skipping SAS attestation: wallet does not support signMessage");
    return undefined;
  }

  let signature: string;
  let message: string;
  try {
    const timestamp = Math.floor(Date.now() / 1000);
    message = `Entros-ATTEST:${walletAddress}:${timestamp}`;
    const messageBytes = new TextEncoder().encode(message);
    // Bounded because this prompt comes after the transaction has confirmed.
    // A wallet that never surfaces it must not be able to hold a successful
    // verification open: the caller stores the local baseline and reports
    // success only once this function returns.
    const sigBytes: Uint8Array = await withTimeout(
      wallet.signMessage(messageBytes),
      ATTESTATION_SIGNATURE_TIMEOUT_MS,
      "wallet did not sign the attestation message",
    );
    signature = Array.from(sigBytes)
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  } catch (err) {
    sdkWarn(
      `[Entros SDK] Attestation signature unavailable, skipping SAS attestation: ${errToString(err)}`,
    );
    return undefined;
  }

  try {
    const attestHeaders: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (relayerApiKey) {
      attestHeaders["X-API-Key"] = relayerApiKey;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15_000);

    const baseUrl = new URL(relayerUrl);
    const attestUrl = `${baseUrl.origin}/attest`;

    const attestBody: Record<string, unknown> = {
      wallet_address: walletAddress,
      nonce: serverNonce,
      signature,
      message,
    };

    const attestRes = await fetch(attestUrl, {
      method: "POST",
      headers: attestHeaders,
      body: JSON.stringify(attestBody),
      signal: controller.signal,
    });

    clearTimeout(timer);

    if (attestRes.ok) {
      const attestData = (await attestRes.json()) as {
        success?: boolean;
        attestation_tx?: string;
      };
      if (attestData.success && attestData.attestation_tx) {
        return attestData.attestation_tx;
      }
    }
  } catch (err) {
    // Attestation is best-effort; on-chain tx already confirmed. Log the
    // failure cause so operators / integrators can distinguish "not
    // configured" (returned undefined silently) from "configured but
    // failed" (network error, 5xx, malformed response).
    const msg = err instanceof Error ? err.message : String(err);
    sdkWarn(`[Entros SDK] SAS attestation request failed: ${msg}`);
  }
  return undefined;
}

/**
 * Submit a proof on-chain via a connected wallet (wallet-connected mode).
 * Uses Anchor SDK to construct and send the transaction.
 *
 * Flow for re-verification: single batched transaction containing
 *   ComputeBudget → create_challenge → verify_proof → update_anchor
 * Flow for first verification: mint_anchor (already 1 transaction)
 */
export async function submitViaWallet(
  proof: SolanaProof,
  commitment: Uint8Array,
  options: {
    wallet: any;
    connection: any;
    isFirstVerification: boolean;
    relayerUrl?: string;
    relayerApiKey?: string;
    /**
     * Validator-signed mint receipt. Consumed only on the first-verification
     * path: when present, the SDK prepends an `Ed25519Program::verify`
     * instruction so on-chain `mint_anchor` can confirm the commitment was
     * endorsed by the configured validator. Re-verification ignores the
     * field entirely — `update_anchor` enforces binding via the
     * VerificationResult PDA instead.
     */
    signedReceipt?: SignedReceiptDto;
    /**
     * Encrypted baseline blob. When present, the SDK
     * appends a `set_encrypted_baseline` instruction at the end of the
     * atomic transaction so the wallet's on-chain baseline is rewritten
     * to reflect the new fingerprint in the same wallet prompt as the
     * mint / re-verify. Omitted when the wallet adapter lacks
     * `signMessage` (the SDK can't derive the AES key without it).
     */
    encryptedBaselineBlob?: Uint8Array;
    /**
     * Stage notifications for the tail of the submission. The caller has
     * already rendered a "submitting" stage, which stops being true the moment
     * the transaction confirms. Everything after that is optional work, and on
     * mobile it is where the user waits longest, so it should not be described
     * as a submission still in flight.
     */
    onProgress?: (stage: string) => void;
  }
): Promise<SubmissionResult> {
  if (options.isFirstVerification && !options.signedReceipt) {
    return {
      success: false,
      error:
        "First verification requires a validator-signed mint receipt. The validator did not return one, so the SDK did not request a wallet transaction.",
      failedAt: "submission",
    };
  }

  try {
    const anchor = await import("@coral-xyz/anchor");
    const {
      PublicKey,
      SystemProgram,
      Transaction,
      ComputeBudgetProgram,
      SYSVAR_INSTRUCTIONS_PUBKEY,
    } = await import("@solana/web3.js");

    const provider = new anchor.AnchorProvider(
      options.connection,
      options.wallet,
      { commitment: "confirmed" }
    );

    const anchorProgramId = new PublicKey(PROGRAM_IDS.entrosAnchor);

    let txSig: string | undefined;
    let serverNonce = false;
    let nonce: number[] = [];

    if (!options.isFirstVerification) {
      const compactProof = compactProofArguments(proof);
      const expectedCommitment = fixedBytes(commitment, 32, "commitment");
      if (
        expectedCommitment.some(
          (byte, index) => byte !== compactProof.commitmentNew[index],
        )
      ) {
        throw new Error("proof commitment does not match the submitted commitment");
      }

      // Re-verification submits challenge creation, proof verification, and
      // identity update in one transaction.
      const verifierProgramId = new PublicKey(PROGRAM_IDS.entrosVerifier);

      // Fetch server-generated nonce (prevents pre-computation attacks).
      // Falls back to client-generated nonce if executor is unreachable.
      if (options.relayerUrl) {
        try {
          const baseUrl = new URL(options.relayerUrl);
          const challengeHeaders: Record<string, string> = {};
          if (options.relayerApiKey) {
            challengeHeaders["X-API-Key"] = options.relayerApiKey;
          }
          const challengeController = new AbortController();
          const challengeTimer = setTimeout(() => challengeController.abort(), 5_000);
          const challengeRes = await fetch(
            `${baseUrl.origin}/challenge?wallet=${provider.wallet.publicKey.toBase58()}`,
            { headers: challengeHeaders, signal: challengeController.signal }
          );
          clearTimeout(challengeTimer);
          if (challengeRes.ok) {
            const challengeData = (await challengeRes.json()) as { nonce?: unknown };
            const validatedNonce = validatedServerNonce(challengeData.nonce);
            if (validatedNonce) {
              nonce = validatedNonce;
              serverNonce = true;
              sdkLog("Using server-generated challenge nonce");
            } else {
              nonce = randomNonce();
              sdkWarn("Server returned invalid nonce, using client-generated");
            }
          } else {
            nonce = randomNonce();
            sdkWarn("Challenge endpoint returned error, using client-generated nonce");
          }
        } catch {
          nonce = randomNonce();
          sdkWarn("Challenge fetch failed, using client-generated nonce");
        }
      } else {
        nonce = randomNonce();
      }

      const [challengePda] = PublicKey.findProgramAddressSync(
        [
          new TextEncoder().encode("challenge"),
          provider.wallet.publicKey.toBuffer(),
          new Uint8Array(nonce),
        ],
        verifierProgramId
      );

      const [verificationPda] = PublicKey.findProgramAddressSync(
        [
          new TextEncoder().encode("verification"),
          provider.wallet.publicKey.toBuffer(),
          new Uint8Array(nonce),
        ],
        verifierProgramId
      );

      const [identityPda] = PublicKey.findProgramAddressSync(
        [new TextEncoder().encode("identity"), provider.wallet.publicKey.toBuffer()],
        anchorProgramId
      );

      const registryProgramId = new PublicKey(PROGRAM_IDS.entrosRegistry);
      const [protocolConfigPda] = PublicKey.findProgramAddressSync(
        [new TextEncoder().encode("protocol_config")],
        registryProgramId
      );
      const [treasuryPda] = PublicKey.findProgramAddressSync(
        [new TextEncoder().encode("protocol_treasury")],
        registryProgramId
      );

      const verifierProgram: any = new anchor.Program(
        entrosVerifierIdl as Idl,
        provider,
      );
      const anchorProgram: any = new anchor.Program(
        entrosAnchorIdl as Idl,
        provider,
      );

      // Build all three instructions without sending
      const createChallengeIx = await verifierProgram.methods
        .createChallenge(nonce)
        .accounts({
          challenger: provider.wallet.publicKey,
          challenge: challengePda,
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      const verifyProofIx = await verifierProgram.methods
        .verifyProofCompact(
          nonce,
          compactProof.proofBytes,
          compactProof.commitmentNew,
          compactProof.commitmentPrev,
          compactProof.threshold,
          compactProof.minDistance,
        )
        .accounts({
          verifier: provider.wallet.publicKey,
          challenge: challengePda,
          verificationResult: verificationPda,
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      const updateAnchorIx = await anchorProgram.methods
        .updateAnchorCompact(nonce)
        .accounts({
          authority: provider.wallet.publicKey,
          identityState: identityPda,
          verificationResult: verificationPda,
          protocolConfig: protocolConfigPda,
          treasury: treasuryPda,
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      // Request more compute when the transaction also creates or updates the
      // encrypted baseline account.
      const tx = new Transaction();
      const computeUnitLimit = options.encryptedBaselineBlob ? 300_000 : 250_000;
      tx.add(ComputeBudgetProgram.setComputeUnitLimit({ units: computeUnitLimit }));
      tx.add(createChallengeIx);
      tx.add(verifyProofIx);
      tx.add(updateAnchorIx);
      if (options.encryptedBaselineBlob) {
        const setBaselineIx = await buildSetEncryptedBaselineIx(
          anchorProgram,
          provider.wallet.publicKey,
          options.encryptedBaselineBlob,
        );
        tx.add(setBaselineIx);
      }

      tx.feePayer = provider.wallet.publicKey;
      tx.recentBlockhash = (
        await options.connection.getLatestBlockhash("confirmed")
      ).blockhash;

      const sent = await sendAndConfirm(options.wallet, options.connection, tx, {
        skipPreflight: true,
      });
      if (!sent.ok) {
        return { success: false, error: sent.error, failedAt: sent.failedAt };
      }
      txSig = sent.txSig;
    } else {
      // First verification: mint anchor. Bundles an `Ed25519Program::verify`
      // instruction before `mint_anchor` when the validator returned a
      // signed receipt. The on-chain program inspects the preceding
      // instruction via the Instructions sysvar to confirm the validator
      // endorsed (wallet, commitment, validated_at) before allowing the
      // mint.
      //
      // The `instructions_sysvar` account is required by the on-chain
      // `MintAnchor` accounts struct unconditionally — it must be present
      // even when no receipt is bundled (the on-chain check is currently
      // log-only, but the Anchor framework itself requires every account
      // listed in the IDL to be supplied).
      const anchorProgram: any = new anchor.Program(
        entrosAnchorIdl as Idl,
        provider,
      );

      const [identityPda] = PublicKey.findProgramAddressSync(
        [new TextEncoder().encode("identity"), provider.wallet.publicKey.toBuffer()],
        anchorProgramId
      );
      const [mintPda] = PublicKey.findProgramAddressSync(
        [new TextEncoder().encode("mint"), provider.wallet.publicKey.toBuffer()],
        anchorProgramId
      );
      const [mintAuthority] = PublicKey.findProgramAddressSync(
        [new TextEncoder().encode("mint_authority")],
        anchorProgramId
      );

      const registryProgramId = new PublicKey(PROGRAM_IDS.entrosRegistry);
      const [protocolConfigPda] = PublicKey.findProgramAddressSync(
        [new TextEncoder().encode("protocol_config")],
        registryProgramId
      );
      const [treasuryPda] = PublicKey.findProgramAddressSync(
        [new TextEncoder().encode("protocol_treasury")],
        registryProgramId
      );

      const ata = deriveToken2022AssociatedAddress(
        mintPda,
        provider.wallet.publicKey,
        PublicKey,
      );
      const token2022ProgramId = new PublicKey(TOKEN_2022_PROGRAM_ADDRESS);
      const associatedTokenProgramId = new PublicKey(
        ASSOCIATED_TOKEN_PROGRAM_ADDRESS,
      );

      const mintAnchorIx = await anchorProgram.methods
        .mintAnchor(Array.from(commitment))
        .accounts({
          user: provider.wallet.publicKey,
          identityState: identityPda,
          mint: mintPda,
          mintAuthority,
          tokenAccount: ata,
          associatedTokenProgram: associatedTokenProgramId,
          tokenProgram: token2022ProgramId,
          systemProgram: SystemProgram.programId,
          protocolConfig: protocolConfigPda,
          treasury: treasuryPda,
          instructionsSysvar: SYSVAR_INSTRUCTIONS_PUBKEY,
        })
        .instruction();

      // Decode the receipt up front so we can hard-fail if the validator
      // returned malformed bytes. Silently falling back to a no-receipt
      // mint when the caller expected a binding would mask validator bugs
      // and, once on-chain enforcement is enabled, would produce a
      // confusing on-chain reject after the user has already approved a
      // wallet signature. Fail-fast lets the caller surface a clear error
      // and retry once the validator is healthy.
      let ed25519Ix: import("@solana/web3.js").TransactionInstruction | null = null;
      if (options.signedReceipt) {
        ed25519Ix = await buildEd25519ReceiptIx(options.signedReceipt);
        if (!ed25519Ix) {
          return {
            success: false,
            error:
              "Validator returned a signed receipt that failed to decode (malformed hex or wrong byte length). Refusing to mint without a valid binding. The validator service may be misconfigured. Check the validation-service logs.",
            failedAt: "submission",
          };
        }
        sdkLog(
          "[Entros SDK] Bundling validator-signed mint receipt before mint_anchor"
        );
      }

      // Transaction shape:
      //   [0] ComputeBudgetProgram.setComputeUnitLimit
      //   [1] Ed25519Program::verify(receipt)
      //   [2] mint_anchor(initial_commitment)
      //   [3] (optional) set_encrypted_baseline(blob)
      //
      // Including an explicit compute-budget ix at index 0 prevents wallet
      // adapters that lazily inject one from inserting it between the
      // Ed25519 ix and `mint_anchor`. The on-chain receipt parser locates
      // the receipt at `current_instruction_index - 1`, so any ix between
      // the Ed25519 prefix and `mint_anchor` would fail the enforced binding.
      //
      // `set_encrypted_baseline` is appended LAST because it depends on the
      // IdentityState PDA created by `mint_anchor`; intra-tx instructions
      // execute sequentially, so the existence check (`data_len() > 0`)
      // passes when `mint_anchor` runs first. A live diagnostic measured this
      // four-instruction shape at 94,440 CU. The Ed25519 precompile runs in
      // the runtime, not against the program's CU budget.
      const tx = new Transaction();
      const computeUnitLimit = options.encryptedBaselineBlob ? 250_000 : 200_000;
      tx.add(ComputeBudgetProgram.setComputeUnitLimit({ units: computeUnitLimit }));
      if (ed25519Ix) tx.add(ed25519Ix);
      tx.add(mintAnchorIx);
      if (options.encryptedBaselineBlob) {
        const setBaselineIx = await buildSetEncryptedBaselineIx(
          anchorProgram,
          provider.wallet.publicKey,
          options.encryptedBaselineBlob,
        );
        tx.add(setBaselineIx);
      }

      tx.feePayer = provider.wallet.publicKey;
      tx.recentBlockhash = (
        await options.connection.getLatestBlockhash("confirmed")
      ).blockhash;

      const sent = await sendAndConfirm(options.wallet, options.connection, tx, {
        skipPreflight: true,
      });
      if (!sent.ok) {
        return { success: false, error: sent.error, failedAt: sent.failedAt };
      }
      txSig = sent.txSig;
    }

    options.onProgress?.("Finishing up...");

    // The transaction has confirmed, so the verification is durable on chain
    // from here. Nothing below may turn it into a failure: the caller stores
    // the local baseline and shows success only on what this returns, and the
    // attestation is best-effort. Isolated rather than left to the outer catch
    // so that stays true if anything else is ever added after the confirm.
    let attestationTx: string | undefined;
    if (options.relayerUrl) {
      try {
        attestationTx = await requestSasAttestation(
          options.wallet,
          provider.wallet.publicKey.toBase58(),
          options.relayerUrl,
          options.relayerApiKey,
          serverNonce ? nonce : undefined,
        );
      } catch (err) {
        sdkWarn(`[Entros SDK] SAS attestation skipped: ${errToString(err)}`);
      }
    }

    return { success: true, txSignature: txSig, attestationTx };
  } catch (err: any) {
    // Everything `sendAndConfirm` handles has already returned, so reaching
    // here means the transaction was never built: an RPC that would not answer,
    // a PDA derivation, a dynamic import. Nothing was spent, and `submission`
    // reports `possible` rather than `none`, which is the one direction it is
    // safe to be wrong in.
    return { success: false, error: errToString(err), failedAt: "submission" };
  }
}

/**
 * Submit a baseline reset on-chain via a connected wallet.
 *
 * Fires when the on-chain IdentityState exists for the wallet but the
 * device's local encrypted fingerprint envelope is unrecoverable. The
 * ZK Hamming proof used by `update_anchor` needs the previous
 * fingerprint's bits as a private witness; without them, re-verification
 * is blocked. `reset_identity_state` rotates `current_commitment`
 * in place, zeroes verification_count / trust_score / recent_timestamps,
 * and sets a 7-day cooldown before the next reset.
 *
 * Transaction shape: single instruction (no challenge / verify_proof /
 * ZK proof required). Humanness evidence comes from the validation
 * pipeline invoked at the /attest step (same as mint and update).
 */
export async function submitResetViaWallet(
  commitment: Uint8Array,
  options: {
    wallet: any;
    connection: any;
    relayerUrl?: string;
    relayerApiKey?: string;
    /** Version asserted against the active on-chain projection policy. */
    projectionVersion?: number;
    /** Validator receipt required by reset transitions on projection version 1 and later. */
    signedReceipt?: SignedReceiptDto;
    /**
     * Encrypted baseline blob. When present, the SDK
     * appends a `set_encrypted_baseline` instruction so the wallet's
     * on-chain baseline is rewritten under the NEW post-reset commitment
     * in the same atomic transaction. Without this, the prior blob would
     * be stale on the next recovery attempt (auth-tag mismatch under the
     * new commitment in AAD) and recovery would fall back to fresh capture.
     */
    encryptedBaselineBlob?: Uint8Array;
    /**
     * Stage notifications for the tail of the submission. The caller has
     * already rendered a "submitting" stage, which stops being true the moment
     * the transaction confirms. Everything after that is optional work, and on
     * mobile it is where the user waits longest, so it should not be described
     * as a submission still in flight.
     */
    onProgress?: (stage: string) => void;
  }
): Promise<SubmissionResult> {
  try {
    const anchor = await import("@coral-xyz/anchor");
    const {
      PublicKey,
      SystemProgram,
      SYSVAR_INSTRUCTIONS_PUBKEY,
      Transaction,
      ComputeBudgetProgram,
    } =
      await import("@solana/web3.js");

    const provider = new anchor.AnchorProvider(
      options.connection,
      options.wallet,
      { commitment: "confirmed" }
    );

    const anchorProgramId = new PublicKey(PROGRAM_IDS.entrosAnchor);
    const registryProgramId = new PublicKey(PROGRAM_IDS.entrosRegistry);

    const [identityPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("identity"), provider.wallet.publicKey.toBuffer()],
      anchorProgramId
    );
    const [protocolConfigPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("protocol_config")],
      registryProgramId
    );
    const [treasuryPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("protocol_treasury")],
      registryProgramId
    );

    const anchorProgram: any = new anchor.Program(
      entrosAnchorIdl as Idl,
      provider,
    );
    const projectionVersion = options.projectionVersion ?? 0;
    let receiptIx: import("@solana/web3.js").TransactionInstruction | undefined;
    if (getProjectionDefinition(projectionVersion).authenticatedTransitions) {
      if (
        !options.signedReceipt ||
        !receiptMatchesBinding(options.signedReceipt, {
          purpose: 3,
          projectionVersion,
          wallet: provider.wallet.publicKey.toBytes(),
          commitment,
        })
      ) {
        return {
          success: false,
          error: "Baseline reset requires a matching validator-signed receipt.",
          failedAt: "submission",
        };
      }
      const builtReceiptIx = await buildEd25519ReceiptIx(options.signedReceipt);
      if (!builtReceiptIx) {
        return {
          success: false,
          error: "Baseline reset receipt is malformed.",
          failedAt: "submission",
        };
      }
      receiptIx = builtReceiptIx;
    }

    let resetBuilder = anchorProgram.methods
      .resetIdentityState(Array.from(commitment), projectionVersion)
      .accounts({
        authority: provider.wallet.publicKey,
        identityState: identityPda,
        protocolConfig: protocolConfigPda,
        treasury: treasuryPda,
        systemProgram: SystemProgram.programId,
      });
    if (getProjectionDefinition(projectionVersion).authenticatedTransitions) {
      resetBuilder = resetBuilder.remainingAccounts([
        {
          pubkey: SYSVAR_INSTRUCTIONS_PUBKEY,
          isSigner: false,
          isWritable: false,
        },
      ]);
    }
    const resetIx = await resetBuilder.instruction();

    // Reset does no ZK verification; budget is well under the 200K default
    // even with the encrypted-baseline ix bundled (~30K reset + ~17K init).
    // Keep an explicit limit for determinism and to match batched-tx ergonomics.
    const tx = new Transaction();
    tx.add(ComputeBudgetProgram.setComputeUnitLimit({ units: 200_000 }));
    if (receiptIx) tx.add(receiptIx);
    tx.add(resetIx);
    if (options.encryptedBaselineBlob) {
      const setBaselineIx = await buildSetEncryptedBaselineIx(
        anchorProgram,
        provider.wallet.publicKey,
        options.encryptedBaselineBlob,
      );
      tx.add(setBaselineIx);
    }

    tx.feePayer = provider.wallet.publicKey;
    tx.recentBlockhash = (
      await options.connection.getLatestBlockhash("confirmed")
    ).blockhash;

    // Preflight left ON deliberately, unlike the verify and mint paths above,
    // which still skip it. Those two work in production, and changing them
    // would be an untested behaviour change on a path that is not broken.
    // This one was.
    //
    // Preflight is what turns a client-side encoding error into a free
    // rejection instead of a paid revert. `skipPreflight: true` here meant
    // that when the bundled IDL fell behind the deployed program, every reset
    // was broadcast, charged, and reverted on chain with
    // `InstructionDidNotDeserialize` rather than being refused for nothing.
    const sent = await sendAndConfirm(options.wallet, options.connection, tx);
    if (!sent.ok) {
      return { success: false, error: sent.error, failedAt: sent.failedAt };
    }
    const txSig = sent.txSig;

    options.onProgress?.("Finishing up...");

    // Request a fresh SAS attestation. The executor's /attest handler closes
    // any prior attestation for this wallet and creates a new one bound to the
    // current commitment. Isolated for the same reason as the verify path: the
    // reset has already confirmed on chain and nothing here may undo that.
    let attestationTx: string | undefined;
    if (options.relayerUrl) {
      try {
        attestationTx = await requestSasAttestation(
          options.wallet,
          provider.wallet.publicKey.toBase58(),
          options.relayerUrl,
          options.relayerApiKey,
          undefined,
        );
      } catch (err) {
        sdkWarn(`[Entros SDK] SAS attestation skipped: ${errToString(err)}`);
      }
    }

    return { success: true, txSignature: txSig, attestationTx };
  } catch (err: any) {
    // Everything `sendAndConfirm` handles has already returned, so reaching
    // here means the transaction was never built: an RPC that would not answer,
    // a PDA derivation, a dynamic import. Nothing was spent, and `submission`
    // reports `possible` rather than `none`, which is the one direction it is
    // safe to be wrong in.
    return { success: false, error: errToString(err), failedAt: "submission" };
  }
}

/** Submit an authenticated projection migration and its encrypted baseline. */
export async function submitRebaselineViaWallet(
  commitment: Uint8Array,
  projectionVersion: number,
  options: {
    wallet: SubmissionWallet;
    connection: SubmissionConnection;
    signedReceipt: SignedReceiptDto;
    encryptedBaselineBlob: Uint8Array;
  },
): Promise<SubmissionResult> {
  try {
    const anchor = await import("@coral-xyz/anchor");
    const {
      ComputeBudgetProgram,
      PublicKey,
      SYSVAR_INSTRUCTIONS_PUBKEY,
      SystemProgram,
      Transaction,
    } = await import("@solana/web3.js");
    const provider = new anchor.AnchorProvider(
      options.connection as ConstructorParameters<typeof anchor.AnchorProvider>[0],
      options.wallet as unknown as Wallet,
      { commitment: "confirmed" },
    );
    const anchorProgramId = new PublicKey(PROGRAM_IDS.entrosAnchor);
    const registryProgramId = new PublicKey(PROGRAM_IDS.entrosRegistry);
    const [identityPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("identity"), provider.wallet.publicKey.toBuffer()],
      anchorProgramId,
    );
    const [protocolConfigPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("protocol_config")],
      registryProgramId,
    );
    const [treasuryPda] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("protocol_treasury")],
      registryProgramId,
    );
    const anchorProgram = new anchor.Program(
      entrosAnchorIdl as Idl,
      provider,
    ) as unknown as RebaselineProgram;

    if (
      !receiptMatchesBinding(options.signedReceipt, {
        purpose: 2,
        projectionVersion,
        wallet: provider.wallet.publicKey.toBytes(),
        commitment,
      })
    ) {
      return {
        success: false,
        error: "Projection migration requires a matching validator-signed receipt.",
        failedAt: "submission",
      };
    }
    const receiptIx = await buildEd25519ReceiptIx(options.signedReceipt);
    if (!receiptIx) {
      return {
        success: false,
        error: "Projection migration receipt is malformed.",
        failedAt: "submission",
      };
    }
    const rebaselineIx = await anchorProgram.methods
      .rebaselineAnchor(Array.from(commitment), projectionVersion)
      .accounts({
        authority: provider.wallet.publicKey,
        identityState: identityPda,
        protocolConfig: protocolConfigPda,
        treasury: treasuryPda,
        instructionsSysvar: SYSVAR_INSTRUCTIONS_PUBKEY,
        systemProgram: SystemProgram.programId,
      })
      .instruction();
    const setBaselineIx = await buildSetEncryptedBaselineIx(
      anchorProgram,
      provider.wallet.publicKey,
      options.encryptedBaselineBlob,
    );

    const tx = new Transaction().add(
      ComputeBudgetProgram.setComputeUnitLimit({ units: 200_000 }),
      receiptIx,
      rebaselineIx,
      setBaselineIx,
    );
    tx.feePayer = provider.wallet.publicKey;
    tx.recentBlockhash = (
      await options.connection.getLatestBlockhash("confirmed")
    ).blockhash;
    const sent = await sendAndConfirm(options.wallet, options.connection, tx);
    return sent.ok
      ? { success: true, txSignature: sent.txSig }
      : { success: false, error: sent.error, failedAt: sent.failedAt };
  } catch (err) {
    return { success: false, error: errToString(err), failedAt: "submission" };
  }
}
