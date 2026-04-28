/* eslint-disable @typescript-eslint/no-explicit-any */
// Anchor program interactions use runtime IDL fetching, requiring dynamic typing.
import type { SolanaProof } from "../proof/types";
import type { SubmissionResult } from "./types";
import { PROGRAM_IDS } from "../config";
import { sdkLog, sdkWarn } from "../log";

/**
 * Best-effort SAS attestation request. POSTs to the executor's `/attest`
 * endpoint with the wallet's public key, an optional server-issued challenge
 * nonce, and an `Entros-ATTEST:{wallet}:{timestamp}` ownership signature.
 *
 * Returns the attestation tx signature on success, `undefined` on any
 * failure (attestation is non-fatal — the on-chain tx has already confirmed
 * by the time this is called).
 */
async function requestSasAttestation(
  wallet: any,
  walletAddress: string,
  relayerUrl: string,
  relayerApiKey: string | undefined,
  serverNonce: number[] | undefined,
): Promise<string | undefined> {
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

    const attestBody: Record<string, unknown> = { wallet_address: walletAddress };
    if (serverNonce) attestBody.nonce = serverNonce;

    if (wallet?.signMessage) {
      try {
        const timestamp = Math.floor(Date.now() / 1000);
        const attestMessage = `Entros-ATTEST:${walletAddress}:${timestamp}`;
        const messageBytes = new TextEncoder().encode(attestMessage);
        const sigBytes: Uint8Array = await wallet.signMessage(messageBytes);
        const sigHex = Array.from(sigBytes)
          .map((b) => b.toString(16).padStart(2, "0"))
          .join("");
        attestBody.signature = sigHex;
        attestBody.message = attestMessage;
      } catch {
        sdkWarn("Wallet signMessage failed, skipping ownership proof");
      }
    }

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
  }
): Promise<SubmissionResult> {
  try {
    const anchor = await import("@coral-xyz/anchor");
    const { PublicKey, SystemProgram, Transaction, ComputeBudgetProgram } =
      await import("@solana/web3.js");

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
      // Re-verification: batch create_challenge + verify_proof + update_anchor
      // into a single transaction (1 wallet prompt instead of 3)
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
            const challengeData = (await challengeRes.json()) as { nonce?: number[] };
            if (challengeData.nonce && challengeData.nonce.length === 32) {
              nonce = challengeData.nonce;
              serverNonce = true;
              sdkLog("Using server-generated challenge nonce");
            } else {
              nonce = Array.from(crypto.getRandomValues(new Uint8Array(32)));
              sdkWarn("Server returned invalid nonce, using client-generated");
            }
          } else {
            nonce = Array.from(crypto.getRandomValues(new Uint8Array(32)));
            sdkWarn("Challenge endpoint returned error, using client-generated nonce");
          }
        } catch {
          nonce = Array.from(crypto.getRandomValues(new Uint8Array(32)));
          sdkWarn("Challenge fetch failed, using client-generated nonce");
        }
      } else {
        nonce = Array.from(crypto.getRandomValues(new Uint8Array(32)));
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

      // Fetch both IDLs
      const [verifierIdl, anchorIdl] = await Promise.all([
        anchor.Program.fetchIdl(verifierProgramId, provider),
        anchor.Program.fetchIdl(anchorProgramId, provider),
      ]);
      if (!verifierIdl) {
        return {
          success: false,
          error: `Failed to fetch entros-verifier IDL from Solana (program ${PROGRAM_IDS.entrosVerifier}). Check your RPC endpoint is reachable and on the correct cluster.`,
        };
      }
      if (!anchorIdl) {
        return {
          success: false,
          error: `Failed to fetch entros-anchor IDL from Solana (program ${PROGRAM_IDS.entrosAnchor}). Check your RPC endpoint is reachable and on the correct cluster.`,
        };
      }

      const verifierProgram: any = new anchor.Program(verifierIdl, provider);
      const anchorProgram: any = new anchor.Program(anchorIdl, provider);
      const { Buffer: SolBuffer } = await import("buffer");

      // Build all three instructions without sending
      const createChallengeIx = await verifierProgram.methods
        .createChallenge(nonce)
        .accounts({
          challenger: provider.wallet.publicKey,
          challenge: challengePda,
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      // Anchor 0.32.1 uses buffer-layout v1.2 which requires Node.js Buffer
      // (not Uint8Array) for Blob.encode on Vec<u8> fields.
      const verifyProofIx = await verifierProgram.methods
        .verifyProof(
          SolBuffer.from(proof.proofBytes),
          proof.publicInputs.map((pi) => SolBuffer.from(pi)),
          nonce
        )
        .accounts({
          verifier: provider.wallet.publicKey,
          challenge: challengePda,
          verificationResult: verificationPda,
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      // updateAnchor post-2026-04-20 binding patch takes the verification
      // nonce as a second arg and requires the VerificationResult PDA as an
      // account. Without these, the instruction would accept any commitment
      // with no biometric proof — see protocol-core AUDIT.md for details.
      const updateAnchorIx = await anchorProgram.methods
        .updateAnchor(Array.from(commitment), nonce)
        .accounts({
          authority: provider.wallet.publicKey,
          identityState: identityPda,
          verificationResult: verificationPda,
          protocolConfig: protocolConfigPda,
          treasury: treasuryPda,
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      // Batch: compute budget + 3 program instructions → 1 wallet prompt
      // Total CU ~205K; request 250K to exceed the 200K default limit.
      const tx = new Transaction();
      tx.add(ComputeBudgetProgram.setComputeUnitLimit({ units: 250_000 }));
      tx.add(createChallengeIx);
      tx.add(verifyProofIx);
      tx.add(updateAnchorIx);

      tx.feePayer = provider.wallet.publicKey;
      tx.recentBlockhash = (
        await options.connection.getLatestBlockhash("confirmed")
      ).blockhash;

      txSig = await options.wallet.sendTransaction(tx, options.connection, {
        skipPreflight: true,
      });
      await options.connection.confirmTransaction(txSig, "confirmed");
    } else {
      // First verification: mint anchor (already 1 transaction, no batching needed)
      const anchorIdl = await anchor.Program.fetchIdl(anchorProgramId, provider);
      if (!anchorIdl) {
        return {
          success: false,
          error: `Failed to fetch entros-anchor IDL from Solana (program ${PROGRAM_IDS.entrosAnchor}). Check your RPC endpoint is reachable and on the correct cluster.`,
        };
      }

      const anchorProgram: any = new anchor.Program(anchorIdl, provider);

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

      const TOKEN_2022_PROGRAM_ID = new PublicKey(
        "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
      );

      const { getAssociatedTokenAddressSync } = await import("@solana/spl-token");
      const ata = getAssociatedTokenAddressSync(
        mintPda,
        provider.wallet.publicKey,
        false,
        TOKEN_2022_PROGRAM_ID
      );

      await anchorProgram.methods
        .mintAnchor(Array.from(commitment))
        .accounts({
          user: provider.wallet.publicKey,
          identityState: identityPda,
          mint: mintPda,
          mintAuthority,
          tokenAccount: ata,
          associatedTokenProgram: new PublicKey(
            "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
          ),
          tokenProgram: TOKEN_2022_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
          protocolConfig: protocolConfigPda,
          treasury: treasuryPda,
        })
        .rpc();
    }

    const attestationTx = options.relayerUrl
      ? await requestSasAttestation(
          options.wallet,
          provider.wallet.publicKey.toBase58(),
          options.relayerUrl,
          options.relayerApiKey,
          serverNonce ? nonce : undefined,
        )
      : undefined;

    return { success: true, txSignature: txSig, attestationTx };
  } catch (err: any) {
    return { success: false, error: err.message ?? String(err) };
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
 * ZK proof required). Humanness evidence comes from the Tier 1
 * validation pipeline invoked at the /attest step (same as mint and
 * update).
 */
export async function submitResetViaWallet(
  commitment: Uint8Array,
  options: {
    wallet: any;
    connection: any;
    relayerUrl?: string;
    relayerApiKey?: string;
  }
): Promise<SubmissionResult> {
  try {
    const anchor = await import("@coral-xyz/anchor");
    const { PublicKey, SystemProgram, Transaction, ComputeBudgetProgram } =
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

    const anchorIdl = await anchor.Program.fetchIdl(anchorProgramId, provider);
    if (!anchorIdl) {
      return {
        success: false,
        error: `Failed to fetch entros-anchor IDL from Solana (program ${PROGRAM_IDS.entrosAnchor}). Check your RPC endpoint is reachable and on the correct cluster.`,
      };
    }
    const anchorProgram: any = new anchor.Program(anchorIdl, provider);

    const resetIx = await anchorProgram.methods
      .resetIdentityState(Array.from(commitment))
      .accounts({
        authority: provider.wallet.publicKey,
        identityState: identityPda,
        protocolConfig: protocolConfigPda,
        treasury: treasuryPda,
        systemProgram: SystemProgram.programId,
      })
      .instruction();

    // Reset does no ZK verification; budget is well under the 200K default.
    // Keep an explicit limit for determinism and to match batched-tx ergonomics.
    const tx = new Transaction();
    tx.add(ComputeBudgetProgram.setComputeUnitLimit({ units: 150_000 }));
    tx.add(resetIx);

    tx.feePayer = provider.wallet.publicKey;
    tx.recentBlockhash = (
      await options.connection.getLatestBlockhash("confirmed")
    ).blockhash;

    const txSig: string = await options.wallet.sendTransaction(
      tx,
      options.connection,
      { skipPreflight: true }
    );
    await options.connection.confirmTransaction(txSig, "confirmed");

    // Request a fresh SAS attestation. The executor's /attest handler
    // closes any prior attestation for this wallet and creates a new one
    // bound to the current commitment.
    const attestationTx = options.relayerUrl
      ? await requestSasAttestation(
          options.wallet,
          provider.wallet.publicKey.toBase58(),
          options.relayerUrl,
          options.relayerApiKey,
          undefined,
        )
      : undefined;

    return { success: true, txSignature: txSig, attestationTx };
  } catch (err: any) {
    return { success: false, error: err.message ?? String(err) };
  }
}
