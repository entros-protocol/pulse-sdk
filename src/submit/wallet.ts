/* eslint-disable @typescript-eslint/no-explicit-any */
// Anchor program interactions use runtime IDL fetching, requiring dynamic typing.
import type { SolanaProof } from "../proof/types";
import type { SubmissionResult } from "./types";
import { PROGRAM_IDS } from "../config";

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

    const anchorProgramId = new PublicKey(PROGRAM_IDS.iamAnchor);

    let txSig: string | undefined;

    if (!options.isFirstVerification) {
      // Re-verification: batch create_challenge + verify_proof + update_anchor
      // into a single transaction (1 wallet prompt instead of 3)
      const verifierProgramId = new PublicKey(PROGRAM_IDS.iamVerifier);

      const nonce = Array.from(crypto.getRandomValues(new Uint8Array(32)));

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

      const registryProgramId = new PublicKey(PROGRAM_IDS.iamRegistry);
      const [protocolConfigPda] = PublicKey.findProgramAddressSync(
        [new TextEncoder().encode("protocol_config")],
        registryProgramId
      );

      // Fetch both IDLs
      const [verifierIdl, anchorIdl] = await Promise.all([
        anchor.Program.fetchIdl(verifierProgramId, provider),
        anchor.Program.fetchIdl(anchorProgramId, provider),
      ]);
      if (!verifierIdl) {
        return { success: false, error: "Failed to fetch verifier IDL" };
      }
      if (!anchorIdl) {
        return { success: false, error: "Failed to fetch IAM Anchor program IDL" };
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

      const updateAnchorIx = await anchorProgram.methods
        .updateAnchor(Array.from(commitment))
        .accounts({
          authority: provider.wallet.publicKey,
          identityState: identityPda,
          protocolConfig: protocolConfigPda,
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
        return { success: false, error: "Failed to fetch IAM Anchor program IDL" };
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
        })
        .rpc();
    }

    // Request SAS attestation from executor (best-effort, non-fatal)
    let attestationTx: string | undefined;
    if (options.relayerUrl) {
      try {
        const attestHeaders: Record<string, string> = {
          "Content-Type": "application/json",
        };
        if (options.relayerApiKey) {
          attestHeaders["X-API-Key"] = options.relayerApiKey;
        }

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 15_000);

        // Derive base URL from relayerUrl (which may include a path like /verify)
        const baseUrl = new URL(options.relayerUrl);
        const attestUrl = `${baseUrl.origin}/attest`;

        const attestRes = await fetch(attestUrl, {
          method: "POST",
          headers: attestHeaders,
          body: JSON.stringify({
            wallet_address: provider.wallet.publicKey.toBase58(),
          }),
          signal: controller.signal,
        });

        clearTimeout(timer);

        if (attestRes.ok) {
          const attestData = (await attestRes.json()) as {
            success?: boolean;
            attestation_tx?: string;
          };
          if (attestData.success && attestData.attestation_tx) {
            attestationTx = attestData.attestation_tx;
          }
        }
      } catch {
        // Attestation is best-effort; verification already succeeded
      }
    }

    return { success: true, txSignature: txSig, attestationTx };
  } catch (err: any) {
    return { success: false, error: err.message ?? String(err) };
  }
}
