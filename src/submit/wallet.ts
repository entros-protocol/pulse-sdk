/* eslint-disable @typescript-eslint/no-explicit-any */
// Anchor program interactions use runtime IDL fetching, requiring dynamic typing.
import type { SolanaProof } from "../proof/types";
import type { SubmissionResult } from "./types";
import { PROGRAM_IDS } from "../config";

/**
 * Submit a proof on-chain via a connected wallet (wallet-connected mode).
 * Uses Anchor SDK to construct and send the transaction.
 *
 * Flow: create_challenge → verify_proof → update_anchor (or mint_anchor for first time)
 */
export async function submitViaWallet(
  proof: SolanaProof,
  commitment: Uint8Array,
  options: {
    wallet: any; // WalletAdapter
    connection: any; // Connection
    isFirstVerification: boolean;
    trustScore?: number;
  }
): Promise<SubmissionResult> {
  try {
    const anchor = await import("@coral-xyz/anchor");
    const { PublicKey, SystemProgram } = await import("@solana/web3.js");

    const provider = new anchor.AnchorProvider(
      options.connection,
      options.wallet,
      { commitment: "confirmed" }
    );

    const verifierProgramId = new PublicKey(PROGRAM_IDS.iamVerifier);
    const anchorProgramId = new PublicKey(PROGRAM_IDS.iamAnchor);

    // Generate nonce for challenge
    const nonce = Array.from(crypto.getRandomValues(new Uint8Array(32)));

    // Derive PDAs
    const [challengePda] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("challenge"),
        provider.wallet.publicKey.toBuffer(),
        Buffer.from(nonce),
      ],
      verifierProgramId
    );

    const [verificationPda] = PublicKey.findProgramAddressSync(
      [
        Buffer.from("verification"),
        provider.wallet.publicKey.toBuffer(),
        Buffer.from(nonce),
      ],
      verifierProgramId
    );

    // Build and send create_challenge + verify_proof transactions
    // These use the raw Anchor program interface
    const verifierIdl = await anchor.Program.fetchIdl(
      verifierProgramId,
      provider
    );
    if (!verifierIdl) {
      return { success: false, error: "Failed to fetch verifier IDL" };
    }

    const verifierProgram: any = new anchor.Program(
      verifierIdl,
      provider
    );

    // 1. Create challenge
    await verifierProgram.methods
      .createChallenge(nonce)
      .accounts({
        challenger: provider.wallet.publicKey,
        challenge: challengePda,
        systemProgram: SystemProgram.programId,
      })
      .rpc();

    // 2. Verify proof
    const txSig = await verifierProgram.methods
      .verifyProof(
        Buffer.from(proof.proofBytes),
        proof.publicInputs.map((pi) => Array.from(pi)),
        nonce
      )
      .accounts({
        verifier: provider.wallet.publicKey,
        challenge: challengePda,
        verificationResult: verificationPda,
        systemProgram: SystemProgram.programId,
      })
      .rpc();

    // 3. Mint or update anchor
    const anchorIdl = await anchor.Program.fetchIdl(anchorProgramId, provider);
    if (!anchorIdl) {
      return { success: false, error: "Failed to fetch IAM Anchor program IDL" };
    }

    {
      const anchorProgram: any = new anchor.Program(anchorIdl, provider);

      if (options.isFirstVerification) {
        const [identityPda] = PublicKey.findProgramAddressSync(
          [Buffer.from("identity"), provider.wallet.publicKey.toBuffer()],
          anchorProgramId
        );
        const [mintPda] = PublicKey.findProgramAddressSync(
          [Buffer.from("mint"), provider.wallet.publicKey.toBuffer()],
          anchorProgramId
        );
        const [mintAuthority] = PublicKey.findProgramAddressSync(
          [Buffer.from("mint_authority")],
          anchorProgramId
        );

        // Token-2022 program ID
        const TOKEN_2022_PROGRAM_ID = new PublicKey(
          "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
        );

        const { getAssociatedTokenAddressSync } = await import(
          "@solana/spl-token"
        );
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
      } else {
        const [identityPda] = PublicKey.findProgramAddressSync(
          [Buffer.from("identity"), provider.wallet.publicKey.toBuffer()],
          anchorProgramId
        );

        await anchorProgram.methods
          .updateAnchor(Array.from(commitment), options.trustScore ?? 0)
          .accounts({
            authority: provider.wallet.publicKey,
            identityState: identityPda,
          })
          .rpc();
      }
    }

    return { success: true, txSignature: txSig };
  } catch (err: any) {
    return { success: false, error: err.message ?? String(err) };
  }
}
