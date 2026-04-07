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
    wallet: any;
    connection: any;
    isFirstVerification: boolean;
    relayerUrl?: string;
    relayerApiKey?: string;
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

    // For re-verification: create challenge → verify proof → update anchor
    // For first verification: skip challenge/proof → mint anchor directly
    let txSig: string | undefined;
    const { Buffer: SolBuffer } = await import("buffer");

    if (!options.isFirstVerification) {
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
      // Anchor 0.32.1 uses buffer-layout v1.2 which requires Node.js Buffer
      // (not Uint8Array) for Blob.encode on Vec<u8> fields.
      txSig = await verifierProgram.methods
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
        .rpc();
    }

    // 3. Mint or update anchor
    const anchorIdl = await anchor.Program.fetchIdl(anchorProgramId, provider);
    if (!anchorIdl) {
      return { success: false, error: "Failed to fetch IAM Anchor program IDL" };
    }

    {
      const anchorProgram: any = new anchor.Program(anchorIdl, provider);

      if (options.isFirstVerification) {
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
          [new TextEncoder().encode("identity"), provider.wallet.publicKey.toBuffer()],
          anchorProgramId
        );

        // Derive iam-registry ProtocolConfig PDA for trust score computation
        const registryProgramId = new PublicKey(PROGRAM_IDS.iamRegistry);
        const [protocolConfigPda] = PublicKey.findProgramAddressSync(
          [new TextEncoder().encode("protocol_config")],
          registryProgramId
        );

        await anchorProgram.methods
          .updateAnchor(Array.from(commitment))
          .accounts({
            authority: provider.wallet.publicKey,
            identityState: identityPda,
            protocolConfig: protocolConfigPda,
          })
          .rpc();
      }
    }

    // 4. Request SAS attestation from executor (best-effort, non-fatal)
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
