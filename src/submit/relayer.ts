import type { SolanaProof } from "../proof/types";
import type { SubmissionResult } from "./types";

/**
 * Submit a proof via the IAM relayer API (walletless mode).
 * The relayer submits the on-chain transaction using the integrator's funded account.
 * The user needs no wallet, no SOL, no crypto knowledge.
 *
 * In Phase 3, the relayer endpoint is configurable (stub until executor-node in Phase 4).
 */
export async function submitViaRelayer(
  proof: SolanaProof,
  commitment: Uint8Array,
  options: {
    relayerUrl: string;
    apiKey?: string;
    isFirstVerification: boolean;
  }
): Promise<SubmissionResult> {
  try {
    const body = {
      proof_bytes: Array.from(proof.proofBytes),
      public_inputs: proof.publicInputs.map((pi) => Array.from(pi)),
      commitment: Array.from(commitment),
      is_first_verification: options.isFirstVerification,
    };

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    if (options.apiKey) {
      headers["X-API-Key"] = options.apiKey;
    }

    const response = await fetch(options.relayerUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return { success: false, error: `Relayer error: ${response.status} ${errorText}` };
    }

    const result = (await response.json()) as {
      success?: boolean;
      tx_signature?: string;
    };
    return {
      success: result.success ?? true,
      txSignature: result.tx_signature,
    };
  } catch (err: any) {
    return { success: false, error: err.message ?? String(err) };
  }
}
