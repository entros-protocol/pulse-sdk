import type { SolanaProof } from "../proof/types";
import type { SubmissionResult } from "./types";

const RELAYER_TIMEOUT_MS = 30_000;

/**
 * Submit a proof via the Entros relayer API (walletless mode).
 * The relayer submits the on-chain transaction using the integrator's funded account.
 * The user needs no wallet, no SOL, no crypto knowledge.
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

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), RELAYER_TIMEOUT_MS);

    const response = await fetch(options.relayerUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timer);

    if (!response.ok) {
      const errorText = await response.text();
      return {
        success: false,
        error: `Relayer returned HTTP ${response.status} from ${options.relayerUrl}: ${errorText}. Check relayerUrl and apiKey in PulseConfig.`,
      };
    }

    const result = (await response.json()) as {
      success?: boolean;
      tx_signature?: string;
      verified?: boolean;
      registered?: boolean;
    };

    if (result.success !== true) {
      return {
        success: false,
        error: "Relayer accepted the request but reported failure. Typically means proof verification failed on-chain — check the relayer logs.",
      };
    }

    return {
      success: true,
      txSignature: result.tx_signature,
    };
  } catch (err: any) {
    if (err.name === "AbortError") {
      return {
        success: false,
        error: `Relayer request timed out after ${RELAYER_TIMEOUT_MS / 1000}s. Check network connectivity and relayerUrl reachability.`,
      };
    }
    return { success: false, error: err.message ?? String(err) };
  }
}
