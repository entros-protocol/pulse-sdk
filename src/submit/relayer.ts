import type { SolanaProof } from "../proof/types";
import type { SubmissionResult } from "./types";
import { errToString } from "./errors";

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

    // Every failure below reports `submission`, never `confirmation`.
    //
    // The relayer builds, signs and sends on the user's behalf, so the SDK
    // never observes the cluster directly and cannot tell a transaction that
    // reverted from one that was never sent. `submission` says the outcome is
    // unknown, which is the truth here. The relayer also pays the fee in this
    // mode, so a host on the walletless path should not render spend copy at
    // all: `phaseSpend` describes what the signer may have paid, and the
    // signer is not the user.
    if (!response.ok) {
      const errorText = await response.text();
      return {
        success: false,
        error: `Relayer returned HTTP ${response.status} from ${options.relayerUrl}: ${errorText}. Check relayerUrl and apiKey in PulseConfig.`,
        failedAt: "submission",
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
        error: "Relayer accepted the request but reported failure. Typically means proof verification failed on-chain. Check the relayer logs.",
        failedAt: "submission",
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
        failedAt: "submission",
      };
    }
    return { success: false, error: errToString(err), failedAt: "submission" };
  }
}
