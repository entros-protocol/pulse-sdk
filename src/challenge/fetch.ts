/**
 * Fetch the server-issued challenge from the executor.
 *
 * The executor's `/challenge` endpoint returns a fresh nonce + 5-word phrase
 * bound to the wallet for a short TTL (default 60s). The phrase is drawn from
 * a curated English-word dictionary (source of truth at
 * `entros-validation/src/word_dict.rs`); shown to the user as the voice challenge
 * and looked up server-side at `/validate-features` to verify the audio
 * matches the issued phrase (master-list #89, phrase content binding).
 *
 * Server-issued phrases are the only safe design for content binding: if the
 * client generated the phrase and sent it to the server alongside the audio,
 * an attacker would submit their own phrase matching whatever content they
 * captured. With server issuance, the phrase is bound to the nonce and the
 * client cannot substitute it.
 */

import { sdkWarn } from "../log";

/**
 * Server-issued challenge artifacts. Returned by `fetchChallenge`.
 */
export interface ChallengeResponse {
  /** 32-byte nonce used for on-chain `create_challenge` and the `/attest` handshake. */
  nonce: Uint8Array;
  /** Server-issued 5-word challenge phrase (drawn from a curated English-word dictionary) the user must speak aloud. */
  phrase: string;
  /** Nonce TTL in seconds (default 60). */
  expiresIn: number;
}

/**
 * Fetch a fresh nonce + phrase from the executor. Throws on network error or
 * non-2xx response so the caller can surface a retry UX.
 *
 * @param executorUrl - Base URL of the executor (e.g. `https://executor.entros.io`).
 * @param walletAddress - Base58-encoded wallet public key.
 * @param apiKey - Optional executor API key (`X-API-Key` header).
 */
export async function fetchChallenge(
  executorUrl: string,
  walletAddress: string,
  apiKey?: string,
): Promise<ChallengeResponse> {
  const base = new URL(executorUrl);
  const url = new URL("/challenge", base.origin);
  url.searchParams.set("wallet", walletAddress);

  const headers: Record<string, string> = { Accept: "application/json" };
  if (apiKey) headers["X-API-Key"] = apiKey;

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5_000);
  let response: Response;
  try {
    response = await fetch(url.toString(), {
      method: "GET",
      headers,
      signal: controller.signal,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    sdkWarn(`[Entros SDK] /challenge fetch failed: ${msg}`);
    throw new Error(`Unable to fetch challenge from executor: ${msg}`);
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    throw new Error(
      `Executor returned ${response.status} for /challenge. Check the wallet address and try again.`,
    );
  }

  const body = (await response.json()) as {
    nonce: number[];
    expires_in: number;
    phrase: string;
  };

  if (!Array.isArray(body.nonce) || body.nonce.length !== 32) {
    throw new Error("Executor returned malformed nonce; expected 32-byte array");
  }
  if (typeof body.phrase !== "string" || body.phrase.trim().length === 0) {
    throw new Error("Executor returned empty challenge phrase");
  }

  return {
    nonce: Uint8Array.from(body.nonce),
    phrase: body.phrase,
    expiresIn: body.expires_in ?? 60,
  };
}
