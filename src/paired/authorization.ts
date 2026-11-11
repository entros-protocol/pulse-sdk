/**
 * The wallet's authorization for paired sessions.
 *
 * A paired open and a paired finalize both name a wallet. The relayer serves
 * them only for the wallet's owner, so the wallet signs one challenge from the
 * relayer and the client sends that signed authorization with every open and
 * finalize until it expires. One signature covers a whole visit.
 */

import { ed25519 } from "@noble/curves/ed25519";
import { canonicalKeyBytes } from "../agent/base58";
import { SIGNATURE_TIMEOUT_MS } from "../config";
import { withTimeout } from "../submit/errors";
import { PairedProtocolError, refusalOf } from "./client";
import { toHex } from "./transcript";

/**
 * An authorization with less than this left is treated as absent, so no
 * capture starts on one that expires before the session opens.
 */
export const PAIRED_AUTHORIZATION_MARGIN_MS = 120_000;

/**
 * A grant fetched ahead of the wallet prompt is signed only with at least this
 * much left, so the signed authorization still has a minute of use past the margin.
 */
export const PAIRED_GRANT_REUSE_MS = PAIRED_AUTHORIZATION_MARGIN_MS + 60_000;

/** The longest lifetime a relayer may give an authorization. */
const MAX_LIFETIME_SECS = 3_600;
const REQUEST_MS = 10_000;

/**
 * Both clocks read at one moment. `performance.now()` does not run while a
 * device sleeps on some platforms, and `Date.now()` can be set back, so a
 * lifetime counts whichever of the two has moved further.
 */
export interface PairedClockReading {
  monotonicMs: number;
  wallMs: number;
}

export function readPairedClock(): PairedClockReading {
  return { monotonicMs: performance.now(), wallMs: Date.now() };
}

/** When a grant was requested and how long the relayer said it lasts. */
export interface PairedLifetime {
  /** `performance.now()` before the grant was requested. */
  readonly requestedAtMs: number;
  /** `Date.now()` at the same moment. */
  readonly requestedAtWallMs: number;
  /** The lifetime the relayer gave, in milliseconds. */
  readonly lifetimeMs: number;
}

/**
 * Milliseconds left of `lifetime`. Only the time each clock moved counts, so a
 * wall clock that is wrong by a constant changes nothing, and one that moved
 * backwards leaves the monotonic count in charge.
 */
export function pairedLifetimeRemainingMs(
  lifetime: PairedLifetime,
  now: PairedClockReading = readPairedClock(),
): number {
  const elapsedMs = Math.max(
    now.monotonicMs - lifetime.requestedAtMs,
    now.wallMs - lifetime.requestedAtWallMs,
  );
  return lifetime.lifetimeMs - elapsedMs;
}

/** What the relayer issues for the wallet to sign. */
export interface PairedAuthorizationGrant {
  challenge: string;
  expiresAt: number;
  expiresInSecs: number;
  origin: string;
  mac: string;
}

/** A grant the wallet has signed. The lifetime fields stay on the device. */
export interface PairedAuthorization extends PairedLifetime {
  /** The wallet it authorizes, in base58. */
  readonly wallet: string;
  readonly challenge: string;
  readonly expiresAt: number;
  readonly origin: string;
  readonly mac: string;
  readonly signatureHex: string;
}

/** The minimal wallet surface the authorization needs. */
export interface PairedAuthorizationWallet {
  publicKey: { toBase58(): string };
  signMessage(message: Uint8Array): Promise<Uint8Array>;
}

type Json = Record<string, unknown>;

/**
 * The text the wallet signs. The client builds it from the grant's fields and
 * never signs text a server supplies whole.
 */
export function pairedAuthorizationMessage(
  wallet: string,
  challenge: string,
  expiresAt: number,
): string {
  return (
    "Entros verification\n" +
    "Sign to start verification sessions for this wallet. The authorization expires within the hour. " +
    "It moves no funds and approves no transaction.\n" +
    "\n" +
    `Entros-Paired-Open-v1\nwallet:${wallet}\nchallenge:${challenge}\nexpires:${expiresAt}`
  );
}

const isLowerHex = (value: unknown, bytes: number): value is string =>
  typeof value === "string" && new RegExp(`^[0-9a-f]{${bytes * 2}}$`).test(value);

/** Reads a grant and refuses any field outside its fixed form. */
export function parsePairedAuthorizationGrant(value: unknown): PairedAuthorizationGrant {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new PairedProtocolError("malformed_response");
  }
  const grant = value as Json;
  const { challenge, origin, mac, expires_at: expiresAt, expires_in: expiresIn } = grant;
  if (
    !isLowerHex(challenge, 32) ||
    !isLowerHex(origin, 16) ||
    !isLowerHex(mac, 32) ||
    typeof expiresAt !== "number" ||
    !Number.isSafeInteger(expiresAt) ||
    expiresAt <= 0 ||
    typeof expiresIn !== "number" ||
    !Number.isSafeInteger(expiresIn) ||
    // A grant that would be inside the margin from the start can never be used.
    expiresIn * 1_000 <= PAIRED_AUTHORIZATION_MARGIN_MS ||
    expiresIn > MAX_LIFETIME_SECS
  ) {
    throw new PairedProtocolError("malformed_response");
  }
  return { challenge, expiresAt, expiresInSecs: expiresIn, origin, mac };
}

/** The `authorization` object of an open or a finalize request. */
export function pairedAuthorizationBody(authorization: PairedAuthorization): Json {
  return {
    challenge: authorization.challenge,
    expires_at: authorization.expiresAt,
    origin: authorization.origin,
    mac: authorization.mac,
    signature_hex: authorization.signatureHex,
  };
}

/** True while `authorization` is for `wallet` and has at least the margin left. */
export function pairedAuthorizationUsable(
  authorization: PairedAuthorization | null | undefined,
  wallet: string,
  now: PairedClockReading = readPairedClock(),
): authorization is PairedAuthorization {
  return (
    !!authorization &&
    authorization.wallet === wallet &&
    pairedLifetimeRemainingMs(authorization, now) >= PAIRED_AUTHORIZATION_MARGIN_MS
  );
}

/** Requests a grant from the relayer's `/paired/authorization`. */
export async function requestPairedAuthorizationGrant(
  relayerUrl: string,
  wallet: string,
  apiKey?: string,
): Promise<unknown> {
  const url = new URL("/paired/authorization", new URL(relayerUrl).origin);
  url.searchParams.set("wallet", wallet);
  const headers: Record<string, string> = { Accept: "application/json" };
  if (apiKey) headers["X-API-Key"] = apiKey;
  const controller = new AbortController();
  // The deadline covers the body as well as the headers. A body that stalls
  // after the headers would otherwise hold the request open for good.
  const timer = setTimeout(() => controller.abort(), REQUEST_MS);
  const aborted = new Promise<never>((_, reject) => {
    controller.signal.addEventListener(
      "abort",
      () => reject(new PairedProtocolError("validation_unavailable")),
      { once: true },
    );
  });
  // Nothing awaits it when the request settles first.
  aborted.catch(() => undefined);
  let response: Response;
  let body: Json = {};
  try {
    try {
      response = await Promise.race([
        fetch(url.toString(), { method: "GET", headers, signal: controller.signal }),
        aborted,
      ]);
    } catch {
      throw new PairedProtocolError("validation_unavailable");
    }
    try {
      const parsed: unknown = await Promise.race([response.json(), aborted]);
      if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
        body = parsed as Json;
      }
    } catch {
      if (controller.signal.aborted) {
        response.body?.cancel().catch(() => undefined);
        throw new PairedProtocolError("validation_unavailable");
      }
      // A body that is not JSON carries no reason. The status names the refusal.
    }
  } finally {
    clearTimeout(timer);
  }
  if (!response.ok) {
    throw refusalOf({
      status: response.status,
      body,
      header: (name) => response.headers.get(name),
    });
  }
  return body;
}

/**
 * Has the wallet sign a grant. `requestedAt` is the clock reading from before
 * the grant was requested, so the lifetime the client counts never runs past
 * the one the relayer counts. `address` is the wallet the grant was requested
 * for. A wallet that now reports another key is refused before any prompt.
 *
 * The wallet is asked before the call awaits anything, so a caller that holds
 * the grant already keeps a browser's user gesture in force.
 * Rejects when the prompt does not settle while the grant is usable, when the
 * signature does not verify under the wallet's key, and when the prompt stayed
 * open until the authorization was no longer usable.
 */
export async function signPairedAuthorization(
  wallet: PairedAuthorizationWallet,
  grant: PairedAuthorizationGrant,
  requestedAt: PairedClockReading,
  address: string,
): Promise<PairedAuthorization> {
  const key = canonicalKeyBytes(address);
  if (!key || wallet.publicKey.toBase58() !== address) {
    throw new PairedProtocolError("invalid_request");
  }
  const lifetime: PairedLifetime = {
    requestedAtMs: requestedAt.monotonicMs,
    requestedAtWallMs: requestedAt.wallMs,
    lifetimeMs: grant.expiresInSecs * 1_000,
  };
  const usableMs = pairedLifetimeRemainingMs(lifetime) - PAIRED_AUTHORIZATION_MARGIN_MS;
  if (usableMs <= 0) throw new PairedProtocolError("authorization_required");
  const message = new TextEncoder().encode(
    pairedAuthorizationMessage(address, grant.challenge, grant.expiresAt),
  );
  const signature = await withTimeout(
    wallet.signMessage(message),
    Math.min(SIGNATURE_TIMEOUT_MS, usableMs),
    "Your wallet did not respond to the authorization request. Open your wallet and try again.",
  );
  if (!(signature instanceof Uint8Array) || signature.length !== 64) {
    throw new PairedProtocolError("invalid_request");
  }
  let verified = false;
  try {
    verified = ed25519.verify(signature, message, key);
  } catch {
    verified = false;
  }
  if (!verified) throw new PairedProtocolError("invalid_request");
  const authorization: PairedAuthorization = {
    wallet: address,
    challenge: grant.challenge,
    expiresAt: grant.expiresAt,
    origin: grant.origin,
    mac: grant.mac,
    signatureHex: toHex(signature),
    ...lifetime,
  };
  // The prompt may have stayed open until too little of the lifetime was left.
  if (!pairedAuthorizationUsable(authorization, address)) {
    throw new PairedProtocolError("authorization_required");
  }
  return authorization;
}

/** True for a refusal that means the authorization must be signed again. */
export function isPairedAuthorizationRefusal(reason: string | undefined): boolean {
  return reason === "authorization_required" || reason === "authorization_invalid";
}
