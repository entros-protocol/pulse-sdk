import { ed25519 } from "@noble/curves/ed25519";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  PAIRED_AUTHORIZATION_MARGIN_MS,
  pairedAuthorizationBody,
  pairedAuthorizationMessage,
  pairedAuthorizationUsable,
  parsePairedAuthorizationGrant,
  requestPairedAuthorizationGrant,
  signPairedAuthorization,
} from "../src/paired/authorization";
import { PairedProtocolError } from "../src/paired/client";
import { PulseSDK } from "../src/pulse";

/**
 * The values the executor and the mobile client pin in their own tests. A
 * change here is a protocol change for every client and for the server.
 */
const VECTOR = {
  seed: new Uint8Array(32).fill(7),
  wallet: "GmaDrppBC7P5ARKV8g3djiwP89vz1jLK23V2GBjuAEGB",
  challenge: "22".repeat(32),
  expiresAt: 1_800_000_000,
  origin: "4bd598d57ec9d5aacaffc4f0ca573ece",
  mac: "92dd5d5e281678bc041d28c30a9d49e2878f0e87b18710fec211dc70f61379fb",
  message:
    "Entros verification\n" +
    "Sign to start verification sessions for this wallet. The authorization expires within the hour. It moves no funds and approves no transaction.\n" +
    "\n" +
    "Entros-Paired-Open-v1\n" +
    "wallet:GmaDrppBC7P5ARKV8g3djiwP89vz1jLK23V2GBjuAEGB\n" +
    "challenge:2222222222222222222222222222222222222222222222222222222222222222\n" +
    "expires:1800000000",
  signatureHex:
    "a50c16c0bac5a61d9c017184d803de45e7451b782e25be1c2d684e6c6814cff0" +
    "bea246c5d0e2a4484a4798034f376c1a2db2498b75428045ba1a7cc803bd7a08",
};

const grantJson = (overrides: Record<string, unknown> = {}) => ({
  challenge: VECTOR.challenge,
  expires_at: VECTOR.expiresAt,
  expires_in: 900,
  origin: VECTOR.origin,
  mac: VECTOR.mac,
  ...overrides,
});

function vectorWallet() {
  const signed: Uint8Array[] = [];
  return {
    signed,
    wallet: {
      publicKey: { toBase58: () => VECTOR.wallet },
      signMessage: async (message: Uint8Array) => {
        signed.push(message);
        return ed25519.sign(message, VECTOR.seed);
      },
    },
  };
}

/** A reading of both clocks, `wall` milliseconds after the vector's request. */
const WALL_START = 1_799_999_000_000;
const at = (monotonicMs: number, wallMs = monotonicMs) => ({
  monotonicMs,
  wallMs: WALL_START + wallMs,
});

/** Drives `performance.now()` and `Date.now()` from one test-owned reading. */
function controlledClocks(start = at(0)) {
  const clock = { ...start };
  vi.spyOn(performance, "now").mockImplementation(() => clock.monotonicMs);
  vi.spyOn(Date, "now").mockImplementation(() => clock.wallMs);
  return clock;
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("paired authorization", () => {
  it("builds the signed text of the shared vector", () => {
    const message = pairedAuthorizationMessage(VECTOR.wallet, VECTOR.challenge, VECTOR.expiresAt);
    expect(message).toBe(VECTOR.message);
    expect(new TextEncoder().encode(message)).toHaveLength(331);
  });

  it("signs the shared vector and records when the grant was requested", async () => {
    const { wallet, signed } = vectorWallet();
    controlledClocks(at(5_000));
    const authorization = await signPairedAuthorization(
      wallet,
      parsePairedAuthorizationGrant(grantJson()),
      at(5_000),
      VECTOR.wallet,
    );
    expect(new TextDecoder().decode(signed[0])).toBe(VECTOR.message);
    expect(authorization).toEqual({
      wallet: VECTOR.wallet,
      challenge: VECTOR.challenge,
      expiresAt: VECTOR.expiresAt,
      origin: VECTOR.origin,
      mac: VECTOR.mac,
      signatureHex: VECTOR.signatureHex,
      requestedAtMs: 5_000,
      requestedAtWallMs: WALL_START + 5_000,
      lifetimeMs: 900_000,
    });
    expect(pairedAuthorizationBody(authorization)).toEqual({
      challenge: VECTOR.challenge,
      expires_at: VECTOR.expiresAt,
      origin: VECTOR.origin,
      mac: VECTOR.mac,
      signature_hex: VECTOR.signatureHex,
    });
  });

  it("refuses a grant with any field outside its fixed form", () => {
    expect(parsePairedAuthorizationGrant(grantJson())).toEqual({
      challenge: VECTOR.challenge,
      expiresAt: VECTOR.expiresAt,
      expiresInSecs: 900,
      origin: VECTOR.origin,
      mac: VECTOR.mac,
    });
    const malformed: unknown[] = [
      null,
      [],
      "grant",
      grantJson({ challenge: "AB".repeat(32) }),
      grantJson({ challenge: VECTOR.challenge.slice(2) }),
      grantJson({ challenge: `${VECTOR.challenge}\nexpires:9999999999` }),
      grantJson({ origin: "zz".repeat(16) }),
      grantJson({ mac: 7 }),
      grantJson({ expires_at: "1800000000" }),
      grantJson({ expires_at: 1.5 }),
      grantJson({ expires_at: 0 }),
      grantJson({ expires_in: 0 }),
      grantJson({ expires_in: 3_601 }),
      grantJson({ expires_in: undefined }),
    ];
    for (const value of malformed) {
      expect(() => parsePairedAuthorizationGrant(value)).toThrow(PairedProtocolError);
    }
  });

  it("refuses a grant whose lifetime does not exceed the client margin", () => {
    const marginSecs = PAIRED_AUTHORIZATION_MARGIN_MS / 1_000;
    for (const expiresIn of [1, marginSecs - 1, marginSecs]) {
      expect(() => parsePairedAuthorizationGrant(grantJson({ expires_in: expiresIn }))).toThrow(
        expect.objectContaining({ reason: "malformed_response" }),
      );
    }
    expect(parsePairedAuthorizationGrant(grantJson({ expires_in: marginSecs + 1 })).expiresInSecs).toBe(
      marginSecs + 1,
    );
  });

  it("refuses a wallet that returns anything but a 64-byte signature", async () => {
    controlledClocks();
    const grant = parsePairedAuthorizationGrant(grantJson());
    const wallet = {
      publicKey: { toBase58: () => VECTOR.wallet },
      signMessage: async () => new Uint8Array(63),
    };
    await expect(signPairedAuthorization(wallet, grant, at(0), VECTOR.wallet)).rejects.toMatchObject({
      reason: "invalid_request",
    });
    const unnamed = { ...wallet, publicKey: { toBase58: () => "not-a-wallet" } };
    await expect(signPairedAuthorization(unnamed, grant, at(0), "not-a-wallet")).rejects.toMatchObject({
      reason: "invalid_request",
    });
  });

  it("refuses a signature that does not verify under the wallet's key", async () => {
    controlledClocks();
    const grant = parsePairedAuthorizationGrant(grantJson());
    const wallet = {
      publicKey: { toBase58: () => VECTOR.wallet },
      signMessage: async (message: Uint8Array) => ed25519.sign(message, new Uint8Array(32).fill(8)),
    };
    await expect(signPairedAuthorization(wallet, grant, at(0), VECTOR.wallet)).rejects.toMatchObject({
      reason: "invalid_request",
    });
  });

  it("refuses to sign for a wallet other than the one the grant was requested for", async () => {
    controlledClocks();
    const { wallet, signed } = vectorWallet();
    await expect(
      signPairedAuthorization(
        wallet,
        parsePairedAuthorizationGrant(grantJson()),
        at(0),
        "11111111111111111111111111111111",
      ),
    ).rejects.toMatchObject({ reason: "invalid_request" });
    expect(signed).toHaveLength(0);
  });

  it("ends a wallet prompt that never settles in an error", async () => {
    vi.useFakeTimers();
    const wallet = {
      publicKey: { toBase58: () => VECTOR.wallet },
      signMessage: () => new Promise<Uint8Array>(() => {}),
    };
    let outcome: unknown = "pending";
    signPairedAuthorization(
      wallet,
      parsePairedAuthorizationGrant(grantJson()),
      { monotonicMs: performance.now(), wallMs: Date.now() },
      VECTOR.wallet,
    ).then(
      (value) => (outcome = value),
      (error: unknown) => (outcome = error),
    );
    await vi.advanceTimersByTimeAsync(780_001);
    expect(outcome).toBeInstanceOf(Error);
    expect((outcome as Error).name).toBe("EntrosTimeoutError");
  });

  it("refuses an authorization the wallet signed after it became unusable", async () => {
    const clock = controlledClocks();
    const wallet = {
      publicKey: { toBase58: () => VECTOR.wallet },
      signMessage: async (message: Uint8Array) => {
        // The prompt stayed open until less than the margin was left.
        clock.monotonicMs = 900_000 - PAIRED_AUTHORIZATION_MARGIN_MS + 1;
        clock.wallMs = WALL_START + clock.monotonicMs;
        return ed25519.sign(message, VECTOR.seed);
      },
    };
    await expect(
      signPairedAuthorization(wallet, parsePairedAuthorizationGrant(grantJson()), at(0), VECTOR.wallet),
    ).rejects.toMatchObject({ reason: "authorization_required" });
  });

  it("treats an authorization inside the margin, or for another wallet, as absent", async () => {
    controlledClocks();
    const { wallet } = vectorWallet();
    const authorization = await signPairedAuthorization(
      wallet,
      parsePairedAuthorizationGrant(grantJson()),
      at(0),
      VECTOR.wallet,
    );
    const lastUsableMs = 900_000 - PAIRED_AUTHORIZATION_MARGIN_MS;
    expect(pairedAuthorizationUsable(authorization, VECTOR.wallet, at(lastUsableMs))).toBe(true);
    expect(pairedAuthorizationUsable(authorization, VECTOR.wallet, at(lastUsableMs + 1))).toBe(false);
    expect(
      pairedAuthorizationUsable(authorization, "11111111111111111111111111111111", at(0)),
    ).toBe(false);
    expect(pairedAuthorizationUsable(null, VECTOR.wallet, at(0))).toBe(false);
  });

  it("counts the larger of the monotonic and the wall-clock time since the request", async () => {
    controlledClocks();
    const { wallet } = vectorWallet();
    const authorization = await signPairedAuthorization(
      wallet,
      parsePairedAuthorizationGrant(grantJson()),
      at(0),
      VECTOR.wallet,
    );
    const lastUsableMs = 900_000 - PAIRED_AUTHORIZATION_MARGIN_MS;
    // A device that slept: the monotonic clock paused while the wall clock ran on.
    expect(pairedAuthorizationUsable(authorization, VECTOR.wallet, at(1_000, lastUsableMs + 1))).toBe(
      false,
    );
    // A wall clock that moved backwards does not extend the lifetime.
    expect(pairedAuthorizationUsable(authorization, VECTOR.wallet, at(lastUsableMs + 1, -3_600_000))).toBe(
      false,
    );
    // A wall clock that is wrong by a constant has no effect: only its delta counts.
    const skewed = { ...authorization, requestedAtWallMs: authorization.requestedAtWallMs + 86_400_000 };
    expect(pairedAuthorizationUsable(skewed, VECTOR.wallet, at(lastUsableMs, lastUsableMs + 86_400_000))).toBe(
      true,
    );
    expect(
      pairedAuthorizationUsable(skewed, VECTOR.wallet, at(lastUsableMs + 1, lastUsableMs + 1 + 86_400_000)),
    ).toBe(false);
  });

  it("requests the grant for the wallet with the API key", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(grantJson()), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const body = await requestPairedAuthorizationGrant(
      "https://executor.example/base",
      VECTOR.wallet,
      "key",
    );
    expect(body).toEqual(grantJson());
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe(`https://executor.example/paired/authorization?wallet=${VECTOR.wallet}`);
    expect(init.method).toBe("GET");
    expect((init.headers as Record<string, string>)["X-API-Key"]).toBe("key");
  });

  it("keeps the grant request deadline until the body is read", async () => {
    vi.useFakeTimers();
    // Headers arrive at once. The body never does.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(new ReadableStream<Uint8Array>({ start() {} }), { status: 200 })),
    );
    let outcome: unknown = "pending";
    requestPairedAuthorizationGrant("https://executor.example", VECTOR.wallet).then(
      (value) => (outcome = value),
      (error: unknown) => (outcome = error),
    );
    await vi.advanceTimersByTimeAsync(10_001);
    expect(outcome).toMatchObject({ reason: "validation_unavailable" });
  });

  it("reports a refused or failed grant request as a protocol error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({ reason: "ip_rate_limited", retry_after: 9 }), { status: 429 })),
    );
    await expect(
      requestPairedAuthorizationGrant("https://executor.example", VECTOR.wallet),
    ).rejects.toMatchObject({ reason: "ip_rate_limited", status: 429, retryAfterSecs: 9 });

    vi.stubGlobal("fetch", vi.fn(async () => new Response("not found", { status: 404 })));
    await expect(
      requestPairedAuthorizationGrant("https://executor.example", VECTOR.wallet),
    ).rejects.toMatchObject({ reason: "unsupported_session", status: 404 });

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("offline");
      }),
    );
    await expect(
      requestPairedAuthorizationGrant("https://executor.example", VECTOR.wallet),
    ).rejects.toMatchObject({ reason: "validation_unavailable" });
  });
});

describe("PulseSDK paired authorization", () => {
  it("has the wallet sign once and holds the authorization for that wallet", async () => {
    const clock = controlledClocks(at(1_000));
    const sdk = new PulseSDK({ cluster: "devnet" });
    const { wallet, signed } = vectorWallet();
    expect(sdk.pairedAuthorization(VECTOR.wallet)).toBeNull();

    const requested: string[] = [];
    const authorization = await sdk.authorizePairedSession(wallet, {
      requestGrant: async (address) => {
        requested.push(address);
        return grantJson();
      },
    });
    expect(requested).toEqual([VECTOR.wallet]);
    expect(signed).toHaveLength(1);
    expect(authorization.signatureHex).toBe(VECTOR.signatureHex);
    expect(sdk.pairedAuthorization(VECTOR.wallet)).toBe(authorization);
    expect(sdk.pairedAuthorization("11111111111111111111111111111111")).toBeNull();

    // Inside the margin the host must sign again before it starts a capture.
    clock.monotonicMs = 1_000 + 900_000 - PAIRED_AUTHORIZATION_MARGIN_MS + 1;
    clock.wallMs = WALL_START + clock.monotonicMs;
    expect(sdk.pairedAuthorization(VECTOR.wallet)).toBeNull();
  });

  it("needs a relayer or a host that requests the grant", async () => {
    const sdk = new PulseSDK({ cluster: "devnet" });
    await expect(sdk.authorizePairedSession(vectorWallet().wallet)).rejects.toMatchObject({
      reason: "validation_unavailable",
    });
    await expect(sdk.preparePairedAuthorization(VECTOR.wallet)).rejects.toMatchObject({
      reason: "validation_unavailable",
    });
  });

  it("holds nothing when the wallet declines to sign", async () => {
    const sdk = new PulseSDK({ cluster: "devnet" });
    const wallet = {
      publicKey: { toBase58: () => VECTOR.wallet },
      signMessage: async () => {
        throw new Error("User rejected the request.");
      },
    };
    await expect(
      sdk.authorizePairedSession(wallet, { requestGrant: async () => grantJson() }),
    ).rejects.toThrow("User rejected the request.");
    expect(sdk.pairedAuthorization(VECTOR.wallet)).toBeNull();
  });

  it("holds nothing when the connected wallet changes while the grant is requested", async () => {
    const sdk = new PulseSDK({ cluster: "devnet" });
    const { wallet, signed } = vectorWallet();
    let address = VECTOR.wallet;
    const switching = { ...wallet, publicKey: { toBase58: () => address } };
    await expect(
      sdk.authorizePairedSession(switching, {
        requestGrant: async () => {
          address = "11111111111111111111111111111111";
          return grantJson();
        },
      }),
    ).rejects.toMatchObject({ reason: "invalid_request" });
    expect(signed).toHaveLength(0);
    expect(sdk.pairedAuthorization(VECTOR.wallet)).toBeNull();
    expect(sdk.pairedAuthorization(address)).toBeNull();
  });

  it("holds nothing when the signature does not verify", async () => {
    const sdk = new PulseSDK({ cluster: "devnet" });
    const wallet = {
      publicKey: { toBase58: () => VECTOR.wallet },
      signMessage: async (message: Uint8Array) => ed25519.sign(message, new Uint8Array(32).fill(8)),
    };
    await expect(
      sdk.authorizePairedSession(wallet, { requestGrant: async () => grantJson() }),
    ).rejects.toMatchObject({ reason: "invalid_request" });
    expect(sdk.pairedAuthorization(VECTOR.wallet)).toBeNull();
  });

  it("holds nothing when the prompt stayed open past the usable lifetime", async () => {
    const clock = controlledClocks();
    const sdk = new PulseSDK({ cluster: "devnet" });
    const wallet = {
      publicKey: { toBase58: () => VECTOR.wallet },
      signMessage: async (message: Uint8Array) => {
        clock.monotonicMs = 900_000 - PAIRED_AUTHORIZATION_MARGIN_MS + 1;
        clock.wallMs = WALL_START + clock.monotonicMs;
        return ed25519.sign(message, VECTOR.seed);
      },
    };
    await expect(
      sdk.authorizePairedSession(wallet, { requestGrant: async () => grantJson() }),
    ).rejects.toMatchObject({ reason: "authorization_required" });
    expect(sdk.pairedAuthorization(VECTOR.wallet)).toBeNull();
  });

  it("signs a prepared grant without waiting on anything before the wallet prompt", async () => {
    controlledClocks();
    const sdk = new PulseSDK({ cluster: "devnet" });
    const { wallet, signed } = vectorWallet();
    const requested: string[] = [];
    const requestGrant = async (address: string) => {
      requested.push(address);
      return grantJson();
    };
    await sdk.preparePairedAuthorization(VECTOR.wallet, { requestGrant });
    expect(requested).toEqual([VECTOR.wallet]);
    expect(signed).toHaveLength(0);

    const pending = sdk.authorizePairedSession(wallet, { requestGrant });
    // The prompt opens in the same task as the call, so a browser still counts the user's gesture.
    expect(signed).toHaveLength(1);
    const authorization = await pending;
    expect(authorization.signatureHex).toBe(VECTOR.signatureHex);
    expect(requested).toEqual([VECTOR.wallet]);

    // A prepared grant serves one authorization.
    await sdk.authorizePairedSession(wallet, { requestGrant });
    expect(requested).toEqual([VECTOR.wallet, VECTOR.wallet]);
  });

  it("requests a new grant when the prepared one is for another wallet or nearly spent", async () => {
    const clock = controlledClocks();
    const sdk = new PulseSDK({ cluster: "devnet" });
    const { wallet } = vectorWallet();
    const requested: string[] = [];
    const requestGrant = async (address: string) => {
      requested.push(address);
      return grantJson();
    };

    await sdk.preparePairedAuthorization("11111111111111111111111111111111", { requestGrant });
    await sdk.authorizePairedSession(wallet, { requestGrant });
    expect(requested).toEqual(["11111111111111111111111111111111", VECTOR.wallet]);

    // Exactly the margin plus one minute left: still used.
    requested.length = 0;
    await sdk.preparePairedAuthorization(VECTOR.wallet, { requestGrant });
    clock.monotonicMs = 900_000 - PAIRED_AUTHORIZATION_MARGIN_MS - 60_000;
    clock.wallMs = WALL_START + clock.monotonicMs;
    await sdk.authorizePairedSession(wallet, { requestGrant });
    expect(requested).toEqual([VECTOR.wallet]);

    // One millisecond less: a fresh grant is requested.
    requested.length = 0;
    clock.monotonicMs = 0;
    clock.wallMs = WALL_START;
    await sdk.preparePairedAuthorization(VECTOR.wallet, { requestGrant });
    clock.monotonicMs = 900_000 - PAIRED_AUTHORIZATION_MARGIN_MS - 60_000 + 1;
    clock.wallMs = WALL_START + clock.monotonicMs;
    await sdk.authorizePairedSession(wallet, { requestGrant });
    expect(requested).toEqual([VECTOR.wallet, VECTOR.wallet]);
  });

  it("refuses to prepare a grant for an address that is not a wallet", async () => {
    const sdk = new PulseSDK({ cluster: "devnet" });
    const requestGrant = vi.fn(async () => grantJson());
    await expect(sdk.preparePairedAuthorization("not-a-wallet", { requestGrant })).rejects.toMatchObject({
      reason: "invalid_request",
    });
    expect(requestGrant).not.toHaveBeenCalled();
  });
});
