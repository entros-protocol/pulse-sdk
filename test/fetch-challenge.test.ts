import { describe, expect, it, afterEach, vi } from "vitest";
import { fetchChallenge } from "../src/challenge/fetch";

const NONCE_BYTES = Array.from({ length: 32 }, (_, i) => i);

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("fetchChallenge", () => {
  it("returns a conservative challenge deadline on 200", async () => {
    vi.spyOn(performance, "now").mockReturnValue(1_234);
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        nonce: NONCE_BYTES,
        expires_in: 60,
        phrase: "bada lita mupe ruso poto",
      }),
    } as Response);
    vi.stubGlobal("fetch", mockFetch);

    const result = await fetchChallenge(
      "https://executor.example.com",
      "So11111111111111111111111111111111111111112",
    );

    expect(result.nonce).toEqual(Uint8Array.from(NONCE_BYTES));
    expect(result.phrase).toBe("bada lita mupe ruso poto");
    expect(result.expiresIn).toBe(60);
    expect(result.expiresAtMs).toBe(61_234);

    const url = mockFetch.mock.calls[0]![0] as string;
    expect(url).toContain("/challenge?wallet=");
  });

  it("parses server-issued curve when provided", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        nonce: NONCE_BYTES,
        expires_in: 60,
        phrase: "bada lita mupe ruso poto",
        curve: {
          a: 2,
          b: 3,
          delta: 1.57,
          points: 200,
          anchor_x: 100,
          anchor_y: 50,
        },
      }),
    } as Response);
    vi.stubGlobal("fetch", mockFetch);

    const result = await fetchChallenge(
      "https://executor.example.com",
      "So11111111111111111111111111111111111111112",
    );

    expect(result.curve).toBeDefined();
    expect(result.curve?.a).toBe(2);
    expect(result.curve?.b).toBe(3);
    expect(result.curve?.delta).toBe(1.57);
    expect(result.curve?.anchorX).toBe(100);
    expect(result.curve?.anchorY).toBe(50);
  });

  it("sends X-API-Key header when apiKey provided", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ nonce: NONCE_BYTES, expires_in: 60, phrase: "ba da" }),
    } as Response);
    vi.stubGlobal("fetch", mockFetch);

    await fetchChallenge("https://executor.example.com", "wallet", "secret-key");

    const init = mockFetch.mock.calls[0]![1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    expect(headers["X-API-Key"]).toBe("secret-key");
  });

  it("throws on non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 400 } as Response),
    );

    await expect(
      fetchChallenge("https://executor.example.com", "wallet"),
    ).rejects.toThrow(/400/);
  });

  it("throws when nonce is not a 32-byte array", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ nonce: [1, 2, 3], expires_in: 60, phrase: "ba" }),
      } as Response),
    );

    await expect(
      fetchChallenge("https://executor.example.com", "wallet"),
    ).rejects.toThrow(/malformed nonce/);
  });

  it.each([
    ["fractional", 1.5],
    ["negative", -1],
    ["oversized", 256],
  ])("throws when a nonce byte is %s", async (_label, invalidByte) => {
    const nonce = [...NONCE_BYTES];
    nonce[10] = invalidByte;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ nonce, expires_in: 60, phrase: "ba" }),
      } as Response),
    );

    await expect(
      fetchChallenge("https://executor.example.com", "wallet"),
    ).rejects.toThrow(/malformed nonce/);
  });

  it("throws when phrase is empty", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ nonce: NONCE_BYTES, expires_in: 60, phrase: "" }),
      } as Response),
    );

    await expect(
      fetchChallenge("https://executor.example.com", "wallet"),
    ).rejects.toThrow(/empty challenge phrase/);
  });

  it("throws when the challenge lifetime is malformed", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ nonce: NONCE_BYTES, expires_in: 0, phrase: "ba" }),
      } as Response),
    );

    await expect(
      fetchChallenge("https://executor.example.com", "wallet"),
    ).rejects.toThrow(/malformed challenge lifetime/);
  });

  it("surfaces network errors with context", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("ECONNREFUSED")),
    );

    await expect(
      fetchChallenge("https://executor.example.com", "wallet"),
    ).rejects.toThrow(/Unable to fetch challenge/);
  });
});
