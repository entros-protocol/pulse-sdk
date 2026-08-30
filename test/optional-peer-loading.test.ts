import { describe, expect, it, vi } from "vitest";

vi.mock("@coral-xyz/anchor", () => {
  throw new Error("Anchor loaded during package import");
});

vi.mock("@solana/web3.js", () => {
  throw new Error("Solana web3 loaded during package import");
});

describe("optional Solana peers", () => {
  it("does not load them when importing the SDK", async () => {
    await expect(import("../src/index")).resolves.toBeDefined();
  });
});
