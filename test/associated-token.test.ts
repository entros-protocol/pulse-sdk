import { describe, expect, it } from "vitest";
import { Keypair, PublicKey } from "@solana/web3.js";
import { deriveToken2022AssociatedAddress } from "../src/submit/associated-token";

const VECTORS = [
  {
    owner: "AKnL4NNf3DGWZJS6cPknBuEGnVsV4A4m5tgebLHaRSZ9",
    mint: "9hSR6S7WPtxmTojgo6GG3k4yDPecgJY292j7xrsUGWBu",
    address: "2meyc5EGZPx66Kzdtmz41zWPRYa9tRqf8pdBGQMowdfT",
  },
  {
    owner: "GmaDrppBC7P5ARKV8g3djiwP89vz1jLK23V2GBjuAEGB",
    mint: "J2xccRtuG43drESLYznHhLhQkLTdfepcKYbiQ9BsJVaf",
    address: "8JYggxs3wb2uA1uYKy9ZdxJsQfHgMDfmCeW7qWLnHh5C",
  },
  {
    owner: "5WcE8o73vmsSZXeeWTLm3ty3fAJKCnBWRF6VuKUme5nu",
    mint: "F25s3DdjXdCxYBhh2z8FBusVEMT4b9bGNFVKJi3wFoF4",
    address: "3hvkh6dWgebu4vZS4MPeKq5VFfvaWMAJ6W1e51J5Liyv",
  },
] as const;

describe("Token-2022 associated token address", () => {
  it.each(VECTORS)("matches the SPL Token derivation for $owner", (vector) => {
    expect(
      deriveToken2022AssociatedAddress(
        new PublicKey(vector.mint),
        new PublicKey(vector.owner),
        PublicKey,
      ).toBase58(),
    ).toBe(vector.address);
  });

  it("rejects an off-curve owner", () => {
    const programId = Keypair.generate().publicKey;
    const [owner] = PublicKey.findProgramAddressSync(
      [new TextEncoder().encode("owner")],
      programId,
    );

    expect(() =>
      deriveToken2022AssociatedAddress(
        Keypair.generate().publicKey,
        owner,
        PublicKey,
      ),
    ).toThrow("Associated token account owner must be on curve");
  });
});
