import { PublicKey } from "@solana/web3.js";
import { describe, expect, it } from "vitest";
import { SAS_CONFIG } from "../src/config";

describe("SAS configuration", () => {
  it("exports the reviewed Entros credential and schema", () => {
    expect(SAS_CONFIG).toEqual({
      programId: "22zoJMtdu4tQc2PzL74ZUT7FrwgB1Udec8DdW4yw4BdG",
      entrosCredentialPda: "AMBtabCgRFwGLjoZ21Z2LhSKJ6c47NckxUkMogJ3Lpuw",
      entrosSchemaPda: "5LNc7syFW7USPLveVLcNcjjY1xqS7QTXVjHZ7CQCbAMQ",
    });

    const addresses = Object.values(SAS_CONFIG).map((value) =>
      new PublicKey(value).toBase58(),
    );
    expect(new Set(addresses).size).toBe(addresses.length);
  });
});
