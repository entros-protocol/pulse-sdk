import { describe, expect, it } from "vitest";
import { PublicKey } from "@solana/web3.js";
import { PROGRAM_IDS } from "../src/config";
import { fetchProjectionPolicy, type ProjectionPolicy } from "../src/identity/anchor";
import { PulseSDK } from "../src/pulse";

function versionedAccount(current: number, minimum: number) {
  const data = Buffer.alloc(113);
  data.writeUInt16LE(current, 109);
  data.writeUInt16LE(minimum, 111);
  return {
    data,
    owner: new PublicKey(PROGRAM_IDS.entrosRegistry),
  };
}

describe("projection policy", () => {
  it("reads the versioned registry fields", async () => {
    const connection = {
      getAccountInfo: async () => versionedAccount(1, 0),
    };
    await expect(fetchProjectionPolicy(connection)).resolves.toEqual({
      current: 1,
      minimum: 0,
    });
  });

  it("rejects a chain version newer than the SDK", async () => {
    const connection = {
      getAccountInfo: async () => versionedAccount(2, 0),
    };
    await expect(fetchProjectionPolicy(connection)).rejects.toThrow(/supports projection versions/i);
  });

  it("does not let a stale prefetch override the supplied connection", async () => {
    const session = new PulseSDK({
      cluster: "devnet",
      relayerUrl: "https://executor.test",
    }).createSession();
    const internal = session as unknown as {
      projectionPolicyPromise: Promise<ProjectionPolicy | null>;
      resolveProjectionPolicy(connection: unknown): Promise<ProjectionPolicy>;
    };
    internal.projectionPolicyPromise = Promise.resolve({ current: 0, minimum: 0 });

    const explicitConnection = {
      getAccountInfo: async () => versionedAccount(1, 0),
    };
    await expect(internal.resolveProjectionPolicy(explicitConnection)).resolves.toEqual({
      current: 1,
      minimum: 0,
    });
  });
});
