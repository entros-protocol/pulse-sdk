import { describe, expect, it } from "vitest";
import { PublicKey } from "@solana/web3.js";
import { PROGRAM_IDS } from "../src/config";
import {
  fetchProjectionPolicy,
  type ProjectionPolicy,
} from "../src/identity/anchor";
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

interface InternalSession {
  pinnedProjectionPolicyPromise: Promise<ProjectionPolicy> | null;
  readProjectionPolicy(connection?: unknown): Promise<ProjectionPolicy>;
  pinProjectionPolicy(connection?: unknown): Promise<ProjectionPolicy>;
}

function internalSession(studyProjection?: 1 | 2): InternalSession {
  const studyContext = studyProjection
    ? {
        token: "A".repeat(43),
        record_id: "1".repeat(32),
        capture_class: "web-desktop" as const,
        feature_schema_version: studyProjection === 1 ? 4 : 5,
        projection_version: studyProjection,
      }
    : undefined;
  return new PulseSDK({
    cluster: "devnet",
    relayerUrl: "https://executor.test",
  }).createSession(undefined, studyContext) as unknown as InternalSession;
}

describe("projection policy", () => {
  it("reads supported registry fields and rejects newer versions", async () => {
    const supported = {
      getAccountInfo: async () => versionedAccount(2, 0),
    };
    await expect(fetchProjectionPolicy(supported)).resolves.toEqual({
      current: 2,
      minimum: 0,
    });

    const newer = {
      getAccountInfo: async () => versionedAccount(3, 0),
    };
    await expect(fetchProjectionPolicy(newer)).rejects.toThrow(
      /supports projection versions/i,
    );
  });

  it("keeps one projection policy for the full session", async () => {
    const session = internalSession();
    session.readProjectionPolicy = async () => ({ current: 1, minimum: 0 });

    await expect(session.pinProjectionPolicy()).resolves.toEqual({
      current: 1,
      minimum: 0,
    });
    const newerConnection = {
      getAccountInfo: async () => versionedAccount(2, 0),
    };
    await expect(session.pinProjectionPolicy(newerConnection)).resolves.toEqual({
      current: 1,
      minimum: 0,
    });

    const fresh = internalSession();
    await expect(fresh.pinProjectionPolicy(newerConnection)).resolves.toEqual({
      current: 2,
      minimum: 0,
    });
  });

  it("reads the active policy only when the first capture stage pins it", async () => {
    const session = internalSession();
    let reads = 0;
    session.readProjectionPolicy = async () => {
      reads += 1;
      return { current: 2, minimum: 1 };
    };

    expect(reads).toBe(0);
    await expect(session.pinProjectionPolicy()).resolves.toEqual({
      current: 2,
      minimum: 1,
    });
    expect(reads).toBe(1);
  });

  it("retries a failed policy read without falling back to projection 0", async () => {
    const session = internalSession();
    let reads = 0;
    session.readProjectionPolicy = async () => {
      reads += 1;
      if (reads === 1) throw new Error("temporary RPC failure");
      return { current: 2, minimum: 1 };
    };

    await expect(session.pinProjectionPolicy()).rejects.toThrow(
      "temporary RPC failure",
    );
    expect(session.pinnedProjectionPolicyPromise).toBeNull();
    await expect(session.pinProjectionPolicy()).resolves.toEqual({
      current: 2,
      minimum: 1,
    });
    expect(reads).toBe(2);
  });

  it("shares one in-flight policy read across capture stages", async () => {
    const session = internalSession();
    let release!: (policy: ProjectionPolicy) => void;
    const pending = new Promise<ProjectionPolicy>((resolve) => {
      release = resolve;
    });
    session.readProjectionPolicy = () => pending;

    const first = session.pinProjectionPolicy();
    const second = session.pinProjectionPolicy();
    expect(second).toBe(first);
    release({ current: 2, minimum: 1 });
    await expect(first).resolves.toEqual({ current: 2, minimum: 1 });
  });

  it("rejects a study grant from another projection", async () => {
    const session = internalSession(1);
    session.readProjectionPolicy = async () => ({ current: 2, minimum: 1 });
    await expect(session.pinProjectionPolicy()).rejects.toThrow(
      "Study projection does not match the active projection policy",
    );
    expect(session.pinnedProjectionPolicyPromise).toBeNull();
  });
});
