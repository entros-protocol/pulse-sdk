import { readFileSync } from "node:fs";
import { PublicKey, type AccountInfo } from "@solana/web3.js";
import { describe, expect, it } from "vitest";

import {
  getAgentHumanOperator,
  readAgentOperatorSnapshot,
  type AgentSnapshotConnection,
} from "../src/agent/anchor";

const capture = JSON.parse(
  readFileSync(new URL("./fixtures/agent-registry-devnet.json", import.meta.url), "utf8"),
) as {
  accounts: Record<"asset" | "legacySnapshot", { address: string; owner: string; dataBase64: string }>;
};
const AGENT = capture.accounts.asset.address;
const REGISTRY = new PublicKey("8oo4J9tBB3Hna1jRQ3rWvJjojqM5DYTDJo5cejUuJy3C");
const original = Buffer.from(capture.accounts.legacySnapshot.dataBase64, "base64");

function entry(value: string, change: (data: Buffer) => void = () => {}): Buffer {
  const key = Buffer.from("entros:human-operator");
  const body = Buffer.from(value);
  const data = Buffer.alloc(Math.max(original.length, 50 + key.length + body.length));
  original.copy(data, 0, 0, 42);
  data.writeUInt32LE(key.length, 42);
  key.copy(data, 46);
  data.writeUInt32LE(body.length, 46 + key.length);
  body.copy(data, 50 + key.length);
  change(data);
  return data;
}

function connection(
  data: Buffer | null,
  options: { owner?: PublicKey; fail?: boolean } = {},
): AgentSnapshotConnection & { requested: string[] } {
  const requested: string[] = [];
  return {
    requested,
    async getAccountInfo(address) {
      requested.push(address.toBase58());
      if (options.fail) throw new Error("Synthetic RPC failure");
      if (!data) return null;
      const account: AccountInfo<Buffer> = {
        data,
        executable: false,
        lamports: 1,
        owner: options.owner ?? REGISTRY,
        rentEpoch: 0,
      };
      return account;
    },
  };
}

describe("historical operator snapshot", () => {
  it("reads the captured devnet entry as a historical snapshot", async () => {
    const rpc = connection(original);
    expect(await readAgentOperatorSnapshot(AGENT, { connection: rpc })).toEqual({
      status: "present",
      snapshot: {
        kind: "historical_snapshot",
        anchorPda: "6kwsTrMM2TXJ6VLGt3ksGQR4GJxNKt1A9qkpGtE5CoZn",
        trustScore: 821,
        verifiedAt: 1777131922,
        wallet: "2jowceXRayDC3ufU3kRjf67zT3cLv48q4gbcdhA1SNQw",
        immutable: true,
      },
    });
    expect(rpc.requested).toEqual([capture.accounts.legacySnapshot.address]);
  });

  it("reports what any owner wrote, including a claim that names another wallet", async () => {
    const claimed = new PublicKey(new Uint8Array(32).fill(9)).toBase58();
    const forged = JSON.stringify({ anchorPda: claimed, trustScore: 10000, verifiedAt: 1, wallet: claimed });
    const result = await readAgentOperatorSnapshot(AGENT, { connection: connection(entry(forged)) });
    expect(result).toMatchObject({ status: "present", snapshot: { wallet: claimed, trustScore: 10000 } });
  });

  it.each([
    ["an extra key", JSON.stringify({ anchorPda: AGENT, trustScore: 1, verifiedAt: 1, wallet: AGENT, extra: 1 })],
    ["a string score", JSON.stringify({ anchorPda: AGENT, trustScore: "1", verifiedAt: 1, wallet: AGENT })],
    ["a score above 10000", JSON.stringify({ anchorPda: AGENT, trustScore: 10001, verifiedAt: 1, wallet: AGENT })],
    ["a negative time", JSON.stringify({ anchorPda: AGENT, trustScore: 1, verifiedAt: -1, wallet: AGENT })],
    ["a non-canonical wallet", JSON.stringify({ anchorPda: AGENT, trustScore: 1, verifiedAt: 1, wallet: "1" + AGENT })],
    ["an array", "[]"],
    ["text that is not JSON", "not json"],
  ])("rejects %s", async (_name, value) => {
    expect(await readAgentOperatorSnapshot(AGENT, { connection: connection(entry(value)) })).toEqual({
      status: "invalid",
    });
  });

  it("rejects malformed accounts", async () => {
    const cases: [Buffer, { owner?: PublicKey }][] = [
      [original, { owner: new PublicKey(new Uint8Array(32).fill(5)) }],
      [entry(JSON.stringify({}), (data) => void (data[0] ^= 1)), {}],
      [entry(JSON.stringify({}), (data) => void (data[8] ^= 1)), {}],
      [entry(JSON.stringify({}), (data) => void (data[40] = 2)), {}],
      [entry(JSON.stringify({}), (data) => void data.write("entros:human-operatoR", 46)), {}],
      [entry(JSON.stringify({}), (data) => void data.writeUInt32LE(251, 67)), {}],
      [original.subarray(0, 60), {}],
      [Buffer.concat([original.subarray(0, 71), Buffer.from([0xff, 0xfe])]), {}],
    ];
    for (const [data, options] of cases) {
      expect(await readAgentOperatorSnapshot(AGENT, { connection: connection(data, options) })).toEqual({
        status: "invalid",
      });
    }
  });

  it("separates an absent entry, a failed read and a malformed agent", async () => {
    expect(await readAgentOperatorSnapshot(AGENT, { connection: connection(null) })).toEqual({ status: "absent" });
    expect(
      await readAgentOperatorSnapshot(AGENT, { connection: connection(original, { fail: true }) }),
    ).toEqual({ status: "unavailable" });
    const rpc = connection(original);
    expect(await readAgentOperatorSnapshot("1" + AGENT, { connection: rpc })).toEqual({ status: "invalid" });
    expect(rpc.requested).toEqual([]);
  });

  it("keeps the deprecated reader returning the snapshot or null", async () => {
    expect(await getAgentHumanOperator(AGENT, connection(original))).toMatchObject({
      kind: "historical_snapshot",
      trustScore: 821,
    });
    expect(await getAgentHumanOperator(AGENT, connection(null))).toBeNull();
    expect(await getAgentHumanOperator(AGENT, connection(original, { fail: true }))).toBeNull();
  });
});
