import { readFileSync } from "node:fs";
import { PublicKey, type AccountInfo } from "@solana/web3.js";
import { afterEach, describe, expect, it, vi } from "vitest";

import { readAgentState, type AgentStateConnection } from "../src/agent/state";
import { INTEGRATOR_DEVNET_GENESIS_HASH } from "../src/identity/integrator";

interface CapturedAccount {
  address: string;
  owner: string;
  executable: boolean;
  lamports: number;
  dataBase64: string;
}

const capture = JSON.parse(
  readFileSync(new URL("./fixtures/agent-registry-devnet.json", import.meta.url), "utf8"),
) as { contextSlot: number; accounts: Record<"asset" | "agentAccount" | "legacySnapshot", CapturedAccount> };
const AGENT = capture.accounts.asset.address;
const OWNER = "2jowceXRayDC3ufU3kRjf67zT3cLv48q4gbcdhA1SNQw";
const OTHER = new PublicKey(new Uint8Array(32).fill(5)).toBase58();
const AGENT_WALLET = new PublicKey(new Uint8Array(32).fill(6)).toBase58();

afterEach(() => vi.useRealTimers());

function account(captured: CapturedAccount): AccountInfo<Buffer> {
  return {
    data: Buffer.from(captured.dataBase64, "base64"),
    executable: captured.executable,
    lamports: captured.lamports,
    owner: new PublicKey(captured.owner),
    rentEpoch: 0,
  };
}

function withAgentWallet(data: Buffer, wallet: string): Buffer {
  const changed = Buffer.concat([
    data.subarray(0, 138),
    Buffer.from([1]),
    new PublicKey(wallet).toBuffer(),
    data.subarray(139, data.length - 32),
  ]);
  expect(data.subarray(data.length - 32).every((byte) => byte === 0)).toBe(true);
  return changed;
}

function setup(
  change: (accounts: { asset: AccountInfo<Buffer> | null; agentAccount: AccountInfo<Buffer> | null }) => void = () => {},
  options: { genesis?: string; slot?: number; fail?: boolean } = {},
) {
  const accounts = {
    asset: account(capture.accounts.asset) as AccountInfo<Buffer> | null,
    agentAccount: account(capture.accounts.agentAccount) as AccountInfo<Buffer> | null,
  };
  change(accounts);
  const requested: string[][] = [];
  const configs: unknown[] = [];
  const connection: AgentStateConnection = {
    async getGenesisHash() {
      return options.genesis ?? INTEGRATOR_DEVNET_GENESIS_HASH;
    },
    async getMultipleAccountsInfoAndContext(keys, config) {
      requested.push(keys.map((key) => key.toBase58()));
      configs.push(config);
      if (options.fail) throw new Error("Synthetic RPC failure");
      return {
        context: { slot: options.slot ?? capture.contextSlot },
        value: [accounts.asset, accounts.agentAccount],
      };
    },
  };
  return { connection, requested, configs, accounts };
}

describe("agent state from captured devnet accounts", () => {
  it("reads the Core owner and the registry cache at one slot", async () => {
    const { connection, requested, configs } = setup();
    expect(await readAgentState({ agent: AGENT, connection })).toEqual({
      status: "available",
      evidence: {
        cluster: "devnet",
        genesisHash: INTEGRATOR_DEVNET_GENESIS_HASH,
        coreProgram: "CoREENxT6tW1HoK8ypY1SxRMZTcVPm7R94rH4PZNhX7d",
        agentRegistryProgram: "8oo4J9tBB3Hna1jRQ3rWvJjojqM5DYTDJo5cejUuJy3C",
        agentCollection: "6CTyGPcn8dMwKEqgtvx2XCpkGUd7uqCVK6937RSM5bhA",
        agent: AGENT,
        agentAccount: capture.accounts.agentAccount.address,
        owner: OWNER,
        cachedOwner: OWNER,
        agentWallet: null,
        agentWalletStatus: "unbound",
        readContextSlot: capture.contextSlot,
      },
    });
    expect(requested).toEqual([[AGENT, capture.accounts.agentAccount.address]]);
    expect(requested.flat()).not.toContain(capture.accounts.legacySnapshot.address);
    expect(configs).toEqual([{ commitment: "confirmed" }]);
  });

  it("passes a minimum context slot and rejects a response below it", async () => {
    const low = setup(() => {}, { slot: 10 });
    expect(await readAgentState({ agent: AGENT, connection: low.connection, minContextSlot: 11 })).toEqual({
      status: "invalid",
      reason: "agent_invalid",
    });
    expect(low.configs).toEqual([{ commitment: "confirmed", minContextSlot: 11 }]);
  });

  it("reports a bound agent wallet", async () => {
    const { connection } = setup(({ agentAccount }) => {
      agentAccount!.data = withAgentWallet(agentAccount!.data, AGENT_WALLET);
    });
    const result = await readAgentState({ agent: AGENT, connection });
    expect(result).toMatchObject({
      status: "available",
      evidence: { agentWallet: AGENT_WALLET, agentWalletStatus: "bound", owner: OWNER },
    });
  });

  it("follows a direct Core transfer and marks the cached wallet stale", async () => {
    const { connection } = setup(({ asset, agentAccount }) => {
      new PublicKey(OTHER).toBuffer().copy(asset!.data, 1);
      agentAccount!.data = withAgentWallet(agentAccount!.data, AGENT_WALLET);
    });
    expect(await readAgentState({ agent: AGENT, connection })).toMatchObject({
      status: "available",
      evidence: { owner: OTHER, cachedOwner: OWNER, agentWallet: AGENT_WALLET, agentWalletStatus: "stale" },
    });
  });

  it("follows a registry transfer that reset the agent wallet", async () => {
    const { connection } = setup(({ asset, agentAccount }) => {
      new PublicKey(OTHER).toBuffer().copy(asset!.data, 1);
      new PublicKey(OTHER).toBuffer().copy(agentAccount!.data, 72);
    });
    expect(await readAgentState({ agent: AGENT, connection })).toMatchObject({
      status: "available",
      evidence: { owner: OTHER, cachedOwner: OTHER, agentWallet: null, agentWalletStatus: "unbound" },
    });
  });

  it("ignores plugin data after the asset prefix", async () => {
    const { connection } = setup(({ asset }) => {
      asset!.data = Buffer.concat([asset!.data, Buffer.from([3, 1, 2, 3, 4, 5])]);
    });
    expect((await readAgentState({ agent: AGENT, connection })).status).toBe("available");
  });
});

describe("agent state rejections", () => {
  const cases: [string, Parameters<typeof setup>[0], string][] = [
    ["missing asset", (a) => void (a.asset = null), "agent_missing"],
    [
      "burned asset",
      (a) => void (a.asset!.data = Buffer.from([0])),
      "agent_missing",
    ],
    ["hashed asset", (a) => void (a.asset!.data[0] = 2), "agent_invalid"],
    ["uninitialized key with data", (a) => void (a.asset!.data[0] = 0), "agent_invalid"],
    [
      "asset owned by another program",
      (a) => void (a.asset!.owner = new PublicKey(OTHER)),
      "agent_invalid",
    ],
    ["executable asset", (a) => void (a.asset!.executable = true), "agent_invalid"],
    ["no update authority", (a) => void (a.asset!.data[33] = 0), "agent_unregistered"],
    ["address update authority", (a) => void (a.asset!.data[33] = 1), "agent_unregistered"],
    ["unknown update authority", (a) => void (a.asset!.data[33] = 3), "agent_invalid"],
    [
      "another collection",
      (a) => void new PublicKey(OTHER).toBuffer().copy(a.asset!.data, 34),
      "agent_unregistered",
    ],
    [
      "name length past the account",
      (a) => void a.asset!.data.writeUInt32LE(10_000, 66),
      "agent_invalid",
    ],
    [
      "unknown sequence tag",
      (a) => void (a.asset!.data[a.asset!.data.length - 1] = 2),
      "agent_invalid",
    ],
    [
      "truncated sequence",
      (a) => {
        a.asset!.data[a.asset!.data.length - 1] = 1;
      },
      "agent_invalid",
    ],
    ["missing registry account", (a) => void (a.agentAccount = null), "agent_invalid"],
    [
      "registry account of another length",
      (a) => void (a.agentAccount!.data = a.agentAccount!.data.subarray(0, 747)),
      "agent_invalid",
    ],
    ["registry discriminator", (a) => void (a.agentAccount!.data[0] ^= 1), "agent_invalid"],
    [
      "registry collection",
      (a) => void new PublicKey(OTHER).toBuffer().copy(a.agentAccount!.data, 8),
      "agent_invalid",
    ],
    [
      "registry asset",
      (a) => void new PublicKey(OTHER).toBuffer().copy(a.agentAccount!.data, 104),
      "agent_invalid",
    ],
    ["registry bump", (a) => void (a.agentAccount!.data[136] ^= 1), "agent_invalid"],
    ["registry flag", (a) => void (a.agentAccount!.data[137] = 2), "agent_invalid"],
    ["agent wallet tag", (a) => void (a.agentAccount!.data[138] = 2), "agent_invalid"],
    [
      "registry account owned by another program",
      (a) => void (a.agentAccount!.owner = new PublicKey(OTHER)),
      "agent_invalid",
    ],
  ];

  it.each(cases)("rejects %s", async (_name, change, reason) => {
    const { connection } = setup(change);
    expect(await readAgentState({ agent: AGENT, connection })).toEqual({ status: "invalid", reason });
  });

  it("rejects another cluster", async () => {
    const { connection } = setup(() => {}, { genesis: "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d" });
    expect(await readAgentState({ agent: AGENT, connection })).toEqual({
      status: "invalid",
      reason: "wrong_cluster",
    });
  });

  it("rejects malformed input before any RPC call", async () => {
    const { connection, requested } = setup();
    for (const agent of ["", "1" + AGENT, AGENT + "1", "0".repeat(44), "1".repeat(31)]) {
      expect(await readAgentState({ agent, connection })).toEqual({ status: "invalid", reason: "invalid_request" });
    }
    expect(await readAgentState({ agent: AGENT, connection, minContextSlot: -1 })).toEqual({
      status: "invalid",
      reason: "invalid_request",
    });
    expect(requested).toEqual([]);
  });

  it("reports RPC failure and a stalled RPC as unavailable", async () => {
    const failing = setup(() => {}, { fail: true });
    expect(await readAgentState({ agent: AGENT, connection: failing.connection })).toEqual({
      status: "unavailable",
      reason: "rpc_unavailable",
    });
    vi.useFakeTimers();
    let started: () => void = () => {};
    const reached = new Promise<void>((resolve) => {
      started = resolve;
    });
    const stalled: AgentStateConnection = {
      getGenesisHash: () => new Promise<string>(() => {}),
      getMultipleAccountsInfoAndContext: () => {
        started();
        return new Promise(() => {});
      },
    };
    const pending = readAgentState({ agent: AGENT, connection: stalled });
    // The reader loads web3.js before it calls RPC. Advance the clock only after the call starts.
    await reached;
    await vi.advanceTimersByTimeAsync(3000);
    expect(await pending).toEqual({ status: "unavailable", reason: "rpc_unavailable" });
  });
});

describe("agent state from real program output", () => {
  const localnet = JSON.parse(
    readFileSync(new URL("./fixtures/agent-registry-localnet.json", import.meta.url), "utf8"),
  ) as {
    agent: string;
    agentAccount: string;
    ownerA: string;
    ownerB: string;
    agentWallets: { first: string; successor: string };
    stages: Record<string, { asset: string; agentAccount: string }>;
  };
  const expected: Record<string, [string, string, string | null, string]> = {
    registered: [localnet.ownerA, localnet.ownerA, null, "unbound"],
    bound: [localnet.ownerA, localnet.ownerA, localnet.agentWallets.first, "bound"],
    coreTransfer: [localnet.ownerB, localnet.ownerA, localnet.agentWallets.first, "stale"],
    reboundByB: [localnet.ownerB, localnet.ownerB, localnet.agentWallets.successor, "bound"],
    registryTransfer: [localnet.ownerA, localnet.ownerA, null, "unbound"],
  };

  it.each(Object.entries(expected))("decodes the %s stage", async (stage, [owner, cachedOwner, agentWallet, status]) => {
    const bytes = localnet.stages[stage]!;
    const connection: AgentStateConnection = {
      getGenesisHash: async () => INTEGRATOR_DEVNET_GENESIS_HASH,
      getMultipleAccountsInfoAndContext: async () => ({
        context: { slot: 1 },
        value: [
          {
            data: Buffer.from(bytes.asset, "base64"),
            executable: false,
            lamports: 1,
            owner: new PublicKey("CoREENxT6tW1HoK8ypY1SxRMZTcVPm7R94rH4PZNhX7d"),
            rentEpoch: 0,
          },
          {
            data: Buffer.from(bytes.agentAccount, "base64"),
            executable: false,
            lamports: 1,
            owner: new PublicKey("8oo4J9tBB3Hna1jRQ3rWvJjojqM5DYTDJo5cejUuJy3C"),
            rentEpoch: 0,
          },
        ],
      }),
    };
    expect(await readAgentState({ agent: localnet.agent, connection })).toMatchObject({
      status: "available",
      evidence: {
        agentAccount: localnet.agentAccount,
        owner,
        cachedOwner,
        agentWallet,
        agentWalletStatus: status,
      },
    });
  });
});
