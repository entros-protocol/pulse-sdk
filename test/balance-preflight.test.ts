import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Keypair,
  SystemProgram,
  Transaction,
  VersionedTransaction,
} from "@solana/web3.js";
import { preflightWalletBalance } from "../src/submit/balance";

const wallet = Keypair.fromSeed(new Uint8Array(32).fill(19)).publicKey;
const receiver = Keypair.fromSeed(new Uint8Array(32).fill(20)).publicKey;

function transaction() {
  const tx = new Transaction().add(
    SystemProgram.transfer({ fromPubkey: wallet, toPubkey: receiver, lamports: 1 }),
  );
  tx.feePayer = wallet;
  tx.recentBlockhash = "11111111111111111111111111111111";
  return tx;
}

function connection() {
  return {
    getBalance: vi.fn(async () => 768_203),
    getFeeForMessage: vi.fn(async () => ({ value: 5_000 as number | null })),
    simulateTransaction: vi.fn(
      async (_tx: VersionedTransaction, _config: unknown) => ({
        value: { err: null as unknown, logs: [] as string[] | null },
      }),
    ),
  };
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("wallet funding preflight", () => {
  it("simulates the actual unsigned message without changing its bytes", async () => {
    const tx = transaction();
    const before = tx.compileMessage().serialize();
    const rpc = connection();
    await preflightWalletBalance(rpc, tx);
    expect(rpc.getBalance).toHaveBeenCalledWith(wallet, "confirmed");
    expect(rpc.getFeeForMessage).toHaveBeenCalledWith(tx.compileMessage(), "confirmed");
    const [simulated, options] = rpc.simulateTransaction.mock.calls[0]!;
    expect(simulated).toBeInstanceOf(VersionedTransaction);
    expect(simulated.message.serialize()).toEqual(before);
    expect(simulated.signatures.every((signature) => signature.every((byte) => byte === 0))).toBe(true);
    expect(options).toEqual({
      commitment: "confirmed",
      sigVerify: false,
      replaceRecentBlockhash: true,
    });
    expect(tx.compileMessage().serialize()).toEqual(before);
  });

  it.each([0, 4_999])("stops a %i-lamport wallet before simulation", async (balance) => {
    const rpc = connection();
    rpc.getBalance.mockResolvedValue(balance);
    await expect(preflightWalletBalance(rpc, transaction())).rejects.toThrow("Insufficient funds");
    expect(rpc.simulateTransaction).not.toHaveBeenCalled();
  });

  it("does not impose a fixed minimum or safety margin", async () => {
    const rpc = connection();
    rpc.getBalance.mockResolvedValue(5_000);
    await expect(preflightWalletBalance(rpc, transaction())).resolves.toBeUndefined();
  });

  it("names insufficient rent even when the wallet can pay its network fee", async () => {
    const rpc = connection();
    rpc.simulateTransaction.mockResolvedValue({
      value: {
        err: { InstructionError: [3, { Custom: 1 }] },
        logs: [
          "Program log: Transfer: insufficient lamports 768203, need 1574800",
          "Program 11111111111111111111111111111111 failed: custom program error: 0x1",
        ],
      },
    });
    await expect(preflightWalletBalance(rpc, transaction())).rejects.toThrow(
      "account rent and network fees. No transaction was sent.",
    );
  });

  it.each([
    "InsufficientFundsForFee",
    "InsufficientFundsForRent",
    { InsufficientFundsForRent: { account_index: 0 } },
  ])("recognizes the structured funding error %j", async (err) => {
    const rpc = connection();
    rpc.simulateTransaction.mockResolvedValue({ value: { err, logs: null } });
    await expect(preflightWalletBalance(rpc, transaction())).rejects.toThrow("Insufficient funds");
  });

  it.each([
    { InstructionError: [3, { Custom: 1 }] },
    { InstructionError: [3, { Custom: 6099 }] },
    "BlockhashNotFound",
    null,
  ])("preserves existing send policy for a non-funding simulation outcome %j", async (err) => {
    const rpc = connection();
    rpc.simulateTransaction.mockResolvedValue({
      value: { err, logs: ["Program example failed: custom program error: 0x1"] },
    });
    await expect(preflightWalletBalance(rpc, transaction())).resolves.toBeUndefined();
  });

  it.each([NaN, Infinity, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    "refuses an invalid balance %s without calling it insufficient",
    async (balance) => {
      const rpc = connection();
      rpc.getBalance.mockResolvedValue(balance);
      await expect(preflightWalletBalance(rpc, transaction())).rejects.toThrow("could not check");
      expect(rpc.simulateTransaction).not.toHaveBeenCalled();
    },
  );

  it.each([null, NaN, Infinity, -1, 1.5, Number.MAX_SAFE_INTEGER + 1])(
    "refuses an unavailable or invalid network fee %s",
    async (value) => {
      const rpc = connection();
      rpc.getFeeForMessage.mockResolvedValue({ value });
      await expect(preflightWalletBalance(rpc, transaction())).rejects.toThrow("could not check");
    },
  );

  it.each(["getBalance", "getFeeForMessage", "simulateTransaction"] as const)(
    "keeps a %s RPC failure distinct from insufficient SOL",
    async (method) => {
      const rpc = connection();
      rpc[method].mockRejectedValue(new Error("private RPC diagnostic"));
      await expect(preflightWalletBalance(rpc, transaction())).rejects.toThrow("could not check");
      await expect(preflightWalletBalance(rpc, transaction())).rejects.not.toThrow("private RPC diagnostic");
    },
  );

  it("bounds a stalled RPC without starting a late simulation", async () => {
    vi.useFakeTimers();
    const rpc = connection();
    let resolveBalance: ((value: number) => void) | undefined;
    rpc.getBalance.mockImplementation(
      () => new Promise((resolve) => { resolveBalance = resolve; }),
    );
    const checked = preflightWalletBalance(rpc, transaction());
    const rejected = expect(checked).rejects.toThrow("No transaction was sent");
    await vi.advanceTimersByTimeAsync(10_000);
    await rejected;
    expect(rpc.simulateTransaction).not.toHaveBeenCalled();
    resolveBalance?.(768_203);
    await vi.advanceTimersByTimeAsync(1);
    expect(rpc.simulateTransaction).not.toHaveBeenCalled();
  });
});
