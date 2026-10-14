import type {
  Commitment,
  Message,
  PublicKey,
  SimulateTransactionConfig,
  Transaction,
  VersionedTransaction,
} from "@solana/web3.js";

import { withTimeout } from "./errors";

const PREFLIGHT_TIMEOUT_MS = 10_000;
const UNAVAILABLE =
  "We could not check this wallet's SOL balance and transaction costs. No transaction was sent. Try again.";
const INSUFFICIENT =
  "Insufficient funds: this wallet needs more SOL for protocol fees, account rent and network fees. No transaction was sent.";
const SYSTEM_FUNDS_FAILURE =
  "Program 11111111111111111111111111111111 failed: custom program error: 0x1";

export interface BalancePreflightConnection {
  getBalance(wallet: PublicKey, commitment: Commitment): Promise<number>;
  getFeeForMessage(
    message: Message,
    commitment: Commitment,
  ): Promise<{ value: number | null }>;
  simulateTransaction(
    transaction: VersionedTransaction,
    config: SimulateTransactionConfig,
  ): Promise<{ value: { err: unknown; logs: string[] | null } }>;
}

class BalancePreflightError extends Error {}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function fundingFailure(error: unknown, logs: string[] | null): boolean {
  if (
    error === "InsufficientFundsForFee" ||
    error === "InsufficientFundsForRent"
  ) {
    return true;
  }
  if (!record(error)) return false;
  if (Object.prototype.hasOwnProperty.call(error, "InsufficientFundsForRent")) return true;
  const instructionError = error.InstructionError;
  return (
    Array.isArray(instructionError) &&
    record(instructionError[1]) &&
    instructionError[1].Custom === 1 &&
    (logs?.includes(SYSTEM_FUNDS_FAILURE) ?? false)
  );
}

/**
 * Check funding before requesting a wallet transaction. Simulate the actual
 * message so account creation, rent top-ups, protocol fees and refunds follow
 * the deployed instructions instead of a second client-side cost model.
 *
 * Only funding errors change the existing submission policy. Other simulated
 * program errors still follow the caller's existing send/preflight behavior.
 * RPC failures stop before sending and reveal no simulation logs.
 */
export async function preflightWalletBalance(
  connection: Partial<BalancePreflightConnection>,
  transaction: Transaction,
): Promise<void> {
  let finished = false;
  try {
    await withTimeout(
      (async () => {
        if (
          !transaction.feePayer ||
          !connection.getBalance ||
          !connection.getFeeForMessage ||
          !connection.simulateTransaction
        ) {
          throw new BalancePreflightError(UNAVAILABLE);
        }
        const message = transaction.compileMessage();
        const [balance, fee] = await Promise.all([
          connection.getBalance(transaction.feePayer, "confirmed"),
          connection.getFeeForMessage(message, "confirmed"),
        ]);
        if (
          !Number.isSafeInteger(balance) ||
          balance < 0 ||
          fee.value === null ||
          !Number.isSafeInteger(fee.value) ||
          fee.value < 0
        ) {
          throw new BalancePreflightError(UNAVAILABLE);
        }
        if (balance < fee.value) {
          throw new BalancePreflightError(INSUFFICIENT);
        }
        const { VersionedTransaction } = await import("@solana/web3.js");
        if (finished) return;
        const simulation = await connection.simulateTransaction(
          new VersionedTransaction(message),
          {
            commitment: "confirmed",
            sigVerify: false,
            replaceRecentBlockhash: true,
          },
        );
        if (fundingFailure(simulation.value.err, simulation.value.logs)) {
          throw new BalancePreflightError(INSUFFICIENT);
        }
      })(),
      PREFLIGHT_TIMEOUT_MS,
      UNAVAILABLE,
    );
  } catch (error) {
    throw error instanceof BalancePreflightError
      ? error
      : new BalancePreflightError(UNAVAILABLE);
  } finally {
    finished = true;
  }
}
