import type { BalancePreflightConnection } from "../../src/submit/balance";

export function fundedPreflightConnection(): BalancePreflightConnection {
  return {
    getBalance: async () => 10_000_000_000,
    getFeeForMessage: async () => ({ value: 5_000 }),
    simulateTransaction: async () => ({ value: { err: null, logs: [] } }),
  };
}
