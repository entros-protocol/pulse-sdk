import { afterEach, describe, expect, it, vi } from "vitest";
import { readIntegratorEvidence } from "../src/identity/integrator";
import { createEvidenceFixture } from "./fixtures/integrator-evidence";

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

async function futureFixture(offset = 1) {
  const fixture = await createEvidenceFixture();
  fixture.sasAccount.data.writeBigInt64LE(
    BigInt(fixture.input.nowSeconds + offset),
    104,
  );
  await readIntegratorEvidence(fixture.input);
  fixture.calls.length = 0;
  vi.useFakeTimers({
    toFake: ["setTimeout", "clearTimeout", "Date", "performance"],
  });
  vi.setSystemTime(fixture.input.nowSeconds * 1000);
  return fixture;
}

const liveClock = () => Math.floor(Date.now() / 1000);

async function finishRead(
  fixture: Awaited<ReturnType<typeof futureFixture>>,
  clock: number | (() => number) = liveClock,
) {
  const pending = readIntegratorEvidence({
    ...fixture.input,
    nowSeconds: clock,
  });
  let settled = false;
  void pending.finally(() => {
    settled = true;
  });
  for (let turn = 0; turn < 1000 && !settled; turn += 1) {
    await new Promise<void>((resolve) => setImmediate(resolve));
    await vi.runAllTimersAsync();
  }
  expect(settled).toBe(true);
  return pending;
}

describe("attestation clock reconciliation", () => {
  it.each([1, 3])(
    "re-reads after the real clock reaches +%i seconds",
    async (offset) => {
      const fixture = await futureFixture(offset);
      const result = await finishRead(fixture);
      expect(result.status).toBe("available");
      if (result.status === "available")
        expect(result.evidence.attestation.status).toBe("present");
      expect(fixture.calls).toEqual([
        "genesis",
        "status",
        "transaction",
        "identity",
        "attestation",
        "genesis",
        "status",
        "transaction",
        "identity",
        "attestation",
      ]);
      expect(performance.now()).toBe(offset * 1000);
    },
  );

  it("rejects +4 seconds without waiting or re-reading", async () => {
    const fixture = await futureFixture(4);
    const result = await finishRead(fixture);
    expect(result).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "invalid" } },
    });
    expect(fixture.calls).toHaveLength(5);
    expect(performance.now()).toBe(0);
  });

  it("rejects a clock that catches up after the monotonic deadline", async () => {
    const fixture = await futureFixture();
    vi.spyOn(performance, "now").mockImplementation(() =>
      liveClock() === fixture.input.nowSeconds ? 0 : 4000,
    );
    const result = await finishRead(fixture);
    expect(result).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "invalid" } },
    });
    expect(fixture.calls).toHaveLength(5);
  });

  it("preserves immediate rejection with a fixed snapshot clock", async () => {
    const fixture = await futureFixture();
    const result = await finishRead(fixture, fixture.input.nowSeconds);
    expect(result).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "invalid" } },
    });
    expect(fixture.calls).toHaveLength(5);
    expect(performance.now()).toBe(0);
  });

  it("bounds a frozen clock without downgrading optional evidence", async () => {
    const fixture = await futureFixture();
    const result = await finishRead(fixture, () => fixture.input.nowSeconds);
    expect(result).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "invalid" } },
    });
    expect(fixture.calls).toHaveLength(5);
    expect(performance.now()).toBe(3000);
  });

  it.each(["backwards", "throws", "fractional", "nan"])(
    "rejects a clock that becomes %s during waiting",
    async (failure) => {
      const fixture = await futureFixture();
      const result = await finishRead(fixture, () => {
        if (performance.now() === 0) return fixture.input.nowSeconds;
        if (failure === "throws") throw new Error("Synthetic clock failure");
        if (failure === "backwards") return fixture.input.nowSeconds - 1;
        return failure === "fractional" ? fixture.input.nowSeconds + 0.5 : NaN;
      });
      expect(result).toEqual({ status: "invalid", reason: "invalid_request" });
      expect(fixture.calls).toHaveLength(5);
    },
  );

  it.each(["signer", "token", "expiry"])(
    "never waits for an otherwise invalid %s",
    async (field) => {
      const fixture = await futureFixture();
      if (field === "signer") fixture.sasAccount.data.fill(0, 132, 164);
      if (field === "token") fixture.sasAccount.data[172] = 1;
      if (field === "expiry")
        fixture.sasAccount.data.writeBigInt64LE(
          BigInt(fixture.input.nowSeconds + 1),
          164,
        );
      const result = await finishRead(fixture);
      expect(result).toMatchObject({
        status: "available",
        evidence: { attestation: { status: "invalid" } },
      });
      expect(fixture.calls).toHaveLength(5);
      expect(performance.now()).toBe(0);
    },
  );

  it.each(["expiry", "reset", "commitment", "rebaseline", "revocation"])(
    "rechecks %s after waiting",
    async (change) => {
      const fixture = await futureFixture();
      const replacement = await createEvidenceFixture({
        lastVerificationTimestamp: fixture.input.nowSeconds + 1,
        lastResetTimestamp:
          change === "reset" ? fixture.input.nowSeconds + 1 : 0,
        lastRebaselineTimestamp:
          change === "rebaseline" ? fixture.input.nowSeconds + 1 : 0,
      });
      const getAccount = fixture.connection.getAccountInfoAndContext;
      fixture.connection.getAccountInfoAndContext = async (...args) => {
        if (performance.now() > 0) {
          if (change === "reset" || change === "rebaseline")
            fixture.identityAccount.data = replacement.identityAccount.data;
          if (change === "commitment") fixture.identityAccount.data[62] = 4;
          if (change === "expiry")
            fixture.sasAccount.data.writeBigInt64LE(
              BigInt(fixture.input.nowSeconds + 1),
              164,
            );
          if (
            change === "revocation" &&
            args[0].equals(fixture.attestationPda)
          ) {
            const response = await getAccount(...args);
            return { ...response, value: null };
          }
        }
        return getAccount(...args);
      };
      const result = await finishRead(fixture);
      if (change === "revocation" || change === "expiry") {
        expect(result).toMatchObject({
          status: "available",
          evidence: {
            attestation: {
              status: change === "revocation" ? "missing" : "invalid",
            },
          },
        });
      } else {
        expect(result.status).toBe("invalid");
      }
      expect(
        fixture.calls.filter((call) => call === "transaction"),
      ).toHaveLength(2);
    },
  );

  it("does not wait again for a newly future-dated attestation", async () => {
    const fixture = await futureFixture();
    const getAccount = fixture.connection.getAccountInfoAndContext;
    fixture.connection.getAccountInfoAndContext = async (...args) => {
      if (performance.now() > 0)
        fixture.sasAccount.data.writeBigInt64LE(BigInt(liveClock() + 1), 104);
      return getAccount(...args);
    };
    const result = await finishRead(fixture);
    expect(result).toMatchObject({
      status: "available",
      evidence: { attestation: { status: "invalid" } },
    });
    expect(fixture.calls).toHaveLength(10);
    expect(performance.now()).toBe(1000);
  });

  it.each(["before", "after"])(
    "shares one propagation retry %s clock reconciliation",
    async (order) => {
      const fixture = await futureFixture();
      const getTransaction = fixture.connection.getParsedTransaction;
      let reads = 0;
      fixture.connection.getParsedTransaction = async (...args) => {
        const transaction = await getTransaction(...args);
        reads += 1;
        if (
          (order === "before" && reads === 1) ||
          (order === "after" && reads === 2)
        )
          return null;
        return transaction;
      };
      const result = await finishRead(fixture);
      expect(result).toMatchObject({
        status: "available",
        evidence: { attestation: { status: "present" } },
      });
      expect(reads).toBe(3);
      expect(performance.now()).toBe(1000);
    },
  );

  it("does not reset the propagation retry after reconciliation", async () => {
    const fixture = await futureFixture();
    const getTransaction = fixture.connection.getParsedTransaction;
    let reads = 0;
    fixture.connection.getParsedTransaction = async (...args) => {
      const transaction = await getTransaction(...args);
      reads += 1;
      return reads === 2 ? transaction : null;
    };
    const result = await finishRead(fixture);
    expect(result).toEqual({
      status: "unavailable",
      reason: "transaction_unavailable",
    });
    expect(reads).toBe(3);
  });
});
