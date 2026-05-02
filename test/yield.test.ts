import { describe, expect, it } from "vitest";
import { yieldToMainThread } from "../src/yield";

describe("yieldToMainThread", () => {
  it("returns a Promise that resolves to undefined", async () => {
    await expect(yieldToMainThread()).resolves.toBeUndefined();
  });

  it("yields after pending microtasks have flushed", async () => {
    // Promise.resolve() schedules microtasks; yieldToMainThread should
    // schedule a macrotask via MessageChannel (or setTimeout in fallback
    // environments). All microtasks the event loop has queued before the
    // macrotask runs must flush first — so the order array below records
    // microtask values before the yield's resolution.
    const order: string[] = [];
    const yielded = yieldToMainThread().then(() => order.push("macrotask"));
    await Promise.resolve().then(() => order.push("microtask-1"));
    await Promise.resolve().then(() => order.push("microtask-2"));
    await yielded;
    expect(order).toEqual(["microtask-1", "microtask-2", "macrotask"]);
  });

  it("supports concurrent calls without cross-talk", async () => {
    // Each call allocates its own MessageChannel — verify two parallel
    // yields both resolve and don't share resolution by accident.
    const calls = await Promise.all([
      yieldToMainThread(),
      yieldToMainThread(),
      yieldToMainThread(),
    ]);
    expect(calls).toHaveLength(3);
    for (const value of calls) expect(value).toBeUndefined();
  });
});
