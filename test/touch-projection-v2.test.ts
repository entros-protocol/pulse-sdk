import { afterEach, describe, expect, it, vi } from "vitest";
import {
  canonicalizeTouchSamples,
  captureTouch,
  captureTouchWithCompatibility,
} from "../src/sensor/touch";
import type { TouchSample } from "../src/sensor/types";
import { PulseSDK } from "../src/pulse";

type PointerListener = (event: PointerEvent) => void;

class FakeElement {
  readonly tagName = "DIV";
  private readonly listeners = new Map<string, Set<PointerListener>>();
  rectReads = 0;

  constructor(
    private rect = { left: 0, top: 0, width: 200, height: 100 },
  ) {}

  addEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    if (typeof listener !== "function") return;
    const listeners = this.listeners.get(type) ?? new Set<PointerListener>();
    listeners.add(listener as PointerListener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void {
    if (typeof listener !== "function") return;
    this.listeners.get(type)?.delete(listener as PointerListener);
  }

  getBoundingClientRect(): DOMRect {
    this.rectReads += 1;
    return this.rect as DOMRect;
  }

  setRect(rect: { left: number; top: number; width: number; height: number }): void {
    this.rect = rect;
  }

  emit(type: string, event: Partial<PointerEvent>): void {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event as PointerEvent);
    }
  }

  listenerCount(): number {
    return [...this.listeners.values()].reduce(
      (count, listeners) => count + listeners.size,
      0,
    );
  }
}

function event(overrides: Partial<PointerEvent> = {}): Partial<PointerEvent> {
  return {
    clientX: 100,
    clientY: 50,
    pressure: 0.5,
    width: 20,
    height: 10,
    pointerId: 1,
    ...overrides,
  };
}

function periodicSamples(rateHz: number, durationMs: number): TouchSample[] {
  const count = Math.round((durationMs * rateHz) / 1_000);
  return Array.from({ length: count + 1 }, (_, index) => {
    const timestamp = (index * durationMs) / count;
    const progress = timestamp / durationMs;
    return {
      timestamp: 50_000 + timestamp,
      x: 0.1 + progress * 0.7,
      y: 0.8 - progress * 0.4,
      pressure: 0.3 + progress * 0.2,
      width: 12,
      height: 8,
    };
  });
}

interface InternalTouchSession {
  readProjectionPolicy(): Promise<{ current: number; minimum: number }>;
  touchStageState: string;
}

function projectionTwoSession() {
  const session = new PulseSDK({
    cluster: "devnet",
    relayerUrl: "https://executor.test",
  }).createSession();
  const internal = session as unknown as InternalTouchSession;
  internal.readProjectionPolicy = async () => ({ current: 2, minimum: 1 });
  return { session, internal };
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("projection 2 touch grid", () => {
  it("preserves projections 0 and 1 exactly", () => {
    const samples = periodicSamples(60, 4_000);
    expect(canonicalizeTouchSamples(samples, 0)).toBe(samples);
    expect(canonicalizeTouchSamples(samples, 1)).toBe(samples);
  });

  it.each([
    [4_000, 121],
    [12_000, 361],
    [60_000, 1_801],
  ])("maps %i ms onto %i endpoint-preserving points", (durationMs, count) => {
    const output = canonicalizeTouchSamples(
      periodicSamples(60, durationMs),
      2,
    );
    expect(output).toHaveLength(count);
    expect(output[0]!.timestamp).toBe(50_000);
    expect(output.at(-1)!.timestamp).toBe(50_000 + durationMs);
    expect(output.every((sample) => sample.width === 1 && sample.height === 1)).toBe(true);
    const interval = durationMs / (count - 1);
    for (let index = 1; index < output.length; index++) {
      expect(output[index]!.timestamp - output[index - 1]!.timestamp).toBeCloseTo(
        interval,
        10,
      );
    }
  });

  it("keeps the canonical path stable across source event rates", () => {
    const thirty = canonicalizeTouchSamples(periodicSamples(30, 4_000), 2);
    for (const rate of [60, 120]) {
      const candidate = canonicalizeTouchSamples(periodicSamples(rate, 4_000), 2);
      expect(candidate).toHaveLength(thirty.length);
      for (let index = 0; index < thirty.length; index++) {
        expect(candidate[index]!.x).toBeCloseTo(thirty[index]!.x, 12);
        expect(candidate[index]!.y).toBeCloseTo(thirty[index]!.y, 12);
        expect(candidate[index]!.pressure).toBeCloseTo(
          thirty[index]!.pressure,
          12,
        );
      }
    }
  });

  it("is idempotent once samples are on the canonical grid", () => {
    const first = canonicalizeTouchSamples(periodicSamples(120, 4_000), 2);
    const second = canonicalizeTouchSamples(first, 2);

    expect(second).toEqual(first);
  });

  it("coalesces equal timestamps and rejects missing evidence", () => {
    expect(() => canonicalizeTouchSamples([], 2)).toThrow(
      "requires source samples",
    );

    const duplicate = periodicSamples(60, 4_000);
    duplicate.splice(2, 0, { ...duplicate[1]!, x: 0.75 });
    expect(canonicalizeTouchSamples(duplicate, 2)).toHaveLength(121);

    const toleratedGap = periodicSamples(60, 4_000);
    toleratedGap.splice(20, 7);
    expect(canonicalizeTouchSamples(toleratedGap, 2)).toHaveLength(121);

    const interrupted = periodicSamples(60, 4_000);
    interrupted.splice(20, 20);
    expect(() => canonicalizeTouchSamples(interrupted, 2)).toThrow(
      "capture clock was interrupted",
    );

    const decreasing = periodicSamples(60, 4_000);
    decreasing[3] = { ...decreasing[3]!, timestamp: decreasing[1]!.timestamp };
    expect(() => canonicalizeTouchSamples(decreasing, 2)).toThrow(
      "timestamps must be monotonic",
    );

    const nonFinite = periodicSamples(60, 4_000);
    nonFinite[3] = { ...nonFinite[3]!, x: Number.NaN };
    expect(() => canonicalizeTouchSamples(nonFinite, 2)).toThrow(
      "non-finite value",
    );

    const invalidPressure = periodicSamples(60, 4_000);
    invalidPressure[3] = { ...invalidPressure[3]!, pressure: 1.01 };
    expect(() => canonicalizeTouchSamples(invalidPressure, 2)).toThrow(
      "pressure must stay inside the unit interval",
    );

    expect(canonicalizeTouchSamples(periodicSamples(120, 60_000), 2)).toHaveLength(
      1_801,
    );
    const oversized = periodicSamples(241, 60_000);
    expect(() => canonicalizeTouchSamples(oversized, 2)).toThrow(
      "exceeded the source sample limit",
    );
  });
});

describe("projection 2 browser touch capture", () => {
  it("retains the raw pointer scale only for on-device compatibility extraction", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement({ left: 50, top: 100, width: 400, height: 200 });
    const controller = new AbortController();
    const capture = captureTouchWithCompatibility(
      target as unknown as HTMLElement,
      {
        signal: controller.signal,
        minDurationMs: 0,
        maxDurationMs: 1_000,
        projectionVersion: 2,
        coordinateSurface: surface as unknown as HTMLElement,
      },
    );

    target.emit(
      "pointerdown",
      event({ clientX: 150, clientY: 150, width: 20, height: 10 }),
    );
    for (let index = 1; index <= 16; index += 1) {
      target.emit(
        "pointermove",
        event({
          clientX: 150 + index * 10,
          clientY: 150 + index * 5,
          width: 20 + index,
          height: 10 + index,
        }),
      );
      await vi.advanceTimersByTimeAsync(20);
    }
    controller.abort();
    await vi.runAllTimersAsync();

    const result = await capture;
    expect(result.samples[0]).toMatchObject({
      x: 0.25,
      y: 0.25,
      width: 1,
      height: 1,
    });
    expect(result.compatibilitySamples).toHaveLength(17);
    expect(result.compatibilitySamples![0]).toMatchObject({
      x: 150,
      y: 150,
      width: 20,
      height: 10,
    });
    expect(result.compatibilitySamples!.at(-1)).toMatchObject({
      x: 310,
      y: 230,
      width: 36,
      height: 26,
    });
  });

  it("separates the event target from the unit coordinate surface", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement({ left: 50, top: 100, width: 400, height: 200 });
    const controller = new AbortController();
    const capture = captureTouch(target as unknown as HTMLElement, {
      signal: controller.signal,
      minDurationMs: 0,
      maxDurationMs: 1_000,
      projectionVersion: 2,
      coordinateSurface: surface as unknown as HTMLElement,
    });

    target.emit("pointerdown", event({ clientX: 150, clientY: 150, pointerId: 7 }));
    target.emit("pointerdown", event({ clientX: 450, clientY: 300, pointerId: 9 }));
    await vi.advanceTimersByTimeAsync(200);
    target.emit("pointermove", event({ clientX: 350, clientY: 250, pointerId: 7 }));
    await vi.advanceTimersByTimeAsync(200);
    controller.abort();
    await vi.runAllTimersAsync();

    const samples = await capture;
    expect(samples[0]).toMatchObject({ x: 0.25, y: 0.25, width: 1, height: 1 });
    expect(samples.at(-1)).toMatchObject({ x: 0.75, y: 0.75, width: 1, height: 1 });
    expect(samples.length).toBeGreaterThanOrEqual(10);
    expect(target.listenerCount()).toBe(0);
    expect(surface.listenerCount()).toBe(0);
  });

  it("captures equivalent unit paths on translated and scaled surfaces", async () => {
    vi.useFakeTimers();

    const capturePath = async (rect: {
      left: number;
      top: number;
      width: number;
      height: number;
    }) => {
      const target = new FakeElement();
      const surface = new FakeElement(rect);
      const controller = new AbortController();
      const capture = captureTouch(target as unknown as HTMLElement, {
        signal: controller.signal,
        minDurationMs: 0,
        maxDurationMs: 1_000,
        projectionVersion: 2,
        coordinateSurface: surface as unknown as HTMLElement,
      });

      target.emit(
        "pointerdown",
        event({
          clientX: rect.left + rect.width * 0.2,
          clientY: rect.top + rect.height * 0.3,
        }),
      );
      await vi.advanceTimersByTimeAsync(200);
      target.emit(
        "pointermove",
        event({
          clientX: rect.left + rect.width * 0.8,
          clientY: rect.top + rect.height * 0.7,
        }),
      );
      await vi.advanceTimersByTimeAsync(200);
      controller.abort();
      await vi.runAllTimersAsync();
      return capture;
    };

    const compact = await capturePath({
      left: 10,
      top: 20,
      width: 200,
      height: 100,
    });
    const expanded = await capturePath({
      left: 400,
      top: 300,
      width: 800,
      height: 600,
    });

    expect(expanded).toHaveLength(compact.length);
    for (let index = 0; index < compact.length; index++) {
      expect(expanded[index]!.x).toBeCloseTo(compact[index]!.x, 12);
      expect(expanded[index]!.y).toBeCloseTo(compact[index]!.y, 12);
      expect(expanded[index]!.pressure).toBeCloseTo(
        compact[index]!.pressure,
        12,
      );
    }
  });

  it("fails before capture for an invalid surface and rejects resizing", async () => {
    const target = new FakeElement();
    const invalid = new FakeElement({ left: 0, top: 0, width: 0, height: 100 });
    expect(() =>
      captureTouch(target as unknown as HTMLElement, {
        projectionVersion: 2,
        coordinateSurface: invalid as unknown as HTMLElement,
      }),
    ).toThrow("finite positive dimensions");
    expect(target.listenerCount()).toBe(0);

    vi.useFakeTimers();
    const surface = new FakeElement();
    const controller = new AbortController();
    const capture = captureTouch(target as unknown as HTMLElement, {
      signal: controller.signal,
      minDurationMs: 0,
      maxDurationMs: 1_000,
      projectionVersion: 2,
      coordinateSurface: surface as unknown as HTMLElement,
    });
    const resized = expect(capture).rejects.toThrow("changed during capture");
    target.emit("pointerdown", event());
    surface.setRect({ left: 0, top: 0, width: 300, height: 100 });
    controller.abort();
    await vi.runAllTimersAsync();
    await resized;
    expect(target.listenerCount()).toBe(0);
  });

  it("rejects a transient surface change even after geometry returns", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement();
    const controller = new AbortController();
    const capture = captureTouch(target as unknown as HTMLElement, {
      signal: controller.signal,
      minDurationMs: 0,
      maxDurationMs: 1_000,
      projectionVersion: 2,
      coordinateSurface: surface as unknown as HTMLElement,
    });
    const rejected = expect(capture).rejects.toThrow("changed during capture");

    target.emit("pointerdown", event());
    surface.setRect({ left: 100, top: 0, width: 200, height: 100 });
    await vi.advanceTimersByTimeAsync(20);
    surface.setRect({ left: 0, top: 0, width: 200, height: 100 });
    controller.abort();
    await vi.runAllTimersAsync();

    await rejected;
    expect(target.listenerCount()).toBe(0);
  });

  it("keeps the session reusable after an invalid coordinate surface", async () => {
    const target = new FakeElement();
    const invalid = new FakeElement({ left: 0, top: 0, width: 0, height: 100 });
    const { session, internal } = projectionTwoSession();

    await expect(
      session.startTouch({
        eventTarget: target as unknown as HTMLElement,
        coordinateSurface: invalid as unknown as HTMLElement,
      }),
    ).rejects.toThrow("finite positive dimensions");

    expect(internal.touchStageState).toBe("idle");
    expect(() => session.skipTouch()).not.toThrow();
    expect(target.listenerCount()).toBe(0);
  });

  it("marks an interrupted session as failed", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement();
    const { session, internal } = projectionTwoSession();

    await session.startTouch({
      eventTarget: target as unknown as HTMLElement,
      coordinateSurface: surface as unknown as HTMLElement,
    });
    target.emit("pointerdown", event());
    surface.setRect({ left: 0, top: 0, width: 300, height: 100 });
    const stopped = session.stopTouch();
    const rejected = expect(stopped).rejects.toThrow("changed during capture");
    await vi.runAllTimersAsync();
    await rejected;

    expect(internal.touchStageState).toBe("failed");
    await expect(session.complete()).resolves.toMatchObject({
      success: false,
      failedAt: "capture",
    });
    expect(target.listenerCount()).toBe(0);
  });

  it("consumes timer-driven capture failures until the host stops capture", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement();
    const { session, internal } = projectionTwoSession();

    await session.startTouch({
      eventTarget: target as unknown as HTMLElement,
      coordinateSurface: surface as unknown as HTMLElement,
    });
    await vi.runAllTimersAsync();

    await expect(session.stopTouch()).rejects.toThrow("requires source samples");
    expect(internal.touchStageState).toBe("failed");
    expect(target.listenerCount()).toBe(0);
  });

  it("removes the abort listener after timer-driven completion", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement();
    const controller = new AbortController();
    const removeListener = vi.spyOn(controller.signal, "removeEventListener");
    const capture = captureTouch(target as unknown as HTMLElement, {
      signal: controller.signal,
      minDurationMs: 0,
      maxDurationMs: 400,
      projectionVersion: 2,
      coordinateSurface: surface as unknown as HTMLElement,
    });

    target.emit("pointerdown", event());
    await vi.runAllTimersAsync();
    await expect(capture).resolves.toHaveLength(13);

    expect(removeListener).toHaveBeenCalledWith("abort", expect.any(Function));
    expect(target.listenerCount()).toBe(0);
  });

  it("does not treat lifted contact as continued touch evidence", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement();
    const capture = captureTouch(target as unknown as HTMLElement, {
      minDurationMs: 0,
      maxDurationMs: 1_000,
      projectionVersion: 2,
      coordinateSurface: surface as unknown as HTMLElement,
    });
    const rejected = expect(capture).rejects.toThrow("insufficient duration");

    target.emit("pointerdown", event());
    await vi.advanceTimersByTimeAsync(100);
    target.emit("pointerup", event());
    await vi.runAllTimersAsync();

    await rejected;
    expect(target.listenerCount()).toBe(0);
  });

  it("rejects a second stroke instead of interpolating across a lift", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement();
    const controller = new AbortController();
    const capture = captureTouch(target as unknown as HTMLElement, {
      signal: controller.signal,
      minDurationMs: 0,
      maxDurationMs: 1_000,
      projectionVersion: 2,
      coordinateSurface: surface as unknown as HTMLElement,
    });
    const rejected = expect(capture).rejects.toThrow("one continuous pointer contact");

    target.emit("pointerdown", event());
    await vi.advanceTimersByTimeAsync(200);
    target.emit("pointerup", event());
    target.emit("pointerdown", event({ pointerId: 2 }));
    controller.abort();
    await vi.runAllTimersAsync();

    await rejected;
    expect(target.listenerCount()).toBe(0);
  });

  it.each([
    ["pointermove", event({ clientX: 250 })],
    ["pointercancel", event()],
  ])("rejects an interrupted trace after %s", async (eventType, pointerEvent) => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement();
    const controller = new AbortController();
    const capture = captureTouch(target as unknown as HTMLElement, {
      signal: controller.signal,
      minDurationMs: 0,
      maxDurationMs: 1_000,
      projectionVersion: 2,
      coordinateSurface: surface as unknown as HTMLElement,
    });
    const rejected = expect(capture).rejects.toThrow(
      /left the coordinate surface|was interrupted/,
    );

    target.emit("pointerdown", event());
    target.emit(eventType, pointerEvent);
    controller.abort();
    await vi.runAllTimersAsync();

    await rejected;
    expect(target.listenerCount()).toBe(0);
  });

  it("tolerates subpixel layout noise and rejects invalid durations", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement();
    const controller = new AbortController();
    const capture = captureTouch(target as unknown as HTMLElement, {
      signal: controller.signal,
      minDurationMs: 0,
      maxDurationMs: 1_000,
      projectionVersion: 2,
      coordinateSurface: surface as unknown as HTMLElement,
    });

    target.emit("pointerdown", event());
    await vi.advanceTimersByTimeAsync(400);
    surface.setRect({ left: 0.2, top: 0.2, width: 200.2, height: 100.2 });
    controller.abort();
    await vi.runAllTimersAsync();
    await expect(capture).resolves.toHaveLength(13);

    for (const durations of [
      { minDurationMs: -1, maxDurationMs: 1_000 },
      { minDurationMs: 2_000, maxDurationMs: 1_000 },
      { minDurationMs: 0, maxDurationMs: 60_001 },
    ]) {
      expect(() =>
        captureTouch(target as unknown as HTMLElement, {
          ...durations,
          projectionVersion: 2,
          coordinateSurface: surface as unknown as HTMLElement,
        }),
      ).toThrow("outside the supported range");
    }
    expect(target.listenerCount()).toBe(0);
  });

  it("bounds source samples and avoids layout reads in pointer handlers", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement();
    const controller = new AbortController();
    const capture = captureTouchWithCompatibility(
      target as unknown as HTMLElement,
      {
        signal: controller.signal,
        minDurationMs: 0,
        maxDurationMs: 1_000,
        projectionVersion: 2,
        coordinateSurface: surface as unknown as HTMLElement,
      },
    );

    target.emit("pointerdown", event());
    const readsBeforeStorm = surface.rectReads;
    for (let index = 0; index < 1_500; index++) {
      target.emit(
        "pointermove",
        event({
          clientX: 20 + (index % 160),
          clientY: 20 + (index % 60),
        }),
      );
    }
    expect(surface.rectReads).toBe(readsBeforeStorm);
    await vi.advanceTimersByTimeAsync(400);
    controller.abort();
    await vi.runAllTimersAsync();

    const result = await capture;
    expect(result.samples).toHaveLength(13);
    expect(result.compatibilitySamples!.length).toBeGreaterThanOrEqual(10);
    expect(result.compatibilitySamples!.length).toBeLessThanOrEqual(248);
    expect(
      result.compatibilitySamples!.every((sample) =>
        [
          sample.timestamp,
          sample.x,
          sample.y,
          sample.pressure,
          sample.width,
          sample.height,
        ].every(Number.isFinite),
      ),
    ).toBe(true);
    expect(surface.rectReads).toBeLessThanOrEqual(30);
    expect(target.listenerCount()).toBe(0);
  });

  it("bounds a maximum-duration capture under high-rate pointer pressure", async () => {
    vi.useFakeTimers();
    const target = new FakeElement();
    const surface = new FakeElement();
    const controller = new AbortController();
    const capture = captureTouchWithCompatibility(
      target as unknown as HTMLElement,
      {
        signal: controller.signal,
        minDurationMs: 0,
        maxDurationMs: 60_000,
        projectionVersion: 2,
        coordinateSurface: surface as unknown as HTMLElement,
      },
    );

    target.emit("pointerdown", event());
    for (let index = 0; index < 100_000; index += 1) {
      target.emit(
        "pointermove",
        event({
          clientX: 20 + (index % 160),
          clientY: 20 + (index % 60),
        }),
      );
    }
    await vi.advanceTimersByTimeAsync(400);
    controller.abort();
    await vi.runAllTimersAsync();

    const result = await capture;
    expect(result.samples).toHaveLength(13);
    expect(result.compatibilitySamples!.length).toBeLessThanOrEqual(14_408);
    expect(target.listenerCount()).toBe(0);
  });
});
