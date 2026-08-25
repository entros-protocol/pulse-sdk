import { afterEach, describe, expect, it, vi } from "vitest";
import { captureTouch } from "../src/sensor/touch";

type PointerListener = (event: PointerEvent) => void;

class FakeTouchElement {
  readonly tagName = "DIV";
  private readonly listeners = new Map<string, Set<PointerListener>>();

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

function pointer(overrides: Partial<PointerEvent> = {}): Partial<PointerEvent> {
  return {
    clientX: 120,
    clientY: 240,
    pressure: 0.6,
    width: 11,
    height: 9,
    pointerId: 1,
    ...overrides,
  };
}

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("touch capture boundary", () => {
  it("records viewport coordinates and CSS-pixel contact geometry", async () => {
    vi.useFakeTimers();
    vi.spyOn(performance, "now")
      .mockReturnValueOnce(1_000)
      .mockReturnValueOnce(1_010)
      .mockReturnValueOnce(1_020)
      .mockReturnValue(1_020);
    const element = new FakeTouchElement();
    const controller = new AbortController();
    const capture = captureTouch(element as unknown as HTMLElement, {
      signal: controller.signal,
      minDurationMs: 0,
      maxDurationMs: 1_000,
    });

    element.emit("pointerdown", pointer());
    element.emit(
      "pointermove",
      pointer({ clientX: 180, clientY: 260, pressure: 0.7, width: 12 }),
    );
    controller.abort();
    await vi.runAllTimersAsync();

    await expect(capture).resolves.toEqual([
      {
        timestamp: 1_010,
        x: 120,
        y: 240,
        pressure: 0.6,
        width: 11,
        height: 9,
      },
      {
        timestamp: 1_020,
        x: 180,
        y: 260,
        pressure: 0.7,
        width: 12,
        height: 9,
      },
    ]);
    expect(element.listenerCount()).toBe(0);
  });

  it("documents that the current capture mixes active pointer identifiers", async () => {
    vi.useFakeTimers();
    const element = new FakeTouchElement();
    const controller = new AbortController();
    const capture = captureTouch(element as unknown as HTMLElement, {
      signal: controller.signal,
      minDurationMs: 0,
      maxDurationMs: 1_000,
    });

    element.emit("pointerdown", pointer({ pointerId: 7, clientX: 20 }));
    element.emit("pointerdown", pointer({ pointerId: 9, clientX: 900 }));
    controller.abort();
    await vi.runAllTimersAsync();

    const samples = await capture;
    expect(samples.map((sample) => sample.x)).toEqual([20, 900]);
    expect(element.listenerCount()).toBe(0);
  });
});
