import type { TouchSample } from "./types";

const CAPTURE_DURATION_MS = 7000;

/**
 * Capture touch/pointer data (position, pressure, contact area) for the specified duration.
 * Uses PointerEvent for cross-platform support (touch, pen, mouse).
 */
export function captureTouch(
  element: HTMLElement,
  durationMs: number = CAPTURE_DURATION_MS
): Promise<TouchSample[]> {
  const samples: TouchSample[] = [];

  return new Promise((resolve) => {
    const handler = (e: PointerEvent) => {
      samples.push({
        timestamp: performance.now(),
        x: e.clientX,
        y: e.clientY,
        pressure: e.pressure,
        width: e.width,
        height: e.height,
      });
    };

    element.addEventListener("pointermove", handler);
    element.addEventListener("pointerdown", handler);

    setTimeout(() => {
      element.removeEventListener("pointermove", handler);
      element.removeEventListener("pointerdown", handler);
      resolve(samples);
    }, durationMs);
  });
}
