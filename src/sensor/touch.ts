import type { TouchSample, CaptureOptions } from "./types";
import { MIN_CAPTURE_MS, MAX_CAPTURE_MS } from "../config";
import { sdkLog } from "../log";

/**
 * Capture touch/pointer data (position, pressure, contact area) until signaled to stop.
 * Uses PointerEvent for cross-platform support (touch, pen, mouse).
 */
export function captureTouch(
  element: HTMLElement,
  options: CaptureOptions = {}
): Promise<TouchSample[]> {
  const {
    signal,
    minDurationMs = MIN_CAPTURE_MS,
    maxDurationMs = MAX_CAPTURE_MS,
  } = options;

  const samples: TouchSample[] = [];
  const startTime = performance.now();

  return new Promise((resolve) => {
    let stopped = false;

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

    function stopCapture() {
      if (stopped) return;
      stopped = true;
      clearTimeout(maxTimer);
      element.removeEventListener("pointermove", handler);
      element.removeEventListener("pointerdown", handler);
      sdkLog(`[Entros SDK] Touch capture stopped: ${samples.length} samples collected`);
      resolve(samples);
    }

    element.addEventListener("pointermove", handler);
    element.addEventListener("pointerdown", handler);
    sdkLog(`[Entros SDK] Touch capture started on <${element.tagName}>, listening for pointer events`);

    const maxTimer = setTimeout(stopCapture, maxDurationMs);

    if (signal) {
      if (signal.aborted) {
        setTimeout(stopCapture, minDurationMs);
      } else {
        signal.addEventListener(
          "abort",
          () => {
            const elapsed = performance.now() - startTime;
            const remaining = Math.max(0, minDurationMs - elapsed);
            setTimeout(stopCapture, remaining);
          },
          { once: true }
        );
      }
    }
  });
}
