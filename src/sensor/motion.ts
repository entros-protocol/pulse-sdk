import type { MotionSample, CaptureOptions } from "./types";
import { MIN_CAPTURE_MS, MAX_CAPTURE_MS } from "../config";

/**
 * Request motion sensor permission (required on iOS 13+).
 * No-op on Android/Chrome where permission is implicit.
 */
export async function requestMotionPermission(): Promise<boolean> {
  const DME = (globalThis as any).DeviceMotionEvent;
  if (!DME) return false;

  if (typeof DME.requestPermission === "function") {
    const permission = await DME.requestPermission();
    return permission === "granted";
  }

  // Android/Chrome: permission is implicit
  return true;
}

/**
 * Capture accelerometer + gyroscope data until signaled to stop.
 * Samples at the device's native rate (typically ~60-100Hz).
 */
export async function captureMotion(
  options: CaptureOptions = {}
): Promise<MotionSample[]> {
  const {
    signal,
    minDurationMs = MIN_CAPTURE_MS,
    maxDurationMs = MAX_CAPTURE_MS,
  } = options;

  const hasPermission = await requestMotionPermission();
  if (!hasPermission) return [];

  const samples: MotionSample[] = [];
  const startTime = performance.now();

  return new Promise((resolve) => {
    let stopped = false;

    const handler = (e: DeviceMotionEvent) => {
      samples.push({
        timestamp: performance.now(),
        ax: e.acceleration?.x ?? 0,
        ay: e.acceleration?.y ?? 0,
        az: e.acceleration?.z ?? 0,
        gx: e.rotationRate?.alpha ?? 0,
        gy: e.rotationRate?.beta ?? 0,
        gz: e.rotationRate?.gamma ?? 0,
      });
    };

    function stopCapture() {
      if (stopped) return;
      stopped = true;
      clearTimeout(maxTimer);
      window.removeEventListener("devicemotion", handler);
      resolve(samples);
    }

    window.addEventListener("devicemotion", handler);

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
