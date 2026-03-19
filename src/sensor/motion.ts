import type { MotionSample } from "./types";

const CAPTURE_DURATION_MS = 7000;

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
 * Capture accelerometer + gyroscope data for the specified duration.
 * Samples at the device's native rate (typically ~60-100Hz).
 */
export async function captureMotion(
  durationMs: number = CAPTURE_DURATION_MS
): Promise<MotionSample[]> {
  const hasPermission = await requestMotionPermission();
  if (!hasPermission) return [];

  const samples: MotionSample[] = [];

  return new Promise((resolve) => {
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

    window.addEventListener("devicemotion", handler);

    setTimeout(() => {
      window.removeEventListener("devicemotion", handler);
      resolve(samples);
    }, durationMs);
  });
}
