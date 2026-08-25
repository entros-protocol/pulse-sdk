import type { TouchSample, CaptureOptions } from "./types";
import { MIN_CAPTURE_MS, MAX_CAPTURE_MS } from "../config";
import { sdkLog } from "../log";
import { getProjectionDefinition } from "../projection";

const TOUCH_CAPTURE_RATE_HZ = 60;
const TOUCH_MAX_INPUT_RATE_HZ = 240;
const TOUCH_CANONICAL_RATE_HZ = 30;
const TOUCH_MAX_DURATION_MS = 60_000;
const TOUCH_MAX_SOURCE_SAMPLES =
  Math.ceil((TOUCH_MAX_DURATION_MS * TOUCH_MAX_INPUT_RATE_HZ) / 1_000) + 2;
const TOUCH_MAX_SOURCE_GAP_MS = 250;
const TOUCH_MIN_CANONICAL_POINTS = 10;
const TOUCH_RECT_TOLERANCE_PX = 0.25;

interface FrozenRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

function readValidRect(surface: HTMLElement): FrozenRect {
  const { left, top, width, height } = surface.getBoundingClientRect();
  if (
    ![left, top, width, height].every(Number.isFinite) ||
    width <= 0 ||
    height <= 0
  ) {
    throw new Error("Touch coordinate surface must have finite positive dimensions");
  }
  return { left, top, width, height };
}

function sameRect(left: FrozenRect, right: FrozenRect): boolean {
  return (
    Math.abs(left.left - right.left) <= TOUCH_RECT_TOLERANCE_PX &&
    Math.abs(left.top - right.top) <= TOUCH_RECT_TOLERANCE_PX &&
    Math.abs(left.width - right.width) <= TOUCH_RECT_TOLERANCE_PX &&
    Math.abs(left.height - right.height) <= TOUCH_RECT_TOLERANCE_PX
  );
}

function validateNormalizedCaptureDuration(
  minDurationMs: number,
  maxDurationMs: number,
): void {
  if (
    !Number.isFinite(minDurationMs) ||
    !Number.isFinite(maxDurationMs) ||
    minDurationMs < 0 ||
    maxDurationMs <= 0 ||
    minDurationMs > maxDurationMs ||
    maxDurationMs > TOUCH_MAX_DURATION_MS
  ) {
    throw new Error(
      "Normalized touch capture duration is outside the supported range",
    );
  }
}

function canonicalPointCount(durationMs: number): number {
  return Math.floor((durationMs * TOUCH_CANONICAL_RATE_HZ) / 1_000) + 1;
}

/** Normalize projection-2 touch samples onto an endpoint-preserving 30 Hz grid. */
export function canonicalizeTouchSamples(
  samples: TouchSample[],
  projectionVersion: number,
): TouchSample[] {
  const definition = getProjectionDefinition(projectionVersion);
  if (definition.featurePipeline !== "normalized-touch") return samples;
  if (samples.length === 0) {
    throw new Error("Normalized touch capture requires source samples");
  }
  if (samples.length > TOUCH_MAX_SOURCE_SAMPLES) {
    throw new Error("Normalized touch capture exceeded the source sample limit");
  }
  if (samples.length < 2) {
    throw new Error("Normalized touch capture has insufficient source samples");
  }

  const source: TouchSample[] = [];
  for (const sample of samples) {
    if (
      ![
        sample.timestamp,
        sample.x,
        sample.y,
        sample.pressure,
        sample.width,
        sample.height,
      ].every(Number.isFinite)
    ) {
      throw new Error("Normalized touch capture contains a non-finite value");
    }
    if (sample.x < 0 || sample.x > 1 || sample.y < 0 || sample.y > 1) {
      throw new Error("Normalized touch coordinates must stay inside the unit surface");
    }
    if (sample.pressure < 0 || sample.pressure > 1) {
      throw new Error("Normalized touch pressure must stay inside the unit interval");
    }

    const previous = source[source.length - 1];
    if (previous && sample.timestamp < previous.timestamp) {
      throw new Error("Normalized touch timestamps must be monotonic");
    }
    const normalized = { ...sample, width: 1, height: 1 };
    if (previous && sample.timestamp === previous.timestamp) {
      source[source.length - 1] = normalized;
    } else {
      source.push(normalized);
    }
  }

  if (source.length < 2) {
    throw new Error("Normalized touch capture has insufficient distinct timestamps");
  }
  for (let index = 1; index < source.length; index++) {
    if (source[index]!.timestamp - source[index - 1]!.timestamp > TOUCH_MAX_SOURCE_GAP_MS) {
      throw new Error("Normalized touch capture clock was interrupted");
    }
  }

  const firstAt = source[0]!.timestamp;
  const lastAt = source[source.length - 1]!.timestamp;
  const durationMs = lastAt - firstAt;
  if (durationMs <= 0 || durationMs > TOUCH_MAX_DURATION_MS) {
    throw new Error("Normalized touch capture duration is outside the supported range");
  }
  const pointCount = canonicalPointCount(durationMs);
  if (pointCount < TOUCH_MIN_CANONICAL_POINTS) {
    throw new Error("Normalized touch capture has insufficient duration");
  }

  const output = new Array<TouchSample>(pointCount);
  let cursor = 0;
  for (let index = 0; index < pointCount; index++) {
    const timestamp =
      index === pointCount - 1
        ? lastAt
        : firstAt + (index * durationMs) / (pointCount - 1);
    while (
      cursor + 1 < source.length &&
      source[cursor + 1]!.timestamp < timestamp
    ) {
      cursor += 1;
    }

    const left = source[cursor]!;
    const right = source[Math.min(cursor + 1, source.length - 1)]!;
    if (timestamp === right.timestamp) {
      output[index] = { ...right, width: 1, height: 1 };
      continue;
    }
    const span = right.timestamp - left.timestamp;
    const fraction = span > 0 ? (timestamp - left.timestamp) / span : 0;
    output[index] = {
      timestamp,
      x: left.x + (right.x - left.x) * fraction,
      y: left.y + (right.y - left.y) * fraction,
      pressure: left.pressure + (right.pressure - left.pressure) * fraction,
      width: 1,
      height: 1,
    };
  }
  return output;
}

/**
 * Capture touch/pointer data (position, pressure, contact area) until signaled to stop.
 * Uses PointerEvent for cross-platform support (touch, pen, mouse).
 */
export function captureTouch(
  element: HTMLElement,
  options: CaptureOptions = {}
): Promise<TouchSample[]> {
  return captureTouchWithCompatibility(element, options).then(
    (capture) => capture.samples,
  );
}

export interface TouchCaptureResult {
  samples: TouchSample[];
  compatibilitySamples?: TouchSample[];
}

export function captureTouchWithCompatibility(
  element: HTMLElement,
  options: CaptureOptions = {},
): Promise<TouchCaptureResult> {
  const {
    signal,
    minDurationMs = MIN_CAPTURE_MS,
    maxDurationMs = MAX_CAPTURE_MS,
    projectionVersion = 0,
    coordinateSurface,
  } = options;

  if (
    getProjectionDefinition(projectionVersion).featurePipeline ===
    "normalized-touch"
  ) {
    if (!coordinateSurface) {
      return Promise.reject(
        new Error("Normalized touch capture requires a coordinate surface"),
      );
    }
    return captureNormalizedTouch(element, coordinateSurface, {
      signal,
      minDurationMs,
      maxDurationMs,
      projectionVersion,
    });
  }

  const samples: TouchSample[] = [];
  const startTime = performance.now();

  return new Promise((resolve) => {
    let stopped = false;
    // See motion.ts for the abortTimer rationale — same pattern across
    // all three sensor modules.
    let abortTimer: ReturnType<typeof setTimeout> | null = null;

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
      if (abortTimer !== null) clearTimeout(abortTimer);
      element.removeEventListener("pointermove", handler);
      element.removeEventListener("pointerdown", handler);
      sdkLog(`[Entros SDK] Touch capture stopped: ${samples.length} samples collected`);
      resolve({ samples });
    }

    element.addEventListener("pointermove", handler);
    element.addEventListener("pointerdown", handler);
    sdkLog(`[Entros SDK] Touch capture started on <${element.tagName}>, listening for pointer events`);

    const maxTimer = setTimeout(stopCapture, maxDurationMs);

    if (signal) {
      if (signal.aborted) {
        abortTimer = setTimeout(stopCapture, minDurationMs);
      } else {
        signal.addEventListener(
          "abort",
          () => {
            const elapsed = performance.now() - startTime;
            const remaining = Math.max(0, minDurationMs - elapsed);
            abortTimer = setTimeout(stopCapture, remaining);
          },
          { once: true }
        );
      }
    }
  });
}

function captureNormalizedTouch(
  eventTarget: HTMLElement,
  coordinateSurface: HTMLElement,
  options: Required<
    Pick<
      CaptureOptions,
      "minDurationMs" | "maxDurationMs" | "projectionVersion"
    >
  > &
    Pick<CaptureOptions, "signal">,
): Promise<TouchCaptureResult> {
  const { signal, minDurationMs, maxDurationMs, projectionVersion } = options;
  validateNormalizedCaptureDuration(minDurationMs, maxDurationMs);
  const frozenRect = readValidRect(coordinateSurface);
  if (typeof eventTarget.setPointerCapture !== "function") {
    throw new Error(
      "Normalized touch capture requires pointer capture support",
    );
  }
  const samples: TouchSample[] = [];
  const compatibilitySamples: TouchSample[] = [];
  const startTime = performance.now();
  const sourceSampleLimit =
    Math.ceil((maxDurationMs * TOUCH_CAPTURE_RATE_HZ) / 1_000) + 2;
  const compatibilitySampleLimit = sourceSampleLimit * 4;

  return new Promise((resolve, reject) => {
    let stopped = false;
    let captureError: Error | null = null;
    let activePointerId: number | null = null;
    let capturedPointerId: number | null = null;
    let contactEnded = false;
    let latest: Omit<TouchSample, "timestamp"> | null = null;
    let abortTimer: ReturnType<typeof setTimeout> | null = null;
    let abortListener: (() => void) | null = null;
    let compatibilityEventCount = 0;
    let compatibilityStride = 1;

    const appendLatest = (timestamp: number): void => {
      if (!latest || captureError) return;
      if (samples.length >= sourceSampleLimit) {
        captureError = new Error(
          "Normalized touch capture exceeded the source sample limit",
        );
        return;
      }
      samples.push({ timestamp, ...latest });
    };

    const updateLatest = (event: PointerEvent): boolean => {
      if (
        ![
          event.clientX,
          event.clientY,
          event.pressure,
        ].every(Number.isFinite)
      ) {
        latest = null;
        return false;
      }
      const x = (event.clientX - frozenRect.left) / frozenRect.width;
      const y = (event.clientY - frozenRect.top) / frozenRect.height;
      if (
        x < 0 ||
        x > 1 ||
        y < 0 ||
        y > 1 ||
        event.pressure < 0 ||
        event.pressure > 1
      ) {
        latest = null;
        return false;
      }
      latest = {
        x,
        y,
        pressure: event.pressure,
        width: 1,
        height: 1,
      };
      return true;
    };

    const appendCompatibilitySample = (event: PointerEvent): void => {
      if (
        ![
          event.clientX,
          event.clientY,
          event.pressure,
          event.width,
          event.height,
        ].every(Number.isFinite)
      ) {
        captureError = new Error(
          "Normalized touch capture contains invalid compatibility data",
        );
        return;
      }
      compatibilityEventCount += 1;
      if (compatibilityEventCount % compatibilityStride !== 0) return;
      if (compatibilitySamples.length >= compatibilitySampleLimit) {
        // Preserve the capture span while bounding high-rate pointer streams.
        let writeIndex = 1;
        for (
          let readIndex = 2;
          readIndex < compatibilitySamples.length;
          readIndex += 2
        ) {
          compatibilitySamples[writeIndex] = compatibilitySamples[readIndex]!;
          writeIndex += 1;
        }
        compatibilitySamples.length = writeIndex;
        compatibilityStride *= 2;
        if (compatibilityEventCount % compatibilityStride !== 0) return;
      }
      compatibilitySamples.push({
        timestamp: performance.now(),
        x: event.clientX,
        y: event.clientY,
        pressure: event.pressure,
        width: event.width,
        height: event.height,
      });
    };

    const onPointerDown = (event: PointerEvent) => {
      if (activePointerId !== null) return;
      if (contactEnded) {
        captureError = new Error(
          "Normalized touch capture requires one continuous pointer contact",
        );
        return;
      }
      if (!updateLatest(event)) return;
      activePointerId = event.pointerId;
      try {
        eventTarget.setPointerCapture(event.pointerId);
        capturedPointerId = event.pointerId;
      } catch {
        activePointerId = null;
        latest = null;
        captureError = new Error(
          "Normalized touch capture could not capture the pointer",
        );
        return;
      }
      appendCompatibilitySample(event);
      appendLatest(performance.now());
    };
    const onPointerMove = (event: PointerEvent) => {
      if (event.pointerId !== activePointerId) return;
      if (!updateLatest(event)) {
        captureError = new Error(
          "Normalized touch capture left the coordinate surface",
        );
      } else {
        appendCompatibilitySample(event);
      }
    };
    const onPointerEnd = (event: PointerEvent) => {
      if (event.pointerId !== activePointerId) return;
      if (updateLatest(event)) {
        appendLatest(performance.now());
      } else {
        captureError = new Error(
          "Normalized touch capture left the coordinate surface",
        );
      }
      activePointerId = null;
      capturedPointerId = null;
      contactEnded = true;
      latest = null;
    };
    const onPointerCancel = (event: PointerEvent) => {
      if (event.pointerId !== activePointerId) return;
      activePointerId = null;
      capturedPointerId = null;
      contactEnded = true;
      latest = null;
      captureError = new Error("Normalized touch capture was interrupted");
    };
    const onLostPointerCapture = (event: PointerEvent) => {
      if (event.pointerId !== activePointerId) return;
      activePointerId = null;
      capturedPointerId = null;
      contactEnded = true;
      latest = null;
      captureError = new Error(
        "Normalized touch capture lost pointer capture",
      );
    };

    const sampleTimer = setInterval(() => {
      if (!captureError) {
        try {
          if (!sameRect(frozenRect, readValidRect(coordinateSurface))) {
            captureError = new Error(
              "Touch coordinate surface changed during capture",
            );
          }
        } catch (error) {
          captureError =
            error instanceof Error ? error : new Error(String(error));
        }
      }
      if (captureError) {
        latest = null;
        return;
      }
      appendLatest(performance.now());
    }, 1_000 / TOUCH_CAPTURE_RATE_HZ);

    const removeListeners = () => {
      eventTarget.removeEventListener("pointerdown", onPointerDown);
      eventTarget.removeEventListener("pointermove", onPointerMove);
      eventTarget.removeEventListener("pointerup", onPointerEnd);
      eventTarget.removeEventListener("pointercancel", onPointerCancel);
      eventTarget.removeEventListener(
        "lostpointercapture",
        onLostPointerCapture,
      );
      if (signal && abortListener) {
        signal.removeEventListener("abort", abortListener);
        abortListener = null;
      }
    };

    const stopCapture = () => {
      if (stopped) return;
      stopped = true;
      clearTimeout(maxTimer);
      clearInterval(sampleTimer);
      if (abortTimer !== null) clearTimeout(abortTimer);
      removeListeners();
      if (
        capturedPointerId !== null &&
        typeof eventTarget.releasePointerCapture === "function"
      ) {
        const pointerId = capturedPointerId;
        capturedPointerId = null;
        try {
          eventTarget.releasePointerCapture(pointerId);
        } catch {}
      }

      if (captureError) {
        reject(captureError);
        return;
      }

      let currentRect: FrozenRect;
      try {
        currentRect = readValidRect(coordinateSurface);
      } catch (error) {
        reject(error);
        return;
      }
      if (!sameRect(frozenRect, currentRect)) {
        reject(new Error("Touch coordinate surface changed during capture"));
        return;
      }
      const now = performance.now();
      if (latest && (!samples.length || now > samples[samples.length - 1]!.timestamp)) {
        appendLatest(now);
      }
      sdkLog(
        `[Entros SDK] Normalized touch capture stopped: ${samples.length} source samples collected`,
      );
      try {
        resolve({
          samples: canonicalizeTouchSamples(samples, projectionVersion),
          compatibilitySamples,
        });
      } catch (error) {
        reject(error);
      }
    };

    eventTarget.addEventListener("pointerdown", onPointerDown);
    eventTarget.addEventListener("pointermove", onPointerMove);
    eventTarget.addEventListener("pointerup", onPointerEnd);
    eventTarget.addEventListener("pointercancel", onPointerCancel);
    eventTarget.addEventListener(
      "lostpointercapture",
      onLostPointerCapture,
    );
    sdkLog(
      `[Entros SDK] Normalized touch capture started on <${eventTarget.tagName}>`,
    );

    const maxTimer = setTimeout(stopCapture, maxDurationMs);
    if (signal) {
      if (signal.aborted) {
        abortTimer = setTimeout(stopCapture, minDurationMs);
      } else {
        abortListener = () => {
          const remaining = Math.max(
            0,
            minDurationMs - (performance.now() - startTime),
          );
          abortTimer = setTimeout(stopCapture, remaining);
        };
        signal.addEventListener("abort", abortListener, { once: true });
      }
    }
  });
}
