/** Raw audio samples captured during the Pulse challenge */
export interface AudioCapture {
  samples: Float32Array;
  sampleRate: number;
  duration: number;
  virtualDevice?: boolean;
}


/** Single IMU reading */
export interface MotionSample {
  timestamp: number;
  ax: number;
  ay: number;
  az: number;
  gx: number;
  gy: number;
  gz: number;
}

/** Single touch reading */
export interface TouchSample {
  timestamp: number;
  x: number;
  y: number;
  pressure: number;
  width: number;
  height: number;
}

/**
 * Single raw curve-trace sample from the "trace the curve" challenge, in the
 * 200x200 viewBox coordinate frame. Captured on-device only; resampled to a
 * coarse, timestamp-free `CurveTraceOutline` before anything leaves the device.
 */
export interface CurveTracePoint {
  x: number;
  y: number;
  /**
   * `performance.now()` timestamp (ms); used only for on-device equal-time
   * resampling, never transmitted.
   */
  t: number;
}

/**
 * Coarse, equal-time-resampled curve outline sent to the validation service for
 * touch content-binding. Tuple points keep the wire form compact and match the
 * executor's `curve_trace.points: [[x,y],...]`; no per-point timestamps.
 */
export interface CurveTraceOutline {
  points: [number, number][];
  duration_ms: number;
}

/** Options for event-driven sensor capture */
export interface CaptureOptions {
  /** AbortSignal to stop capture. If omitted, captures for maxDurationMs. */
  signal?: AbortSignal;
  /** Minimum capture duration in ms. Capture continues until this even if signal fires early. Default: 2000 */
  minDurationMs?: number;
  /** Maximum capture duration in ms. Auto-stops if signal hasn't fired. Default: 60000 */
  maxDurationMs?: number;
  /** Called with RMS audio level (0-1) on each buffer during audio capture (~4x per second). */
  onAudioLevel?: (rms: number) => void;
  /**
   * Called once, the first time a real audio frame is delivered (capture is
   * live). Lets callers gate the "speak now" prompt on audio actually flowing
   * rather than a fixed delay, avoiding the first-attempt cold-start gap where
   * the AudioContext/mic isn't producing samples yet.
   */
  onReady?: () => void;
  /** Pre-acquired MediaStream. If provided, captureAudio skips getUserMedia. */
  stream?: MediaStream;
  /** If true, captureMotion skips requestMotionPermission (already acquired). */
  permissionGranted?: boolean;
}

/** Stage of a capture session */
export type CaptureStage = "audio" | "motion" | "touch";

/** State of an individual capture stage */
export type StageState = "idle" | "capturing" | "captured" | "skipped";

/** Combined sensor data from a Pulse capture session */
export interface SensorData {
  audio: AudioCapture | null;
  motion: MotionSample[];
  touch: TouchSample[];
  /**
   * Raw curve-trace outline from the "trace the curve" UI (wallet-connected
   * verify only). On-device only — resampled to a coarse `CurveTraceOutline` at
   * send time. Absent for reset and walletless flows.
   */
  curveTrace?: CurveTracePoint[];
  modalities: {
    audio: boolean;
    motion: boolean;
    touch: boolean;
  };
}
