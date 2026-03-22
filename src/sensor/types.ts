/** Raw audio samples captured during the Pulse challenge */
export interface AudioCapture {
  samples: Float32Array;
  sampleRate: number;
  duration: number;
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
  modalities: {
    audio: boolean;
    motion: boolean;
    touch: boolean;
  };
}
