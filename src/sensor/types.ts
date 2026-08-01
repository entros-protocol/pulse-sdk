/** Raw audio samples captured during the Pulse challenge */
export interface AudioCapture {
  /**
   * Mono PCM at {@link sampleRate}, band-limited and decimated to the canonical
   * rate, with any pre-prompt lead-in already discarded. This is the exact
   * buffer feature extraction reads and the exact buffer transmitted, which is
   * what keeps the client's claimed features and the validator's re-derivation
   * describing the same signal.
   */
  samples: Float32Array;
  sampleRate: number;
  /**
   * Length of {@link samples} in seconds. This is how much audio is being
   * handed over, not how long the recorder ran. The two differ once the
   * lead-in is trimmed or the transmitted-length cap bites.
   */
  duration: number;
  /**
   * Wall-clock instant, in the `performance.now()` domain, of the first sample
   * in {@link samples}. This is the recorder's own start plus whatever the
   * lead-in trim and the transmitted-length cap removed, so it tracks the audio
   * actually handed over rather than the audio recorded.
   *
   * Every other modality has to be aligned to this. {@link MotionSample.timestamp}
   * uses the same clock, so the two compare directly with no conversion. See
   * `extractAccelerationMagnitude`, which correlates against the F0 contour
   * derived from this exact buffer.
   */
  windowStartMs: number;
  /** Wall-clock instant just past the last sample. `windowStartMs + duration * 1000`. */
  windowEndMs: number;
  /**
   * What the microphone delivered over this window, measured **before**
   * {@link samples} was normalised. Reading the level of `samples` instead
   * reports the normalisation target, not the capture.
   *
   * See `describeInputLevel` for how to read `gainClipped` against
   * `voicedFrameRatio`, which is what separates a genuinely quiet microphone
   * from a host warning stricter than this pipeline's own tolerance.
   */
  inputLevel: {
    rms: number;
    peak: number;
    gain: number;
    gainClipped: boolean;
    voicedFrameRatio: number;
  };
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
  /**
   * Fires when the capture window opens, i.e. when the speak prompt appears.
   * Everything recorded before it is discarded.
   *
   * The recorder starts early on purpose, so `onReady` can gate the prompt on
   * audio genuinely flowing rather than on a fixed delay. The cost is that the
   * buffer opens with dead air covering the challenge fetch and the countdown,
   * and without this signal that silence is extracted from, fingerprinted and
   * transmitted as though it were speech.
   *
   * An `AbortSignal` rather than a callback so it matches `signal` above, is
   * host-to-SDK like `stream`, and can only fire once, so idempotence is a
   * property of the type instead of a guard someone can delete. Note the two
   * signals mean opposite things: `signal` closes the capture, this opens the
   * window inside it.
   *
   * The trim happens before feature extraction, so `features`, `f0_contour`
   * and the transmitted audio all derive from one buffer. Trimming only the
   * transmitted copy would shift the validator's f0 frames against the
   * client's by hundreds, against a five-frame correlation search.
   */
  captureWindowSignal?: AbortSignal;
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
