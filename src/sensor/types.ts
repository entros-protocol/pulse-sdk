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

/** Combined sensor data from a 7-second Pulse capture */
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
