/** Capture-level gain, shared by normalization and paired pacing. */
export const TARGET_CAPTURE_RMS = 0.05;
export const MIN_RMS_FOR_NORMALIZATION = 1e-4;
export const MAX_NORMALIZATION_GAIN = 50;

export function captureNormalizationGain(rms: number): number {
  return rms < MIN_RMS_FOR_NORMALIZATION
    ? 1
    : Math.min(TARGET_CAPTURE_RMS / rms, MAX_NORMALIZATION_GAIN);
}
