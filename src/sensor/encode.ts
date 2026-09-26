/**
 * Encode captured Float32 audio samples as base64 int16 PCM for transmission
 * to the validation service.
 *
 * Audio is captured as `Float32Array` with values in `[-1.0, 1.0]` by the
 * Pulse SDK (`sensor/audio.ts`). The validation service decodes the base64
 * payload and feeds the audio into server-side transcription. int16 is the
 * standard compact representation: 2 bytes per sample vs 4 for f32, halving
 * wire size without perceptible quality loss for 16kHz speech.
 *
 * Byte layout: little-endian int16 samples, contiguous, no header.
 */

/**
 * Convert Float32 PCM samples to base64-encoded 16-bit little-endian PCM.
 * Samples are clamped to [-1, 1] and scaled. Uses `btoa`, a DOM global
 * available in browser runtimes and in Node 16+.
 */
export function encodeAudioAsBase64(samples: Float32Array): string {
  return bytesToBase64(encodePcm16(samples));
}

/**
 * Convert Float32 samples to 16-bit little-endian PCM bytes. Samples are
 * clamped to [-1, 1]. Negatives scale by 32768 and positives by 32767, and
 * `Math.round` sends ties toward positive infinity.
 */
export function encodePcm16(samples: Float32Array): Uint8Array {
  const buf = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]!));
    const int16 = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
    view.setInt16(i * 2, int16, true);
  }
  return new Uint8Array(buf);
}

/**
 * Convert 16-bit little-endian PCM bytes back to Float32 samples. Divides by
 * 32768 for every sample, as the validator does, so each value is exact.
 */
export function decodePcm16(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(Math.floor(bytes.byteLength / 2));
  for (let i = 0; i < out.length; i++) {
    out[i] = view.getInt16(i * 2, true) / 0x8000;
  }
  return out;
}

export function bytesToBase64(bytes: Uint8Array): string {
  // `btoa` is a DOM global and is also available as a Node global since
  // Node 16 (2021), which covers every runtime the SDK ships into. Chunk
  // the input to avoid "maximum call stack size" on large arrays — btoa
  // needs a string, and `String.fromCharCode(...bytes)` blows the stack
  // for Uint8Array length > ~128KB.
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}
