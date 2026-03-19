import type { AudioCapture } from "./types";

const TARGET_SAMPLE_RATE = 16000;
const CAPTURE_DURATION_MS = 7000;

/**
 * Capture audio at 16kHz for the specified duration.
 * Uses AudioWorklet for raw PCM sample access.
 * Falls back to ScriptProcessorNode if AudioWorklet is unavailable.
 */
export async function captureAudio(
  durationMs: number = CAPTURE_DURATION_MS
): Promise<AudioCapture> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      sampleRate: TARGET_SAMPLE_RATE,
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });

  const ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
  const source = ctx.createMediaStreamSource(stream);
  const chunks: Float32Array[] = [];

  return new Promise((resolve, reject) => {
    const bufferSize = 4096;
    // ScriptProcessorNode is deprecated but has wider support than AudioWorklet
    // and gives direct sample access. Sufficient for a 7-second capture.
    const processor = ctx.createScriptProcessor(bufferSize, 1, 1);

    processor.onaudioprocess = (e: AudioProcessingEvent) => {
      const data = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(data));
    };

    source.connect(processor);
    processor.connect(ctx.destination);

    setTimeout(() => {
      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t: MediaStreamTrack) => t.stop());
      ctx.close().catch(() => {});

      // Concatenate all chunks
      const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
      const samples = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        samples.set(chunk, offset);
        offset += chunk.length;
      }

      resolve({
        samples,
        sampleRate: ctx.sampleRate,
        duration: totalLength / ctx.sampleRate,
      });
    }, durationMs);
  });
}
