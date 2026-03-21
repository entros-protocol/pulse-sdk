import type { AudioCapture, CaptureOptions } from "./types";
import { MIN_CAPTURE_MS, MAX_CAPTURE_MS } from "../config";

const TARGET_SAMPLE_RATE = 16000;

/**
 * Capture audio at 16kHz until signaled to stop.
 * Uses ScriptProcessorNode for raw PCM sample access.
 *
 * Stop behavior:
 * - If signal fires before minDurationMs, capture continues until minimum is reached.
 * - If signal never fires, capture auto-stops at maxDurationMs.
 * - If no signal provided, captures for maxDurationMs.
 */
export async function captureAudio(
  options: CaptureOptions = {}
): Promise<AudioCapture> {
  const {
    signal,
    minDurationMs = MIN_CAPTURE_MS,
    maxDurationMs = MAX_CAPTURE_MS,
    onAudioLevel,
  } = options;

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
  const startTime = performance.now();

  return new Promise((resolve) => {
    let stopped = false;
    const bufferSize = 4096;
    const processor = ctx.createScriptProcessor(bufferSize, 1, 1);

    processor.onaudioprocess = (e: AudioProcessingEvent) => {
      const data = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(data));

      if (onAudioLevel) {
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i]! * data[i]!;
        onAudioLevel(Math.sqrt(sum / data.length));
      }
    };

    source.connect(processor);
    processor.connect(ctx.destination);

    function stopCapture() {
      if (stopped) return;
      stopped = true;
      clearTimeout(maxTimer);

      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t: MediaStreamTrack) => t.stop());
      ctx.close().catch(() => {});

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
    }

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
