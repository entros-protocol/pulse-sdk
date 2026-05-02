import type { AudioCapture, CaptureOptions } from "./types";
import { MIN_CAPTURE_MS, MAX_CAPTURE_MS } from "../config";

const TARGET_SAMPLE_RATE = 16000;

/**
 * Capture audio at 16kHz until signaled to stop.
 * Uses ScriptProcessorNode for raw PCM sample access.
 *
 * @privacyGuarantee Raw audio samples returned from this function are processed
 * locally by the SDK's feature extraction pipeline. The 134-feature derived
 * statistical summary is the only audio-related signal that crosses the
 * device boundary. The single sanctioned exception is the encoded base64
 * audio bytes sent to the validator's `/validate-features` endpoint for
 * server-side verification, which the validator processes ephemerally —
 * see entros.io for the privacy and threat model.
 *
 * NOTE: ScriptProcessorNode is deprecated in favor of AudioWorklet.
 * Migration planned for v1.0. ScriptProcessorNode is used because it
 * provides synchronous access to raw PCM samples without requiring a
 * separate worker file, which simplifies SDK distribution. All current
 * browsers still support it.
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
    stream: preAcquiredStream,
  } = options;

  const stream = preAcquiredStream ?? await navigator.mediaDevices.getUserMedia({
    audio: {
      sampleRate: TARGET_SAMPLE_RATE,
      channelCount: 1,
      // Capture without browser-side audio processing — preserves the
      // raw microphone signal for the SDK's downstream feature extraction
      // and for server-side validation. Audio cleanup intended for the
      // transcription path runs server-side, on a parallel path that
      // never feeds back to feature extraction. Matches the mobile SDK's
      // choice of Android's `MIC` source over `VOICE_RECOGNITION` —
      // same architectural decision, two platforms.
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      // OS-level voice isolation request (W3C Media Capture Extensions,
      // 2024). Activates the platform DSP on Chrome 124+ / ChromeOS and
      // surfaces Apple Voice Isolation Mic Mode on Safari macOS Sonoma+
      // / iOS 17+ when the user has it enabled in Control Center.
      // Silently ignored on browsers/OSes without support, so the
      // constraint costs nothing where it doesn't help. Distinct
      // mechanism from `noiseSuppression` above — that flag controls
      // WebRTC's hand-tuned AudioProcessingModule, this requests the
      // OS-native neural effect.
      // @ts-expect-error -- W3C Media Capture Extensions property; not
      // yet in lib.dom.d.ts as of TypeScript 6.0. Removing this directive
      // becomes a compile error once lib.dom catches up, signaling that
      // it can be deleted.
      voiceIsolation: true,
    },
  });

  // If anything between `getUserMedia` and the Promise constructor throws
  // (AudioContext construction, ctx.resume(), createMediaStreamSource) the
  // stream we just acquired would leak indefinitely. Wrap the setup in a
  // try-on-error path that stops the stream tracks before re-throwing.
  let ctx: AudioContext;
  let source: MediaStreamAudioSourceNode;
  let capturedSampleRate: number;
  try {
    ctx = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    await ctx.resume(); // Required on iOS — AudioContext may be suspended outside user gesture
    capturedSampleRate = ctx.sampleRate;
    source = ctx.createMediaStreamSource(stream);
  } catch (err) {
    // Stop tracks we already acquired; we can't acquire them again so leaks
    // would persist for the page lifetime if we don't release here.
    if (!preAcquiredStream) {
      stream.getTracks().forEach((t: MediaStreamTrack) => t.stop());
    }
    throw err;
  }
  const chunks: Float32Array[] = [];
  const startTime = performance.now();

  return new Promise((resolve) => {
    let stopped = false;
    // See motion.ts for the abortTimer rationale.
    let abortTimer: ReturnType<typeof setTimeout> | null = null;
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
      if (abortTimer !== null) clearTimeout(abortTimer);

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
        sampleRate: capturedSampleRate,
        duration: totalLength / capturedSampleRate,
      });
    }

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
