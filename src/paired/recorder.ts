/**
 * One continuous microphone recording for a paired session.
 *
 * The recorder canonicalises audio as it arrives and cuts it into 800-sample
 * frames for the round tracker. Every position is a canonical sample index, so
 * a round's mark, its trim and its segment all refer to one clock. Audio before
 * the current round is released once that round's segment is built, which
 * bounds memory by one round.
 */

import {
  audioCaptureConstraints,
  detectVirtualAudioInput,
  readVoiceIsolationApplied,
} from "../sensor/audio";
import { CANONICAL_SAMPLE_RATE, createStreamingCanonicalizer } from "../sensor/resample";
import { FRAME_SAMPLES, frameRms } from "./tracker";

export interface FrameListener {
  /** One frame's RMS and the canonical sample index just past its end. */
  (level: number, endSample: number): void;
}

export interface PairedRecorder {
  readonly ready: Promise<void>;
  markNow(): number;
  readonly virtualDevice: boolean;
  readonly voiceIsolationApplied: boolean | null;
  /** The end of the last whole frame the tracker has read. Marks fall here. */
  framedSamples(): number;
  /** The canonical sample index that was live at a `performance.now()` instant. */
  sampleIndexAt(timeMs: number): number;
  /** The `performance.now()` instant of a canonical sample index. */
  timeAt(sampleIndex: number): number;
  /** Canonical samples in `[start, end)`. Throws for audio already released. */
  slice(start: number, end: number): Float32Array;
  /** Releases audio before `sampleIndex`. */
  releaseBefore(sampleIndex: number): void;
  /** Stops the microphone and returns once the last samples are canonical. */
  stop(): Promise<void>;
}

const BUFFER_SIZE = 4096;

/**
 * Starts recording. `onFrame` runs inside the audio callback for every full
 * frame, so it must stay cheap: the tracker's work per frame is bounded.
 */
export async function startPairedRecorder(onFrame: FrameListener): Promise<PairedRecorder> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: audioCaptureConstraints(1),
  });
  const voiceIsolationApplied = readVoiceIsolationApplied(stream);
  const virtualDevice = await detectVirtualAudioInput(stream);

  let context: AudioContext;
  let source: MediaStreamAudioSourceNode;
  try {
    context = new AudioContext({ sampleRate: CANONICAL_SAMPLE_RATE });
    await context.resume();
    source = context.createMediaStreamSource(stream);
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    throw error;
  }

  // A browser can deliver a lower rate than it was asked for. Paired segments
  // are 16 kHz by contract, so such a device cannot take part.
  let canonicalizer: ReturnType<typeof createStreamingCanonicalizer> | null;
  try {
    canonicalizer = createStreamingCanonicalizer(context.sampleRate);
  } catch (error) {
    stream.getTracks().forEach((track) => track.stop());
    await context.close().catch(() => undefined);
    throw error;
  }

  let buffer = new Float32Array(CANONICAL_SAMPLE_RATE * 4);
  let bufferStart = 0;
  let total = 0;
  let framed = 0;
  let nativeReceived = 0;
  let originMs = Number.POSITIVE_INFINITY;
  let resolveReady: () => void = () => undefined;
  const ready = new Promise<void>((resolve) => { resolveReady = resolve; });
  const sampleIndexAt = (timeMs: number): number => Number.isFinite(originMs)
    ? Math.max(0, Math.floor(((timeMs - originMs) * CANONICAL_SAMPLE_RATE) / 1000)) : 0;
  let stopped = false;

  const append = (samples: Float32Array): void => {
    const used = total - bufferStart;
    if (used + samples.length > buffer.length) {
      const grown = new Float32Array(Math.max(buffer.length * 2, used + samples.length));
      grown.set(buffer.subarray(0, used));
      buffer = grown;
    }
    buffer.set(samples, used);
    total += samples.length;
    while (!stopped && framed + FRAME_SAMPLES <= total) {
      const start = framed - bufferStart;
      const level = frameRms(buffer.subarray(start, start + FRAME_SAMPLES));
      framed += FRAME_SAMPLES;
      resolveReady();
      onFrame(level, framed);
    }
  };

  const processor = context.createScriptProcessor(BUFFER_SIZE, 1, 1);
  processor.onaudioprocess = (event: AudioProcessingEvent) => {
    if (stopped || !canonicalizer) return;
    const input = new Float32Array(event.inputBuffer.getChannelData(0));
    // A buffer arrives once it is full, so its first sample was captured one
    // buffer's duration before this callback.
    nativeReceived += input.length;
    originMs = Math.min(originMs, performance.now() - (nativeReceived * 1000) / context.sampleRate);
    append(canonicalizer.push(input));
  };
  source.connect(processor);
  processor.connect(context.destination);

  return {
    ready,
    markNow: () => sampleIndexAt(performance.now()),
    virtualDevice,
    voiceIsolationApplied,
    framedSamples: () => framed,
    sampleIndexAt,
    timeAt(sampleIndex) {
      return originMs + (sampleIndex * 1000) / CANONICAL_SAMPLE_RATE;
    },
    slice(start, end) {
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) throw new RangeError("Sample indices must be integers.");
      if (start < bufferStart || end > total || start > end) {
        throw new RangeError(`samples ${start}..${end} are not held`);
      }
      return buffer.slice(start - bufferStart, end - bufferStart);
    },
    releaseBefore(sampleIndex) {
      const discard = Math.min(sampleIndex, framed) - bufferStart;
      if (discard <= 0) return;
      buffer.copyWithin(0, discard, total - bufferStart);
      buffer.fill(0, total - bufferStart - discard, total - bufferStart);
      bufferStart += discard;
    },
    async stop() {
      if (stopped) return;
      stopped = true;
      processor.onaudioprocess = null;
      source.disconnect();
      processor.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      buffer = new Float32Array(0);
      bufferStart = total;
      canonicalizer = null;
      resolveReady();
      await context.close().catch(() => undefined);
    },
  };
}
