import type { AudioCapture, CaptureOptions } from "./types";
import { MIN_CAPTURE_MS, MAX_CAPTURE_MS, MAX_TRANSMITTED_CAPTURE_MS } from "../config";
import { realFFT } from "../extraction/fft";
import { sdkWarn } from "../log";
import { CANONICAL_SAMPLE_RATE, toCanonicalCapture } from "./resample";

/**
 * Target RMS level the captured audio is normalized to before being
 * surfaced as `AudioCapture.samples`. Stays well inside Float32's
 * `[-1, 1]` range (no clipping risk on transients) and lands at the low
 * end of Whisper-tiny.en's amplitude sweet spot of `[0.05, 0.15]`. The
 * validator's VAD gate then applies a clean ~2× gain to reach its own
 * `0.1` target instead of the ~30× it had to apply to raw mic input.
 */
const TARGET_CAPTURE_RMS = 0.05;

/**
 * Below this RMS, the capture is treated as effective silence and the
 * normalization step is skipped — never amplify the noise floor of a
 * muted or unplugged mic into apparent signal. Mirrors
 * `entros-validation::vad::VAD_NORMALIZE_MIN_RMS`.
 */
const MIN_RMS_FOR_NORMALIZATION = 1e-4;

/**
 * Cap on the gain factor applied during normalization. Mirrors
 * `entros-validation::vad::VAD_NORMALIZE_MAX_GAIN` so a capture that the
 * SDK can't fully normalize also gets the same partial-gain treatment
 * server-side without further amplification surprises.
 */
const MAX_NORMALIZATION_GAIN = 50;

/**
 * Scale the capture buffer so its RMS lands at `TARGET_CAPTURE_RMS`,
 * with safety guards: empty input passes through unchanged, near-silent
 * input passes through unchanged (so we don't amplify noise floor), and
 * the per-sample multiply is clamped to `[-1, 1]` so a transient peak
 * × gain can't overflow into clipping artifacts.
 *
 * Rationale: prior to this normalization, `extractSpeakerFeaturesDetailed`
 * computed amplitude features from raw mic input — which carries mic-
 * setup identity (gain × distance × mic class) rather than biological
 * identity. After normalization, all downstream feature extraction
 * operates on amplitude-stable input and the validator's Whisper path
 * sees consistent loudness across captures. Exported so the unit test
 * suite can pin the contract directly without driving the full
 * `getUserMedia` stack.
 */
export function normalizeCaptureRMS(samples: Float32Array): Float32Array {
  if (samples.length === 0) return samples;
  let sumSq = 0;
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    sumSq += s * s;
  }
  const rms = Math.sqrt(sumSq / samples.length);
  if (rms < MIN_RMS_FOR_NORMALIZATION) return samples;
  const gain = Math.min(TARGET_CAPTURE_RMS / rms, MAX_NORMALIZATION_GAIN);
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    out[i] = Math.max(-1, Math.min(1, samples[i]! * gain));
  }
  return out;
}

/**
 * Speech-presence threshold, matched to the value the host UI uses to decide
 * whether the user has started speaking. Kept here so
 * {@link describeInputLevel} reports against the same bar the warning fires
 * on, which is the only way to tell a genuinely quiet microphone from a
 * warning stricter than the pipeline it warns about.
 */
const VOICED_FRAME_RMS = 0.008;

/** 10ms at the canonical rate, matching the F0 extractor's hop. */
const VOICED_FRAME_SAMPLES = 160;

/**
 * Measure what the microphone actually delivered, before any gain.
 *
 * Must run on the pre-normalisation buffer. {@link normalizeCaptureRMS}
 * rescales toward {@link TARGET_CAPTURE_RMS}, so the transmitted buffer's
 * level is a property of that target rather than of the capture, and reading
 * it tells you nothing about the microphone.
 *
 * The pair that matters is `gainClipped` and `voicedFrameRatio`. Gain applied
 * but not clipped, with a low voiced ratio, means the capture was recoverable
 * and a warning threshold is simply stricter than this pipeline's tolerance:
 * normalisation rescues input down to `TARGET_CAPTURE_RMS / MAX_NORMALIZATION_GAIN`,
 * which is 0.001, while hosts commonly warn at {@link VOICED_FRAME_RMS} of
 * 0.008. Gain clipped means the input really was below what can be recovered.
 */
export function describeInputLevel(samples: Float32Array): {
  rms: number;
  peak: number;
  gain: number;
  gainClipped: boolean;
  voicedFrameRatio: number;
} {
  if (samples.length === 0) {
    return { rms: 0, peak: 0, gain: 1, gainClipped: false, voicedFrameRatio: 0 };
  }

  let sumSq = 0;
  let peak = 0;
  let voicedFrames = 0;
  let totalFrames = 0;
  let frameSumSq = 0;
  let frameLen = 0;

  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]!;
    const sq = s * s;
    sumSq += sq;
    const abs = s < 0 ? -s : s;
    if (abs > peak) peak = abs;

    frameSumSq += sq;
    if (++frameLen === VOICED_FRAME_SAMPLES) {
      if (Math.sqrt(frameSumSq / frameLen) > VOICED_FRAME_RMS) voicedFrames++;
      totalFrames++;
      frameSumSq = 0;
      frameLen = 0;
    }
  }
  // A trailing partial frame counts, so a capture shorter than one frame
  // still reports a ratio rather than dividing by zero.
  if (frameLen > 0) {
    if (Math.sqrt(frameSumSq / frameLen) > VOICED_FRAME_RMS) voicedFrames++;
    totalFrames++;
  }

  const rms = Math.sqrt(sumSq / samples.length);
  // Mirrors `normalizeCaptureRMS` exactly rather than re-deriving it, so the
  // reported gain is the gain that was actually applied.
  const gain =
    rms < MIN_RMS_FOR_NORMALIZATION ? 1 : Math.min(TARGET_CAPTURE_RMS / rms, MAX_NORMALIZATION_GAIN);

  return {
    rms,
    peak,
    gain,
    gainClipped: rms >= MIN_RMS_FOR_NORMALIZATION && TARGET_CAPTURE_RMS / rms > MAX_NORMALIZATION_GAIN,
    voicedFrameRatio: totalFrames > 0 ? voicedFrames / totalFrames : 0,
  };
}

/**
 * Capture audio at 16kHz until signaled to stop.
 * Uses ScriptProcessorNode for raw PCM sample access.
 *
 * @privacyGuarantee Raw audio samples returned from this function are processed
 * locally by the SDK's feature extraction pipeline. The derived statistical
 * summary (308-element vector under the v3 pipeline) is the only audio-
 * related signal that crosses the device boundary. The single sanctioned
 * exception is the encoded base64 audio bytes sent to the validator's
 * `/validate-features` endpoint for server-side verification, which the
 * validator processes ephemerally — see entros.io for the privacy and
 * threat model.
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
    onReady,
    captureWindowSignal,
    stream: preAcquiredStream,
  } = options;

  const stream = preAcquiredStream ?? await navigator.mediaDevices.getUserMedia({
    audio: {
      sampleRate: CANONICAL_SAMPLE_RATE,
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

  let isVirtual = false;
  try {
    const track = stream.getAudioTracks()[0];
    if (track) {
      const label = track.label.toLowerCase();
      const virtualKeywords = ["blackhole", "vb-audio", "loopback", "virtual", "soundflower", "cable", "vac ", "audio cable"];
      if (virtualKeywords.some(kw => label.includes(kw))) {
        isVirtual = true;
      }
    }
    if (!isVirtual && typeof navigator !== "undefined" && navigator.mediaDevices?.enumerateDevices) {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const virtualKeywords = ["blackhole", "vb-audio", "loopback", "virtual", "soundflower", "cable", "vac ", "audio cable"];
      for (const d of devices) {
        if (d.kind === "audioinput") {
          const label = d.label.toLowerCase();
          if (virtualKeywords.some(kw => label.includes(kw))) {
            isVirtual = true;
            break;
          }
        }
      }
    }
  } catch {
    // Ignore any device enumeration/label query errors to prevent capture blocker
  }

  // If anything between `getUserMedia` and the Promise constructor throws
  // (AudioContext construction, ctx.resume(), createMediaStreamSource) the
  // stream we just acquired would leak indefinitely. Wrap the setup in a
  // try-on-error path that stops the stream tracks before re-throwing.
  let ctx: AudioContext;
  let source: MediaStreamAudioSourceNode;
  let capturedSampleRate: number;
  try {
    ctx = new AudioContext({ sampleRate: CANONICAL_SAMPLE_RATE });
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

    // Sample index at which the capture window opened. Everything before it is
    // dead air recorded while the prompt was still coming up. Null means the
    // host never signalled, in which case the whole buffer is kept.
    //
    // Same shape as the `signal` handling below, deliberately: check whether it
    // has already fired, otherwise subscribe once. An AbortSignal cannot fire
    // twice, so the mark is idempotent by construction rather than by a guard.
    let markedAtSample: number | null = null;
    const markWindowOpen = () => {
      // Keyed on `stopped`, not on the session's stage state: the capture can
      // close on its own `maxDurationMs` timer, which leaves the session still
      // reading "capturing" so its guard stays silent. Without this the mark is
      // recorded, never read, and the entire lead-in ships with no signal that
      // the trim did not happen.
      if (stopped) {
        sdkWarn(
          "[Entros SDK] Capture window signalled after the capture had already closed. Pre-prompt audio was not trimmed.",
        );
        return;
      }
      markedAtSample = chunks.reduce((sum, c) => sum + c.length, 0);
    };
    if (captureWindowSignal) {
      if (captureWindowSignal.aborted) markWindowOpen();
      else captureWindowSignal.addEventListener("abort", markWindowOpen, { once: true });
    }

    let firstFrameSeen = false;
    // Wall-clock instant of `collected[0]`, which is what every other modality
    // gets aligned to. `onaudioprocess` fires once a buffer is full, so the
    // audio in the first callback began one buffer before it arrived. Without
    // this the only timestamps available are the recorder's own start, which
    // precedes the microphone by a cold start of unknown length.
    let audioEpochMs = startTime;
    processor.onaudioprocess = (e: AudioProcessingEvent) => {
      const data = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(data));

      // Signal "capture is live" the moment real samples start flowing, so
      // callers can gate the speak prompt on actual audio rather than a fixed
      // delay. The first onaudioprocess proves the AudioContext + mic are
      // delivering frames — this is what fixes the first-attempt cold-start
      // miss where the start of the phrase fell into dead air.
      if (!firstFrameSeen) {
        firstFrameSeen = true;
        audioEpochMs = performance.now() - (data.length / capturedSampleRate) * 1000;
        onReady?.();
      }

      if (onAudioLevel) {
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i]! * data[i]!;
        onAudioLevel(Math.sqrt(sum / data.length));
      }
    };

    source.connect(processor);
    processor.connect(ctx.destination);

    async function stopCapture() {
      if (stopped) return;
      stopped = true;
      clearTimeout(maxTimer);
      if (abortTimer !== null) clearTimeout(abortTimer);

      processor.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t: MediaStreamTrack) => t.stop());
      ctx.close().catch(() => {});

      // Everything from here allocates, and this function is only ever reached
      // from a `setTimeout` callback, so a throw would not reject the promise.
      // it would leave it unsettled and hang the capture with no error and no
      // timeout. `maxTimer` is already cleared, so nothing would ever wake it.
      // Resolving an empty capture instead fails `MIN_AUDIO_SAMPLES` honestly.
      try {
        const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
        const collected = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
          collected.set(chunk, offset);
          offset += chunk.length;
        }
        // The chunk list is a second full copy of the audio, and the abort
        // listener closes over it, so it outlives this function unless dropped.
        // At 48 kHz for 60 s that is 11.5 MB held for nothing.
        chunks.length = 0;

        // Drop the lead-in before anything reads the buffer, so extraction and
        // transmission see the same samples.
        //
        // `slice`, not `subarray`. A view would keep the whole pre-trim buffer
        // alive behind it and hand callers a `Float32Array` whose `.buffer` is
        // larger than its contents, which is a trap for anything reaching past
        // `.length`. One copy per capture is not worth that.
        //
        // No upper guard on `markedAtSample`: `slice` clamps, and an empty
        // result is the honest answer to "the window opened at the very end".
        // Falling back to the untrimmed buffer there would fail open, silently
        // transmitting the whole lead-in.
        const raw =
          markedAtSample !== null ? collected.slice(markedAtSample) : collected;

        // Bring every capture to one canonical rate and bandwidth before
        // anything reads it. The AudioContext above requests 16 kHz but
        // browsers treat that as a hint, and feature extraction is rate-aware,
        // so without this the same voice yields a different fingerprint
        // depending on the browser. Every capture is filtered, including one
        // already at 16 kHz: a browser that honoured the request has already
        // resampled with its own filter, so short-circuiting would leave the
        // last band-limiting step browser-dependent. See `resample.ts`.
        //
        // Strictly before `normalizeCaptureRMS`. Normalization targets an RMS
        // the validator's VAD is calibrated against, and filtering a
        // normalized buffer would move the level back off that target.
        const { samples, sampleRate } = await toCanonicalCapture(
          raw,
          capturedSampleRate,
        );

        // Bound what leaves the device, mirroring the validator's own limit.
        //
        // Which end to keep depends on where the phrase is. With a mark,
        // index 0 is the prompt and the phrase is at the front. Without one,
        // every integrator who has not adopted `markCaptureStart`, index 0 is
        // recorder start and the phrase is at the end, so keeping the head
        // would delete the speech and leave the validator transcribing
        // silence.
        const maxSamples = Math.round((MAX_TRANSMITTED_CAPTURE_MS / 1000) * sampleRate);
        const overruns =
          Number.isFinite(maxSamples) && maxSamples > 0 && samples.length > maxSamples;
        const bounded = !overruns
          ? samples
          : markedAtSample !== null
            ? samples.slice(0, maxSamples)
            : samples.slice(samples.length - maxSamples);

        // Measured on `bounded`, which is the transmitted window before gain.
        // After `normalizeCaptureRMS` the level describes the target, not the
        // microphone, and the question this answers is about the microphone.
        const inputLevel = describeInputLevel(bounded);

        const normalized = normalizeCaptureRMS(bounded);
        const duration = normalized.length / sampleRate;

        // Where the transmitted buffer sits on the wall clock. Derived from
        // `markedAtSample`, which is exact, rather than from the instant the
        // mark fired, which is only accurate to one 4096-sample buffer. That
        // is 85ms at 48kHz, wider than the validator's whole lag search.
        //
        // Taking the far edge from `duration` rather than from the raw sample
        // count makes canonicalisation and the transmitted-length bound fall
        // out for free, whatever either did to the buffer. Which edge anchors
        // depends on which end `bounded` kept: with a mark it slices from the
        // head, without one from the tail.
        const windowStartMs =
          markedAtSample !== null
            ? audioEpochMs + (markedAtSample / capturedSampleRate) * 1000
            : audioEpochMs + (totalLength / capturedSampleRate) * 1000 - duration * 1000;

        resolve({
          samples: normalized,
          sampleRate,
          duration,
          windowStartMs,
          windowEndMs: windowStartMs + duration * 1000,
          inputLevel,
          virtualDevice: isVirtual,
        });
      } catch {
        resolve({
          samples: new Float32Array(0),
          sampleRate: capturedSampleRate,
          duration: 0,
          // An empty capture covers no window. Equal bounds are rejected
          // downstream, so no contour gets built against a buffer that failed.
          windowStartMs: audioEpochMs,
          windowEndMs: audioEpochMs,
          inputLevel: describeInputLevel(new Float32Array(0)),
          virtualDevice: isVirtual,
        });
      }
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

export function analyzeAcousticRealism(
  samples: Float32Array,
  sampleRate: number
): { flatness: number; centroid: number } {
  if (samples.length === 0) {
    return { flatness: 0, centroid: 0 };
  }

  const frameSize = 1024;
  const numFrames = Math.floor(samples.length / frameSize);
  if (numFrames === 0) {
    return { flatness: 0, centroid: 0 };
  }

  let totalFlatness = 0;
  let totalCentroid = 0;
  let validFrames = 0;

  for (let f = 0; f < numFrames; f++) {
    const frameData: number[] = [];
    const offset = f * frameSize;
    for (let i = 0; i < frameSize; i++) {
      frameData.push(samples[offset + i]!);
    }

    const { real, imag } = realFFT(frameData, frameSize);
    const numBins = frameSize / 2 + 1;
    const magnitudes = new Float32Array(numBins);
    const power = new Float32Array(numBins);

    let magSum = 0;
    for (let k = 0; k < numBins; k++) {
      const r = real[k]!;
      const im = imag[k]!;
      const m = Math.sqrt(r * r + im * im);
      magnitudes[k] = m;
      power[k] = m * m + 1e-10;
      magSum += m;
    }

    if (magSum < 1e-6) {
      continue;
    }

    // Compute Centroid
    let centroidNumerator = 0;
    for (let k = 0; k < numBins; k++) {
      const freq = (k * sampleRate) / frameSize;
      centroidNumerator += freq * magnitudes[k]!;
    }
    const frameCentroid = centroidNumerator / magSum;

    // Compute Flatness (Wiener Entropy)
    let lnSum = 0;
    let powerSum = 0;
    for (let k = 0; k < numBins; k++) {
      const p = power[k]!;
      lnSum += Math.log(p);
      powerSum += p;
    }
    const geomMean = Math.exp(lnSum / numBins);
    const arithMean = powerSum / numBins;
    const frameFlatness = arithMean > 0 ? geomMean / arithMean : 0;

    totalCentroid += frameCentroid;
    totalFlatness += frameFlatness;
    validFrames++;
  }

  if (validFrames === 0) {
    return { flatness: 0, centroid: 0 };
  }

  return {
    flatness: totalFlatness / validFrames,
    centroid: totalCentroid / validFrames
  };
}
