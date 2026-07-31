import { yieldToMainThread } from "../yield";

/**
 * Brings every capture at or above 16 kHz to one canonical audio format
 * before anything reads it.
 *
 * Why this exists.
 *
 * Browsers treat `new AudioContext({ sampleRate })` as a request, not an
 * instruction. Chromium honours 16 kHz; WebKit commonly returns the hardware
 * default of 48 kHz. Feature extraction is rate-aware: `getFrameSize` doubles
 * at 48 kHz while the hop stays at 10 ms, the MFCC mel bank spans
 * `0..sampleRate/2`, and the LPC order is fixed at 12, so the same voice
 * captured at the two rates yields two incomparable feature vectors. Since
 * `fuseFeatures` z-scores the whole 170-dimension audio block together, that
 * difference reaches every audio feature and the SimHash.
 *
 * Why every capture is filtered, including one already at 16 kHz.
 *
 * This is the part that is easy to get wrong, and the first version of this
 * module did. When Chromium honours the 16 kHz request against 48 kHz
 * hardware, **the browser has already resampled, using its own filter**. If
 * this module then short-circuits, the last band-limiting step applied to the
 * audio is Chromium's filter on one browser and this one on another, so the
 * fingerprint still depends on the browser's DSP. Sharpening this filter
 * cannot fix that, because the difference is not in this filter, it is in
 * whose filter runs last.
 *
 * Filtering every capture at or above the canonical rate makes this the final
 * step on those paths. Whatever
 * a browser's resampler did above the cutoff is removed here, and the paths
 * converge. It costs the "already 16 kHz means untouched" property, which
 * means existing baselines move. That is a deliberate, one-time trade taken
 * while the protocol is on devnet and the anchor population is test data.
 *
 * Why a hand-written FIR rather than `OfflineAudioContext`.
 *
 * The browser-native path would be shorter, but each engine ships its own
 * resampler with its own filter design, which is the dependence this module
 * exists to remove. A fixed algorithm varies across engines only at the level
 * of floating-point rounding, orders of magnitude below the natural variation
 * between two captures of the same person.
 *
 * Ported from `entros-validation/src/f0_recheck.rs::design_lowpass_fir` and
 * `::resample_fir`, so the client and the validator share one filter design.
 * The cutoff differs deliberately, see {@link CUTOFF_FRACTION}.
 */

/**
 * The one rate and bandwidth every capture is brought to before extraction or
 * transmission.
 *
 * Matches `entros-validation`'s Whisper and VAD rate (`MODEL_SAMPLE_RATE`,
 * `VAD_SAMPLE_RATE`), so the server's own resample becomes a no-op.
 */
export const CANONICAL_SAMPLE_RATE = 16000;

/**
 * Cutoff as a fraction of the canonical rate, leaving a transition band below
 * Nyquist. 0.475 puts it at 7600 Hz.
 *
 * The Rust reference uses `min(to_rate / 2, 3800)`, correct for its purpose:
 * it feeds YIN pitch detection, where everything above 3.8 kHz is noise.
 * Reusing that here would be a serious mistake. This audio still has to serve
 * the MFCC bank, the LTAS features and Whisper transcription, all of which
 * read the full speech band.
 *
 * The exact value matters less than the fact that every capture meets the same
 * one. A capture band-limited at 7600 Hz by this filter is comparable with any
 * other capture band-limited at 7600 Hz by this filter, whatever the
 * microphone or browser did beforehand.
 */
const CUTOFF_FRACTION = 0.475;

/**
 * Filter length at the canonical rate. Odd, so the group delay is an exact
 * integer of samples.
 *
 * A Hamming-windowed sinc has a transition width of roughly `3.3 * fs / N`, so
 * at 127 taps and 16 kHz the response is flat to about 7.4 kHz, -6 dB at
 * 7600 Hz, and into the window's -53 dB floor by 7.8 kHz.
 */
const TAPS_AT_CANONICAL = 127;

/**
 * Group delay at the canonical rate, in samples. The filter is symmetric, so
 * its delay is exactly half its span.
 */
const DELAY_AT_CANONICAL = (TAPS_AT_CANONICAL - 1) / 2;

/**
 * Filter length for a given source rate, scaled so the response is the same
 * curve in Hz whatever the rate.
 *
 * This is the detail that makes unconditional filtering actually deliver
 * convergence, and it is easy to miss. A fixed tap count means a fixed
 * transition width in *fractions of the sample rate*, so 127 taps spans 416 Hz
 * at 16 kHz but 1247 Hz at 48 kHz. Two captures of one voice would then be
 * band-limited by two different curves and would still not be comparable -
 * less badly than before, but still by the browser.
 *
 * Specifying the filter by its time span instead fixes the response. 127 taps
 * at 16 kHz is 7.9 ms; the same 7.9 ms is 381 taps at 48 kHz. Cost scales with
 * it, which is why {@link resampleTo} yields.
 */
function tapsForRate(sampleRate: number): number {
  // Derived from the delay rather than the span, so the group delay lands on
  // the same instant in time at every rate. Scaling the span directly gives
  // 381 taps at 48 kHz, a delay of 190 source samples, which is 3.9583 ms
  // against 3.9375 ms at 16 kHz. Twenty microseconds is nothing to the 10 ms
  // frame hop, but it is most of a period at the top of the band and it shows
  // up as a phase difference between the two paths.
  const delay = Math.round((DELAY_AT_CANONICAL * sampleRate) / CANONICAL_SAMPLE_RATE);
  // At least 3 taps. A single tap makes `m` zero, so the Hamming window
  // divides by zero, every tap comes out NaN, and the DC guard skips
  // normalization because `Math.abs(NaN) > 1e-9` is false, the whole output
  // would be NaN. Unreachable through `toCanonicalCapture`, but `resampleTo`
  // is exported as the general form.
  return Math.max(3, 2 * delay + 1);
}

/**
 * Multiply-accumulates between cooperative yields.
 *
 * Cost is `outputLength * tapsForRate(fromRate)` multiply-accumulates, doubled
 * for a non-integer ratio. A 15-second capture is about 31 million at 16 kHz
 * and 94 million at 48 kHz, measured at 95 ms and 285 ms on desktop V8 and
 * several times that on Hermes. It runs the instant capture stops, when the UI is showing the
 * hand-off to processing, so it yields on the same principle as
 * `speaker.ts`'s F0 loop rather than blocking the paint.
 */
const YIELD_EVERY_N_MACS = 1_000_000;

/** A capture and the rate it is actually at, which are never separable. */
export interface CanonicalCapture {
  samples: Float32Array;
  sampleRate: number;
}

/**
 * Design a lowpass FIR by the windowed-sinc method with a Hamming window,
 * normalized to unit gain at DC.
 *
 * `cutoffHz` is normalized against `sampleRate`, matching the Rust reference.
 */
function designLowpassFir(
  sampleRate: number,
  cutoffHz: number,
  numTaps: number
): Float64Array {
  const taps = new Float64Array(numTaps);
  const m = numTaps - 1;
  const fc = cutoffHz / sampleRate;
  const center = m / 2;

  for (let n = 0; n < numTaps; n++) {
    const diff = n - center;
    const sinc =
      Math.abs(diff) < 1e-9
        ? 2 * fc
        : Math.sin(2 * Math.PI * fc * diff) / (Math.PI * diff);
    const window = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / m);
    taps[n] = sinc * window;
  }

  // Unit gain at DC, so this cannot shift the capture's RMS and disturb the
  // gain hand-off to the validator's VAD.
  let sum = 0;
  for (let n = 0; n < numTaps; n++) sum += taps[n]!;
  if (Math.abs(sum) > 1e-9) {
    for (let n = 0; n < numTaps; n++) {
      const tap = taps[n]!;
      taps[n] = tap / sum;
    }
  }
  return taps;
}

/**
 * Bring a capture to the canonical rate and bandwidth, and report the rate it
 * ended up at.
 *
 * Returns both together on purpose. An earlier shape had one function
 * transform the buffer and another label it, which meant two copies of one
 * condition and a real divergence on the empty-buffer path. Audio tagged with
 * a rate it is not at would make the validator resample from the wrong source.
 *
 * Passes the capture through untouched, at its original rate, when the rate is
 * not a finite number above zero, or is below the canonical rate. Neither can
 * arise from `captureAudio`, an `AudioContext` returns either the requested
 * rate or a hardware rate, and hardware rates are 44.1 or 48 kHz, and
 * upsampling would invent detail that was never captured. Degrading rather
 * than throwing is load-bearing: the only caller runs inside a `setTimeout`
 * within a `Promise` executor, where a throw would leave the promise unsettled
 * and hang the capture with no error and no timeout. Passing through preserves
 * the path where `extractSpeakerFeaturesDetailed` validates the rate, warns,
 * and degrades to a zero feature vector.
 */
export async function toCanonicalCapture(
  input: Float32Array,
  fromRate: number
): Promise<CanonicalCapture> {
  if (!(fromRate >= CANONICAL_SAMPLE_RATE) || !Number.isFinite(fromRate)) {
    return { samples: input, sampleRate: fromRate };
  }
  if (input.length === 0) {
    return { samples: input, sampleRate: CANONICAL_SAMPLE_RATE };
  }
  return {
    samples: await resampleTo(input, fromRate, CANONICAL_SAMPLE_RATE),
    sampleRate: CANONICAL_SAMPLE_RATE,
  };
}

/**
 * Lowpass and, where the rates differ, decimate. Exposed for tests; prefer
 * {@link toCanonicalCapture}, which also reports the resulting rate.
 *
 * `fromRate === toRate` is a filter with no decimation, not a no-op. That is
 * the whole point of the module, see the note on unconditional filtering at
 * the top of the file.
 */
export async function resampleTo(
  input: Float32Array,
  fromRate: number,
  toRate: number
): Promise<Float32Array> {
  if (!(fromRate >= toRate) || !Number.isFinite(fromRate) || input.length === 0) {
    return input;
  }

  const ratio = fromRate / toRate;
  const outputLength = Math.round(input.length / ratio);
  const output = new Float32Array(outputLength);

  const numTaps = tapsForRate(fromRate);
  const taps = designLowpassFir(fromRate, toRate * CUTOFF_FRACTION, numTaps);
  const delay = (numTaps - 1) / 2;

  // Yield on a work budget, not a sample count. Cost per output sample is
  // `numTaps`, doubled for a non-integer ratio, and `tapsForRate` scales with
  // the source rate, so a fixed sample interval blocks the main thread three
  // times longer at 48 kHz and five times longer at 44.1 kHz, on exactly the
  // browsers this module exists for.
  const macsPerSample = numTaps * (Number.isInteger(ratio) ? 1 : 2);
  const yieldEvery = Math.max(1, Math.ceil(YIELD_EVERY_N_MACS / macsPerSample));

  for (let i = 0; i < outputLength; i++) {
    // Output sample i sits at input position `i * ratio`, an integer only when
    // the ratio is. Rounding to the nearest input sample, what the Rust
    // reference does, because it only ever runs at exact ratios of 2 and 6 -
    // leaves a periodic wobble of up to half a source sample. At 44.1 kHz that
    // is 11.3 microseconds, about 0.14% of a 120 Hz pitch period, inside the
    // range of the jitter features the pipeline measures. Measured cost of
    // rounding instead of interpolating: 20 dB of error at 200 Hz, 34 dB at
    // 997 Hz. It would put a device-dependent term back into the fingerprint,
    // which is the defect this module exists to remove.
    const pos = i * ratio;
    const base = Math.floor(pos);
    const frac = pos - base;

    let sum = 0;
    for (let j = 0; j < numTaps; j++) {
      const idx = base + delay - j;
      if (idx >= 0 && idx < input.length) sum += input[idx]! * taps[j]!;
    }

    // Exact ratios (1 and 3 into 16 kHz, the two that occur in practice) never
    // take this branch, so the common paths cost one convolution.
    if (frac !== 0) {
      let next = 0;
      for (let j = 0; j < numTaps; j++) {
        const idx = base + 1 + delay - j;
        if (idx >= 0 && idx < input.length) next += input[idx]! * taps[j]!;
      }
      sum += (next - sum) * frac;
    }

    output[i] = sum;

    if (i > 0 && i < outputLength - 1 && i % yieldEvery === 0) {
      await yieldToMainThread();
    }
  }
  return output;
}
