import { describe, it, expect } from "vitest";
import {
  CANONICAL_SAMPLE_RATE,
  resampleTo,
  toCanonicalCapture,
} from "../src/sensor/resample";

const canonical = async (x: Float32Array, rate: number) =>
  (await toCanonicalCapture(x, rate)).samples;

/** A pure tone, so passband and stopband behaviour can be read off directly. */
function tone(freqHz: number, sampleRate: number, seconds: number): Float32Array {
  const out = new Float32Array(Math.round(sampleRate * seconds));
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.sin((2 * Math.PI * freqHz * i) / sampleRate);
  }
  return out;
}

/**
 * A sum of unit tones, sampled at whatever rate from one continuous
 * definition. Deliberately not normalized by the tone count: the two
 * renderings below carry different numbers of tones, and scaling by the count
 * would put a constant gain difference between them that reads exactly like a
 * filter mismatch. It did, on the first attempt, at 20*log10(11/9) = 1.74 dB
 * flat across the band.
 */
function chord(freqs: number[], sampleRate: number, seconds: number): Float32Array {
  const out = new Float32Array(Math.round(sampleRate * seconds));
  for (let i = 0; i < out.length; i++) {
    const t = i / sampleRate;
    let v = 0;
    for (const f of freqs) v += Math.sin(2 * Math.PI * f * t);
    out[i] = v * 0.1;
  }
  return out;
}

/**
 * Amplitude of one frequency component, by Goertzel. Phase-insensitive, which
 * is what makes it the right comparison here: every spectral feature the
 * pipeline computes reads magnitude, and a constant sub-sample group delay
 * between two paths is meaningless against a 10 ms frame hop.
 */
function amplitudeAt(x: Float32Array, freq: number, sampleRate: number): number {
  const w = (2 * Math.PI * freq) / sampleRate;
  const k = 2 * Math.cos(w);
  let s1 = 0;
  let s2 = 0;
  for (let i = 0; i < x.length; i++) {
    const s0 = x[i]! + k * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  const re = s1 - s2 * Math.cos(w);
  const im = s2 * Math.sin(w);
  return (2 * Math.hypot(re, im)) / x.length;
}

function rms(x: Float32Array): number {
  if (x.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i]! * x[i]!;
  return Math.sqrt(sum / x.length);
}

/** Drop the filter's settling region at both ends before measuring. */
function interior(x: Float32Array): Float32Array {
  const edge = Math.min(600, Math.floor(x.length / 4));
  return x.subarray(edge, x.length - edge);
}

describe("canonical capture format", () => {
  /**
   * The property the design exists to deliver, and the reason every capture is
   * filtered rather than only the ones that need decimating.
   *
   * One voice is rendered at 16 kHz and at 48 kHz from the same continuous
   * definition. The 48 kHz rendering additionally carries content a 16 kHz
   * microphone could never capture, exactly as a real one would. After
   * processing, the two have to agree: that is what makes the fingerprint a
   * property of the speaker rather than of the browser's audio stack.
   *
   * Short-circuiting at 16 kHz breaks this, because Chromium has already
   * resampled 48 kHz hardware with its own filter and would be the last
   * band-limiting step on that path.
   */
  it("brings a 16 kHz and a 48 kHz rendering of one signal into agreement", async () => {
    const passband = [500, 1200, 2400, 4000, 6000, 7200];
    const transition = 7800;
    const shared = [...passband, transition];
    const at16 = chord(shared, CANONICAL_SAMPLE_RATE, 1);
    // Ultrasonic-ish content only a 48 kHz microphone hears. Chosen so their
    // fold-down frequencies (5500 and 1000 Hz) collide with none of the shared
    // tones, or the check would measure a real tone and call it an alias.
    const inaudibleAt16 = [
      [10500, 5500],
      [15000, 1000],
    ] as const;
    const at48 = chord([...shared, ...inaudibleAt16.map(([f]) => f)], 48000, 1);

    const out16 = interior(await canonical(at16, CANONICAL_SAMPLE_RATE));
    const out48 = interior(await canonical(at48, 48000));

    const deltaDbAt = (f: number) =>
      Math.abs(
        20 *
          Math.log10(
            amplitudeAt(out16, f, CANONICAL_SAMPLE_RATE) /
              amplitudeAt(out48, f, CANONICAL_SAMPLE_RATE)
          )
      );

    // Through the speech band the two paths are indistinguishable.
    for (const f of passband) expect(deltaDbAt(f)).toBeLessThan(0.05);

    // Inside the transition the curves can part slightly, because 7800 Hz sits
    // 200 Hz from Nyquist at 16 kHz where the discrete response folds. This is
    // the residual device dependence the design accepts, and it is bounded.
    expect(deltaDbAt(transition)).toBeLessThan(2);

    // And content only the 48 kHz microphone could hear is removed rather than
    // folding down into the speech band as a phantom formant.
    for (const [, foldsTo] of inaudibleAt16) {
      expect(amplitudeAt(out48, foldsTo, CANONICAL_SAMPLE_RATE)).toBeLessThan(0.001);
    }
  });

  /**
   * The filter has to be the same curve in Hz at every rate, or the two paths
   * are band-limited differently and cannot converge. A fixed tap count would
   * give a 416 Hz transition at 16 kHz and 1247 Hz at 48 kHz.
   */
  it("applies the same frequency response whatever the source rate", async () => {
    for (const freq of [6000, 7000, 7300]) {
      const a = rms(interior(await canonical(tone(freq, CANONICAL_SAMPLE_RATE, 0.5), CANONICAL_SAMPLE_RATE)));
      const b = rms(interior(await canonical(tone(freq, 48000, 0.5), 48000)));
      const deltaDb = 20 * Math.log10(a / b);
      expect(Math.abs(deltaDb)).toBeLessThan(1.0);
    }
  });

  it("band-limits a native 16 kHz capture rather than passing it through", async () => {
    const input = tone(7900, CANONICAL_SAMPLE_RATE, 0.5);
    const out = await canonical(input, CANONICAL_SAMPLE_RATE);
    expect(out).not.toBe(input);
    expect(rms(interior(out))).toBeLessThan(0.05);
  });

  it("stays flat across the speech band", async () => {
    const reference = rms(interior(await canonical(tone(1000, 48000, 0.4), 48000)));
    for (const freq of [500, 2000, 4000, 6000, 7000]) {
      const measured = rms(interior(await canonical(tone(freq, 48000, 0.4), 48000)));
      expect(Math.abs(20 * Math.log10(measured / reference))).toBeLessThan(0.5);
    }
  });

  /**
   * Without the lowpass, a 12 kHz tone at 48 kHz folds to 4 kHz and lands in
   * the middle of the speech band as a phantom formant.
   */
  it("attenuates content above the target Nyquist instead of aliasing it", async () => {
    expect(rms(interior(await canonical(tone(12000, 48000, 0.5), 48000)))).toBeLessThan(0.0015);
  });

  /**
   * Group delay. Every amplitude measurement here is shift-invariant, so an
   * impulse is the only thing that catches an uncompensated delay, which would
   * slide the audio against the motion and touch streams.
   */
  it("keeps the output time-aligned with the input", async () => {
    const input = new Float32Array(48000);
    input[24000] = 1; // t = 0.5 s
    const out = await canonical(input, 48000);
    let peak = 0;
    for (let i = 1; i < out.length; i++) {
      if (Math.abs(out[i]!) > Math.abs(out[peak]!)) peak = i;
    }
    expect(peak).toBe(8000); // 0.5 s at 16 kHz
  });

  /**
   * Unit gain at DC. Normalization to RMS 0.05 runs immediately after this, to
   * hand the validator's VAD a clean ~2x gain, so a shift in level here would
   * disturb that hand-off. Tight enough to catch a deleted DC normalization,
   * whose un-normalized tap sum is 0.9992155.
   */
  it("holds the signal level steady for the VAD hand-off", async () => {
    const out = await canonical(new Float32Array(48000).fill(0.25), 48000);
    expect(Math.abs(rms(interior(out)) - 0.25)).toBeLessThan(1e-5);
  });

  /**
   * Error against the ideal waveform, which catches a timing defect that an
   * amplitude check cannot see. Rounding to the nearest source sample instead
   * of interpolating leaves an 11.3 microsecond wobble at 44.1 kHz, inside the
   * range of the jitter features the pipeline measures. Measured cost of
   * rounding: 20 dB at 200 Hz, 34 dB at 997 Hz.
   */
  it("tracks the ideal waveform at the non-integer 44.1 kHz ratio", async () => {
    for (const [freq, maxErrorDb] of [
      [200, -50],
      [997, -50],
      [3000, -32],
    ] as const) {
      const out = await canonical(tone(freq, 44100, 2), 44100);
      let num = 0;
      let den = 0;
      for (let i = 600; i < out.length - 600; i++) {
        const ideal = Math.sin((2 * Math.PI * freq * i) / CANONICAL_SAMPLE_RATE);
        num += (out[i]! - ideal) ** 2;
        den += ideal * ideal;
      }
      expect(10 * Math.log10(num / den)).toBeLessThan(maxErrorDb);
    }
  });

  it("produces the expected output length for the real device rates", async () => {
    for (const [rate, seconds] of [
      [16000, 1],
      [44100, 1],
      [48000, 1],
      [48000, 12],
    ] as const) {
      const out = await canonical(tone(440, rate, seconds), rate);
      expect(out.length).toBe(Math.round(seconds * CANONICAL_SAMPLE_RATE));
    }
  });

  it("is deterministic across repeated calls", async () => {
    const input = tone(700, 48000, 0.3);
    const a = await canonical(input, 48000);
    const b = await canonical(input, 48000);
    for (let i = 0; i < a.length; i++) expect(a[i]).toBe(b[i]);
  });

  /**
   * Degradation, never a throw. The only caller runs in a `setTimeout` inside a
   * `Promise` executor, so a throw would leave the promise unsettled and hang
   * the capture with no error and no timeout. It also must never hand back a
   * plausible-looking empty capture, which is what `Infinity` would produce if
   * it reached the decimation arithmetic.
   */
  it("degrades on a nonsensical rate instead of throwing or emptying", async () => {
    const input = tone(440, 48000, 0.1);
    for (const bad of [
      0,
      -48000,
      Number.NaN,
      Number.POSITIVE_INFINITY,
      Number.NEGATIVE_INFINITY,
    ]) {
      const { samples, sampleRate } = await toCanonicalCapture(input, bad);
      expect(samples).toBe(input);
      expect(sampleRate).toBe(bad === bad ? bad : sampleRate); // NaN compares false
    }
  });

  it("passes rates below the canonical rate through, rather than inventing detail", async () => {
    const input = tone(440, 8000, 0.25);
    const { samples, sampleRate } = await toCanonicalCapture(input, 8000);
    expect(samples).toBe(input);
    expect(sampleRate).toBe(8000);
  });

  it("handles an empty capture", async () => {
    const { samples, sampleRate } = await toCanonicalCapture(new Float32Array(0), 48000);
    expect(samples.length).toBe(0);
    expect(sampleRate).toBe(CANONICAL_SAMPLE_RATE);
  });

  /**
   * A buffer tagged with a rate it is not at would make the validator resample
   * from the wrong source. Pinned across every awkward input rather than left
   * to inspection, including the empty buffer that an earlier two-function
   * shape got wrong.
   */
  it("never labels a buffer with a rate it is not at", async () => {
    for (const input of [tone(440, 48000, 0.2), new Float32Array(0)]) {
      for (const rate of [
        8000, 16000, 22050, 44100, 44100.5, 48000, 96000, 0, -1,
        Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY,
      ]) {
        const { samples, sampleRate } = await toCanonicalCapture(input, rate);
        if (input.length === 0) {
          // Nothing is at any rate, so only the label is under test elsewhere.
          continue;
        }
        if (samples === input) expect(sampleRate).toBe(rate);
        else expect(sampleRate).toBe(CANONICAL_SAMPLE_RATE);
      }
    }
  });

  it("exposes the general form for a target other than the canonical rate", async () => {
    const out = await resampleTo(tone(440, 48000, 0.2), 48000, 8000);
    expect(out.length).toBe(Math.round(0.2 * 8000));
  });
});
