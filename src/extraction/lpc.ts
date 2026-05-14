/**
 * Linear Predictive Coding (LPC) for formant detection.
 *
 * Implements Levinson-Durbin recursion for LPC coefficient computation
 * and polynomial root-finding for formant frequency estimation.
 */

/**
 * Compute autocorrelation of a signal for lags 0..order.
 */
function autocorrelate(signal: Float32Array, order: number): number[] {
  const r: number[] = [];
  for (let lag = 0; lag <= order; lag++) {
    let sum = 0;
    for (let i = 0; i < signal.length - lag; i++) {
      sum += signal[i]! * signal[i + lag]!;
    }
    r.push(sum);
  }
  return r;
}

/**
 * Levinson-Durbin recursion to compute LPC coefficients from autocorrelation.
 * Returns the LPC coefficients a[1..order] (a[0] is implicitly 1).
 */
function levinsonDurbin(r: number[], order: number): number[] {
  const a: number[] = new Array(order + 1).fill(0);
  const aTemp: number[] = new Array(order + 1).fill(0);
  a[0] = 1;

  let error = r[0]!;
  if (error === 0) return new Array(order).fill(0);

  for (let i = 1; i <= order; i++) {
    let lambda = 0;
    for (let j = 1; j < i; j++) {
      lambda += a[j]! * r[i - j]!;
    }
    lambda = -(r[i]! + lambda) / error;

    for (let j = 1; j < i; j++) {
      aTemp[j] = a[j]! + lambda * a[i - j]!;
    }
    aTemp[i] = lambda;

    for (let j = 1; j <= i; j++) {
      a[j] = aTemp[j]!;
    }

    error *= 1 - lambda * lambda;
    if (error <= 0) break;
  }

  return a.slice(1);
}

/**
 * Find roots of a polynomial using the Durand-Kerner method.
 * The polynomial is 1 + a[0]*z^-1 + a[1]*z^-2 + ... + a[n-1]*z^-n
 * which is equivalent to z^n + a[0]*z^(n-1) + ... + a[n-1] = 0.
 *
 * Returns complex roots as [real, imag] pairs.
 */
function findRoots(coefficients: number[], maxIterations: number = 50): [number, number][] {
  const n = coefficients.length;
  if (n === 0) return [];

  // Initial guesses: points on a circle of radius 0.9
  const roots: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const angle = (2 * Math.PI * i) / n + 0.1;
    roots.push([0.9 * Math.cos(angle), 0.9 * Math.sin(angle)]);
  }

  for (let iter = 0; iter < maxIterations; iter++) {
    let maxShift = 0;

    for (let i = 0; i < n; i++) {
      // Evaluate polynomial at roots[i]: z^n + a[0]*z^(n-1) + ... + a[n-1]
      let pReal = 1;
      let pImag = 0;
      let zPowReal = 1;
      let zPowImag = 0;

      // Compute z^n by repeated multiplication
      const [rr, ri] = roots[i]!;
      let curReal = 1;
      let curImag = 0;

      // Evaluate as: z^n + sum(a[k] * z^(n-1-k))
      // Start with z^n
      let znReal = 1;
      let znImag = 0;
      for (let k = 0; k < n; k++) {
        const newReal = znReal * rr - znImag * ri;
        const newImag = znReal * ri + znImag * rr;
        znReal = newReal;
        znImag = newImag;
      }
      pReal = znReal;
      pImag = znImag;

      // Add coefficient terms: a[k] * z^(n-1-k)
      zPowReal = 1;
      zPowImag = 0;
      for (let k = n - 1; k >= 0; k--) {
        pReal += coefficients[k]! * zPowReal;
        pImag += coefficients[k]! * zPowImag;
        const newReal = zPowReal * rr - zPowImag * ri;
        const newImag = zPowReal * ri + zPowImag * rr;
        zPowReal = newReal;
        zPowImag = newImag;
      }

      // Compute product of (roots[i] - roots[j]) for j != i
      let denomReal = 1;
      let denomImag = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const diffReal = rr - roots[j]![0];
        const diffImag = ri - roots[j]![1];
        const newReal = denomReal * diffReal - denomImag * diffImag;
        const newImag = denomReal * diffImag + denomImag * diffReal;
        denomReal = newReal;
        denomImag = newImag;
      }

      // Divide p / denom
      const denomMag2 = denomReal * denomReal + denomImag * denomImag;
      if (denomMag2 < 1e-30) continue;

      const shiftReal = (pReal * denomReal + pImag * denomImag) / denomMag2;
      const shiftImag = (pImag * denomReal - pReal * denomImag) / denomMag2;

      roots[i] = [rr - shiftReal, ri - shiftImag];
      maxShift = Math.max(maxShift, Math.sqrt(shiftReal * shiftReal + shiftImag * shiftImag));
    }

    if (maxShift < 1e-10) break;
  }

  return roots;
}

/**
 * Per-frame analysis output: the 12 LPC coefficients always exist (one
 * autocorrelate + Levinson-Durbin pass per frame), and the formant tuple
 * exists when at least 3 valid candidates were detected within the
 * speech-formant band. Bandwidths are paired with the formants — they're
 * the imaginary-axis decay rates of the same complex roots that produce
 * F1/F2/F3.
 */
interface FrameAnalysis {
  /** LPC coefficients a[1..order]; a[0] = 1 implicitly. Always populated. */
  lpcCoefficients: number[];
  /** [F1, F2, F3] in Hz, or null if fewer than 3 valid candidates. */
  formants: [number, number, number] | null;
  /** [B1, B2, B3] in Hz, paired with formants. Null when formants are null. */
  bandwidths: [number, number, number] | null;
}

/**
 * Per-frame LPC analysis surfacing the full information set used downstream
 * by the v2 feature pipeline (LPC coefficients, formant frequencies,
 * formant bandwidths). Replaces the legacy `extractFormants()` which only
 * returned formant frequencies and discarded both the LPC coefficients
 * and the bandwidths.
 */
function extractFrameAnalysis(
  frame: Float32Array,
  sampleRate: number,
  lpcOrder: number = 12,
): FrameAnalysis {
  const r = autocorrelate(frame, lpcOrder);
  const coeffs = levinsonDurbin(r, lpcOrder);

  const roots = findRoots(coeffs);

  // Pair each candidate frequency with its corresponding bandwidth so we
  // can preserve the F1↔B1, F2↔B2, F3↔B3 alignment after sorting by
  // frequency. The LPC pole's imaginary-axis decay rate maps to formant
  // bandwidth via the same complex-magnitude → log relation; capturing it
  // is information-free vs. the existing computation.
  const candidates: { freq: number; bandwidth: number }[] = [];

  for (const [real, imag] of roots) {
    if (imag <= 0) continue; // Keep only positive-frequency roots

    const freq = (Math.atan2(imag, real) / (2 * Math.PI)) * sampleRate;
    const bandwidth =
      (-sampleRate / (2 * Math.PI)) *
      Math.log(Math.sqrt(real * real + imag * imag));

    // Filter: formants are in 200-5000Hz range with reasonable bandwidth.
    if (freq > 200 && freq < 5000 && bandwidth < 500) {
      candidates.push({ freq, bandwidth });
    }
  }

  candidates.sort((a, b) => a.freq - b.freq);

  if (candidates.length < 3) {
    return { lpcCoefficients: coeffs, formants: null, bandwidths: null };
  }

  const formants: [number, number, number] = [
    candidates[0]!.freq,
    candidates[1]!.freq,
    candidates[2]!.freq,
  ];
  const bandwidths: [number, number, number] = [
    candidates[0]!.bandwidth,
    candidates[1]!.bandwidth,
    candidates[2]!.bandwidth,
  ];

  return { lpcCoefficients: coeffs, formants, bandwidths };
}

/**
 * Backward-compatible thin wrapper. Existing callers that only need
 * formant frequencies keep their signatures stable; the v2 feature pipeline
 * uses `extractFrameAnalysis` directly to surface the additional LPC and
 * bandwidth signal that the old wrapper discarded.
 */
function extractFormants(
  frame: Float32Array,
  sampleRate: number,
  lpcOrder: number = 12,
): [number, number, number] | null {
  return extractFrameAnalysis(frame, sampleRate, lpcOrder).formants;
}

/**
 * Extract formant ratio time series (F1/F2 and F2/F3) from audio.
 * Returns { f1f2: number[], f2f3: number[] } — one ratio per frame where formants were detected.
 */
export function extractFormantRatios(
  samples: Float32Array,
  sampleRate: number,
  frameSize: number,
  hopSize: number
): { f1f2: number[]; f2f3: number[] } {
  const f1f2: number[] = [];
  const f2f3: number[] = [];
  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    // Read-only — windowed below is a fresh allocation that copies values
    // out, so a zero-copy view here is bit-equivalent and saves a Float32Array.
    const frame = samples.subarray(start, start + frameSize);

    // Apply Hamming window
    const windowed = new Float32Array(frameSize);
    for (let j = 0; j < frameSize; j++) {
      windowed[j] = (frame[j] ?? 0) * (0.54 - 0.46 * Math.cos((2 * Math.PI * j) / (frameSize - 1)));
    }

    const formants = extractFormants(windowed, sampleRate);
    if (formants) {
      const [f1, f2, f3] = formants;
      if (f2 > 0) f1f2.push(f1 / f2);
      if (f3 > 0) f2f3.push(f2 / f3);
    }
  }

  return { f1f2, f2f3 };
}

/**
 * Session-level output of the v2 LPC analysis. Each per-coefficient time
 * series has length `numFramesAnalyzed` (every frame contributes); the
 * formant time series may be shorter because frames with fewer than 3
 * valid candidates are skipped (matches `extractFormantRatios` behavior).
 */
export interface LpcAnalysis {
  /** lpcCoefficients[c][t] = c-th LPC coefficient at frame t. Shape: [order × frames]. */
  lpcCoefficients: number[][];
  /** F1/F2/F3 absolute frequencies in Hz, one entry per detected-formant frame. */
  f1: number[];
  f2: number[];
  f3: number[];
  /** B1/B2/B3 corresponding bandwidths in Hz, aligned with f1/f2/f3 by index. */
  b1: number[];
  b2: number[];
  b3: number[];
  /** Existing formant ratios, retained for backward-compat with the original 44-feature audio block. */
  f1f2: number[];
  f2f3: number[];
  /** Diagnostic: number of frames the analyzer ran over (independent of formant detection). */
  numFramesAnalyzed: number;
}

/**
 * Comprehensive session-level LPC + formant analysis. Surfaces the full
 * information set the v2 feature pipeline consumes: per-frame LPC
 * coefficients (12 time series), absolute formant frequencies (3 time
 * series), formant bandwidths (3 time series), and the existing F1/F2 +
 * F2/F3 ratios.
 *
 * Single pass over the samples — autocorrelate + Levinson-Durbin runs
 * once per frame, with all downstream signals derived from that single
 * coefficient set. CPU cost is identical to the legacy
 * `extractFormantRatios` (the formant filtering, root-finding, and ratio
 * derivation were always running; this function just RETAINS the
 * intermediate values that the legacy function discarded).
 */
export function extractLpcAnalysis(
  samples: Float32Array,
  sampleRate: number,
  frameSize: number,
  hopSize: number,
  lpcOrder: number = 12,
): LpcAnalysis {
  const lpcCoefficients: number[][] = Array.from({ length: lpcOrder }, () => []);
  const f1: number[] = [];
  const f2: number[] = [];
  const f3: number[] = [];
  const b1: number[] = [];
  const b2: number[] = [];
  const b3: number[] = [];
  const f1f2: number[] = [];
  const f2f3: number[] = [];

  const numFrames = Math.floor((samples.length - frameSize) / hopSize) + 1;
  let numFramesAnalyzed = 0;

  if (numFrames < 1) {
    return {
      lpcCoefficients,
      f1, f2, f3, b1, b2, b3, f1f2, f2f3,
      numFramesAnalyzed: 0,
    };
  }

  // Pre-allocate the windowed frame buffer once (matches the optimization
  // in speaker.ts::computeLTAS — saves ~1200 Float32Array allocations
  // per session at ~10ms hop over 12 seconds).
  const windowed = new Float32Array(frameSize);

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    const frame = samples.subarray(start, start + frameSize);

    // Apply Hamming window in-place to the pre-allocated buffer.
    for (let j = 0; j < frameSize; j++) {
      windowed[j] =
        (frame[j] ?? 0) *
        (0.54 - 0.46 * Math.cos((2 * Math.PI * j) / (frameSize - 1)));
    }

    const analysis = extractFrameAnalysis(windowed, sampleRate, lpcOrder);
    numFramesAnalyzed++;

    // LPC coefficients are always populated.
    for (let c = 0; c < lpcOrder; c++) {
      const coeff = analysis.lpcCoefficients[c];
      if (Number.isFinite(coeff)) {
        lpcCoefficients[c]!.push(coeff!);
      }
    }

    // Formants and bandwidths are populated only when 3+ valid candidates
    // were detected. Push the triplet atomically (all three or none) so
    // index alignment between f1/f2/f3 and b1/b2/b3 is preserved.
    if (analysis.formants && analysis.bandwidths) {
      const [F1, F2, F3] = analysis.formants;
      const [B1, B2, B3] = analysis.bandwidths;
      f1.push(F1);
      f2.push(F2);
      f3.push(F3);
      b1.push(B1);
      b2.push(B2);
      b3.push(B3);
      if (F2 > 0) f1f2.push(F1 / F2);
      if (F3 > 0) f2f3.push(F2 / F3);
    }
  }

  return {
    lpcCoefficients,
    f1, f2, f3, b1, b2, b3, f1f2, f2f3,
    numFramesAnalyzed,
  };
}
