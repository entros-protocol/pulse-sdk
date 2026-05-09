/**
 * Real-input radix-2 Cooley-Tukey FFT and frequency-band energy / peak
 * helpers used by the v2 kinematic feature pipeline.
 *
 * Used to extract physiological-tremor signatures and motion frequency-band
 * energies from IMU axes (~50–200 Hz sample rates, ~600–2400 sample
 * windows). See `docs/master/BLUEPRINT-feature-pipeline-v2.md` §2.2.
 *
 * Why a custom FFT and not Meyda: Meyda's spectral extractors are
 * tuned for audio frame sizes (≥ 512 samples at 16 kHz) and bake in
 * audio-specific assumptions (Hanning window, magnitude scaling, frame
 * indexing). Kinematic signals are short, low-rate, and analyzed once
 * per session — a self-contained radix-2 implementation is simpler
 * and avoids leaking Meyda's global state into the motion path
 * (which would interfere with the speaker-extractor invariants set up
 * in speaker.ts / mfcc.ts / voice-quality.ts).
 *
 * @privacyGuarantee This module operates on already-on-device sensor
 * arrays. Outputs are reduced to a small number of band-energy /
 * peak-frequency scalars before leaving the SDK; the FFT itself is
 * never transmitted.
 */

/** Round up to the next power of two, minimum 2. */
function nextPow2(n: number): number {
  if (n <= 2) return 2;
  let p = 2;
  while (p < n) p <<= 1;
  return p;
}

/**
 * Compute the radix-2 FFT of a real-valued input. Input is zero-padded (or
 * truncated) to the requested `size`, which must be a power of two — use
 * `nextPow2(input.length)` to pick a sensible size.
 *
 * Returns `{ real, imag }` arrays of length `size`. Bin k corresponds to
 * frequency `k × sampleRate / size` Hz. By Hermitian symmetry only the
 * first `size / 2 + 1` bins are physically meaningful for real input;
 * downstream helpers (`bandEnergy`, `peakInBand`) account for this
 * automatically.
 */
export function realFFT(
  input: number[],
  size: number,
): { real: number[]; imag: number[] } {
  if (size <= 0 || (size & (size - 1)) !== 0) {
    throw new Error(`FFT size must be a positive power of two, got ${size}`);
  }

  const real = new Array<number>(size);
  const imag = new Array<number>(size).fill(0);

  for (let i = 0; i < size; i++) {
    real[i] = i < input.length ? (input[i] ?? 0) : 0;
  }

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < size; i++) {
    let bit = size >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = real[i]!;
      real[i] = real[j]!;
      real[j] = tr;
      // imag is all-zero pre-permutation, so no swap needed there.
    }
  }

  // Cooley-Tukey butterflies.
  for (let halfSize = 1; halfSize < size; halfSize <<= 1) {
    const fullSize = halfSize << 1;
    const phaseStep = -Math.PI / halfSize;
    for (let chunkStart = 0; chunkStart < size; chunkStart += fullSize) {
      for (let k = 0; k < halfSize; k++) {
        const phase = phaseStep * k;
        const wr = Math.cos(phase);
        const wi = Math.sin(phase);
        const ar = real[chunkStart + k]!;
        const ai = imag[chunkStart + k]!;
        const br = real[chunkStart + k + halfSize]!;
        const bi = imag[chunkStart + k + halfSize]!;
        const tr = wr * br - wi * bi;
        const ti = wr * bi + wi * br;
        real[chunkStart + k] = ar + tr;
        imag[chunkStart + k] = ai + ti;
        real[chunkStart + k + halfSize] = ar - tr;
        imag[chunkStart + k + halfSize] = ai - ti;
      }
    }
  }

  return { real, imag };
}

/**
 * Sum the squared magnitudes of FFT bins falling inside the half-open
 * frequency interval [`fLow`, `fHigh`) Hz, NORMALIZED by `N²` so the
 * result is a per-bin power density in the same units as the squared
 * input signal (not Parseval N²-scaled energy).
 *
 * Why normalize: an unscaled `Σ|X[k]|²` grows as O(N²·A²) for a sine
 * of amplitude A on N samples, which would put band energies 10,000×
 * larger than other motion features (O(0.01-1)) and dominate the
 * z-score normalization in `statistics.ts::normalizeGroup`. Dividing
 * by `N²` aligns the band energy with the squared-signal scale and
 * lets it sit alongside other motion moments cleanly.
 *
 * Out-of-range or NaN/Infinity inputs return 0 — never throw — so a
 * malformed sensor capture downstream produces a deterministic zero
 * feature instead of a crash.
 */
export function bandEnergy(
  real: number[],
  imag: number[],
  sampleRate: number,
  fLow: number,
  fHigh: number,
): number {
  const N = real.length;
  if (
    N === 0 ||
    !Number.isFinite(sampleRate) ||
    sampleRate <= 0 ||
    fLow >= fHigh ||
    fLow < 0
  ) {
    return 0;
  }

  // Bin k → frequency k × sampleRate / N. Only positive frequencies are
  // physically meaningful for real input.
  const binHz = sampleRate / N;
  const kLow = Math.max(0, Math.ceil(fLow / binHz));
  const kHigh = Math.min(Math.floor(N / 2), Math.floor((fHigh - 1e-9) / binHz));

  let energy = 0;
  for (let k = kLow; k <= kHigh; k++) {
    const re = real[k] ?? 0;
    const im = imag[k] ?? 0;
    energy += re * re + im * im;
  }
  return energy / (N * N);
}

/**
 * Return the dominant frequency (Hz) and its squared-magnitude amplitude
 * (normalized by `N²` for the same reason as `bandEnergy`) in the
 * half-open frequency interval [`fLow`, `fHigh`) Hz.
 *
 * If no bin falls inside the band (e.g. capture is too short to resolve
 * the requested band), returns `{ freq: 0, amplitude: 0 }`. Used by the
 * kinematic tremor block to detect physiological-tremor signatures in
 * the 4–12 Hz range.
 */
export function peakInBand(
  real: number[],
  imag: number[],
  sampleRate: number,
  fLow: number,
  fHigh: number,
): { freq: number; amplitude: number } {
  const N = real.length;
  if (
    N === 0 ||
    !Number.isFinite(sampleRate) ||
    sampleRate <= 0 ||
    fLow >= fHigh ||
    fLow < 0
  ) {
    return { freq: 0, amplitude: 0 };
  }

  const binHz = sampleRate / N;
  const kLow = Math.max(0, Math.ceil(fLow / binHz));
  const kHigh = Math.min(Math.floor(N / 2), Math.floor((fHigh - 1e-9) / binHz));

  let bestK = -1;
  let bestAmp = -Infinity;
  for (let k = kLow; k <= kHigh; k++) {
    const re = real[k] ?? 0;
    const im = imag[k] ?? 0;
    const amp = re * re + im * im;
    if (amp > bestAmp) {
      bestAmp = amp;
      bestK = k;
    }
  }
  if (bestK < 0) return { freq: 0, amplitude: 0 };
  return { freq: bestK * binHz, amplitude: bestAmp / (N * N) };
}

export { nextPow2 };
