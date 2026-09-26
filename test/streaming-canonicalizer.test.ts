import { describe, expect, it } from "vitest";
import { createStreamingCanonicalizer, resampleTo } from "../src/sensor/resample";

function signal(rate: number, seconds: number, seed: number): Float32Array {
  const out = new Float32Array(Math.round(rate * seconds));
  let state = seed;
  for (let i = 0; i < out.length; i++) {
    state = (state * 1_103_515_245 + 12_345) % 2_147_483_648;
    out[i] = 0.4 * Math.sin((2 * Math.PI * 220 * i) / rate) + (state / 2_147_483_648 - 0.5) * 0.2;
  }
  return out;
}

function streamed(input: Float32Array, rate: number, chunkSizes: number[]): Float32Array {
  const stream = createStreamingCanonicalizer(rate);
  const parts: Float32Array[] = [];
  let offset = 0;
  let turn = 0;
  while (offset < input.length) {
    const size = chunkSizes[turn % chunkSizes.length]!;
    parts.push(stream.push(input.subarray(offset, offset + size)));
    offset += size;
    turn++;
  }
  parts.push(stream.flush());
  const total = parts.reduce((sum, part) => sum + part.length, 0);
  const out = new Float32Array(total);
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.length;
  }
  return out;
}

describe("streaming canonicaliser", () => {
  for (const rate of [16_000, 44_100, 48_000]) {
    it(`matches the batch resampler value for value at ${rate} Hz`, async () => {
      const input = signal(rate, 6.3, rate);
      const batch = await resampleTo(input, rate, 16_000);
      for (const chunks of [[4_096], [1, 7, 4_096, 333], [128], [100_000]]) {
        const stream = streamed(input, rate, chunks);
        expect(stream.length).toBe(batch.length);
        expect(Buffer.from(stream.buffer).equals(Buffer.from(batch.buffer))).toBe(true);
      }
    });
  }

  it("refuses a rate below the canonical rate", () => {
    expect(() => createStreamingCanonicalizer(8_000)).toThrow(RangeError);
  });
});
