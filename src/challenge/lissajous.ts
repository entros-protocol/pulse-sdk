/**
 * Generate Lissajous curve points for the touch tracing challenge.
 * The user traces this shape on screen while speaking the phrase.
 *
 * x(t) = A * sin(a*t + delta)
 * y(t) = B * sin(b*t)
 */
export interface LissajousParams {
  a: number;
  b: number;
  delta: number;
  points: number;
}

export interface Point2D {
  x: number;
  y: number;
}

/**
 * Generate random Lissajous parameters for a challenge.
 */
export function randomLissajousParams(): LissajousParams {
  const ratios = [
    [1, 2],
    [2, 3],
    [3, 4],
    [3, 5],
    [4, 5],
  ];
  const arr = new Uint32Array(2);
  crypto.getRandomValues(arr);
  const pair = ratios[arr[0]! % ratios.length]!;
  return {
    a: pair[0]!,
    b: pair[1]!,
    delta: Math.PI * (0.25 + (arr[1]! / 0xFFFFFFFF) * 0.5),
    points: 200,
  };
}

/**
 * Generate Lissajous curve points normalized to [0, 1] range.
 */
export function generateLissajousPoints(params: LissajousParams): Point2D[] {
  const { a, b, delta, points } = params;
  const result: Point2D[] = [];

  for (let i = 0; i < points; i++) {
    const t = (i / points) * 2 * Math.PI;
    result.push({
      x: (Math.sin(a * t + delta) + 1) / 2,
      y: (Math.sin(b * t) + 1) / 2,
    });
  }

  return result;
}

/**
 * Generate a sequence of Lissajous curves for dynamic mid-session switching.
 * Each curve uses different parameters, preventing pre-computation.
 */
export function generateLissajousSequence(
  count: number = 2
): { params: LissajousParams; points: Point2D[] }[] {
  const allRatios: [number, number][] = [
    [1, 2], [2, 3], [3, 4], [3, 5], [4, 5],
    [1, 3], [2, 5], [5, 6], [3, 7], [4, 7],
  ];

  // Fisher-Yates shuffle with crypto randomness
  const shuffled = [...allRatios];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const arr = new Uint32Array(1);
    crypto.getRandomValues(arr);
    const j = arr[0]! % (i + 1);
    [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
  }

  const sequence: { params: LissajousParams; points: Point2D[] }[] = [];

  for (let i = 0; i < count; i++) {
    const pair = shuffled[i % shuffled.length]!;
    const deltaArr = new Uint32Array(1);
    crypto.getRandomValues(deltaArr);
    const params: LissajousParams = {
      a: pair[0],
      b: pair[1],
      delta: Math.PI * (0.1 + (deltaArr[0]! / 0xFFFFFFFF) * 0.8),
      points: 200,
    };
    sequence.push({ params, points: generateLissajousPoints(params) });
  }

  return sequence;
}
