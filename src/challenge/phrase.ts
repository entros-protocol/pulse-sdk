// Phonetically-balanced nonsense syllables for the voice challenge — FALLBACK ONLY.
//
// The authoritative challenge phrase is server-issued by the executor's
// `/challenge` endpoint (a 5-word phrase drawn from a curated English-word
// dictionary — see `entros-validation/src/word_dict.rs` for the source of truth).
// This client-side generator only fires when the executor is unreachable; in
// that path the server has no record of the phrase and validation skips
// phrase content binding entirely (Tier 1 acoustic + Tier 2 cross-modal still
// run). The fallback intentionally stays nonsense to avoid shipping the
// curated dictionary client-side — the JS bundle stays lean, and a degraded
// session is visually distinct from a normal one for users / contributors
// debugging.
const SYLLABLES = [
  "ba", "da", "fa", "ga", "ha", "ja", "ka", "la", "ma", "na",
  "pa", "ra", "sa", "ta", "wa", "za", "be", "de", "fe", "ge",
  "ke", "le", "me", "ne", "pe", "re", "se", "te", "we", "ze",
  "bi", "di", "fi", "gi", "ki", "li", "mi", "ni", "pi", "ri",
  "si", "ti", "wi", "zi", "bo", "do", "fo", "go", "ko", "lo",
  "mo", "no", "po", "ro", "so", "to", "wo", "zo", "bu", "du",
  "fu", "gu", "ku", "lu", "mu", "nu", "pu", "ru", "su", "tu",
];

/** Cryptographically random integer in [0, max) */
function secureRandom(max: number): number {
  const arr = new Uint32Array(1);
  crypto.getRandomValues(arr);
  return arr[0]! % max;
}

/**
 * FALLBACK challenge-phrase generator. Used only when the executor's
 * `/challenge` endpoint is unreachable; the authoritative phrase comes from
 * the server (5 real words drawn from a curated English-word dictionary). On
 * this fallback path, validation skips server-side phrase content binding —
 * Tier 1 acoustic + Tier 2 cross-modal still run.
 *
 * Output is 5-6 syllable pairs, forming nonsensical but speakable words.
 * Uses crypto.getRandomValues for unpredictable challenge generation.
 */
export function generatePhrase(wordCount: number = 5): string {
  const words: string[] = [];
  for (let w = 0; w < wordCount; w++) {
    const syllableCount = 2 + secureRandom(2);
    let word = "";
    for (let s = 0; s < syllableCount; s++) {
      word += SYLLABLES[secureRandom(SYLLABLES.length)];
    }
    words.push(word);
  }
  return words.join(" ");
}

/**
 * Generate a sequence of phrases for dynamic mid-session switching.
 * Each phrase uses a different syllable subset to prevent pre-computation.
 */
export function generatePhraseSequence(
  count: number = 3,
  wordCount: number = 4
): string[] {
  const subsetSize = Math.floor(SYLLABLES.length / count);
  const phrases: string[] = [];

  for (let p = 0; p < count; p++) {
    const start = (p * subsetSize) % SYLLABLES.length;
    const subset = [
      ...SYLLABLES.slice(start, start + subsetSize),
      ...SYLLABLES.slice(0, Math.max(0, (start + subsetSize) - SYLLABLES.length)),
    ];

    const words: string[] = [];
    for (let w = 0; w < wordCount; w++) {
      const syllableCount = 2 + secureRandom(2);
      let word = "";
      for (let s = 0; s < syllableCount; s++) {
        word += subset[secureRandom(subset.length)];
      }
      words.push(word);
    }
    phrases.push(words.join(" "));
  }

  return phrases;
}
