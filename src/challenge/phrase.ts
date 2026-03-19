// Phonetically-balanced nonsense syllables for the voice challenge.
// Designed to elicit diverse vocal patterns while preventing dictionary-based deepfake attacks.
const SYLLABLES = [
  "ba", "da", "fa", "ga", "ha", "ja", "ka", "la", "ma", "na",
  "pa", "ra", "sa", "ta", "wa", "za", "be", "de", "fe", "ge",
  "ke", "le", "me", "ne", "pe", "re", "se", "te", "we", "ze",
  "bi", "di", "fi", "gi", "ki", "li", "mi", "ni", "pi", "ri",
  "si", "ti", "wi", "zi", "bo", "do", "fo", "go", "ko", "lo",
  "mo", "no", "po", "ro", "so", "to", "wo", "zo", "bu", "du",
  "fu", "gu", "ku", "lu", "mu", "nu", "pu", "ru", "su", "tu",
];

/**
 * Generate a random phonetically-balanced phrase for the voice challenge.
 * Each phrase is 5-6 syllable pairs, forming nonsensical but speakable words.
 */
export function generatePhrase(wordCount: number = 5): string {
  const words: string[] = [];
  for (let w = 0; w < wordCount; w++) {
    const syllableCount = 2 + Math.floor(Math.random() * 2); // 2-3 syllables per word
    let word = "";
    for (let s = 0; s < syllableCount; s++) {
      const idx = Math.floor(Math.random() * SYLLABLES.length);
      word += SYLLABLES[idx];
    }
    words.push(word);
  }
  return words.join(" ");
}
