const ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

export function base58Encode(bytes: Uint8Array): string {
  let number = 0n;
  for (const byte of bytes) number = (number << 8n) | BigInt(byte);
  let text = "";
  while (number > 0n) {
    text = ALPHABET[Number(number % 58n)] + text;
    number /= 58n;
  }
  let zeros = 0;
  while (zeros < bytes.length && bytes[zeros] === 0) zeros++;
  return "1".repeat(zeros) + text;
}

export function base58Decode(text: string): Uint8Array | null {
  if (text.length === 0) return null;
  let number = 0n;
  for (const character of text) {
    const digit = ALPHABET.indexOf(character);
    if (digit < 0) return null;
    number = number * 58n + BigInt(digit);
  }
  const body: number[] = [];
  while (number > 0n) {
    body.unshift(Number(number & 0xffn));
    number >>= 8n;
  }
  let zeros = 0;
  while (zeros < text.length && text[zeros] === "1") zeros++;
  return Uint8Array.from([...new Array<number>(zeros).fill(0), ...body]);
}

/** Decodes a 32-byte key and rejects any text that is not its canonical encoding. */
export function canonicalKeyBytes(text: unknown): Uint8Array | null {
  if (typeof text !== "string" || text.length < 32 || text.length > 44) return null;
  const bytes = base58Decode(text);
  if (!bytes || bytes.length !== 32 || base58Encode(bytes) !== text) return null;
  return bytes;
}
