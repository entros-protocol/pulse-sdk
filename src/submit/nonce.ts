export interface SubmissionNonce {
  readonly bytes: Uint8Array;
  readonly source: "client" | "executor";
}
function localNonce(): Uint8Array {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  if (bytes.every((byte) => byte === 0)) bytes[31] = 1;
  return bytes;
}
export function validateNonce(value: unknown): Uint8Array {
  if (!Array.isArray(value) && !(value instanceof Uint8Array))
    throw new Error("Nonce must contain 32 bytes");
  const bytes: unknown[] = Array.from(value);
  if (
    bytes.length !== 32 ||
    !bytes.every(
      (byte) =>
        typeof byte === "number" &&
        Number.isInteger(byte) &&
        byte >= 0 &&
        byte <= 255,
    ) ||
    bytes.every((byte) => byte === 0)
  )
    throw new Error("Nonce must contain 32 bytes and must not be zero");
  return Uint8Array.from(bytes as number[]);
}
export async function fetchSubmissionNonce(
  wallet: string,
  relayerUrl?: string,
  apiKey?: string,
): Promise<SubmissionNonce> {
  if (relayerUrl) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    try {
      const headers: Record<string, string> = {};
      if (apiKey) headers["X-API-Key"] = apiKey;
      const response = await fetch(
        `${new URL(relayerUrl).origin}/challenge?wallet=${encodeURIComponent(wallet)}`,
        { headers, signal: controller.signal },
      );
      if (response.ok) {
        const value: unknown = await response.json();
        if (value && typeof value === "object" && "nonce" in value)
          return { bytes: validateNonce(value.nonce), source: "executor" };
      }
    } catch {
      // Fallback happens before proof generation fixes the request nonce.
    } finally {
      clearTimeout(timer);
    }
  }
  return { bytes: localNonce(), source: "client" };
}
