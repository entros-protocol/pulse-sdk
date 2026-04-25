/**
 * Client-side encryption for stored verification data.
 *
 * Uses AES-256-GCM via Web Crypto API with a non-extractable CryptoKey
 * stored in IndexedDB. The key cannot be exported or exfiltrated — only
 * the browser's internal crypto engine can use it for encrypt/decrypt
 * on the same origin.
 */

const DB_NAME = "entros-protocol-keystore";
// Bumped to 2 on 2026-04-20 to force `onupgradeneeded` on pre-existing DBs
// whose `keys` object store went missing (observed in the wild — browser
// interrupted an earlier upgrade or a concurrent-tab race left the DB
// partially initialized). Existing users with a healthy v1 DB upgrade
// cleanly — the handler below is idempotent.
const DB_VERSION = 2;
const STORE_NAME = "keys";
const KEY_ID = "encryption-key";

// --- Capability detection ---

export function hasCryptoSupport(): boolean {
  return (
    typeof globalThis.crypto?.subtle !== "undefined" &&
    typeof globalThis.indexedDB !== "undefined"
  );
}

// --- IndexedDB key management ---

function openAtVersion(version: number): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, version);
    request.onupgradeneeded = () => {
      const db = request.result;
      // Idempotent: only create if missing. Preserves existing key+data on
      // version bumps that don't require a schema change.
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function openKeyStore(): Promise<IDBDatabase> {
  const db = await openAtVersion(DB_VERSION);
  // Defensive post-open check: if the DB somehow arrives at the current
  // version without the required store (observed in the wild — browser
  // interrupted an earlier upgrade, concurrent-tab race, manual DevTools
  // deletion), close and reopen at version+1 to force `onupgradeneeded`
  // to fire and recreate the store. Without this, every subsequent
  // `transaction(STORE_NAME, ...)` would throw `NotFoundError` forever.
  if (!db.objectStoreNames.contains(STORE_NAME)) {
    const nextVersion = db.version + 1;
    db.close();
    return openAtVersion(nextVersion);
  }
  return db;
}

function getKey(db: IDBDatabase): Promise<CryptoKey | null> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const request = store.get(KEY_ID);
    request.onsuccess = () => resolve(request.result ?? null);
    request.onerror = () => reject(request.error);
  });
}

function putKey(db: IDBDatabase, key: CryptoKey): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const request = store.put(key, KEY_ID);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

export async function getOrCreateEncryptionKey(): Promise<CryptoKey | null> {
  try {
    const db = await openKeyStore();
    try {
      const existing = await getKey(db);
      if (existing) return existing;

      const key = await crypto.subtle.generateKey(
        { name: "AES-GCM", length: 256 },
        false, // non-extractable
        ["encrypt", "decrypt"]
      );

      await putKey(db, key);
      return key;
    } finally {
      db.close();
    }
  } catch {
    return null;
  }
}

// --- AES-256-GCM encrypt / decrypt ---

export async function encrypt(
  plaintext: string,
  key: CryptoKey
): Promise<{ iv: string; ct: string }> {
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const encoded = new TextEncoder().encode(plaintext);
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    key,
    encoded
  );
  return {
    iv: toBase64(iv),
    ct: toBase64(new Uint8Array(ciphertext)),
  };
}

export async function decrypt(
  iv: string,
  ct: string,
  key: CryptoKey
): Promise<string> {
  const ivBytes = fromBase64(iv);
  const ctBytes = fromBase64(ct);
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: ivBytes.buffer as ArrayBuffer },
    key,
    ctBytes.buffer as ArrayBuffer
  );
  return new TextDecoder().decode(plaintext);
}

// --- Base64 helpers (browser-only, no deps) ---

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return btoa(binary);
}

function fromBase64(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}
