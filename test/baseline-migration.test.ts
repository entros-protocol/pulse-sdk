/**
 * Tests for src/identity/baseline.ts — master-list #98 SDK side.
 *
 * Covers:
 *   - deriveBaselineKey determinism under a mocked deterministic signer
 *   - encrypt/decrypt round-trip with correct AAD
 *   - StaleEncryptedBaselineError on commitment mismatch (the staleness signal)
 *   - StaleEncryptedBaselineError on wallet / PDA / version mismatches
 *   - Malformed blob rejection (wrong length, wrong version, wrong algo)
 *   - deriveEncryptedBaselinePda stability
 */

import { describe, it, expect, beforeEach } from "vitest";
import { Keypair, PublicKey } from "@solana/web3.js";
import {
  bytes32ToBigint,
  bytesToFingerprint,
  clearBaselineKeyCache,
  decryptBaselineBlob,
  deriveBaselineKey,
  deriveEncryptedBaselinePda,
  ENCRYPTED_BASELINE_BLOB_BYTES,
  encryptBaselineBlob,
  fingerprintToBytes,
  getOrDeriveBaselineKey,
  StaleEncryptedBaselineError,
  type BaselineWallet,
} from "../src/identity/baseline";
import { bigintToBytes32 } from "../src/hashing/poseidon";

// --- Helpers ---

/**
 * A mock wallet with deterministic `signMessage`. Returns the same
 * 64-byte "signature" for the same input message, mimicking RFC 8032
 * Ed25519 determinism. The bytes themselves are fixed-keyed pseudorandom
 * (derived from a stable seed) — sufficient for HKDF input under test.
 */
function makeMockWallet(seed: number): BaselineWallet {
  const kp = Keypair.fromSeed(new Uint8Array(32).fill(seed));
  return {
    publicKey: kp.publicKey,
    signMessage: async (msg: Uint8Array) => {
      // Construct a deterministic 64-byte "signature" by hashing
      // (seed || message) twice. Cryptographically meaningless, but
      // deterministic — which is the only property the tests need.
      const seedPrefix = new Uint8Array([seed]);
      const buf = new Uint8Array(seedPrefix.length + msg.length);
      buf.set(seedPrefix, 0);
      buf.set(msg, seedPrefix.length);
      const h1 = new Uint8Array(
        await crypto.subtle.digest("SHA-256", buf as Uint8Array<ArrayBuffer>)
      );
      const h2 = new Uint8Array(
        await crypto.subtle.digest("SHA-256", h1 as Uint8Array<ArrayBuffer>)
      );
      const sig = new Uint8Array(64);
      sig.set(h1, 0);
      sig.set(h2, 32);
      return sig;
    },
  };
}

function fixedBytes(seed: number, length: number): Uint8Array {
  const bytes = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    bytes[i] = (seed + i) & 0xff;
  }
  return bytes;
}

async function setup(seed = 1) {
  const wallet = makeMockWallet(seed);
  const key = await deriveBaselineKey(wallet);
  const [baselinePda] = await deriveEncryptedBaselinePda(wallet.publicKey);
  return { wallet, key, baselinePda };
}

// --- deriveBaselineKey ---

describe("deriveBaselineKey", () => {
  it("produces a usable AES-GCM CryptoKey", async () => {
    const wallet = makeMockWallet(7);
    const key = await deriveBaselineKey(wallet);
    expect(key).toBeDefined();
    // crypto.subtle.encrypt with this key should not throw.
    const ct = await crypto.subtle.encrypt(
      { name: "AES-GCM", iv: new Uint8Array(12) },
      key,
      new Uint8Array(16)
    );
    expect(ct.byteLength).toBeGreaterThan(0);
  });

  it("is deterministic — same wallet produces the same key (round-trip stability)", async () => {
    const wallet = makeMockWallet(7);
    const key1 = await deriveBaselineKey(wallet);
    const key2 = await deriveBaselineKey(wallet);
    // Keys are non-extractable, so identity is checked via behavior:
    // encrypt with key1, decrypt with key2.
    const iv = new Uint8Array(12).fill(0x42);
    const plaintext = new Uint8Array(16).fill(0xaa);
    const ct = await crypto.subtle.encrypt(
      { name: "AES-GCM", iv },
      key1,
      plaintext as Uint8Array<ArrayBuffer>
    );
    const dec = new Uint8Array(
      await crypto.subtle.decrypt(
        { name: "AES-GCM", iv },
        key2,
        ct
      )
    );
    expect([...dec]).toEqual([...plaintext]);
  });

  it("produces different keys for different wallets", async () => {
    const a = makeMockWallet(1);
    const b = makeMockWallet(2);
    const keyA = await deriveBaselineKey(a);
    const keyB = await deriveBaselineKey(b);
    const iv = new Uint8Array(12);
    const plaintext = new Uint8Array(16);
    const ct = await crypto.subtle.encrypt(
      { name: "AES-GCM", iv },
      keyA,
      plaintext as Uint8Array<ArrayBuffer>
    );
    // Decrypting wallet A's ciphertext with wallet B's key must fail.
    await expect(
      crypto.subtle.decrypt({ name: "AES-GCM", iv }, keyB, ct)
    ).rejects.toBeInstanceOf(Error);
  });

  it("throws when wallet lacks signMessage", async () => {
    const bogus = {
      publicKey: Keypair.generate().publicKey,
    } as unknown as BaselineWallet;
    await expect(deriveBaselineKey(bogus)).rejects.toThrow(
      /signMessage/
    );
  });
});

// --- deriveEncryptedBaselinePda ---

describe("deriveEncryptedBaselinePda", () => {
  it("returns a deterministic PDA + bump for a given wallet", async () => {
    const wallet = Keypair.generate().publicKey;
    const [pda1, bump1] = await deriveEncryptedBaselinePda(wallet);
    const [pda2, bump2] = await deriveEncryptedBaselinePda(wallet);
    expect(pda1.toBase58()).toBe(pda2.toBase58());
    expect(bump1).toBe(bump2);
  });

  it("produces distinct PDAs for distinct wallets", async () => {
    const a = Keypair.generate().publicKey;
    const b = Keypair.generate().publicKey;
    const [pdaA] = await deriveEncryptedBaselinePda(a);
    const [pdaB] = await deriveEncryptedBaselinePda(b);
    expect(pdaA.toBase58()).not.toBe(pdaB.toBase58());
  });
});

// --- encrypt + decrypt ---

describe("encryptBaselineBlob + decryptBaselineBlob", () => {
  it("produces a 96-byte blob with version=1, algo=1, and zeroed reserved bytes", async () => {
    const { wallet, key, baselinePda } = await setup();
    const simhash = fixedBytes(1, 32);
    const salt = fixedBytes(2, 32);
    const commitment = fixedBytes(3, 32);
    const blob = await encryptBaselineBlob(
      simhash,
      salt,
      key,
      wallet.publicKey,
      baselinePda,
      commitment
    );
    expect(blob.length).toBe(ENCRYPTED_BASELINE_BLOB_BYTES);
    expect(blob[0]).toBe(0x01); // version
    expect(blob[1]).toBe(0x01); // algorithm
    expect(blob[2]).toBe(0x00); // reserved
    expect(blob[3]).toBe(0x00); // reserved
  });

  it("round-trips simhash + salt under correct AAD", async () => {
    const { wallet, key, baselinePda } = await setup();
    const simhash = fixedBytes(0x11, 32);
    const salt = fixedBytes(0x22, 32);
    const commitment = fixedBytes(0x33, 32);
    const blob = await encryptBaselineBlob(
      simhash,
      salt,
      key,
      wallet.publicKey,
      baselinePda,
      commitment
    );
    const decrypted = await decryptBaselineBlob(
      blob,
      key,
      wallet.publicKey,
      baselinePda,
      commitment
    );
    expect([...decrypted.simhash]).toEqual([...simhash]);
    expect([...decrypted.salt]).toEqual([...salt]);
  });

  it("uses a fresh random IV per encryption — repeated encrypts produce distinct blobs", async () => {
    const { wallet, key, baselinePda } = await setup();
    const simhash = fixedBytes(0x11, 32);
    const salt = fixedBytes(0x22, 32);
    const commitment = fixedBytes(0x33, 32);
    const blob1 = await encryptBaselineBlob(
      simhash, salt, key, wallet.publicKey, baselinePda, commitment
    );
    const blob2 = await encryptBaselineBlob(
      simhash, salt, key, wallet.publicKey, baselinePda, commitment
    );
    expect([...blob1]).not.toEqual([...blob2]);
  });

  it("rejects simhash of wrong length", async () => {
    const { wallet, key, baselinePda } = await setup();
    await expect(
      encryptBaselineBlob(
        new Uint8Array(31),
        fixedBytes(2, 32),
        key,
        wallet.publicKey,
        baselinePda,
        fixedBytes(3, 32)
      )
    ).rejects.toThrow(/simhash must be 32 bytes/);
  });

  it("rejects salt of wrong length", async () => {
    const { wallet, key, baselinePda } = await setup();
    await expect(
      encryptBaselineBlob(
        fixedBytes(1, 32),
        new Uint8Array(31),
        key,
        wallet.publicKey,
        baselinePda,
        fixedBytes(3, 32)
      )
    ).rejects.toThrow(/salt must be 32 bytes/);
  });
});

// --- Staleness detection (the core security property) ---

describe("StaleEncryptedBaselineError — AAD-binding staleness signal", () => {
  it("throws when decrypting under a DIFFERENT commitment (post-reset scenario)", async () => {
    const { wallet, key, baselinePda } = await setup();
    const simhash = fixedBytes(0x11, 32);
    const salt = fixedBytes(0x22, 32);
    const commitmentAtEncryption = fixedBytes(0x33, 32);
    const blob = await encryptBaselineBlob(
      simhash, salt, key, wallet.publicKey, baselinePda, commitmentAtEncryption
    );

    // After a `reset_identity_state`, the on-chain commitment cycles to a new value.
    const newCommitment = fixedBytes(0x44, 32);
    await expect(
      decryptBaselineBlob(blob, key, wallet.publicKey, baselinePda, newCommitment)
    ).rejects.toBeInstanceOf(StaleEncryptedBaselineError);
  });

  it("throws when decrypting with a DIFFERENT wallet pubkey in the AAD", async () => {
    const { wallet, key, baselinePda } = await setup();
    const simhash = fixedBytes(0x11, 32);
    const salt = fixedBytes(0x22, 32);
    const commitment = fixedBytes(0x33, 32);
    const blob = await encryptBaselineBlob(
      simhash, salt, key, wallet.publicKey, baselinePda, commitment
    );

    const wrongWallet = Keypair.generate().publicKey;
    await expect(
      decryptBaselineBlob(blob, key, wrongWallet, baselinePda, commitment)
    ).rejects.toBeInstanceOf(StaleEncryptedBaselineError);
  });

  it("throws when decrypting with a DIFFERENT baseline PDA in the AAD", async () => {
    const { wallet, key, baselinePda } = await setup();
    const simhash = fixedBytes(0x11, 32);
    const salt = fixedBytes(0x22, 32);
    const commitment = fixedBytes(0x33, 32);
    const blob = await encryptBaselineBlob(
      simhash, salt, key, wallet.publicKey, baselinePda, commitment
    );

    const wrongPda = new PublicKey(fixedBytes(0xff, 32));
    await expect(
      decryptBaselineBlob(blob, key, wallet.publicKey, wrongPda, commitment)
    ).rejects.toBeInstanceOf(StaleEncryptedBaselineError);
  });

  it("does NOT throw StaleEncryptedBaselineError for malformed-length blobs (different failure class)", async () => {
    const { wallet, key, baselinePda } = await setup();
    const malformed = new Uint8Array(95); // 1 byte short
    await expect(
      decryptBaselineBlob(malformed, key, wallet.publicKey, baselinePda, fixedBytes(3, 32))
    ).rejects.toThrow(/blob must be 96 bytes/);
  });

  it("rejects unsupported version byte", async () => {
    const { wallet, key, baselinePda } = await setup();
    const simhash = fixedBytes(0x11, 32);
    const salt = fixedBytes(0x22, 32);
    const commitment = fixedBytes(0x33, 32);
    const blob = await encryptBaselineBlob(
      simhash, salt, key, wallet.publicKey, baselinePda, commitment
    );
    // Corrupt the version byte.
    const corrupted = new Uint8Array(blob);
    corrupted[0] = 0x02;
    await expect(
      decryptBaselineBlob(corrupted, key, wallet.publicKey, baselinePda, commitment)
    ).rejects.toThrow(/unsupported blob version/);
  });

  it("rejects unsupported algorithm byte", async () => {
    const { wallet, key, baselinePda } = await setup();
    const simhash = fixedBytes(0x11, 32);
    const salt = fixedBytes(0x22, 32);
    const commitment = fixedBytes(0x33, 32);
    const blob = await encryptBaselineBlob(
      simhash, salt, key, wallet.publicKey, baselinePda, commitment
    );
    // Corrupt the algorithm byte.
    const corrupted = new Uint8Array(blob);
    corrupted[1] = 0x02;
    await expect(
      decryptBaselineBlob(corrupted, key, wallet.publicKey, baselinePda, commitment)
    ).rejects.toThrow(/unsupported algorithm/);
  });

  it("rejects commitment of wrong length", async () => {
    const { wallet, key, baselinePda } = await setup();
    await expect(
      decryptBaselineBlob(
        new Uint8Array(96),
        key,
        wallet.publicKey,
        baselinePda,
        new Uint8Array(31)
      )
    ).rejects.toThrow(/commitment must be 32 bytes/);
  });
});

// --- Bit packing helpers ---

describe("fingerprintToBytes + bytesToFingerprint", () => {
  it("round-trips a known 256-bit pattern", () => {
    const bits = new Array(256).fill(0).map((_, i) => i % 3 === 0 ? 1 : 0);
    const bytes = fingerprintToBytes(bits);
    expect(bytes.length).toBe(32);
    const recovered = bytesToFingerprint(bytes);
    expect(recovered).toEqual(bits);
  });

  it("round-trips all-zeros + all-ones", () => {
    const zeros = new Array(256).fill(0);
    const ones = new Array(256).fill(1);
    expect(bytesToFingerprint(fingerprintToBytes(zeros))).toEqual(zeros);
    expect(bytesToFingerprint(fingerprintToBytes(ones))).toEqual(ones);
  });

  it("rejects wrong-length input", () => {
    expect(() => fingerprintToBytes(new Array(255).fill(0))).toThrow(
      /expected 256-bit fingerprint/
    );
    expect(() => bytesToFingerprint(new Uint8Array(31))).toThrow(
      /expected 32 bytes/
    );
  });
});

describe("bytes32ToBigint", () => {
  it("is the inverse of bigintToBytes32 for a range of values", () => {
    const values = [
      BigInt(0),
      BigInt(1),
      BigInt(255),
      BigInt(256),
      BigInt("0xdeadbeef"),
      BigInt("0x" + "ff".repeat(32)),
    ];
    for (const v of values) {
      const bytes = bigintToBytes32(v);
      const round = bytes32ToBigint(bytes);
      expect(round).toBe(v);
    }
  });

  it("rejects wrong-length input", () => {
    expect(() => bytes32ToBigint(new Uint8Array(31))).toThrow(
      /expected 32 bytes/
    );
  });
});

// --- Session-scoped key cache ---

describe("getOrDeriveBaselineKey + clearBaselineKeyCache", () => {
  beforeEach(() => {
    clearBaselineKeyCache();
  });

  it("returns the same key for the same wallet across calls (cache hit)", async () => {
    const wallet = makeMockWallet(11);
    const key1 = await getOrDeriveBaselineKey(wallet);
    const key2 = await getOrDeriveBaselineKey(wallet);
    // Identity is checked via behavior — the keys must encrypt/decrypt
    // compatibly (same key material).
    const iv = new Uint8Array(12).fill(0x77);
    const plaintext = new Uint8Array(16).fill(0xbb);
    const ct = await crypto.subtle.encrypt(
      { name: "AES-GCM", iv },
      key1,
      plaintext as Uint8Array<ArrayBuffer>
    );
    const dec = new Uint8Array(
      await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key2, ct)
    );
    expect([...dec]).toEqual([...plaintext]);
  });

  it("issues distinct keys for distinct wallets", async () => {
    const a = makeMockWallet(20);
    const b = makeMockWallet(21);
    const keyA = await getOrDeriveBaselineKey(a);
    const keyB = await getOrDeriveBaselineKey(b);
    const iv = new Uint8Array(12);
    const plaintext = new Uint8Array(16);
    const ct = await crypto.subtle.encrypt(
      { name: "AES-GCM", iv },
      keyA,
      plaintext as Uint8Array<ArrayBuffer>
    );
    await expect(
      crypto.subtle.decrypt({ name: "AES-GCM", iv }, keyB, ct)
    ).rejects.toBeInstanceOf(Error);
  });

  it("only prompts signMessage on first call per wallet", async () => {
    let calls = 0;
    const seed = 30;
    const kp = Keypair.fromSeed(new Uint8Array(32).fill(seed));
    const wallet: BaselineWallet = {
      publicKey: kp.publicKey,
      signMessage: async (msg: Uint8Array) => {
        calls += 1;
        const seedPrefix = new Uint8Array([seed]);
        const buf = new Uint8Array(seedPrefix.length + msg.length);
        buf.set(seedPrefix, 0);
        buf.set(msg, seedPrefix.length);
        const h1 = new Uint8Array(
          await crypto.subtle.digest("SHA-256", buf as Uint8Array<ArrayBuffer>)
        );
        const h2 = new Uint8Array(
          await crypto.subtle.digest("SHA-256", h1 as Uint8Array<ArrayBuffer>)
        );
        const sig = new Uint8Array(64);
        sig.set(h1, 0);
        sig.set(h2, 32);
        return sig;
      },
    };

    await getOrDeriveBaselineKey(wallet);
    await getOrDeriveBaselineKey(wallet);
    await getOrDeriveBaselineKey(wallet);
    expect(calls).toBe(1);
  });

  it("clearBaselineKeyCache forces re-derivation on next call", async () => {
    let calls = 0;
    const seed = 40;
    const kp = Keypair.fromSeed(new Uint8Array(32).fill(seed));
    const wallet: BaselineWallet = {
      publicKey: kp.publicKey,
      signMessage: async (msg: Uint8Array) => {
        calls += 1;
        const buf = new Uint8Array(1 + msg.length);
        buf[0] = seed;
        buf.set(msg, 1);
        const h1 = new Uint8Array(
          await crypto.subtle.digest("SHA-256", buf as Uint8Array<ArrayBuffer>)
        );
        const sig = new Uint8Array(64);
        sig.set(h1, 0);
        sig.set(h1, 32);
        return sig;
      },
    };

    await getOrDeriveBaselineKey(wallet);
    expect(calls).toBe(1);
    clearBaselineKeyCache();
    await getOrDeriveBaselineKey(wallet);
    expect(calls).toBe(2);
  });

  it("concurrent calls share one signMessage prompt (race-free cache)", async () => {
    let calls = 0;
    const seed = 50;
    const kp = Keypair.fromSeed(new Uint8Array(32).fill(seed));
    const wallet: BaselineWallet = {
      publicKey: kp.publicKey,
      signMessage: async (msg: Uint8Array) => {
        calls += 1;
        // Yield a tick so a naive cache (resolved-key-only) would race.
        await new Promise((resolve) => setTimeout(resolve, 0));
        const buf = new Uint8Array(1 + msg.length);
        buf[0] = seed;
        buf.set(msg, 1);
        const h1 = new Uint8Array(
          await crypto.subtle.digest("SHA-256", buf as Uint8Array<ArrayBuffer>)
        );
        const sig = new Uint8Array(64);
        sig.set(h1, 0);
        sig.set(h1, 32);
        return sig;
      },
    };

    await Promise.all([
      getOrDeriveBaselineKey(wallet),
      getOrDeriveBaselineKey(wallet),
      getOrDeriveBaselineKey(wallet),
      getOrDeriveBaselineKey(wallet),
    ]);
    expect(calls).toBe(1);
  });

  it("evicts the cache entry on derivation failure so subsequent calls retry", async () => {
    let calls = 0;
    let shouldFail = true;
    const seed = 60;
    const kp = Keypair.fromSeed(new Uint8Array(32).fill(seed));
    const wallet: BaselineWallet = {
      publicKey: kp.publicKey,
      signMessage: async (msg: Uint8Array) => {
        calls += 1;
        if (shouldFail) {
          throw new Error("user cancelled");
        }
        const buf = new Uint8Array(1 + msg.length);
        buf[0] = seed;
        buf.set(msg, 1);
        const h1 = new Uint8Array(
          await crypto.subtle.digest("SHA-256", buf as Uint8Array<ArrayBuffer>)
        );
        const sig = new Uint8Array(64);
        sig.set(h1, 0);
        sig.set(h1, 32);
        return sig;
      },
    };

    await expect(getOrDeriveBaselineKey(wallet)).rejects.toThrow(/user cancelled/);
    expect(calls).toBe(1);
    // Cache must NOT be poisoned with the rejection — the next call re-prompts.
    shouldFail = false;
    const key = await getOrDeriveBaselineKey(wallet);
    expect(key).toBeDefined();
    expect(calls).toBe(2);
  });
});

// --- End-to-end blob round-trip via the integration helpers ---

describe("end-to-end blob round-trip with fingerprint + salt encoding", () => {
  it("recovers the original 256-bit fingerprint and BigInt salt after encrypt + decrypt", async () => {
    const { wallet, key, baselinePda } = await setup(5);

    // Realistic fingerprint: 256 bits with some 1s.
    const fingerprint = new Array(256).fill(0).map((_, i) => (i * 7) % 5 === 0 ? 1 : 0);
    const salt = BigInt(
      "12345678901234567890123456789012345678901234567890"
    );
    const commitment = fixedBytes(0x99, 32);

    const simhashBytes = fingerprintToBytes(fingerprint);
    const saltBytes = bigintToBytes32(salt);
    const blob = await encryptBaselineBlob(
      simhashBytes,
      saltBytes,
      key,
      wallet.publicKey,
      baselinePda,
      commitment
    );

    const decrypted = await decryptBaselineBlob(
      blob,
      key,
      wallet.publicKey,
      baselinePda,
      commitment
    );

    const recoveredFingerprint = bytesToFingerprint(decrypted.simhash);
    const recoveredSalt = bytes32ToBigint(decrypted.salt);

    expect(recoveredFingerprint).toEqual(fingerprint);
    expect(recoveredSalt).toBe(salt);
  });
});
