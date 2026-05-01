# @entros/pulse-sdk

[![npm version](https://img.shields.io/npm/v/@entros/pulse-sdk.svg)](https://www.npmjs.com/package/@entros/pulse-sdk)
[![npm downloads](https://img.shields.io/npm/dm/@entros/pulse-sdk.svg)](https://www.npmjs.com/package/@entros/pulse-sdk)

Client-side SDK for the Entros Protocol. Captures behavioral biometrics (voice, motion, touch), extracts 134 statistical features, generates a Groth16 zero-knowledge proof, and submits for on-chain verification on Solana. Raw biometric data stays on-device — only derived features and the proof are transmitted.

> **Looking for a drop-in?** Most integrators want [`@entros/verify`](https://github.com/entros-protocol/entros-verify) — a popup-pattern React component that wraps this SDK and ships verification in five lines of JSX. Use this package directly when you need to own the verification UX (custom capture canvas, branded loading states, mobile-native).

## Install

```bash
npm install @entros/pulse-sdk
```

## Usage

### Wallet-connected (primary)

The user pays a small protocol fee (~0.005 SOL) and signs the verification transaction. Re-verification is batched into a single transaction (1 wallet prompt).

```typescript
import { PulseSDK } from '@entros/pulse-sdk';

const pulse = new PulseSDK({ cluster: 'devnet' });
const result = await pulse.verify(touchElement, walletAdapter, connection);

if (result.success) {
  console.log('Verified:', result.txSignature);
}
```

### Walletless (liveness-check tier)

For non-crypto users. No wallet, no SOL required. The integrator optionally funds verifications via the relayer API.

```typescript
import { PulseSDK } from '@entros/pulse-sdk';

const pulse = new PulseSDK({
  cluster: 'devnet',
  relayerUrl: 'https://api.entros.io/relay',
  wasmUrl: '/circuits/entros_hamming.wasm',
  zkeyUrl: '/circuits/entros_hamming_final.zkey',
});

const result = await pulse.verify(touchElement);
```

## Pipeline

1. **Capture**: Audio (16kHz), IMU (accelerometer + gyroscope), touch (pressure + area) — event-driven, caller controls duration
2. **Extract**: 134 features — speaker (F0, jitter, shimmer, HNR, formants, LTAS), motion (jerk/jounce), touch (velocity/pressure)
3. **Validate**: Feature summaries sent to Entros validation server for server-side analysis
4. **Hash**: SimHash → 256-bit Temporal Fingerprint → Poseidon commitment
5. **Prove**: Groth16 proof that new fingerprint is within Hamming distance of previous
6. **Submit**: Single batched transaction via wallet (1 prompt) or relayer

## Development

```bash
npm install
npm test          # 60 vitest tests (including 8-phase adversarial pen test)
npm run build     # ESM + CJS output
npm run typecheck # TypeScript strict mode
```

## Migration history

Originally published as `@iam-protocol/pulse-sdk` (deprecated). Renamed during
the IAM → Entros Protocol rebrand on 2026-04-25; full commit history preserved
on the current repository at `github.com/entros-protocol/pulse-sdk`.

## License

MIT
