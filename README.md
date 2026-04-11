# @iam-protocol/pulse-sdk

[![npm version](https://img.shields.io/npm/v/@iam-protocol/pulse-sdk.svg)](https://www.npmjs.com/package/@iam-protocol/pulse-sdk)
[![npm downloads](https://img.shields.io/npm/dm/@iam-protocol/pulse-sdk.svg)](https://www.npmjs.com/package/@iam-protocol/pulse-sdk)

Client-side SDK for the IAM Protocol. Captures behavioral biometrics (voice, motion, touch), extracts 134 statistical features, generates a Groth16 zero-knowledge proof, and submits for on-chain verification on Solana. Raw biometric data stays on-device — only derived features and the proof are transmitted.

## Install

```bash
npm install @iam-protocol/pulse-sdk
```

## Usage

### Wallet-connected (primary)

The user pays a small protocol fee (~0.005 SOL) and signs the verification transaction. Re-verification is batched into a single transaction (1 wallet prompt).

```typescript
import { PulseSDK } from '@iam-protocol/pulse-sdk';

const pulse = new PulseSDK({ cluster: 'devnet' });
const result = await pulse.verify(touchElement, walletAdapter, connection);

if (result.success) {
  console.log('Verified:', result.txSignature);
}
```

### Walletless (liveness-check tier)

For non-crypto users. No wallet, no SOL required. The integrator optionally funds verifications via the relayer API.

```typescript
import { PulseSDK } from '@iam-protocol/pulse-sdk';

const pulse = new PulseSDK({
  cluster: 'devnet',
  relayerUrl: 'https://api.iam-human.io/relay',
  wasmUrl: '/circuits/iam_hamming.wasm',
  zkeyUrl: '/circuits/iam_hamming_final.zkey',
});

const result = await pulse.verify(touchElement);
```

## Pipeline

1. **Capture**: Audio (16kHz), IMU (accelerometer + gyroscope), touch (pressure + area) — event-driven, caller controls duration
2. **Extract**: 134 features — speaker (F0, jitter, shimmer, HNR, formants, LTAS), motion (jerk/jounce), touch (velocity/pressure)
3. **Validate**: Feature summaries sent to IAM validation server for server-side analysis
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

## License

MIT
