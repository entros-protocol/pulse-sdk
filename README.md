# @entros/pulse-sdk

[![npm version](https://img.shields.io/npm/v/@entros/pulse-sdk.svg)](https://www.npmjs.com/package/@entros/pulse-sdk)
[![npm downloads](https://img.shields.io/npm/dm/@entros/pulse-sdk.svg)](https://www.npmjs.com/package/@entros/pulse-sdk)

Client SDK for Entros capture, feature extraction, fingerprinting, proof generation, and Solana submission.

The SDK derives 308 features from phrase audio, motion, and touch. It generates a 256-bit SimHash fingerprint and Poseidon commitment. Re-verification also generates a Groth16 proof.

Raw motion and full-resolution touch streams stay in client memory. Phrase audio leaves transiently for private validation.

Validation also receives the feature summary, F0 contour, acceleration magnitude, wallet, client signals, curve outline, timing, and commitment fields.

The SDK stores the fingerprint, salt, commitment, and timestamp as a baseline. Wallet flows can store an encrypted baseline blob on-chain.

A host-approved fallback can store the local baseline without encryption. Integrators must treat that option as a lower-assurance recovery mode.

> **Looking for a drop-in?** Most integrators want [`@entros/verify`](https://github.com/entros-protocol/entros-verify) — a popup-pattern React component that wraps this SDK and ships verification in five lines of JSX. Use this package directly when you need to own the verification UX (custom capture canvas, branded loading states, mobile-native).

## Install

```bash
npm install @entros/pulse-sdk
```

## Usage

### Wallet-connected (primary)

The user pays the configured SOL fee and signs the verification transaction. Re-verification uses one transaction.

Baseline-key derivation and best-effort SAS issuance can require separate message signatures.

First verification requires a validator-signed receipt bound to the new commitment. Re-verification requires the continuity proof.

```typescript
import { PulseSDK } from '@entros/pulse-sdk';

const pulse = new PulseSDK({ cluster: 'devnet' });
const result = await pulse.verify(touchElement, walletAdapter, connection);

if (result.success) {
  console.log('Verified:', result.txSignature);
}
```

### Walletless (liveness-check tier)

For liveness checking without wallet onboarding. The integrator can fund verification through the relayer API. This path submits protocol transactions through the relayer. It does not issue SAS attestations.

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
2. **Extract**: 170 audio, 81 motion, and 57 touch features
3. **Validate**: Feature summaries sent to Entros validation server for server-side analysis
4. **Hash**: SimHash → 256-bit behavioral fingerprint → Poseidon commitment
5. **Prove**: Re-verification proves Poseidon openings and `min_distance <= distance < threshold`
6. **Submit**: One wallet transaction or one relayer request

## Development

```bash
npm install
npm test          # Vitest suite with a public procedural-attack baseline
npm run build     # ESM + CJS output
npm run typecheck # TypeScript strict mode
```

## Migration history

Originally published as `@iam-protocol/pulse-sdk` (deprecated). Renamed during
the IAM → Entros Protocol rebrand on 2026-04-25; full commit history preserved
on the current repository at `github.com/entros-protocol/pulse-sdk`.

## License

MIT
