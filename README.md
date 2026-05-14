# @entros/pulse-sdk

[![npm version](https://img.shields.io/npm/v/@entros/pulse-sdk.svg)](https://www.npmjs.com/package/@entros/pulse-sdk)
[![npm downloads](https://img.shields.io/npm/dm/@entros/pulse-sdk.svg)](https://www.npmjs.com/package/@entros/pulse-sdk)

Client-side SDK for the Entros Protocol. Captures behavioral biometrics (voice, motion, touch), extracts a 308-dimensional statistical feature vector (v3 expansion: MFCCs with pre-emphasis (C1-C12, MFCC[0] dropped), LPC coefficients, formant trajectories, voice quality, pitch contour shape, IMU FFT-band tremor, cross-axis covariance, mouse-derived FFT / autocorrelation analogues for desktop, touch curvature, gap distribution, path efficiency — see `docs/master/BLUEPRINT-feature-pipeline-v2.md`), generates a Groth16 zero-knowledge proof, and submits for on-chain verification on Solana. Raw biometric data never persists — only derived features and the proof are retained.

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

For liveness checking without wallet onboarding. The integrator optionally funds verifications via the relayer API. Submits proofs to chain through the relayer; **does not issue SAS attestations** — for SAS attestations bound to a verified wallet, use the wallet-connected path above.

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
2. **Extract**: 308 features — speaker block (170): legacy F0 / jitter / shimmer / HNR / formant ratios / LTAS / amplitude (44) plus v3 additions: MFCCs + delta-MFCCs (72 — 12 used coefficients × 4 stats + 12 × 2 deltas, MFCC[0] dropped, pre-emphasis applied), LPC coefficient stats (24), formant absolute trajectories + bandwidths (16), voice quality CPP/tilt/H1-H2/sub-bands (9), pitch contour DCT (5). Motion block (81): legacy jerk + jounce per IMU axis (54) plus v2 additions: cross-axis covariance (6), FFT band energy on accel axes (12), physiological tremor peak (2), direction-reversal stats (3), motion-magnitude autocorrelation (4); desktop captures use mouse-derived analogues for these v2 additions. Touch block (57): legacy velocity + pressure dynamics (36) plus v2 additions: pressure derivative (4), contact aspect ratio + area derivative (4), trajectory curvature (3), velocity autocorrelation (3), inter-touch gap distribution (4), path efficiency + per-stroke length (3).
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
