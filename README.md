# @iam-protocol/pulse-sdk

Client-side SDK for the IAM Protocol. Captures behavioral biometrics (voice, motion, touch), generates a Temporal-Biometric Hash, produces a Groth16 zero-knowledge proof, and submits it for on-chain verification on Solana.

## Install

```bash
npm install @iam-protocol/pulse-sdk
```

## Usage

```typescript
import { PulseSDK } from '@iam-protocol/pulse-sdk';

const pulse = new PulseSDK({
  cluster: 'devnet',
  relayerUrl: 'https://api.iam-human.io/relay', // walletless mode
  wasmUrl: '/circuits/iam_hamming.wasm',
  zkeyUrl: '/circuits/iam_hamming_final.zkey',
});

// Verify (captures sensors, generates proof, submits)
const result = await pulse.verify(touchElement);

if (result.success) {
  console.log('Verified:', result.txSignature);
}
```

### Wallet-connected mode

```typescript
import { PulseSDK } from '@iam-protocol/pulse-sdk';

const pulse = new PulseSDK({ cluster: 'devnet' });
const result = await pulse.verify(touchElement, walletAdapter, connection);
```

## Pipeline

1. **Capture**: Audio (16kHz), IMU (accelerometer + gyroscope), touch (pressure + area) — event-driven, caller controls duration
2. **Extract**: MFCC (voice), jerk/jounce (motion), velocity/pressure (touch)
3. **Hash**: SimHash → 256-bit Temporal Fingerprint → Poseidon commitment
4. **Prove**: Groth16 proof that new fingerprint is within Hamming distance of previous
5. **Submit**: On-chain verification via wallet or relayer

## Development

```bash
npm install
npm test          # 31 vitest tests
npm run build     # ESM + CJS output
npm run typecheck # TypeScript strict mode
```

## License

MIT
