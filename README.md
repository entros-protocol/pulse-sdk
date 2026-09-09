# @entros/pulse-sdk

[![npm version](https://img.shields.io/npm/v/@entros/pulse-sdk.svg)](https://www.npmjs.com/package/@entros/pulse-sdk)
[![npm downloads](https://img.shields.io/npm/dm/@entros/pulse-sdk.svg)](https://www.npmjs.com/package/@entros/pulse-sdk)

Client SDK for Entros capture, feature extraction, fingerprinting, proof generation, and Solana submission.

The SDK derives 308 features from phrase audio, motion, and touch. It generates a 256-bit SimHash fingerprint and Poseidon commitment. Re-verification also generates a Groth16 proof.

Raw motion and full-resolution touch streams stay on device. Projection 2 discards its raw compatibility touch after completion. Phrase audio leaves transiently for private validation.

Validation also receives the feature summary, F0 contour, acceleration magnitude, wallet, client signals, curve outline, timing, and commitment fields.

Projection 2 sends projection 1 compatibility features only for mint, rebaseline, and reset. It never sends raw compatibility touch.

Each projection 2 validation request includes authorization bound to the challenge nonce, request digest, projection, and connected wallet.

The SDK stores the fingerprint, salt, commitment, and timestamp as a baseline. Wallet flows can store an encrypted baseline blob on-chain.

A host-approved fallback can store the local baseline without encryption. Integrators must treat that option as a lower-assurance recovery mode.

> **Looking for a drop-in?** Most integrators want [`@entros/verify`](https://github.com/entros-protocol/entros-verify) — a popup-pattern React component that wraps this SDK and ships verification in five lines of JSX. Use this package directly when you need to own the verification UX (custom capture canvas, branded loading states, mobile-native).

## Install

```bash
npm install @entros/pulse-sdk
```

### Bundler requirement

Re-verification encodes a 64-bit instruction argument through Anchor's Borsh layer, which
reads a global `Buffer`. Next.js and webpack supply that binding to browser code. Supply it
yourself when you bundle with esbuild, Vite, or Rollup:

```ts
import { Buffer } from "buffer";
globalThis.Buffer ??= Buffer;
```

Without it, re-verification throws `ReferenceError: Buffer is not defined` after capture and
before the wallet prompt.

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

### Read evidence for an application policy

`readIntegratorEvidence` reads confirmed devnet transaction and account observations through your configured RPC connection.
It checks wallet binding, program identities, supported account layouts, and the transaction's relationship to current identity state.
It also reads the SAS account and reports its status.

Install the chain-reading peers alongside the SDK:

```bash
npm install @coral-xyz/anchor@^0.32.1 @solana/web3.js@^1.98.0
```

These peers are optional package dependencies because capture-only consumers do not need chain access.
The evidence reader requires both peers. It does not require a wallet adapter.

```typescript
import { Connection } from '@solana/web3.js';
import { readIntegratorEvidence } from '@entros/pulse-sdk';

async function readEvidence(
  rpcUrl: string,
  walletPubkey: string,
  transactionSignature: string,
) {
  const connection = new Connection(rpcUrl, 'confirmed');
  return readIntegratorEvidence({
    walletPubkey,
    transactionSignature,
    connection,
    nowSeconds: () => Math.floor(Date.now() / 1000),
  });
}
```

Supply a clock callback for live reads. A numeric timestamp selects a fixed snapshot comparison and never waits.
For an otherwise valid attestation slightly ahead of the live clock, the reader can wait up to three seconds before reading evidence again.
It performs at most one clock catch-up. The existing propagation retry remains bounded separately, for at most three complete reads.
It never accepts a future timestamp before the supplied clock reaches it. Expiry and all other checks still apply.
Late timer wakeups can exceed the waiting budget and still cause rejection.

The result uses `status: "available"`, `"invalid"`, or `"unavailable"`.
An available result contains identity and transaction evidence, plus a separate attestation status.
A missing or unavailable SAS account does not prevent the reader from returning available identity evidence.
Your application decides whether its policy requires an attestation.

Pair the observation with `evaluatePolicy` from `@entros/verify/policy` in Verify `0.2.0` or later.
The reader does not choose your score floor, freshness requirement, or action authorization.
For protected actions, your service must authenticate the wallet's action request and read current evidence before settlement.
Use your service's policy and RPC configuration. Do not accept browser-supplied observations as settlement authority.

The evidence reports `browser_unattested` assurance and `unmeasured` uniqueness.
These fields do not establish sensor provenance or population uniqueness.
The reader supports the pinned Entros devnet programs. It does not submit a transaction.

### Projection 2 compatibility

The on-chain policy controls projection 2 activation. `CLIENT_PROJECTION_VERSION` reports client support and does not activate policy.

Projection 2 validation requires the server challenge that selected the phrase and touch curve. Present both challenge artifacts during the same capture.

Pass the deadline returned by `fetchChallenge`. Do not recalculate it from `expiresIn` after the request completes.

```typescript
import { fetchChallenge, PulseSDK } from '@entros/pulse-sdk';

const executorUrl = 'https://api.entros.io/relay';
const pulse = new PulseSDK({ cluster: 'devnet', relayerUrl: executorUrl });
const challenge = await fetchChallenge(
  executorUrl,
  walletAdapter.publicKey.toBase58(),
);

const result = await pulse.verify(touchElement, walletAdapter, connection, {
  validationChallengeNonce: challenge.nonce,
  validationChallengeExpiresAtMs: challenge.expiresAtMs,
});
```

For staged capture, call `session.bindValidationChallenge(challenge.nonce, challenge.expiresAtMs)` before `complete()`.

The touch element defines the projection 2 coordinate surface by default. Use `startTouch({ eventTarget, coordinateSurface })` when those elements differ.

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

### Isolated devnet deployments

Supply a trusted `requestBoundManifest` through application configuration to select a separate Anchor and verifier pair.
Both program addresses must differ from the official pair. The SDK keeps the official registry and projection policy unchanged.

Pass the same manifest as the final argument to `fetchIdentityState`, `deriveEncryptedBaselinePda`, `fetchEncryptedBaseline`, and `recoverBaselineFromChain`.
Pass it as the third argument to `storeVerificationData` and `loadVerificationData`.
The SDK separates local baselines by wallet, program pair, deployment domain, and genesis hash.
Existing callers retain their current addresses and storage keys.

Isolated submissions skip shared SAS issuance. The official evidence reader returns `unsupported_deployment` when its `deployment` argument selects an isolated pair.
Do not derive trusted configuration from popup responses. Use independent acceptance checks for an isolated deployment.

### Release metadata

Run `npm run release:prepare -- NEXT_VERSION PREVIOUS_TAG` to update package versions and generate release notes from Git commit subjects.
The generator preserves historical entries. Run it after the release changes have committed, then review the generated section before publication.
Use `--through=SOURCE_COMMIT` to pin the source revision across later metadata commits.
Add `--check` to verify generated metadata without changing files.
