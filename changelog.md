# Changelog

All notable changes to the `@entros/pulse-sdk` package will be documented in this file.

> Entries from 3.12.0 onward are written at release time. Everything below it was
> reconstructed after the fact in `4d79649` (2026-07-07) and the dates were
> inferred rather than recorded. Treat them as approximate and check `git log`
> before relying on one. The 3.3.0 date was wrong by two and a half months and
> that error propagated into master-list #186, where it made a pre-feature
> anchor read as a post-feature anchor that had mysteriously lost its baseline.

## [4.2.0] - 2026-08-01

Includes 4.1.2 and 4.1.3, which were tagged in git but never published.

### Added
- **`onProgress` on `submitViaWallet` and `submitResetViaWallet`**, and a `"Finishing up..."` stage emitted the moment the cluster confirms. The caller renders a "submitting" stage before the transaction goes out, and that stops being true on confirmation, while the optional work after it is exactly where a mobile user waits longest. Telling someone their transaction is still going out after it has landed is what turns a short wait into an apparent hang. Hosts that key on the stage strings should add an entry for it. The existing strings are unchanged.

## [4.1.3] - 2026-08-01

### Fixed
- **The baseline key-derivation prompt was also unbounded.** `deriveBaselineKey` awaited `wallet.signMessage` with no ceiling, so a wallet that never raised the prompt held the verification on whichever stage the host last rendered. It is the first prompt a returning user sees on a new device, and the same failure that stranded the attestation in 4.1.2 applies to it. Bounded at `SIGNATURE_TIMEOUT_MS`, matching the transaction prompt, because the user has to see this one and act on it.

Every wallet round trip in the SDK now has a ceiling. The pattern that produced both bugs was the same: a promise returned by a wallet provider is not guaranteed to settle, and mobile is where that stops being theoretical, because the provider lives in another view and may decline to raise a prompt at all.

## [4.1.2] - 2026-08-01

### Fixed
- **A confirmed verification could be held open forever by the attestation prompt.** The SAS attestation runs after the transaction confirms, is best-effort, and needs a wallet signature. That signature was unbounded. On mobile the prompt arrives after the user has dismissed the wallet on "Sent!", so it never surfaces and the promise never settles: a verification that landed on chain at 22:38:36 left the page on "Submitting to Solana..." with no end, and the executor logs show `/attest` was never called. `ATTESTATION_SIGNATURE_TIMEOUT_MS` bounds it at 20 seconds, and the existing handler already treats a failure as "no attestation", which is the correct outcome for a best-effort step.
- **A confirmed verification is now structurally protected from everything after it.** The attestation call is isolated at both submit sites rather than left to the outer catch, so a throw there cannot report a landed transaction as a failure. On the verify path, a local-storage failure no longer discards the result either: the chain already accepted it, and the next attempt recovers the baseline from chain.

The second half mattered as much as the first. `storeVerificationData` runs only once the submission resolves, so the hang also left the device's local baseline behind the chain it had just advanced.

## [4.1.1] - 2026-07-31

### Fixed
- **Accounts written before `IdentityState` last grew could not be decoded, so cross-device recovery was impossible for almost every anchor.** `decodeIdentityState` handed the raw bytes to Anchor's Borsh coder, which throws on anything shorter than the current 593, and `fetchIdentityState` swallowed that to `null`. `recoverBaselineFromChain` reads the identity **before** it fetches the encrypted baseline, so the blob was never reached. Measured on devnet: 105 of 107 accounts on a legacy layout, 12 of them holding a valid `EncryptedBaseline` that no client could use. Recovery worked on 2026-05-20 and broke on 2026-07-14 when the program appended `projection_version` and `last_rebaseline_timestamp`, the same commit that broke the reset in 4.1.0. A short account is now zero-filled to the current layout before decoding, which is what the program's own realloc writes. The floor is 543 bytes: below that `recent_timestamps` had ten slots rather than fifty-two, so every later offset moves and padding would read one field's bytes as another's.
- The bundled IDL now pins error **codes** as well as instruction and account shapes. Anchor numbers error variants by declaration order, so removing one renumbers every variant after it, and hosts route on the number: `entros.io` sends `Custom 6011` to the stale-baseline screen and `Custom 6012` to the reset cooldown. A shift would have sent each to the wrong surface with no build error anywhere.

### Changed
- `VerificationIntervalTooShort` (6025) is retired. `update_anchor` no longer rate limits verifications on chain, so an integrator can ask for a live verification at the point of a gated action. The variant stays in the enum so the numbering after it holds.

## [4.1.0] - 2026-07-31

### Fixed
- **Baseline reset could not land on chain, for any wallet, since 2026-07-27.** The bundled Anchor IDL was three protocol commits behind, so `reset_identity_state` was encoded with one argument where the deployed program requires two. Every reset was broadcast, charged, and reverted with `InstructionDidNotDeserialize`. The IDL is now synced from `protocol-core/target/idl/`, the SDK passes `PROJECTION_VERSION`, and preflight is left on for the reset path so a client-side encoding error is refused for free rather than paid for. Do not regenerate the IDL from chain: the published IDL account is stale too and reproduces the bug.
- **No timeout existed on the wallet signature or on confirmation.** A wallet that never prompted was reported as "Proof generation timed out", because the only clock in the stack was a single host-side race covering the whole verification. `SIGNATURE_TIMEOUT_MS` and `CONFIRMATION_TIMEOUT_MS` bound each step, and each reports its own phase.
- **On-chain baseline recovery reported one reason out of six.** An anchor minted before on-chain baselines existed, a blob that no longer decrypts, and a wallet that cannot sign all rendered as one screen telling the user their device had lost something. Most had never had it: 13 of 107 devnet anchors carry an `EncryptedBaseline` PDA. Each situation now names its own obstacle and its own way out.
- A wallet adapter without `signMessage` skipped baseline recovery with no trace under any configuration, since both SDK log helpers are gated on `debug`.
- Feature extraction failures escaped `complete()` as a bare throw with no phase and nothing to route on. They return a result like every other failure.

### Added
- **Phase taxonomy** (`VerificationPhase`, `phaseChargesAttempt`, `phaseSpend`, `isVerificationPhase`): which of the eight stages failed, and what a host may conclude from it. Routing by prose put an on-chain revert on the screen that says validation rejected the attempt, because the matcher for a validator rejection also matched `custom program error`.
- **`VerificationResult.failedAt`**, and **`VerificationResult.opaque`** as a second axis over it. A replay-floor rejection in `proving`, an attack-signal rejection in `validation` and a program revert in `confirmation` must render identically, so an honest phase is only safe alongside the flag that says how much may be shown.
- **`VerificationResult.baselineRecovery`**, the reason the on-chain baseline could not be restored.
- **`VerificationResult.portableBaseline`**, false when a verification succeeded and advanced the commitment but wrote no portable copy. Left silent, that is exactly how an identity stops being recoverable anywhere else.
- `phaseChargesAttempt` corrects an attempt budget that charged for every failure carrying no client-origin reason, so three declined wallet prompts hard-failed a user whose capture had passed validation each time.
- `test/idl-parity.test.ts`, which diffs the bundled IDL against `protocol-core`'s build output. It fails 5 of 6 against the IDL that shipped the reset bug.
- **`MAX_VERIFICATION_MS`**, the longest a `complete()` can legitimately run. A host with its own backstop timer must set it above this. Derived from `VALIDATE_DEADLINE_MS`, `SIGNATURE_TIMEOUT_MS` and `CONFIRMATION_TIMEOUT_MS` rather than written down, so raising a clock raises it in step. Three hosts each raced the whole of `complete()` against 120 seconds, which is less than the validate deadline on its own.
- **`isUserRejection`** is exported. The SDK runs it to decide whether a failure was `signing` or `submission`, and a host keeping its own copy could recognise a phrasing the SDK's does not, at which point the two disagree about the same error. Two host copies existed.

## [4.0.0] - 2026-07-31

### Changed
- **BREAKING, canonical 16 kHz capture**: Every capture is band-limited and decimated to 16 kHz before feature extraction, not only captures that arrive at another rate. Browsers treat `new AudioContext({ sampleRate })` as a request: Chromium honours 16 kHz, WebKit commonly returns the hardware's 48 kHz. The 72 MFCC features build their mel bank over `0..sampleRate/2`, the 40 LPC and formant features fix `lpcOrder` at 12, and the jitter, shimmer and HNR windows are defined in samples, so the same voice at the two rates produced two incomparable feature vectors. Filtering unconditionally makes this the last filter in every chain, whatever a browser did upstream. **Every existing fingerprint moves.** Baselines from 3.x do not carry forward.
- **`onProgress` signature widened** to `(stage: string, progress?: UploadProgress)`. Not a breaking change: a `(stage) => void` callback still satisfies the wider type, so no existing caller needs editing. The stage strings are unchanged and remain part of the contract, since the embed popup matches on them.
- **Transport rebuilt around stalls, not deadlines**: `/validate-features` now goes out over `XMLHttpRequest` with upload progress, falling back to `fetch` where no XHR exists. A body that is still moving is never aborted, however slowly. A body that has stopped moving is abandoned quickly. The previous fixed 15-second abort covered the upload as well as the server's work, so a healthy 9.4-second mobile upload was cut off and reported as an unreachable service.

### Added
- **Reason taxonomy** (`reasonDisposition`, `isVerificationReason`, `isClientOriginReason`, `RETRYABLE_REASONS`, `COOLDOWN_REASONS`): one exported source for what a failure means and what a host should do about it. Six copies of this list existed across the web and mobile apps and had already drifted, to the point where the same rejection offered a retry on one and dead-ended on the other.
- **Real status handling**: 413, 429 and 5xx are now distinguished instead of collapsing into one string, `retry_after` reaches the host as `VerificationResult.retryAfterSec`, and a timeout is reported as `validation_timeout` rather than as an unreachable service.
- **`markCaptureStart()`**: marks the point where the speak prompt appears, so the dead air recorded during the challenge fetch and countdown is discarded before extraction rather than fingerprinted and transmitted as speech.
- **`MAX_TRANSMITTED_CAPTURE_MS`**: caps transmitted audio at 20 s, which is the validator's own analysis bound. Audio past it is truncated server-side, so an unbounded capture used to lose its phrase check.
- Exports `CANONICAL_SAMPLE_RATE`, `toCanonicalCapture`, `resampleTo`, `normalizeCaptureRMS` and `postJson`.

## [3.16.0] - 2026-07-25

### Added
- **Curve-trace outline transmission**: the wallet-connected `/validate-features` body carries a coarse equal-time outline of the traced challenge curve for the touch content-binding check. The outline is a downsampled geometric summary of 64 points, with no timestamps and no pressure. The raw touch stream stays on the device.
- `CurveTracePoint` and `CurveTraceOutline` types, and an equal-time resampler that produces the outline.

## [3.15.0] - 2026-07-23

### Added
- Client-side test coverage for the acoustic-realism spectral signals (flatness and centroid) introduced in 3.12.0.

## [3.14.0] - 2026-07-22

### Added
- **Server-issued Lissajous curve challenge**: `fetchChallenge` parses the curve parameters the executor issues alongside the phrase, so the traced curve is server-chosen rather than client-generated.

## [3.13.0] - 2026-07-13

### Changed
- **Cross-Wallet baseline isolation**: Partitioned client-side LocalStorage cache using wallet-specific keys to eliminate signature friction and database corruption when switching wallets.
- **SSR compatibility**: Hardened storage layer with `typeof localStorage` guards to prevent ReferenceErrors during Next.js SSR build and prerender stages.

## [3.12.0] - 2026-07-07

### Added
- **Acoustic Realism (Layer B1)**: Implemented Fast Fourier Transform (FFT) analysis on captured audio buffers to calculate average **Spectral Flatness (Wiener entropy)** and **Spectral Centroid** parameters.
- Exposed `flatness` and `centroid` metrics inside the `client_signals.capture` telemetry payload to protect against virtual loopback audio injection.

## [3.11.0] - 2026-07-06

### Added
- **Virtual Device Detection**: Added checks against `navigator.mediaDevices.enumerateDevices` labels to flag virtual audio loopback drivers (e.g., BlackHole, VB-Audio, Soundflower) in the `client_signals` envelope.

## [3.10.0] - 2026-06-21

### Added
- Extended TypeScript session response types to expose server-side `composite_risk_score`.

### Changed
- Resolved dev-tooling Dependabot advisories via updates to `esbuild`, `ws`, and `underscore`.

## [3.9.0] - 2026-05-18

### Added
- **Automation Detection (Layer A1)**: Injected client environment tells (e.g., Selenium global variables, Puppeteer evaluation keys, and `navigator.webdriver` properties) into the feature validation POST request body under the `client_signals` field.

## [3.8.0] - 2026-05-02

### Changed
- Refactored TBH commitments to use the validator-derived commitment and salt instead of computing it locally, improving cryptographic consensus alignment.

## [3.7.0] - 2026-04-20

### Added
- Pre-checking of Hamming distance bounds before initiating heavy zero-knowledge proving.
- Baseline sync path directly from the on-chain encrypted state.
- Enhanced stringification of raw wallet rejections and relayer errors to return clean messages to the UI.

## [3.5.0] - 2026-04-05

### Added
- Minor version bump adding the `onReady` capture option to allow frontends to align microphone capture start with visual speaker cues.

### Fixed
- Cleared the cold-start timeout race condition when the first audio frame arrives, resolving dead-air delays.

## [3.4.1] - 2026-03-22

### Changed
- Gated `startAudio()` audio processor initialization on the first delivered PCM frame.

## [3.3.1] - 2026-03-10

### Fixed
- Aligned `fetchIdentityState` account decoding with Anchor v0.30+ IDL specifications (mapping to PascalCase account names and snake_case fields).

## [3.3.0] - 2026-05-14

### Added
- **Encrypted Baseline Recovery**: Implemented wallet-keyed AES-GCM-256 baseline encryption bound to AAD commitments.
- Automated bundling of `set_encrypted_baseline` instruction during identity registration transactions.

## [3.0.0] - 2026-02-10

### Changed
- **Separation Hardening**: Drop `MFCC[0]` (cepstral DC term carrying mic-energy bias) to prevent hardware/mic fingerprint leakage.
- Normalized audio output to target RMS 0.05 at the source buffer, stabilizing inputs to the Whisper validation engine.
- Replaced mouse-dynamics zero-padding vectors with 308-dimensional FFT band energy, tremor peaks, cross-axis covariance, reversal stats, angular speed, and autocorrelation markers.

## [2.0.0] - 2026-01-15

### Added
- **V2 Kinematics Extraction**: Expanded motion and touch vector spaces with covariance, curvature, gap distributions, and path efficiency metrics (314-dimensional layout).
- Radix-2 Cooley-Tukey FFT library helper with `bandEnergy`/`peakInBand` primitives.

## [1.5.0] - 2025-11-20

### Added
- Bundled Solana Program IDLs inside build outputs to skip RPC schema fetches on initialization.

## [1.4.0] - 2025-10-08

### Added
- **Validator Receipts**: Wired Ed25519 signature validation receipt checks inside transaction builders. Bundled verification proofs immediately before `mint_anchor` calls.

## [1.3.0] - 2025-08-30

### Changed
- Strengthened IndexedDB storage guards and added fail-safe sensor cleanup routines on all abort/exit paths.

## [1.1.0] - 2025-07-14

### Changed
- **Rebranding**: Migrated npm package name from `@iam-protocol/pulse-sdk` to `@entros/pulse-sdk`.
- Rotated Agent Anchor metadata keys to `entros:human-operator`.

## [0.9.0] - 2025-06-25

### Added
- Added `resetBaseline` workflow to heal corrupted local IndexedDB keys on demand.

## [0.8.0] - 2025-05-18

### Changed
- Forwarded validation result reasons to `updateAnchor` instructions.

## [0.7.12] - 2025-04-30

### Added
- Added F0 pitch contour and acceleration magnitude time-series extractors.

## [0.7.9] - 2025-03-12

### Added
- Initial server-side verification release with SimHash fingerprint checks.
