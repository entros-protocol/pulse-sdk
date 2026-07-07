# Changelog

All notable changes to the `@entros/pulse-sdk` package will be documented in this file.

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

## [3.3.0] - 2026-03-02

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
