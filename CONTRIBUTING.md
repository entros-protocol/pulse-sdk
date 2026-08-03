# Contributing

Thank you for looking at Entros.

**Do not report security problems here.** Read [SECURITY.md](SECURITY.md) first.

## Before you start

Open an issue to describe what you want to change and wait for a reply. This applies to
anything beyond a typo or a broken link. Entros is a protocol with on-chain state, so a
change that looks small can alter the behaviour of a deployed program.

## Pull requests

Open your pull request against `develop`. `main` is the release branch.

Keep one logical change per pull request. A smaller pull request gets reviewed sooner
and is easier to reason about against on-chain state.

Say what changed and how to test it. Commit style and branch naming are yours to
choose.

## Before you open a pull request

Run the checks for the language you touched.

**Rust**

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

**TypeScript**

```bash
npx eslint .
npx tsc --noEmit
npm test
```

**Anchor programs**

```bash
anchor build
anchor test
```

A pull request that fails any of these will not be merged.

## Standards we hold

- No `unwrap()` in production Rust paths. Handle the error.
- No `any` and no `@ts-ignore` in TypeScript. Strict mode stays on.
- Every Anchor instruction validates every account it receives.
- Program state lives in Program Derived Addresses, never raw keypair accounts.
- Never commit a keypair, a `.env` file, a build artifact or `node_modules/`.

## Privacy rules that bind every change

Entros proves personhood without collecting biometrics. Two rules protect that and they
are not negotiable.

**Raw motion and touch recordings never leave the device.** Neither does the derived
behavioural fingerprint. If a change transmits, logs or stores a raw sensor stream, it
will be rejected.

**Captured audio is never persisted.** It is sent for transcription and discarded.

If you are unsure whether a change crosses either line, ask in the issue before writing
the code.

## Licence

By contributing you agree that your contribution is licensed under the same terms as
this repository.
