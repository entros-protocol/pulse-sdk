# Security Policy

## Reporting a vulnerability

**DO NOT OPEN A GITHUB ISSUE to report a security problem.**

Report privately through one of these channels:

1. **GitHub private vulnerability reporting.** Open the repository's **Security**
   tab and select **Report a vulnerability**. This is the preferred route.
2. **Email `security@entros.io`**.

If you use email, do not include exploit details in the message. Send a short notice
and we will open a private advisory for the technical detail.

Please enable two-factor authentication on your GitHub account before you report.

## What to include

Send a clear title, a description of the vulnerability, the affected repository and
commit, and a proof of concept.

**Reports without a proof of concept will be closed without further consideration.**
This is not a judgement on the finder. Entros is maintained by a small team and cannot
triage speculative reports.

## What to expect

We aim to acknowledge a report within five business days. We will tell you whether the
issue is in scope, and we will credit you in the advisory when it is resolved, unless
you ask us not to.

## Current status, stated plainly

Read this before you spend time on a report.

- **Entros runs on Solana devnet. There is no mainnet deployment yet.** No user funds
  are at risk from any finding today.
- **No external security audit has been completed.** One is planned before mainnet.

We would still like to hear about anything you find.

## Scope

**In scope**

- The on-chain Anchor programs in `protocol-core`
- The zero-knowledge circuits in `circuits`
- The mobile proving library in `entros-mopro`
- The client SDK in `pulse-sdk`
- The off-chain relay in `executor-node`
- The verification web application in `entros.io`
- The mobile application in `entros-mobile`
- The integration surface in `entros-verify`
- The Realms voter-weight program in `entros-governance-plugin`

**Out of scope**

- Server-side validation logic, which is not published and cannot be reviewed from
  source
- Findings that require a leaked private key or a compromised device
- Findings that require privileged protocol authority
- Denial of service through ordinary network volume
- Reports produced only by an automated scanner, with no analysis
- Best-practice observations with no demonstrated impact
- Third-party dependencies, which should be reported to their maintainers
- `token-contracts`, which contains documentation only and ships no code

## Disclosure

We ask that you give us the chance to fix an issue before publishing it. We will agree
a disclosure date with you once a fix is ready. We will not take legal action against a
researcher who reports in good faith and follows this policy.
