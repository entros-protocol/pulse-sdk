/**
 * Which stage of a verification failed, and what a host may conclude from it.
 *
 * This module exists because two production failures on 2026-07-31 were
 * reported as the wrong thing, and in both cases the phase was the missing
 * fact:
 *
 *   - An on-chain revert rendered as "Validation rejected this attempt". The
 *     host inferred the stage from English prose, and its opaque-rejection
 *     matcher covered `custom program error`, `-32002` and
 *     `transaction simulation failed`, all of which can only occur after the
 *     validator already returned 200.
 *   - A wallet that never prompted rendered as "Proof generation timed out".
 *     One 120-second clock covered the whole of `complete()` and its message
 *     named proving because proving was the slowest step anyone expected.
 *
 * Prose is not a discriminator. The stage is, so the SDK states it.
 *
 * ## Two axes, not one
 *
 * A phase says where the flow stopped. It does not say how much a user may be
 * told, and conflating the two turns an honest label into a side channel: a
 * replay-floor rejection lives in `proving`, a validator attack-signal
 * rejection lives in `validation`, and an on-chain revert lives in
 * `confirmation`, yet all three must render identically or the difference
 * between them becomes calibration information for an attacker.
 *
 * That is what `VerificationResult.opaque` is for, and a host must consult it
 * before it renders anything derived from `failedAt`. The two compose:
 * `failedAt` decides which matchers may run, `opaque` decides what is shown
 * when none of them produced a user-actionable surface.
 */

/**
 * The stage a verification reached before it failed.
 *
 * Ordered here as the flow runs them, which is documentation only: nothing
 * derives an ordinal, because inserting a stage would silently shift every
 * comparison written against one.
 *
 * - `capture`: sensors. Permissions, silent microphone, too little data.
 * - `extraction`: turning samples into the 308-feature vector.
 * - `validation`: the server's verdict on the capture.
 * - `baseline`: finding a prior fingerprint to prove drift against, locally
 *   or from the on-chain `EncryptedBaseline` PDA.
 * - `proving`: Hamming bounds and Groth16 proof generation.
 * - `signing`: the wallet prompt, up to the point the user approves.
 * - `submission`: building and broadcasting the transaction.
 * - `confirmation`: waiting for the cluster, and the on-chain result.
 *
 * `signing` and `submission` stay separate even though wallet adapters merge
 * them into one `sendTransaction` call. That seam is exactly where both
 * production failures lived, and collapsing it would put a rejected prompt and
 * a failed broadcast in the same bucket, which is the confusion this module
 * was written to end.
 */
export type VerificationPhase =
  | "capture"
  | "extraction"
  | "validation"
  | "baseline"
  | "proving"
  | "signing"
  | "submission"
  | "confirmation";

/**
 * What the user may have paid by the time a phase failed.
 *
 * - `none`: no transaction left the device. Nothing was charged.
 * - `possible`: a transaction was broadcast and its outcome is unknown. A fee
 *   may or may not have been taken, and claiming either way would be a guess.
 * - `certain`: the transaction landed and the program rejected it. The fee was
 *   taken and the state did not change.
 *
 * The `certain` case is not hypothetical. Baseline reset broadcast with
 * `skipPreflight` against a stale IDL for two months, so every attempt was
 * charged for a transaction that could not deserialize, and the interface said
 * only that verification had failed.
 */
export type PhaseSpend = "none" | "possible" | "certain";

interface PhasePolicy {
  spend: PhaseSpend;
  /**
   * Whether reaching this phase means something evaluated whether the capture
   * came from a person.
   */
  judged: boolean;
}

const PHASE_POLICY = {
  capture: { spend: "none", judged: false },
  extraction: { spend: "none", judged: false },
  validation: { spend: "none", judged: true },
  baseline: { spend: "none", judged: false },
  // The replay floor and the drift ceiling are both enforced here, and both
  // are bounds on the capture rather than a verdict on the person. The
  // validator has already accepted or rejected by this point.
  proving: { spend: "none", judged: false },
  // A prompt the user never approved sends nothing.
  signing: { spend: "none", judged: false },
  // Covers building the transaction as well as broadcasting it, and reports
  // `possible` for both. A build failure spent nothing, so the answer is
  // over-cautious there. It errs in the safe direction: this phase is also
  // where every ambiguous outcome lands, including a signature prompt answered
  // after the SDK stopped waiting, and telling that user nothing was charged
  // would be a claim with no basis.
  submission: { spend: "possible", judged: false },
  confirmation: { spend: "certain", judged: false },
} as const satisfies Readonly<Record<VerificationPhase, PhasePolicy>>;

// The `satisfies` clause above is the exhaustiveness check. Add a phase to
// `VerificationPhase` without a row here and the module does not compile,
// which is the only reason the table is spelled out rather than derived.
//
// No set of phase names is exported alongside it. `Extract<VerificationPhase,
// "a" | "b">` looks like the same guarantee and is not one, since `Extract`
// filters a union against a literal list and silently drops a phase added
// later. The two predicates below read the table directly instead.

/**
 * Own keys of {@link PHASE_POLICY}, for exact membership tests.
 *
 * Deliberately not an `in` check against the object. `in` walks the prototype
 * chain, so `"toString" in PHASE_POLICY` is true and the lookup that followed
 * would return `Object.prototype.toString` from a function declared to return
 * a policy. `Set.has` matches own keys only.
 */
const KNOWN_PHASES: ReadonlySet<string> = new Set(Object.keys(PHASE_POLICY));

/**
 * Narrow an untrusted string to a known phase.
 *
 * Hosts construct phases too. Microphone permission, motion permission and
 * challenge-fetch failures are raised before the SDK sees anything, so the
 * value can arrive from a host that is a version ahead or behind.
 */
export function isVerificationPhase(value: unknown): value is VerificationPhase {
  return typeof value === "string" && KNOWN_PHASES.has(value);
}

/**
 * Whether a failure in this phase should count against a host's attempt
 * budget.
 *
 * Only `validation` judged the capture, so only `validation` may charge for
 * it. Every other phase failed for a reason that says nothing about whether a
 * person was present: a denied microphone, a missing baseline, a rejected
 * wallet prompt, an expired blockhash.
 *
 * This corrects a real defect rather than a theoretical one. The budget was
 * charged on every non-transport failure, so three rejected wallet prompts
 * hard-failed a user whose capture had passed validation all three times.
 *
 * Compose it with `isClientOriginReason`: a `validation` phase that failed
 * because the request never arrived did not judge anything either.
 *
 * Unknown phases charge. The budget is a courtesy in any case, since the real
 * per-wallet cap is enforced server-side and a hostile client would not honour
 * a counter it controls, but failing open here would make a stale host
 * grant unlimited attempts.
 */
export function phaseChargesAttempt(phase: string | undefined): boolean {
  return isVerificationPhase(phase) ? PHASE_POLICY[phase].judged : true;
}

/**
 * What the user may have paid, given the phase that failed.
 *
 * Unknown phases report `possible`, which is the honest answer when the stage
 * is unknown: telling someone nothing was charged is a claim, and this
 * function has no basis for it.
 */
export function phaseSpend(phase: string | undefined): PhaseSpend {
  return isVerificationPhase(phase) ? PHASE_POLICY[phase].spend : "possible";
}
