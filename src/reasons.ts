/**
 * The verification-failure reason taxonomy, and what a host should do about
 * each one.
 *
 * This module exists because the same list had been copied into six places:
 * the web app's retryable set, its hint dictionary, its embed popup, the
 * mobile app's soft-reject list, the mobile hint table, and the SDK's own doc
 * comment, with nothing keeping them in step. They had already drifted: the
 * web set carried six entries against mobile's four, so mobile hard-failed
 * `captcha_required` and dead-ended a user the web app would have let retry.
 * The web app had also lost the ability to recognise a rate limit from its
 * code, and recovered the fact by substring-matching the words "too many"
 * against the server's English error prose.
 *
 * A seventh copy would not have fixed that. One exported source does.
 *
 * Reasons arrive from three places and this module is deliberately the only
 * thing that knows the difference:
 *
 *   - The validator's safe-reveal set, forwarded through the executor. These
 *     name a capture-quality problem the user can act on. Every other
 *     validator rejection stays deliberately opaque, so `reason` is absent on
 *     attack-signal rejections and a host must never read its absence as
 *     success.
 *   - The executor's own transport-level rejections: rate limits, cooldowns,
 *     oversized payloads.
 *   - The SDK itself, for failures that never reached a server.
 */

/** Every reason label the SDK will surface. */
export type VerificationReason =
  // Validator safe-reveal. The user can plausibly do better on a retry.
  | "variance_floor"
  | "entropy_bounds"
  | "temporal_coupling_low"
  | "phrase_content_mismatch"
  | "captcha_required"
  // Executor transport-level. Trying again immediately makes things worse.
  | "rate_limited"
  | "ip_rate_limited"
  | "cross_wallet_cooldown"
  | "payload_too_large"
  // SDK-originated. No server ever rendered a verdict on this attempt.
  | "validation_unavailable"
  | "validation_timeout";

/**
 * What a host should do with a reason.
 *
 * - `retry`: offer the user another attempt now.
 * - `wait`: another attempt will be refused until a cooldown elapses. The
 *   result carries `retryAfterSec` when the server supplied one.
 * - `fatal`: retrying changes nothing. Stop and explain.
 *
 * Deliberately three states rather than a retryable boolean. A rate limit is
 * not "not retryable", it is "not retryable yet", and flattening the two
 * loses the countdown that makes the difference legible to the user.
 */
export type ReasonDisposition = "retry" | "wait" | "fatal";

const DISPOSITIONS = {
  variance_floor: "retry",
  entropy_bounds: "retry",
  temporal_coupling_low: "retry",
  phrase_content_mismatch: "retry",
  captcha_required: "retry",
  rate_limited: "wait",
  ip_rate_limited: "wait",
  cross_wallet_cooldown: "wait",
  // An identical body produces an identical rejection, so a retry is pure
  // cost. The capture has to change, which means starting over.
  payload_too_large: "fatal",
  validation_unavailable: "retry",
  validation_timeout: "retry",
} as const satisfies Readonly<Record<VerificationReason, ReasonDisposition>>;

/**
 * The reasons whose disposition is `retry`, derived from the table above
 * rather than listed again.
 *
 * Hosts keying a hint dictionary on this get a real exhaustiveness check: add
 * a retryable reason without writing its copy and the build fails. Spelling
 * the members out by hand, as `Extract<VerificationReason, "a" | "b">` does,
 * looks like the same guarantee and is not one. `Extract` filters the union
 * by a literal list, so a reason added to `VerificationReason` is silently
 * absent from the result and no error is raised. Both apps carried that
 * mistake, each claiming a compile error it did not have.
 */
export type RetryableReason = {
  [K in keyof typeof DISPOSITIONS]: (typeof DISPOSITIONS)[K] extends "retry" ? K : never;
}[keyof typeof DISPOSITIONS];

/**
 * Own keys of {@link DISPOSITIONS}, for exact membership tests.
 *
 * Deliberately not an `in` check against the object. `in` walks the prototype
 * chain, so `"toString" in DISPOSITIONS` is true and the lookup that followed
 * returned `Object.prototype.toString`, a function, from a function declared
 * to return a `ReasonDisposition`. `reason` arrives from a server body, so
 * that is reachable input, not a curiosity. `Set.has` matches own keys only.
 */
const KNOWN_REASONS: ReadonlySet<string> = new Set(Object.keys(DISPOSITIONS));

/**
 * Narrow an untrusted string to a known reason.
 *
 * The SDK forwards `reason` from the server body verbatim, so anything could
 * be in there: an older executor, a newer one, or a proxy's error page.
 * Hosts should gate on this before switching on the value.
 */
export function isVerificationReason(value: unknown): value is VerificationReason {
  return typeof value === "string" && KNOWN_REASONS.has(value);
}

/**
 * Classify a reason. Unrecognised and absent reasons are `fatal`, which
 * matches how hosts already behaved: anything not on the retryable list fell
 * through to the hard-failure surface. Failing closed also means a future
 * server-side reason cannot silently grant retries to an old client.
 */
export function reasonDisposition(reason: string | undefined): ReasonDisposition {
  return isVerificationReason(reason) ? DISPOSITIONS[reason] : "fatal";
}

/**
 * Reasons a host should offer an immediate retry for.
 *
 * Derived from {@link DISPOSITIONS} rather than written out again, so a
 * reason cannot be added to the type and forgotten here.
 */
export const RETRYABLE_REASONS: ReadonlySet<VerificationReason> = new Set(
  (Object.keys(DISPOSITIONS) as VerificationReason[]).filter(
    (reason) => DISPOSITIONS[reason] === "retry",
  ),
);

/**
 * Reasons that carry a cooldown. A host showing a countdown should read
 * `VerificationResult.retryAfterSec` alongside these.
 */
export const COOLDOWN_REASONS: ReadonlySet<VerificationReason> = new Set(
  (Object.keys(DISPOSITIONS) as VerificationReason[]).filter(
    (reason) => DISPOSITIONS[reason] === "wait",
  ),
);

/**
 * Reasons the SDK raised itself, without a server ever seeing the capture.
 *
 * Hosts that meter attempts should not charge the user for these. Nothing was
 * evaluated, so there is nothing to have failed. Metering them means a few
 * dropped connections exhaust a budget and hard-fail someone whose capture was
 * never judged. The budget is a UX affordance in any case, since the real
 * per-wallet cap lives on the server and a hostile client would not honour a
 * counter it controls.
 */
export const CLIENT_ORIGIN_REASONS: ReadonlySet<VerificationReason> = new Set([
  "validation_unavailable",
  "validation_timeout",
]);

/** True when the failure happened before any server rendered a verdict. */
export function isClientOriginReason(reason: string | undefined): boolean {
  return isVerificationReason(reason) && CLIENT_ORIGIN_REASONS.has(reason);
}
