/**
 * Coerce an unknown thrown value into a human- and categorizer-readable string
 * WITHOUT collapsing structured errors to the useless literal "[object Object]".
 *
 * Why this exists: on an on-chain transaction revert the wallet adapter /
 * web3.js can throw a BARE object such as
 *   { InstructionError: [4, { Custom: 6011 }] }
 * whose `.message` is `undefined`. The old `err.message ?? String(err)` idiom
 * then produced "[object Object]", which (a) is meaningless to the user and
 * (b) destroys the `"Custom":6011` substring that entros.io's failure
 * categorizer matches to route the error to its "baseline out of sync"
 * surface. JSON-stringifying objects preserves that substring so the existing
 * mapping fires (and a real human-readable message when one is present).
 */
export function errToString(err: unknown): string {
  if (typeof err === "string") return err;
  if (err instanceof Error) {
    if (typeof err.message === "string" && err.message.length > 0) {
      return err.message;
    }
    // Error with an empty / non-string `.message`: its own props are
    // non-enumerable so JSON would be "{}"; the class name is more useful.
    return err.name || "Error";
  }
  return jsonOrString(err);
}

/**
 * Marker names for the two throws the SDK raises itself.
 *
 * Carried on `Error.name` rather than as subclasses. A bundled SDK and a host
 * can end up with separate copies of a class, and `instanceof` across that
 * boundary is false while the name still matches.
 */
const TIMEOUT_ERROR_NAME = "EntrosTimeoutError";
const CHAIN_REVERT_ERROR_NAME = "EntrosChainRevertError";

/**
 * Build the error {@link isChainRevertError} recognises.
 *
 * Reserved for a transaction that landed and whose execution the cluster
 * reported as failed. It is the one outcome where a fee was definitely taken
 * and the state definitely did not change, and it is the only thing that may
 * be attributed to the `confirmation` phase.
 */
export function chainRevertError(message: string): Error {
  const err = new Error(message);
  err.name = CHAIN_REVERT_ERROR_NAME;
  return err;
}

/** True when the cluster reported a definite on-chain execution failure. */
export function isChainRevertError(err: unknown): boolean {
  return err instanceof Error && err.name === CHAIN_REVERT_ERROR_NAME;
}

/**
 * True when the user dismissed or declined the wallet prompt.
 *
 * The only unambiguous signal that nothing was broadcast. Every other throw
 * out of `sendTransaction` leaves the transaction's fate unknown, so this
 * predicate is what separates a `signing` failure from a `submission` one.
 *
 * Adapters word it differently and none of them expose a code, so the match is
 * on prose. The strings below are the ones Phantom, Solflare and Backpack
 * emit. A miss is safe: the failure is attributed to `submission`, which
 * reports the outcome as unknown rather than claiming nothing was sent.
 */
export function isUserRejection(err: unknown): boolean {
  const e = errToString(err).toLowerCase();
  return (
    e.includes("user rejected") ||
    e.includes("rejected the request") ||
    e.includes("user denied") ||
    e.includes("rejected by user")
  );
}

/**
 * Reject with a timeout when `work` has not settled in `ms`.
 *
 * `work` is not cancelled, because nothing in a wallet adapter or in web3.js
 * can be. It keeps running, and a wallet prompt approved after the clock
 * expires still broadcasts. Callers must therefore treat a timeout as an
 * unknown outcome, never as a failure to send.
 *
 * The timer is always cleared. An uncleared one holds the Node event loop open
 * for the full duration after a fast success, which is a hung test suite
 * rather than a hung verification, but is a defect either way.
 */
export async function withTimeout<T>(
  work: Promise<T>,
  ms: number,
  message: string,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      work,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          const err = new Error(message);
          err.name = TIMEOUT_ERROR_NAME;
          reject(err);
        }, ms);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

function jsonOrString(value: unknown): string {
  try {
    const json = JSON.stringify(value);
    // For a non-Error object/array/primitive, ANY JSON string (incl. "{}")
    // beats "[object Object]". Only `undefined` / circular fall through.
    if (typeof json === "string") return json;
  } catch {
    // circular / non-serializable — fall through to String()
  }
  return String(value);
}
