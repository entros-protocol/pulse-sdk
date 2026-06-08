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
