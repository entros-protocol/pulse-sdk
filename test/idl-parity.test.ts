import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

/**
 * The bundled IDL must match the one `anchor build` produces in protocol-core.
 *
 * This test exists because it did not, and the cost was total. On 2026-07-14 the
 * on-chain program gained a `projection_version: u16` argument on
 * `reset_identity_state`; the deploy landed 2026-07-27. The SDK's copy of the
 * IDL was never re-synced, so the client kept encoding 40-byte instructions
 * where the program required 42. Every baseline reset from every wallet
 * broadcast, was charged, and reverted with `InstructionDidNotDeserialize`.
 * Nothing caught it for two months, and the failure only surfaced when a user
 * with an unrecoverable baseline tried the one escape route the UI offers.
 *
 * Two things make this hard to catch by other means, which is why the check is
 * here rather than somewhere more obvious:
 *
 *   - The SDK and its own IDL agreed with each other. Any self-contained
 *     assertion, including a type check, passes. Only an external source of
 *     truth reveals the drift.
 *   - The IDL account published on chain is stale too, so regenerating from
 *     chain reproduces the bug rather than fixing it. `protocol-core`'s build
 *     output is the only authority.
 */

const SDK_IDL = resolve(__dirname, "../src/protocol/idl/entros_anchor.json");
const CORE_IDL = resolve(__dirname, "../../protocol-core/target/idl/entros_anchor.json");

interface IdlArg {
  name: string;
  type: unknown;
}
interface IdlInstruction {
  name: string;
  args?: IdlArg[];
}
interface Idl {
  address?: string;
  instructions: IdlInstruction[];
  accounts?: { name: string }[];
  types?: { name: string; type?: { fields?: { name: string }[] } }[];
}

function load(path: string): Idl {
  return JSON.parse(readFileSync(path, "utf8")) as Idl;
}

/** Instruction name to its argument list, as `name: type-json` pairs. */
function signatures(idl: Idl): Map<string, string[]> {
  return new Map(
    idl.instructions.map((ix) => [
      ix.name,
      (ix.args ?? []).map((a) => `${a.name}: ${JSON.stringify(a.type)}`),
    ]),
  );
}

// A sibling checkout is not guaranteed. Skipping is the weak point of this
// test and is called out rather than hidden: in a pulse-sdk-only checkout the
// drift this guards against becomes invisible again. Anyone changing the
// on-chain program must run `anchor build` in protocol-core before trusting a
// green run here.
const coreAvailable = existsSync(CORE_IDL);

describe.skipIf(!coreAvailable)("bundled IDL matches the built program", () => {
  it("declares the same program address", () => {
    expect(load(SDK_IDL).address).toBe(load(CORE_IDL).address);
  });

  it("has every instruction the program exposes", () => {
    const sdk = signatures(load(SDK_IDL));
    const core = signatures(load(CORE_IDL));
    const missing = [...core.keys()].filter((name) => !sdk.has(name));
    expect(
      missing,
      `bundled IDL is missing instruction(s) the program exposes. Re-copy ` +
        `protocol-core/target/idl/entros_anchor.json into src/protocol/idl/.`,
    ).toEqual([]);
  });

  it("declares the same arguments for every shared instruction", () => {
    const sdk = signatures(load(SDK_IDL));
    const core = signatures(load(CORE_IDL));
    // Reported as one object so a failure names every drifted instruction at
    // once rather than stopping at the first.
    const drifted: Record<string, { bundled: string[]; built: string[] }> = {};
    for (const [name, builtArgs] of core) {
      const bundledArgs = sdk.get(name);
      if (!bundledArgs) continue; // covered by the previous test
      if (JSON.stringify(bundledArgs) !== JSON.stringify(builtArgs)) {
        drifted[name] = { bundled: bundledArgs, built: builtArgs };
      }
    }
    expect(
      drifted,
      "argument drift encodes instructions the program cannot deserialize",
    ).toEqual({});
  });

  it("declares the same fields on every shared account type", () => {
    const fieldsOf = (idl: Idl) =>
      new Map(
        (idl.types ?? []).map((t) => [
          t.name,
          (t.type?.fields ?? []).map((f) => f.name),
        ]),
      );
    const sdk = fieldsOf(load(SDK_IDL));
    const core = fieldsOf(load(CORE_IDL));
    const drifted: Record<string, { bundled: string[]; built: string[] }> = {};
    for (const [name, builtFields] of core) {
      const bundledFields = sdk.get(name);
      if (!bundledFields) continue;
      if (JSON.stringify(bundledFields) !== JSON.stringify(builtFields)) {
        drifted[name] = { bundled: bundledFields, built: builtFields };
      }
    }
    // Account-field drift is quieter than argument drift: Borsh tolerates
    // trailing fields on read, so a stale account layout decodes without
    // error and simply cannot see the new fields. That is how
    // `projection_version` stayed invisible to the SDK.
    expect(drifted).toEqual({});
  });
});

describe("the reset instruction carries a projection version", () => {
  it("passes two arguments, matching the deployed program", () => {
    const reset = load(SDK_IDL).instructions.find(
      (ix) => ix.name === "reset_identity_state",
    );
    expect(reset, "reset_identity_state missing from the bundled IDL").toBeDefined();
    expect(
      (reset!.args ?? []).map((a) => a.name),
      "a one-argument reset encodes 40 bytes where the program needs 42",
    ).toEqual(["new_commitment", "projection_version"]);
  });

  it("knows about rebaseline_anchor", () => {
    // The instruction that migrates an anchor across projection versions.
    // It has never had a caller partly because the bundled IDL did not
    // declare it. See master-list #215.
    const names = load(SDK_IDL).instructions.map((ix) => ix.name);
    expect(names).toContain("rebaseline_anchor");
  });
});
