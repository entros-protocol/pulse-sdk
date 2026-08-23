import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

/**
 * The bundled IDL must match the one `anchor build` produces in protocol-core.
 *
 * The SDK and a stale bundled IDL can agree with each other. The generated
 * protocol-core IDL supplies the independent interface authority.
 */

const SDK_IDL = resolve(__dirname, "../src/protocol/idl/entros_anchor.json");
const CORE_IDL = resolve(__dirname, "../../protocol-core/target/idl/entros_anchor.json");
const SDK_VERIFIER_IDL = resolve(
  __dirname,
  "../src/protocol/idl/entros_verifier.json",
);
const CORE_VERIFIER_IDL = resolve(
  __dirname,
  "../../protocol-core/target/idl/entros_verifier.json",
);

interface IdlArg {
  name: string;
  type: unknown;
}
interface IdlInstruction {
  name: string;
  args?: IdlArg[];
  accounts?: { name: string }[];
}
interface IdlError {
  code: number;
  name: string;
}
interface Idl {
  address?: string;
  instructions: IdlInstruction[];
  accounts?: { name: string }[];
  types?: { name: string; type?: { fields?: { name: string }[] } }[];
  errors?: IdlError[];
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

// Release validation builds protocol-core before relying on this comparison.
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
    // Report all drifted instructions in one failure.
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
    // Borsh tolerates trailing fields, so a stale decoder can omit new fields.
    expect(drifted).toEqual({});
  });

  it("assigns the same code to every error name", () => {
    // Variant order fixes each Anchor error number. Hosts route on the number.
    // Messages do not affect routing.
    const codes = (idl: Idl) =>
      new Map((idl.errors ?? []).map((e) => [e.name, e.code]));
    const sdk = codes(load(SDK_IDL));
    const core = codes(load(CORE_IDL));
    const drifted: Record<string, { bundled?: number; built: number }> = {};
    for (const [name, builtCode] of core) {
      const bundledCode = sdk.get(name);
      if (bundledCode !== builtCode) {
        drifted[name] = { bundled: bundledCode, built: builtCode };
      }
    }
    expect(
      drifted,
      "an error code moved. Retiring a variant must leave it in place so the " +
        "numbering after it holds.",
    ).toEqual({});
  });
});

describe.skipIf(!existsSync(CORE_VERIFIER_IDL))(
  "bundled verifier IDL matches the built program",
  () => {
    it("matches every generated field", () => {
      expect(load(SDK_VERIFIER_IDL)).toEqual(load(CORE_VERIFIER_IDL));
    });
  },
);

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
    const rebaseline = load(SDK_IDL).instructions.find(
      (ix) => ix.name === "rebaseline_anchor",
    );
    expect(rebaseline, "rebaseline_anchor missing from the bundled IDL").toBeDefined();
    expect((rebaseline!.args ?? []).map((arg) => arg.name)).toEqual([
      "new_commitment",
      "projection_version",
    ]);
    expect((rebaseline!.accounts ?? []).map((account) => account.name)).toContain(
      "instructions_sysvar",
    );
  });

  it("keeps the reset named accounts compatible with version 0 clients", () => {
    const reset = load(SDK_IDL).instructions.find(
      (ix) => ix.name === "reset_identity_state",
    );
    expect((reset!.accounts ?? []).map((account) => account.name)).toEqual([
      "authority",
      "identity_state",
      "protocol_config",
      "treasury",
      "system_program",
    ]);
  });
});
