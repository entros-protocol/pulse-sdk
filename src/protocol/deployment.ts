import type { Idl } from "@coral-xyz/anchor";
import type { RequestBoundDeployment } from "../proof/request";
import { hexBytes } from "../proof/request";
import { PROGRAM_IDS } from "../config";

const DEVNET_GENESIS = "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG";

/** Select programs from trusted application configuration. Never use popup response addresses. */
export function resolveDeployment(deployment?: RequestBoundDeployment) {
  if (!deployment)
    return Object.freeze({
      consumerProgram: PROGRAM_IDS.entrosAnchor,
      verifierProgram: PROGRAM_IDS.entrosVerifier,
      isolated: false,
      storageNamespace: "",
    });
  const {
    generation,
    consumerProgram,
    verifierProgram,
    deploymentDomain,
    genesisHash,
  } = deployment;
  if (generation !== "request-bound-v1")
    throw new Error("Unsupported proof deployment generation");
  hexBytes(deploymentDomain);
  if (
    ![consumerProgram, verifierProgram].every(
      (value) =>
        typeof value === "string" &&
        /^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(value),
    )
  )
    throw new Error("Invalid deployment program address");
  const officialAnchor = consumerProgram === PROGRAM_IDS.entrosAnchor;
  const officialVerifier = verifierProgram === PROGRAM_IDS.entrosVerifier;
  if (
    officialAnchor !== officialVerifier ||
    consumerProgram === verifierProgram ||
    consumerProgram === PROGRAM_IDS.entrosVerifier ||
    verifierProgram === PROGRAM_IDS.entrosAnchor
  )
    throw new Error(
      "Deployment requires a distinct paired Anchor and verifier",
    );
  const isolated = !officialAnchor;
  if (isolated && genesisHash !== DEVNET_GENESIS)
    throw new Error("Isolated deployments require Solana devnet");
  return Object.freeze({
    consumerProgram,
    verifierProgram,
    isolated,
    storageNamespace: isolated
      ? `_${genesisHash}_${consumerProgram}_${verifierProgram}_${deploymentDomain}`
      : "",
  });
}

export async function checkDeploymentChain(
  deployment: RequestBoundDeployment | undefined,
  connection: { getGenesisHash?(): Promise<string> },
): Promise<void> {
  if (!deployment) return;
  const expectedGenesis = deployment.genesisHash;
  resolveDeployment(deployment);
  if (
    !connection.getGenesisHash ||
    (await connection.getGenesisHash()) !== expectedGenesis
  )
    throw new Error("Connected chain does not match the proof manifest");
}

/** Rebind only the paired programs. Registry and token addresses remain fixed. */
export async function deploymentIdl(
  idl: Idl,
  deployment?: RequestBoundDeployment,
): Promise<Idl> {
  const selected = resolveDeployment(deployment);
  if (!selected.isolated) return idl;
  const { PublicKey } = await import("@solana/web3.js");
  const replacements = (
    [
      [PROGRAM_IDS.entrosAnchor, selected.consumerProgram],
      [PROGRAM_IDS.entrosVerifier, selected.verifierProgram],
    ] as const
  ).map(([from, to]) => ({
    from,
    to,
    oldBytes: Array.from(new PublicKey(from).toBytes()),
    newBytes: Array.from(new PublicKey(to).toBytes()),
  }));
  const copy: Idl = structuredClone(idl);
  copy.address =
    replacements.find((pair) => pair.from === idl.address)?.to ?? idl.address;
  const visit = (accounts: Idl["instructions"][number]["accounts"]): void => {
    for (const account of accounts) {
      if ("accounts" in account) {
        visit(account.accounts);
        continue;
      }
      if (account.address)
        account.address =
          replacements.find((pair) => pair.from === account.address)?.to ??
          account.address;
      const program = account.pda?.program;
      if (program?.kind === "const") {
        const pair = replacements.find(
          (pair) =>
            pair.oldBytes.length === program.value.length &&
            pair.oldBytes.every((byte, index) => byte === program.value[index]),
        );
        if (pair) program.value = pair.newBytes;
      }
    }
  };
  for (const instruction of copy.instructions) visit(instruction.accounts);
  return copy;
}
