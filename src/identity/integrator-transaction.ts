import type { Idl } from "@coral-xyz/anchor";
import type {
  ParsedTransactionWithMeta,
  PartiallyDecodedInstruction,
  PublicKey,
} from "@solana/web3.js";
import { PROGRAM_IDS } from "../config";
import { entrosAnchorIdl, entrosVerifierIdl } from "../protocol/idl";
import type { IntegratorTransactionEvidence } from "./integrator-types";

export function commitmentHex(value: unknown): string | null {
  if (!(value instanceof Uint8Array) && !Array.isArray(value)) return null;
  const bytes: unknown[] = Array.from(value);
  if (
    bytes.length !== 32 ||
    !bytes.every(
      (n) => typeof n === "number" && Number.isInteger(n) && n >= 0 && n <= 255,
    )
  )
    return null;
  return bytes.map((n) => (n as number).toString(16).padStart(2, "0")).join("");
}

/** Successful execution supplies the program's receipt and proof checks. */
export async function qualifyingTransaction(
  transaction: ParsedTransactionWithMeta,
  signature: string,
  wallet: PublicKey,
  identity: PublicKey,
): Promise<
  (IntegratorTransactionEvidence & { projectionVersion?: number }) | null
> {
  const anchor = await import("@coral-xyz/anchor");
  const { PublicKey } = await import("@solana/web3.js");
  const anchorCoder = new anchor.BorshInstructionCoder(entrosAnchorIdl as Idl);
  const verifierCoder = new anchor.BorshInstructionCoder(
    entrosVerifierIdl as Idl,
  );
  const instructions = transaction.transaction.message.instructions;
  if (
    !transaction.transaction.message.accountKeys.some(
      (key) => key.signer && key.pubkey.equals(wallet),
    )
  )
    return null;
  if (transaction.transaction.signatures[0] !== signature) return null;
  if (
    transaction.version !== undefined &&
    transaction.version !== "legacy" &&
    transaction.version !== 0
  )
    return null;
  if (transaction.meta?.err !== null || transaction.blockTime == null)
    return null;

  function decode(
    ix: PartiallyDecodedInstruction,
    coder: typeof anchorCoder,
  ): { name: string; data: Record<string, unknown> } | null {
    try {
      const decoded = coder.decode(ix.data, "base58");
      if (!decoded || typeof decoded.data !== "object" || decoded.data === null)
        return null;
      const encoded = coder.encode(decoded.name, decoded.data);
      if (anchor.utils.bytes.bs58.encode(encoded) !== ix.data) return null;
      return {
        name: decoded.name,
        data: decoded.data as Record<string, unknown>,
      };
    } catch {
      return null;
    }
  }

  const candidates: (IntegratorTransactionEvidence & {
    projectionVersion?: number;
  })[] = [];
  for (const [index, instruction] of instructions.entries()) {
    if (!instruction.programId.equals(new PublicKey(PROGRAM_IDS.entrosAnchor)))
      continue;
    if (!("accounts" in instruction)) return null;
    const decoded = decode(instruction, anchorCoder);
    if (!decoded) return null;
    const names = [
      "mint_anchor",
      "update_anchor",
      "update_anchor_compact",
      "rebaseline_anchor",
    ];
    if (!names.includes(decoded.name)) {
      if (decoded.name === "reset_identity_state") return null;
      continue;
    }
    const definition = entrosAnchorIdl.instructions.find(
      (ix) => ix.name === decoded.name,
    );
    if (
      !definition ||
      instruction.accounts.length !== definition.accounts.length
    )
      return null;
    if (
      !instruction.accounts[0]?.equals(wallet) ||
      !instruction.accounts[1]?.equals(identity)
    )
      return null;
    let commitment = commitmentHex(
      decoded.data.initial_commitment ?? decoded.data.new_commitment,
    );
    if (decoded.name === "update_anchor_compact") {
      const nonce = commitmentHex(decoded.data.verification_nonce);
      if (!nonce) return null;
      const nonceBytes = Uint8Array.from(nonce.match(/../g) ?? [], (byte) =>
        Number.parseInt(byte, 16),
      );
      const [verificationPda] = PublicKey.findProgramAddressSync(
        [
          new TextEncoder().encode("verification"),
          wallet.toBytes(),
          nonceBytes,
        ],
        new PublicKey(PROGRAM_IDS.entrosVerifier),
      );
      if (!instruction.accounts[2]?.equals(verificationPda)) return null;
      for (const earlier of instructions.slice(0, index)) {
        if (
          !earlier.programId.equals(
            new PublicKey(PROGRAM_IDS.entrosVerifier),
          ) ||
          !("accounts" in earlier)
        )
          continue;
        const proof = decode(earlier, verifierCoder);
        if (
          !proof ||
          !["verify_proof", "verify_proof_compact"].includes(proof.name)
        )
          continue;
        const proofDefinition = entrosVerifierIdl.instructions.find(
          (ix) => ix.name === proof.name,
        );
        if (earlier.accounts.length !== proofDefinition?.accounts.length)
          return null;
        if (
          !earlier.accounts[0]?.equals(wallet) ||
          !earlier.accounts[2]?.equals(verificationPda)
        )
          continue;
        if (commitmentHex(proof.data.nonce) !== nonce) continue;
        const inputs = proof.data.public_inputs;
        commitment =
          proof.name === "verify_proof_compact"
            ? commitmentHex(proof.data.commitment_new)
            : Array.isArray(inputs)
              ? commitmentHex(inputs[0])
              : null;
      }
    }
    if (!commitment) return null;
    candidates.push({
      signature,
      slot: transaction.slot,
      blockTime: transaction.blockTime,
      commitment,
      ...(decoded.name === "rebaseline_anchor"
        ? { projectionVersion: decoded.data.projection_version as number }
        : {}),
      kind:
        decoded.name === "mint_anchor"
          ? "mint"
          : decoded.name === "rebaseline_anchor"
            ? "rebaseline"
            : "update",
    });
  }
  return candidates.length === 1 ? (candidates[0] ?? null) : null;
}
