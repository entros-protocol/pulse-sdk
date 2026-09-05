import type { Connection, PublicKey } from "@solana/web3.js";
import type {
  AnchorUpdateAction,
  PreparedProofRequest,
  RequestBoundManifest,
} from "../proof/request";
import {
  bytesHex,
  hexBytes,
  prepareProofRequest,
  validateRequestBoundManifest,
} from "../proof/request";
import { decodeIdentityState } from "../identity/anchor";
import { PROGRAM_IDS, MAX_THRESHOLD, MIN_DISTANCE_FLOOR } from "../config";
import { sha256 } from "@noble/hashes/sha256";
import { fetchSubmissionNonce, validateNonce } from "./nonce";

export class IdentityLayoutUpgradeRequiredError extends Error {
  readonly code = "identity_layout_upgrade_required";
  constructor(readonly currentLength: number) {
    super(
      "The identity account requires a signed layout upgrade before proving",
    );
    this.name = "IdentityLayoutUpgradeRequiredError";
  }
}

export interface PrepareWalletProofOptions {
  readonly commitmentNew: Uint8Array;
  readonly commitmentPrevious: Uint8Array;
  readonly threshold: number;
  readonly minDistance: number;
  readonly lifetimeSeconds?: number;
  readonly nonce?: Uint8Array;
  readonly relayerUrl?: string;
  readonly relayerApiKey?: string;
}
export async function prepareWalletProofRequest(
  connection: Pick<
    Connection,
    "getGenesisHash" | "getMultipleAccountsInfoAndContext"
  >,
  wallet: PublicKey,
  manifest: RequestBoundManifest,
  options: PrepareWalletProofOptions,
): Promise<PreparedProofRequest> {
  validateRequestBoundManifest(manifest);
  if (
    manifest.verifierProgram !== PROGRAM_IDS.entrosVerifier ||
    manifest.consumerProgram !== PROGRAM_IDS.entrosAnchor
  )
    throw new Error("Unsupported request-bound deployment programs");
  if (
    options.threshold > MAX_THRESHOLD ||
    options.minDistance < MIN_DISTANCE_FLOOR
  )
    throw new Error("Unsupported proof distance bounds");
  const lifetime = options.lifetimeSeconds ?? 180;
  if (!Number.isInteger(lifetime) || lifetime <= 0 || lifetime > 300)
    throw new Error("Invalid proof request lifetime");
  const { PublicKey, SystemProgram, SYSVAR_CLOCK_PUBKEY } =
    await import("@solana/web3.js");
  const anchor = new PublicKey(manifest.consumerProgram);
  const [identity, identityBump] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("identity"), wallet.toBytes()],
    anchor,
  );
  const [state, bump] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("proof_request_state"), wallet.toBytes()],
    anchor,
  );
  const submissionNonce = options.nonce
    ? { bytes: validateNonce(options.nonce), source: "client" as const }
    : await fetchSubmissionNonce(
        wallet.toBase58(),
        options.relayerUrl,
        options.relayerApiKey,
      );
  const [genesis, accounts] = await Promise.all([
    connection.getGenesisHash(),
    connection.getMultipleAccountsInfoAndContext(
      [identity, state, SYSVAR_CLOCK_PUBKEY],
      { commitment: "confirmed" },
    ),
  ]);
  if (genesis !== manifest.genesisHash)
    throw new Error("Connected chain does not match the proof manifest");
  const [identityAccount, stateAccount, clockAccount] = accounts.value;
  if (
    !identityAccount ||
    identityAccount.executable ||
    !identityAccount.owner.equals(anchor) ||
    ![543, 551, 583, 593].includes(identityAccount.data.length) ||
    identityAccount.data[126] !== identityBump
  )
    throw new Error("Invalid request-bound identity account");
  const decoded = await decodeIdentityState(identityAccount.data);
  if (
    !decoded ||
    decoded.owner !== wallet.toBase58() ||
    bytesHex(decoded.currentCommitment) !== bytesHex(options.commitmentPrevious)
  )
    throw new Error("Identity does not match the proof baseline");
  const [mint] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("mint"), wallet.toBytes()],
    anchor,
  );
  if (decoded.mint !== mint.toBase58())
    throw new Error("Identity mint does not match its PDA");
  if (identityAccount.data.length !== 593)
    throw new IdentityLayoutUpgradeRequiredError(identityAccount.data.length);
  let counter = 0n;
  const uninitializedState =
    stateAccount &&
    !stateAccount.executable &&
    stateAccount.owner.equals(SystemProgram.programId) &&
    stateAccount.data.length === 0;
  if (stateAccount && !uninitializedState) {
    const data = stateAccount.data;
    const discriminator = sha256(
      new TextEncoder().encode("account:ProofRequestState"),
    ).slice(0, 8);
    if (
      stateAccount.executable ||
      !stateAccount.owner.equals(anchor) ||
      data.length !== 50 ||
      bytesHex(data.subarray(0, 8)) !== bytesHex(discriminator) ||
      data[8] !== 1 ||
      bytesHex(data.subarray(9, 41)) !== bytesHex(wallet.toBytes()) ||
      data[49] !== bump
    )
      throw new Error("Invalid proof request state");
    counter = new DataView(
      data.buffer,
      data.byteOffset,
      data.byteLength,
    ).getBigUint64(41, true);
  }
  if (counter === (1n << 64n) - 1n)
    throw new Error("Proof request counter exhausted");
  if (
    !clockAccount ||
    clockAccount.executable ||
    clockAccount.data.length !== 40 ||
    clockAccount.owner.toBase58() !==
      "Sysvar1111111111111111111111111111111111111"
  )
    throw new Error("Invalid chain clock");
  const now = new DataView(
    clockAccount.data.buffer,
    clockAccount.data.byteOffset,
    40,
  ).getBigInt64(32, true);
  if (now <= 0n) throw new Error("Invalid chain clock timestamp");
  const nonce = submissionNonce.bytes;
  const action: AnchorUpdateAction = {
    identity: bytesHex(identity.toBytes()),
    mint: bytesHex(new PublicKey(decoded.mint).toBytes()),
    counter,
    projectionVersion: decoded.projectionVersion,
    commitmentNew: bytesHex(options.commitmentNew),
    commitmentPrevious: bytesHex(options.commitmentPrevious),
    threshold: options.threshold,
    minDistance: options.minDistance,
    validUntil: now + BigInt(lifetime),
  };
  hexBytes(bytesHex(nonce));
  return prepareProofRequest(
    {
      deploymentDomain: manifest.deploymentDomain,
      verifier: bytesHex(new PublicKey(manifest.verifierProgram).toBytes()),
      consumer: bytesHex(anchor.toBytes()),
      wallet: bytesHex(wallet.toBytes()),
      nonce: bytesHex(nonce),
      actionKind: 1,
      action,
    },
    submissionNonce.source,
  );
}
