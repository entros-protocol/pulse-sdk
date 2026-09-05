import type { AccountInfo, PublicKey } from "@solana/web3.js";
import { BN254_SCALAR_FIELD, PROGRAM_IDS, SAS_CONFIG } from "../config";
import { HIGHEST_SUPPORTED_PROJECTION_VERSION } from "../projection";
import { decodeIdentityState } from "./anchor";
import { commitmentHex, qualifyingTransaction } from "./integrator-transaction";
import type {
  IntegratorAttestationEvidence,
  IntegratorEvidenceReadResult,
  IntegratorIdentityEvidence,
  ReadIntegratorEvidenceInput,
} from "./integrator-types";

export const INTEGRATOR_DEVNET_GENESIS_HASH =
  "EtWTRABZaYq6iMfeYKouRu166VU2xqa1wcaWoxPkrZBG";
export const INTEGRATOR_PROGRAM_IDS = {
  anchor: PROGRAM_IDS.entrosAnchor,
  verifier: PROGRAM_IDS.entrosVerifier,
  registry: PROGRAM_IDS.entrosRegistry,
  sas: SAS_CONFIG.programId,
  credential: SAS_CONFIG.entrosCredentialPda,
  schema: SAS_CONFIG.entrosSchemaPda,
} as const;

async function boundedRpc<T>(operation: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("Evidence RPC deadline exceeded")),
          3000,
        );
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

function integer(
  value: number,
  minimum = 0,
  maximum = Number.MAX_SAFE_INTEGER,
): boolean {
  return Number.isSafeInteger(value) && value >= minimum && value <= maximum;
}

async function strictIdentity(
  account: AccountInfo<Uint8Array>,
  wallet: PublicKey,
  identityPda: PublicKey,
  now: number,
): Promise<IntegratorIdentityEvidence | null> {
  const { PublicKey } = await import("@solana/web3.js");
  if (
    account.executable ||
    !account.owner.equals(new PublicKey(PROGRAM_IDS.entrosAnchor))
  )
    return null;
  if (![543, 551, 583, 593].includes(account.data.length)) return null;
  const state = await decodeIdentityState(account.data);
  if (!state || state.owner !== wallet.toBase58()) return null;
  const commitment = commitmentHex(state.currentCommitment);
  const lastRebaseline = state.lastRebaselineTimestamp;
  if (
    !commitment ||
    lastRebaseline === undefined ||
    BigInt("0x" + commitment) === 0n ||
    BigInt("0x" + commitment) >= BN254_SCALAR_FIELD
  )
    return null;
  if (
    !integer(state.creationTimestamp, 1, now) ||
    !integer(state.lastVerificationTimestamp, state.creationTimestamp, now)
  )
    return null;
  if (
    !integer(state.verificationCount, 0, 0xffffffff) ||
    !integer(state.trustScore, 0, 10000)
  )
    return null;
  if (
    !integer(state.projectionVersion, 0, HIGHEST_SUPPORTED_PROJECTION_VERSION)
  )
    return null;
  if (
    !integer(state.lastResetTimestamp, 0, state.lastVerificationTimestamp) ||
    !integer(lastRebaseline, 0, state.lastVerificationTimestamp)
  )
    return null;
  if (
    state.lastResetTimestamp !== 0 &&
    state.lastResetTimestamp < state.creationTimestamp
  )
    return null;
  if (lastRebaseline !== 0 && lastRebaseline < state.creationTimestamp)
    return null;
  const [mint] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("mint"), wallet.toBytes()],
    new PublicKey(PROGRAM_IDS.entrosAnchor),
  );
  if (state.mint !== mint.toBase58()) return null;
  return {
    walletPubkey: wallet.toBase58(),
    identityPda: identityPda.toBase58(),
    creationTimestamp: state.creationTimestamp,
    lastVerificationTimestamp: state.lastVerificationTimestamp,
    verificationCount: state.verificationCount,
    trustScore: state.trustScore,
    currentCommitment: commitment,
    projectionVersion: state.projectionVersion,
    lastResetTimestamp: state.lastResetTimestamp,
    lastRebaselineTimestamp: lastRebaseline,
    mint: state.mint,
  };
}

async function strictAttestation(
  account: AccountInfo<Uint8Array> | null,
  wallet: PublicKey,
  address: PublicKey,
  slot: number,
  now: number,
  onFutureAttestation: (verifiedAt: number) => void,
): Promise<IntegratorAttestationEvidence> {
  if (!account) return { status: "missing" };
  const invalid = { status: "invalid" } as const;
  const { PublicKey } = await import("@solana/web3.js");
  if (
    account.executable ||
    !account.owner.equals(new PublicKey(SAS_CONFIG.programId))
  )
    return invalid;
  const raw = account.data;
  // The pinned SAS issuer uses one discriminator byte and revokes by closing the account.
  if (raw.length < 173 || raw[0] !== 2) return invalid;
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const length = view.getUint32(97, true);
  const mode = new TextEncoder().encode("wallet-connected");
  if (length !== 15 + mode.length || raw.length !== 173 + length)
    return invalid;
  if (!new PublicKey(raw.subarray(1, 33)).equals(wallet)) return invalid;
  if (
    !new PublicKey(raw.subarray(33, 65)).equals(
      new PublicKey(SAS_CONFIG.entrosCredentialPda),
    )
  )
    return invalid;
  if (
    !new PublicKey(raw.subarray(65, 97)).equals(
      new PublicKey(SAS_CONFIG.entrosSchemaPda),
    )
  )
    return invalid;
  if (
    raw[101] !== 1 ||
    view.getUint16(102, true) > 10000 ||
    view.getUint32(112, true) !== mode.length
  )
    return invalid;
  if (!mode.every((byte, index) => raw[116 + index] === byte)) return invalid;
  const verifiedAt = Number(view.getBigInt64(104, true));
  const expiry = Number(view.getBigInt64(133 + length, true));
  if (!integer(verifiedAt, 1) || !integer(expiry)) return invalid;
  if (expiry !== 0 && (expiry <= now || expiry <= verifiedAt)) return invalid;
  if (raw.subarray(101 + length, 133 + length).every((byte) => byte === 0))
    return invalid;
  if (!raw.subarray(141 + length).every((byte) => byte === 0)) return invalid;
  if (verifiedAt > now) {
    onFutureAttestation(verifiedAt);
    return invalid;
  }
  return {
    status: "present",
    address: address.toBase58(),
    readContextSlot: slot,
    expiresAt: expiry === 0 ? null : expiry,
  };
}

async function readOnce(
  input: ReadIntegratorEvidenceInput,
  readNow: () => number | null,
  onFutureAttestation: (verifiedAt: number) => void,
): Promise<IntegratorEvidenceReadResult> {
  const { PublicKey } = await import("@solana/web3.js");
  const { connection, transactionSignature: signature } = input;
  let now = readNow();
  if (now === null || !/^[1-9A-HJ-NP-Za-km-z]{64,88}$/.test(signature))
    return { status: "invalid", reason: "invalid_request" };
  let wallet: PublicKey;
  try {
    wallet = new PublicKey(input.walletPubkey);
    if (wallet.toBase58() !== input.walletPubkey)
      return { status: "invalid", reason: "invalid_request" };
    const anchor = await import("@coral-xyz/anchor");
    if (anchor.utils.bytes.bs58.decode(signature).length !== 64)
      return { status: "invalid", reason: "invalid_request" };
  } catch {
    return { status: "invalid", reason: "invalid_request" };
  }
  const [identityPda] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("identity"), wallet.toBytes()],
    new PublicKey(PROGRAM_IDS.entrosAnchor),
  );
  const [genesisHash, statuses, parsed] = await Promise.all([
    boundedRpc(connection.getGenesisHash()),
    boundedRpc(
      connection.getSignatureStatuses([signature], {
        searchTransactionHistory: true,
      }),
    ),
    boundedRpc(
      connection.getParsedTransaction(signature, {
        commitment: "confirmed",
        maxSupportedTransactionVersion: 0,
      }),
    ),
  ]);
  now = readNow();
  if (now === null) return { status: "invalid", reason: "invalid_request" };
  if (genesisHash !== INTEGRATOR_DEVNET_GENESIS_HASH)
    return { status: "invalid", reason: "wrong_cluster" };
  const status = statuses.value[0];
  if (
    !status ||
    !parsed ||
    !["confirmed", "finalized"].includes(status.confirmationStatus ?? "")
  )
    return { status: "unavailable", reason: "transaction_unavailable" };
  if (
    status.err !== null ||
    parsed.meta?.err !== null ||
    status.slot !== parsed.slot
  )
    return { status: "invalid", reason: "transaction_invalid" };
  if (parsed.blockTime == null)
    return { status: "unavailable", reason: "transaction_unavailable" };
  if (!integer(parsed.blockTime, 1, now) || !integer(parsed.slot, 1))
    return { status: "invalid", reason: "transaction_invalid" };
  const qualifying = await qualifyingTransaction(
    parsed,
    signature,
    wallet,
    identityPda,
  );
  if (!qualifying) return { status: "invalid", reason: "transaction_invalid" };
  const { projectionVersion: transactionProjection, ...transaction } =
    qualifying;
  const snapshot = await boundedRpc(
    connection.getAccountInfoAndContext(identityPda, {
      commitment: "confirmed",
      minContextSlot: transaction.slot,
    }),
  );
  now = readNow();
  if (now === null) return { status: "invalid", reason: "invalid_request" };
  if (!integer(snapshot.context.slot, transaction.slot))
    return { status: "unavailable", reason: "snapshot_unavailable" };
  if (!snapshot.value) return { status: "invalid", reason: "identity_missing" };
  const identity = await strictIdentity(
    snapshot.value,
    wallet,
    identityPda,
    now,
  );
  if (!identity) return { status: "invalid", reason: "identity_invalid" };
  if (
    (transactionProjection !== undefined &&
      transactionProjection !== identity.projectionVersion) ||
    identity.lastRebaselineTimestamp > transaction.blockTime ||
    identity.currentCommitment !== transaction.commitment ||
    identity.lastResetTimestamp >= transaction.blockTime ||
    identity.lastVerificationTimestamp < transaction.blockTime ||
    identity.creationTimestamp > transaction.blockTime
  )
    return { status: "invalid", reason: "transaction_invalid" };
  if (
    identity.verificationCount === 0 &&
    !(
      transaction.kind === "mint" &&
      identity.trustScore === 0 &&
      identity.creationTimestamp === transaction.blockTime &&
      identity.lastVerificationTimestamp === transaction.blockTime &&
      identity.lastResetTimestamp === 0 &&
      identity.lastRebaselineTimestamp === 0
    )
  )
    return { status: "invalid", reason: "identity_invalid" };
  const [attestationPda] = PublicKey.findProgramAddressSync(
    [
      new TextEncoder().encode("attestation"),
      new PublicKey(SAS_CONFIG.entrosCredentialPda).toBytes(),
      new PublicKey(SAS_CONFIG.entrosSchemaPda).toBytes(),
      wallet.toBytes(),
    ],
    new PublicKey(SAS_CONFIG.programId),
  );
  let attestation: IntegratorAttestationEvidence;
  try {
    const account = await boundedRpc(
      connection.getAccountInfoAndContext(attestationPda, {
        commitment: "confirmed",
        minContextSlot: snapshot.context.slot,
      }),
    );
    now = readNow();
    if (now === null) return { status: "invalid", reason: "invalid_request" };
    attestation = integer(account.context.slot, snapshot.context.slot)
      ? await strictAttestation(
          account.value,
          wallet,
          attestationPda,
          account.context.slot,
          now,
          onFutureAttestation,
        )
      : { status: "unavailable" };
  } catch {
    attestation = { status: "unavailable" };
  }
  if (readNow() === null)
    return { status: "invalid", reason: "invalid_request" };
  return {
    status: "available",
    evidence: {
      identity,
      transaction,
      attestation,
      readContextSlot: snapshot.context.slot,
      genesisHash,
      cluster: "devnet",
      assuranceTier: "browser_unattested",
      uniquenessStatus: "unmeasured",
      programIds: INTEGRATOR_PROGRAM_IDS,
    },
  };
}

async function waitForAttestationClock(
  verifiedAt: number,
  readNow: () => number | null,
): Promise<"ready" | "timeout" | "invalid"> {
  const budgetMs = 3000;
  const startedAt = performance.now();
  while (true) {
    const now = readNow();
    if (now === null) return "invalid";
    const elapsedMs = performance.now() - startedAt;
    if (elapsedMs > budgetMs || verifiedAt - now > budgetMs / 1000)
      return "timeout";
    if (now >= verifiedAt) return "ready";
    const remainingMs = budgetMs - elapsedMs;
    if (remainingMs <= 0) return "timeout";
    await new Promise<void>((resolve) => {
      setTimeout(resolve, Math.min(remainingMs, (verifiedAt - now) * 1000));
    });
  }
}

/** Read confirmed application evidence without submitting a transaction or changing protocol policy. */
export async function readIntegratorEvidence(
  input: ReadIntegratorEvidenceInput,
): Promise<IntegratorEvidenceReadResult> {
  let previousNow = 0;
  const readNow = (): number | null => {
    try {
      const now =
        typeof input.nowSeconds === "function"
          ? input.nowSeconds()
          : input.nowSeconds;
      if (!integer(now, 1) || now < previousNow) return null;
      previousNow = now;
      return now;
    } catch {
      return null;
    }
  };
  let propagationRetried = false;
  let clockReconciled = false;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const pending: { verifiedAt: number | null } = { verifiedAt: null };
      const result = await readOnce(input, readNow, (verifiedAt) => {
        pending.verifiedAt = verifiedAt;
      });
      if (result.status === "invalid") return result;
      if (
        pending.verifiedAt !== null &&
        typeof input.nowSeconds === "function" &&
        !clockReconciled
      ) {
        clockReconciled = true;
        const clock = await waitForAttestationClock(
          pending.verifiedAt,
          readNow,
        );
        if (clock === "invalid")
          return { status: "invalid", reason: "invalid_request" };
        if (clock === "ready") continue;
        return result;
      }
      if (result.status !== "unavailable" || propagationRetried) return result;
      propagationRetried = true;
    } catch {
      return { status: "unavailable", reason: "rpc_unavailable" };
    }
  }
  return { status: "unavailable", reason: "rpc_unavailable" };
}
