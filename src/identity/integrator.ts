import { resolveDeployment } from "../protocol/deployment";
import type { AccountInfo, Connection, PublicKey } from "@solana/web3.js";
import { BN254_SCALAR_FIELD, PROGRAM_IDS, SAS_CONFIG } from "../config";
import { HIGHEST_SUPPORTED_PROJECTION_VERSION } from "../projection";
import { decodeIdentityState } from "./anchor";
import { commitmentHex, qualifyingTransaction } from "./integrator-transaction";
import {
  migrationRecordSeeds,
  readMigrationRecord,
  type MigrationRecordRead,
} from "./migration";
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

/**
 * Seconds the RPC block time of a slot may differ from the Clock sysvar time
 * the program stored in that slot. Devnet has served block times one second
 * after the stored clock.
 */
export const BLOCK_TIME_TOLERANCE_SECONDS = 2;

/**
 * Seconds a reader waits in total for a device clock that trails the chain.
 * Devices that sync time automatically have been measured several seconds slow.
 */
export const CLOCK_CATCH_UP_SECONDS = 15;

type ChainTimeSource = "transaction" | "identity" | "attestation";

/** Rejects after three seconds so one stalled RPC call cannot hold a reader open. */
export async function boundedRpc<T>(operation: Promise<T>): Promise<T> {
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
    !integer(state.creationTimestamp, 1) ||
    !integer(state.lastVerificationTimestamp, state.creationTimestamp)
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

/** What the attestation must postdate for it to stand for the identity the wallet holds now. */
interface AttestationContext {
  creationTimestamp: number;
  lastResetTimestamp: number;
  migration: MigrationRecordRead;
}

async function strictAttestation(
  account: AccountInfo<Uint8Array> | null,
  wallet: PublicKey,
  address: PublicKey,
  slot: number,
  now: number,
  context: AttestationContext,
  onFutureAttestation: (verifiedAt: number) => void,
): Promise<IntegratorAttestationEvidence> {
  if (!account) return { status: "missing" };
  const invalid = { status: "invalid" } as const;
  const { PublicKey, SystemProgram } = await import("@solana/web3.js");
  // Anyone can send lamports to this address. That creates an account the
  // System Program owns, with no data, and no attestation exists there.
  if (
    !account.executable &&
    account.owner.equals(SystemProgram.programId) &&
    account.data.length === 0
  )
    return { status: "missing" };
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
  if (expiry !== 0 && expiry <= verifiedAt) return invalid;
  if (raw.subarray(101 + length, 133 + length).every((byte) => byte === 0))
    return invalid;
  if (!raw.subarray(141 + length).every((byte) => byte === 0)) return invalid;
  if (context.migration.status === "invalid") return invalid;
  if (verifiedAt > now) {
    onFutureAttestation(verifiedAt);
    return invalid;
  }
  // The cases below are well-formed attestations that no longer stand for the
  // identity this wallet holds, so each reads as no attestation. A policy that
  // does not require one can still allow. All times come from the program's
  // clock: the issuer writes the identity's verification time into the attestation.
  //
  // An expired attestation lapsed.
  if (expiry !== 0 && expiry <= now) return { status: "missing" };
  // A reset replaces the baseline, so an attestation from before it stands for
  // a capture the identity no longer holds.
  if (verifiedAt < context.lastResetTimestamp) return { status: "missing" };
  // The wallet closed an earlier identity and created this one afterwards.
  if (verifiedAt < context.creationTimestamp) return { status: "missing" };
  // The identity arrived from another wallet, and the attestation was issued
  // to this wallet for what it held before. A migration in the same second as
  // the verification counts as later, since the program clock cannot order them.
  if (
    context.migration.status === "migrated" &&
    verifiedAt <= context.migration.migratedAt
  )
    return { status: "missing" };
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
  onFutureTime: (source: ChainTimeSource, time: number) => void,
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
  if (!integer(parsed.blockTime, 1) || !integer(parsed.slot, 1))
    return { status: "invalid", reason: "transaction_invalid" };
  const qualifying = await qualifyingTransaction(
    parsed,
    signature,
    wallet,
    identityPda,
  );
  if (!qualifying) return { status: "invalid", reason: "transaction_invalid" };
  if (parsed.blockTime > now) {
    onFutureTime("transaction", parsed.blockTime);
    return { status: "invalid", reason: "transaction_invalid" };
  }
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
  const identity = await strictIdentity(snapshot.value, wallet, identityPda);
  if (!identity) return { status: "invalid", reason: "identity_invalid" };
  // The program stores its own clock, which can trail or lead the RPC block
  // time. A time that must precede the transaction is checked against the
  // earliest clock it could have had, and one it wrote against the latest.
  const earliestClock = transaction.blockTime - BLOCK_TIME_TOLERANCE_SECONDS;
  const latestClock = transaction.blockTime + BLOCK_TIME_TOLERANCE_SECONDS;
  if (
    (transactionProjection !== undefined &&
      transactionProjection !== identity.projectionVersion) ||
    identity.lastRebaselineTimestamp >
      (transaction.kind === "rebaseline" ? latestClock : earliestClock) ||
    identity.currentCommitment !== transaction.commitment ||
    identity.lastResetTimestamp >= earliestClock ||
    identity.lastVerificationTimestamp < earliestClock ||
    identity.creationTimestamp > latestClock
  )
    return { status: "invalid", reason: "transaction_invalid" };
  if (
    identity.verificationCount === 0 &&
    !(
      transaction.kind === "mint" &&
      identity.trustScore === 0 &&
      identity.creationTimestamp === identity.lastVerificationTimestamp &&
      identity.lastVerificationTimestamp >= earliestClock &&
      identity.lastVerificationTimestamp <= latestClock &&
      identity.lastResetTimestamp === 0 &&
      identity.lastRebaselineTimestamp === 0
    )
  )
    return { status: "invalid", reason: "identity_invalid" };
  if (identity.lastVerificationTimestamp > now) {
    onFutureTime("identity", identity.lastVerificationTimestamp);
    return { status: "invalid", reason: "identity_invalid" };
  }
  const [attestationPda] = PublicKey.findProgramAddressSync(
    [
      new TextEncoder().encode("attestation"),
      new PublicKey(SAS_CONFIG.entrosCredentialPda).toBytes(),
      new PublicKey(SAS_CONFIG.entrosSchemaPda).toBytes(),
      wallet.toBytes(),
    ],
    new PublicKey(SAS_CONFIG.programId),
  );
  const anchorProgram = new PublicKey(PROGRAM_IDS.entrosAnchor);
  const [migrationPda] = PublicKey.findProgramAddressSync(
    migrationRecordSeeds(wallet),
    anchorProgram,
  );
  let attestation: IntegratorAttestationEvidence;
  try {
    // The migration record is read beside the attestation, in parallel and at
    // or after the identity's slot, so the read takes no longer than before.
    const [account, migration] = await Promise.all([
      boundedRpc(
        connection.getAccountInfoAndContext(attestationPda, {
          commitment: "confirmed",
          minContextSlot: snapshot.context.slot,
        }),
      ),
      boundedRpc(
        connection.getAccountInfoAndContext(migrationPda, {
          commitment: "confirmed",
          minContextSlot: snapshot.context.slot,
        }),
      ),
    ]);
    now = readNow();
    if (now === null) return { status: "invalid", reason: "invalid_request" };
    attestation =
      integer(account.context.slot, snapshot.context.slot) &&
      integer(migration.context.slot, snapshot.context.slot)
        ? await strictAttestation(
            account.value,
            wallet,
            attestationPda,
            account.context.slot,
            now,
            {
              creationTimestamp: identity.creationTimestamp,
              lastResetTimestamp: identity.lastResetTimestamp,
              migration: readMigrationRecord(migration.value, anchorProgram),
            },
            (verifiedAt) => onFutureTime("attestation", verifiedAt),
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

/** Waits for the device clock to reach a chain time, within one budget shared by every wait of a read. */
async function waitForChainClock(
  target: number,
  readNow: () => number | null,
  startedAt: number,
): Promise<"ready" | "timeout" | "invalid"> {
  const budgetMs = CLOCK_CATCH_UP_SECONDS * 1000;
  while (true) {
    const now = readNow();
    if (now === null) return "invalid";
    const remainingMs = budgetMs - (performance.now() - startedAt);
    if (remainingMs < 0 || (target - now) * 1000 > remainingMs)
      return "timeout";
    if (now >= target) return "ready";
    await new Promise<void>((resolve) => {
      setTimeout(resolve, Math.min(remainingMs, (target - now) * 1000));
    });
  }
}

/** Read confirmed application evidence without submitting a transaction or changing protocol policy. */
export async function readIntegratorEvidence(
  input: ReadIntegratorEvidenceInput,
): Promise<IntegratorEvidenceReadResult> {
  try {
    if (resolveDeployment(input.deployment).isolated) return { status: "invalid", reason: "unsupported_deployment" };
  } catch {
    return { status: "invalid", reason: "invalid_request" };
  }
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
  // A device clock behind the chain makes fresh evidence look future-dated.
  // Each chain time gets one wait, so a moving target cannot extend the read.
  const waited = new Set<ChainTimeSource>();
  let clockStartedAt: number | null = null;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const pending: { source: ChainTimeSource; time: number }[] = [];
      const result = await readOnce(input, readNow, (source, time) => {
        pending.push({ source, time });
      });
      const future = pending[0];
      if (
        future !== undefined &&
        typeof input.nowSeconds === "function" &&
        !waited.has(future.source)
      ) {
        waited.add(future.source);
        clockStartedAt ??= performance.now();
        const clock = await waitForChainClock(
          future.time,
          readNow,
          clockStartedAt,
        );
        if (clock === "invalid")
          return { status: "invalid", reason: "invalid_request" };
        if (clock === "ready") continue;
        return result;
      }
      if (result.status === "invalid") return result;
      if (result.status !== "unavailable" || propagationRetried) return result;
      propagationRetried = true;
    } catch {
      return { status: "unavailable", reason: "rpc_unavailable" };
    }
  }
  return { status: "unavailable", reason: "rpc_unavailable" };
}

export type VerificationTransactionSearchResult =
  | { status: "found"; signature: string }
  | { status: "none" | "invalid" | "unavailable" };

/** Signatures one page of the history holds. The RPC serves no more. */
const SEARCH_PAGE_SIZE = 1_000;
/** Pages a search reads before it reports that it could not finish. */
const SEARCH_MAX_PAGES = 5;
/** Transactions a search fetches inside the window of the verification time. */
const SEARCH_MAX_CANDIDATES = 50;
/** Candidate signatures a search checks before it reads the history. */
const MAX_CANDIDATE_SIGNATURES = 10;
const SIGNATURE_PATTERN = /^[1-9A-HJ-NP-Za-km-z]{64,88}$/;

/**
 * Finds the newest confirmed Entros transaction that qualifies as verification evidence for a
 * wallet. Pass the signature to `readIntegratorEvidence`, which checks it against current state.
 *
 * Pass `verifiedAt`, the identity account's last verification time. The history of an identity
 * account lists every transaction that names it, and any party can send such a transaction.
 * Without the time, the search reads the newest `limit` signatures alone, so later
 * transactions can push the verification out of reach.
 *
 * The search stops at its bounds, so enough later transactions can still hide the verification
 * from it. Pass `candidateSignatures` with `verifiedAt` when the signature may already be known,
 * for example from the client that submitted the verification. Each candidate is checked first,
 * with the checks the search applies to a transaction it finds, and the first that is the
 * verification `verifiedAt` names is returned. The search runs when none is.
 */
export async function findLatestVerificationTransaction(input: {
  walletPubkey: string;
  connection: Pick<Connection, "getSignaturesForAddress" | "getParsedTransaction">;
  /** Without `verifiedAt`: signatures to inspect, newest first. From 1 to 25, default 10. */
  limit?: number;
  /**
   * The identity's last verification time in seconds. The search pages back to it and
   * fetches only the transactions whose block time lies within the tolerance of it.
   */
  verifiedAt?: number;
  /**
   * Signatures to check before the search, up to 10. Needs `verifiedAt`. A candidate counts
   * only when it is a qualifying transaction whose block time lies within the tolerance of
   * `verifiedAt`. Candidates are untrusted, so one that does not count changes nothing.
   */
  candidateSignatures?: readonly string[];
}): Promise<VerificationTransactionSearchResult> {
  const limit = input.limit ?? 10;
  if (!integer(limit, 1, 25)) return { status: "invalid" };
  const { verifiedAt, candidateSignatures = [] } = input;
  if (verifiedAt !== undefined && !integer(verifiedAt, 1)) return { status: "invalid" };
  if (
    !Array.isArray(candidateSignatures) ||
    candidateSignatures.length > MAX_CANDIDATE_SIGNATURES ||
    (candidateSignatures.length > 0 && verifiedAt === undefined) ||
    !candidateSignatures.every(
      (signature) => typeof signature === "string" && SIGNATURE_PATTERN.test(signature),
    )
  )
    return { status: "invalid" };
  const { PublicKey } = await import("@solana/web3.js");
  let wallet: PublicKey;
  try {
    wallet = new PublicKey(input.walletPubkey);
    if (wallet.toBase58() !== input.walletPubkey) return { status: "invalid" };
  } catch {
    return { status: "invalid" };
  }
  const [identityPda] = PublicKey.findProgramAddressSync(
    [new TextEncoder().encode("identity"), wallet.toBytes()],
    new PublicKey(PROGRAM_IDS.entrosAnchor),
  );
  // A qualifying transaction has succeeded and has a block time.
  const qualifyingAt = async (signature: string): Promise<number | null> => {
    const parsed = await boundedRpc(
      input.connection.getParsedTransaction(signature, {
        commitment: "confirmed",
        maxSupportedTransactionVersion: 0,
      }),
    );
    if (parsed === null) return null;
    const qualifying = await qualifyingTransaction(parsed, signature, wallet, identityPda);
    return qualifying === null ? null : qualifying.blockTime;
  };
  const qualifies = async (signature: string): Promise<boolean> =>
    (await qualifyingAt(signature)) !== null;
  if (verifiedAt !== undefined) {
    for (const signature of candidateSignatures) {
      try {
        const blockTime = await qualifyingAt(signature);
        if (
          blockTime !== null &&
          Math.abs(blockTime - verifiedAt) <= BLOCK_TIME_TOLERANCE_SECONDS
        )
          return { status: "found", signature };
      } catch {
        // A candidate that cannot be read is not evidence. The search decides.
      }
    }
  }
  try {
    if (verifiedAt === undefined) {
      const signatures = await boundedRpc(
        input.connection.getSignaturesForAddress(identityPda, { limit }, "confirmed"),
      );
      for (const entry of signatures.slice(0, limit)) {
        if (entry.err === null && (await qualifies(entry.signature)))
          return { status: "found", signature: entry.signature };
      }
      return { status: "none" };
    }
    // The program wrote its clock into the identity. The block time of that
    // transaction lies within the tolerance of it, so nothing else is fetched.
    const earliest = verifiedAt - BLOCK_TIME_TOLERANCE_SECONDS;
    const latest = verifiedAt + BLOCK_TIME_TOLERANCE_SECONDS;
    let before: string | undefined;
    let candidates = 0;
    for (let page = 0; page < SEARCH_MAX_PAGES; page += 1) {
      const signatures = await boundedRpc(
        input.connection.getSignaturesForAddress(
          identityPda,
          { limit: SEARCH_PAGE_SIZE, before },
          "confirmed",
        ),
      );
      for (const entry of signatures) {
        if (entry.blockTime == null) continue;
        // The history runs newest first, so everything after this entry is older still.
        if (entry.blockTime < earliest) return { status: "none" };
        if (entry.blockTime > latest || entry.err !== null) continue;
        if (candidates >= SEARCH_MAX_CANDIDATES) return { status: "unavailable" };
        candidates += 1;
        if (await qualifies(entry.signature))
          return { status: "found", signature: entry.signature };
      }
      const last = signatures[signatures.length - 1];
      if (signatures.length < SEARCH_PAGE_SIZE || last === undefined) return { status: "none" };
      before = last.signature;
    }
    return { status: "unavailable" };
  } catch {
    return { status: "unavailable" };
  }
}
