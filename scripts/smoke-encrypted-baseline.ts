/**
 * Local-validator smoke for master-list #98.
 *
 * Exercises the pulse-sdk's Anchor TS instruction builder against the live
 * entros_anchor binary preloaded on solana-test-validator at the canonical
 * pubkey. The instruction is constructed VIA THE SDK'S BUILDER (not by hand),
 * so this proves: discriminator + arg encoding + account ordering + PDA
 * seeds all match between the SDK and the deployed program.
 *
 * Expected outcome: a `Custom(6022)` IdentityStateNotFound error from the
 * handler's `require!(identity_state.data_len() > 0)` guard. The 6022 means
 * the program decoded the instruction successfully and rejected only at the
 * intended business-logic check.
 */
import * as anchor from "@coral-xyz/anchor";
import {
  ComputeBudgetProgram,
  Connection,
  Keypair,
  PublicKey,
  SystemProgram,
  Transaction,
} from "@solana/web3.js";
import * as fs from "node:fs";

async function main() {
const PROGRAM_ID = new PublicKey("GZYwTp2ozeuRA5Gof9vs4ya961aANcJBdUzB7LN6q4b2");
const RPC_URL = process.env.SMOKE_RPC_URL ?? "http://127.0.0.1:8899";

const entrosAnchorIdl = JSON.parse(
  fs.readFileSync(
    "/Users/johnny/IAM/pulse-sdk/src/protocol/idl/entros_anchor.json",
    "utf8",
  ),
);

const payerData = JSON.parse(
  fs.readFileSync("/tmp/entros-smoke-test/smoke-payer.json", "utf8"),
);
const payerKp = Keypair.fromSecretKey(new Uint8Array(payerData));
const wallet = new anchor.Wallet(payerKp);

const connection = new Connection(RPC_URL, "confirmed");
const provider = new anchor.AnchorProvider(connection, wallet, {
  commitment: "confirmed",
});

const program: any = new anchor.Program(entrosAnchorIdl as any, provider);

const [identityPda] = PublicKey.findProgramAddressSync(
  [new TextEncoder().encode("identity"), payerKp.publicKey.toBuffer()],
  PROGRAM_ID,
);
const [encryptedBaselinePda] = PublicKey.findProgramAddressSync(
  [new TextEncoder().encode("encrypted_baseline"), payerKp.publicKey.toBuffer()],
  PROGRAM_ID,
);

console.log("payer:               ", payerKp.publicKey.toBase58());
console.log("identityPda:         ", identityPda.toBase58());
console.log("encryptedBaselinePda:", encryptedBaselinePda.toBase58());
console.log("");

const blob = new Uint8Array(96);
blob[0] = 0x01;
blob[1] = 0x01;
for (let i = 4; i < 96; i++) blob[i] = i & 0xff;

const ix = await program.methods
  .setEncryptedBaseline(Array.from(blob))
  .accounts({
    authority: payerKp.publicKey,
    identityState: identityPda,
    encryptedBaseline: encryptedBaselinePda,
    systemProgram: SystemProgram.programId,
  })
  .instruction();

console.log("Encoded ix:");
console.log("  programId:    ", ix.programId.toBase58());
console.log("  keys:");
for (const key of ix.keys) {
  console.log(
    `    ${key.pubkey.toBase58()} signer=${key.isSigner} writable=${key.isWritable}`,
  );
}
console.log("  data length:  ", ix.data.length, "bytes (expect 8 disc + 96 blob = 104)");
console.log("  discriminator:", Array.from(ix.data.slice(0, 8)).join(","));
console.log("  expected disc:10,73,41,36,2,145,87,111");
console.log("");

const tx = new Transaction();
tx.add(ComputeBudgetProgram.setComputeUnitLimit({ units: 50_000 }));
tx.add(ix);
tx.feePayer = payerKp.publicKey;
tx.recentBlockhash = (await connection.getLatestBlockhash("confirmed")).blockhash;
tx.sign(payerKp);

const sim = await connection.simulateTransaction(tx, undefined, true);
const logs = sim.value.logs ?? [];
const err = sim.value.err;

console.log("Simulation logs:");
for (const line of logs) console.log("  " + line);
console.log("");

if (err) {
  const errStr = JSON.stringify(err);
  console.log("Simulation err:", errStr);
  if (errStr.includes("6022")) {
    console.log("");
    console.log("✅ SMOKE PASS — program decoded the SDK-encoded ix and rejected at the");
    console.log("   intended handler guard (IdentityStateNotFound, code 6022). Discriminator,");
    console.log("   account ordering, PDA derivation, and arg encoding are all wire-correct.");
    process.exit(0);
  }
  console.log("❌ SMOKE FAIL — unexpected error class. The ix encoding may be wrong.");
  process.exit(1);
}

console.log("❌ SMOKE FAIL — simulation succeeded; expected IdentityStateNotFound (6022).");
process.exit(1);
}

main().catch((err) => {
  console.error("smoke harness threw:", err);
  process.exit(2);
});
