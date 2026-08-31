import type { PublicKey } from "@solana/web3.js";

export const ASSOCIATED_TOKEN_PROGRAM_ADDRESS =
  "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL";

export const TOKEN_2022_PROGRAM_ADDRESS =
  "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb";

export function deriveToken2022AssociatedAddress(
  mint: PublicKey,
  owner: PublicKey,
  PublicKeyConstructor: typeof import("@solana/web3.js").PublicKey,
): PublicKey {
  if (!PublicKeyConstructor.isOnCurve(owner.toBytes())) {
    throw new Error("Associated token account owner must be on curve");
  }

  const tokenProgramId = new PublicKeyConstructor(TOKEN_2022_PROGRAM_ADDRESS);
  const associatedTokenProgramId = new PublicKeyConstructor(
    ASSOCIATED_TOKEN_PROGRAM_ADDRESS,
  );

  return PublicKeyConstructor.findProgramAddressSync(
    [owner.toBuffer(), tokenProgramId.toBuffer(), mint.toBuffer()],
    associatedTokenProgramId,
  )[0];
}
