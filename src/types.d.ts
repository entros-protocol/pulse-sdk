declare module "snarkjs" {
  export const groth16: {
    fullProve(
      input: import("./proof/types").CircuitInput,
      wasmPath: string | Uint8Array,
      zkeyPath: string | Uint8Array,
    ): Promise<{
      proof: import("./proof/types").RawProof;
      publicSignals: string[];
    }>;
    verify(
      vk: Record<string, unknown>,
      publicSignals: string[],
      proof: import("./proof/types").RawProof,
    ): Promise<boolean>;
  };

  export const zKey: {
    exportVerificationKey(
      zkeyPath: string | Uint8Array,
    ): Promise<Record<string, unknown>>;
  };
}
