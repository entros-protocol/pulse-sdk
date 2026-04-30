// Bundled Anchor IDLs for the on-chain programs the SDK constructs
// instructions against. Copied verbatim from `protocol-core/target/idl/*.json`
// so the on-chain instruction shape is locked at compile time and no
// runtime IDL fetch (Anchor's `Program.fetchIdl` makes an RPC roundtrip
// per call) is needed during submit or read flows.
//
// IDL bundling is the right call here because:
//   - Anchor 0.32 IDLs embed the `address` field, so consumers don't
//     need a separate program-ID arg to `new anchor.Program(idl, provider)`.
//   - Each IDL is ~10–25 KB, ~50 KB total — small impact on the SDK
//     bundle relative to the per-call RPC saved.
//   - Avoids a per-submit / per-read `Program.fetchIdl()` RPC roundtrip
//     (~150–300ms saved each time).
//   - Deterministic across deployments: SDK and on-chain shapes track
//     together via SDK release rather than via runtime fetch.
//
// When the on-chain programs change, re-copy the JSON files from
// `protocol-core/target/idl/` and bump the SDK minor version. Consumers
// of older SDKs targeting newer on-chain programs would see Anchor
// instruction encoding mismatches anyway, so this coupling is already
// load-bearing — bundling just makes it explicit.

import entrosAnchorIdl from "./entros_anchor.json";
import entrosVerifierIdl from "./entros_verifier.json";

export { entrosAnchorIdl, entrosVerifierIdl };
