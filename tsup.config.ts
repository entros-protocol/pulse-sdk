import { defineConfig } from "tsup";

// Internal builds enable the harness test hooks (`PulseSession.__injectSensorData`)
// and emit to dist-internal/. Default builds emit to dist/ with the hooks short-
// circuiting to throw. `package.json#files` only ships dist/, so internal builds
// are structurally excluded from npm tarballs even if both directories exist.
const isInternalBuild = process.env.IAM_INTERNAL_TEST === "1";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: { compilerOptions: { stripInternal: true } },
  splitting: false,
  sourcemap: true,
  clean: true,
  define: {
    __IAM_INTERNAL_TEST__: isInternalBuild ? "true" : "false",
  },
  outDir: isInternalBuild ? "dist-internal" : "dist",
});
