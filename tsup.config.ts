import { defineConfig } from "tsup";

// Internal builds enable the test hooks (`PulseSession.__injectSensorData`),
// emit to dist-internal/ and keep source maps. Default builds emit to dist/
// with the hooks short-circuiting to throw, no source maps and no comments.
// Whitespace and syntax are minified but identifiers are not, so stack traces
// stay readable. `package.json#files` only ships dist/, so internal builds are
// structurally excluded from npm tarballs even if both directories exist.
const isInternalBuild = process.env.ENTROS_INTERNAL_TEST === "1";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: { compilerOptions: { stripInternal: !isInternalBuild } },
  splitting: false,
  sourcemap: isInternalBuild,
  minifyWhitespace: !isInternalBuild,
  minifySyntax: !isInternalBuild,
  clean: true,
  define: {
    __ENTROS_INTERNAL_TEST__: isInternalBuild ? "true" : "false",
  },
  outDir: isInternalBuild ? "dist-internal" : "dist",
});
