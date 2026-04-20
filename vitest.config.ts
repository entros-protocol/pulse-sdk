import { defineConfig } from "vitest/config";

// Mirror tsup's `define` so the build-time constant resolves during source-level
// tests. Default to false (production behaviour, throw test runs); flip with
// IAM_INTERNAL_TEST=1 to exercise the injection path against the real pipeline.
export default defineConfig({
  test: {
    globals: true,
    testTimeout: 60000,
  },
  define: {
    __IAM_INTERNAL_TEST__: process.env.IAM_INTERNAL_TEST === "1" ? "true" : "false",
  },
});
