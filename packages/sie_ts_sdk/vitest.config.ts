import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    // Include all tests EXCEPT integration and browser tests (those run separately)
    include: ["tests/**/*.test.ts"],
    exclude: ["tests/integration/**", "tests/browser/**"],
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
      include: ["src/**/*.ts"],
      exclude: ["src/**/*.d.ts"],
    },
    testTimeout: 10000,
  },
});
