/**
 * Vitest configuration for integration tests.
 *
 * Integration tests run against a real SIE server.
 * The server is started automatically via globalSetup.
 *
 * Usage:
 *   mise run ts test-integration
 *   # or directly:
 *   pnpm test:integration
 */

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Only run integration tests
    include: ["tests/integration/**/*.integration.test.ts"],

    // Global setup/teardown for server lifecycle
    globalSetup: ["./tests/integration/globalSetup.ts"],
    globalTeardown: ["./tests/integration/globalTeardown.ts"],

    // Longer timeouts for server operations
    testTimeout: 60_000, // 60s per test
    hookTimeout: 200_000, // 200s for setup/teardown (server startup)

    // Run tests sequentially (server is shared)
    pool: "forks",
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },

    // Reporter
    reporters: ["verbose"],
  },
});
