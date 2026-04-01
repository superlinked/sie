/**
 * Vitest configuration for integration tests.
 *
 * Integration tests run against a real SIE server.
 * The server is started automatically via the SDK's globalSetup.
 *
 * Usage:
 *   pnpm test:integration
 */

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Only run integration tests
    include: ["tests/integration/**/*.integration.test.ts"],

    // Use the SDK's global setup/teardown for server lifecycle
    globalSetup: ["../../packages/sie_ts_sdk/tests/integration/globalSetup.ts"],
    globalTeardown: ["../../packages/sie_ts_sdk/tests/integration/globalTeardown.ts"],

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
