import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["tests/integration/**/*.integration.test.ts"],
    globalSetup: "../../packages/sie_ts_sdk/tests/integration/globalSetup.ts",
    globalTeardown: "../../packages/sie_ts_sdk/tests/integration/globalTeardown.ts",
    testTimeout: 60_000,
    hookTimeout: 200_000,
    pool: "forks",
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
  },
});
