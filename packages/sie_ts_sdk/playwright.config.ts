import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for browser compatibility tests.
 *
 * Tests that the SDK works correctly in browser environments:
 * - No Node.js-specific APIs are used
 * - fetch, msgpack, and typed arrays work correctly
 * - SDK can communicate with SIE server from browser
 */
export default defineConfig({
  testDir: "./tests/browser",
  testMatch: "**/*.browser.test.ts",

  // Run tests in parallel
  fullyParallel: true,

  // Fail the build on CI if you accidentally left test.only in the source code
  forbidOnly: !!process.env.CI,

  // Retry on CI only
  retries: process.env.CI ? 2 : 0,

  // Reporter
  reporter: process.env.CI ? "github" : "list",

  // Shared settings for all projects
  use: {
    // Base URL for test server
    baseURL: "http://localhost:3456",

    // Collect trace when retrying the failed test
    trace: "on-first-retry",
  },

  // Configure projects for major browsers
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    // Uncomment to test additional browsers:
    // {
    //   name: "firefox",
    //   use: { ...devices["Desktop Firefox"] },
    // },
    // {
    //   name: "webkit",
    //   use: { ...devices["Desktop Safari"] },
    // },
  ],

  // Run local test server before starting the tests
  webServer: {
    command: "pnpm run test:browser:serve",
    url: "http://localhost:3456",
    reuseExistingServer: !process.env.CI,
    timeout: 10000,
  },
});
