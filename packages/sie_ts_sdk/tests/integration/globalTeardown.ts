/**
 * Global teardown for integration tests.
 *
 * Stops the SIE server after running integration tests.
 */

import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { SERVER_INFO_FILE, type ServerInfo } from "./globalSetup.js";

export default async function globalTeardown(): Promise<void> {
  // Read server info from file
  if (!existsSync(SERVER_INFO_FILE)) {
    console.log("\n⚠️ No server info file found - server may not have started\n");
    return;
  }

  let serverInfo: ServerInfo;
  try {
    serverInfo = JSON.parse(readFileSync(SERVER_INFO_FILE, "utf-8"));
  } catch {
    console.log("\n⚠️ Failed to read server info file\n");
    return;
  }

  // Server with pid=0 was pre-existing - we didn't start it, so don't stop it
  if (serverInfo.pid === 0) {
    console.log("\nℹ️ Server was pre-existing, not stopping it\n");
    // Clean up server info file
    try {
      unlinkSync(SERVER_INFO_FILE);
    } catch {
      // Ignore errors
    }
    return;
  }

  // Stop the server we started
  if (serverInfo.pid > 0) {
    console.log(`\n🛑 Stopping SIE server (PID: ${serverInfo.pid})...`);

    try {
      // Send SIGTERM first
      process.kill(serverInfo.pid, "SIGTERM");

      // Wait for graceful shutdown (10s)
      await new Promise<void>((resolve) => {
        let attempts = 0;
        const checkInterval = setInterval(() => {
          attempts++;
          try {
            // Signal 0 checks if process exists without killing
            process.kill(serverInfo.pid, 0);
            if (attempts >= 20) {
              // 10s elapsed, force kill
              console.log("   Forcing kill...");
              try {
                process.kill(serverInfo.pid, "SIGKILL");
              } catch {
                // Process may have already died
              }
              clearInterval(checkInterval);
              resolve();
            }
          } catch {
            // Process no longer exists
            clearInterval(checkInterval);
            resolve();
          }
        }, 500);
      });

      console.log("✅ Server stopped\n");
    } catch {
      // Process may have already died
      console.log("⚠️ Server process not found (may have already stopped)\n");
    }
  }

  // Clean up server info file
  try {
    unlinkSync(SERVER_INFO_FILE);
  } catch {
    // Ignore errors
  }
}
