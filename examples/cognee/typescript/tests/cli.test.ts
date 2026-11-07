import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, expect, it } from "vitest";
import { documentPage, learningRun } from "../cli";
import { namespaceFor, readInput } from "../common";
import { UserError, formatError } from "../errors";

const roots: string[] = [];
afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

it("shows authored guidance while suppressing SDK messages, names and causes", () => {
  expect(formatError(new UserError("Run pnpm fetch-data first."))).toBe(
    "Error: Run pnpm fetch-data first.",
  );
  const sdkError = new Error("Bearer test-secret", { cause: { apiKey: "test-secret" } });
  sdkError.name = "https://example.org?key=test-secret";
  expect(formatError(sdkError)).toBe(
    "Error: The example did not complete. Check the configured model routes, input files and storage settings.",
  );
  expect(formatError("test-secret")).not.toContain("test-secret");
});
it("reports missing sample inputs and invalid namespaces with actionable safe messages", () => {
  expect(() => readInput("missing-test-input.txt")).toThrow(UserError);
  expect(() => readInput("missing-test-input.txt")).toThrow(/pnpm fetch-data/);
  expect(() => namespaceFor({ SIE_COGNEE_NAMESPACE: "../private-value" })).toThrow(
    /SIE_COGNEE_NAMESPACE/,
  );
  expect(namespaceFor({})).toBe("example-v1");
});
it("validates page arguments, missing inputs and size before model access", () => {
  const root = mkdtempSync(join(tmpdir(), "sie-cognee-cli-"));
  roots.push(root);
  const page = join(root, "page.png");
  expect(() => documentPage(["one.png", "two.png"], page, 10)).toThrow(/at most one/);
  expect(() => documentPage(["file.pdf"], page, 10)).toThrow(/render PDF/);
  expect(() => documentPage([], page, 10)).toThrow(/pnpm fetch-data/);
  writeFileSync(page, "example raster bytes");
  expect(() => documentPage([], page, 2)).toThrow(/documentBytes/);
  expect(documentPage(["--", page], page, 100).image.toString()).toBe("example raster bytes");
});
it("accepts teaching and verification usage while rejecting incomplete arguments", () => {
  expect(learningRun([])).toBeUndefined();
  expect(learningRun(["--", "--verify-run", "runs/example"])).toBe("runs/example");
  for (const args of [
    ["--verify-run"],
    ["other"],
    ["--verify-run", ""],
    ["--verify-run", "run", "extra"],
  ])
    expect(() => learningRun(args)).toThrow(/Use pnpm learn/);
});
it.each([
  ["02_read_documents.js", ["one.png", "two.png"], "Supply at most one raster page."],
  ["04_learn.js", ["--verify-run"], "Use pnpm learn, or pnpm learn -- --verify-run PATH."],
])(
  "reports invalid %s usage before connecting to an inference endpoint",
  (script, args, message) => {
    const result = spawnSync(process.execPath, [resolve("dist", script), ...args], {
      env: {
        ...process.env,
        SIE_API_KEY: "",
        SIE_BASE_URL: "http://127.0.0.1:1",
        SIE_COGNEE_NAMESPACE: "cli-check",
      },
      encoding: "utf8",
      timeout: 5000,
    });
    expect(result.error).toBeUndefined();
    expect(result.status).toBe(1);
    expect(result.stderr.trim()).toBe(`Error: ${message}`);
    expect(result.stdout).toBe("");
  },
);
