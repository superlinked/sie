import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { endpointFor } from "../common";
import { checkFile, manifestFiles } from "../fetch";

describe("endpoint credentials", () => {
  it("shares the common key only with the same origin", () => {
    expect(
      endpointFor("llm", {
        SIE_API_KEY: "test-only-key",
        SIE_LLM_BASE_URL: "http://localhost:8080",
      }).apiKey,
    ).toBeUndefined();
    expect(endpointFor("llm", { SIE_API_KEY: "test-only-key" }).apiKey).toBe("test-only-key");
    expect(
      endpointFor("llm", { SIE_API_KEY: "test-only-key", SIE_LLM_API_KEY: "" }).apiKey,
    ).toBeUndefined();
    expect(
      endpointFor("llm", {
        SIE_API_KEY: "test-only-key",
        SIE_LLM_BASE_URL: "https://different.example",
        SIE_LLM_API_KEY: "role-key",
      }).apiKey,
    ).toBe("role-key");
  });
  it("rejects credentials in URLs before writing configuration artifacts", () => {
    expect(() => endpointFor("llm", { SIE_LLM_BASE_URL: "https://secret@example.org" })).toThrow();
  });
});
it("rejects an unpinned manifest and modified downloaded bytes", () => {
  expect(() => manifestFiles(Buffer.from('{"files":{}}'))).toThrow(/checksum/);
  const body = Buffer.from("source");
  const expected = { bytes: body.length, sha256: createHash("sha256").update(body).digest("hex") };
  expect(() => checkFile(body, expected)).not.toThrow();
  expect(() => checkFile(Buffer.from("tamper"), expected)).toThrow();
});
