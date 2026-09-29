/**
 * `SIE_BASE_URL` and `SIE_API_KEY` fallbacks for the client constructor.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { packMessage } from "../src/msgpack.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

async function encodeAuthorization(client: SIEClient): Promise<string | undefined> {
  mockFetch.mockResolvedValueOnce(
    new Response(packMessage({ items: [{ dense: { values: new Float32Array([0.1]) } }] }), {
      status: 200,
      headers: { "Content-Type": "application/msgpack" },
    }),
  );
  await client.encode("bge-m3", { text: "test" });
  const [url, init] = mockFetch.mock.calls[0] ?? [];
  expect(String(url).startsWith(client.getBaseUrl())).toBe(true);
  return (init?.headers as Record<string, string>).Authorization;
}

describe("SIEClient environment fallbacks", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    vi.stubEnv("SIE_BASE_URL", "");
    vi.stubEnv("SIE_API_KEY", "");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("reads the base URL and API key from the environment", async () => {
    vi.stubEnv("SIE_BASE_URL", "https://gateway.example.test/");
    vi.stubEnv("SIE_API_KEY", "env-key");

    const client = new SIEClient();

    expect(client.getBaseUrl()).toBe("https://gateway.example.test");
    expect(await encodeAuthorization(client)).toBe("Bearer env-key");
  });

  it("prefers explicit arguments over the environment", async () => {
    vi.stubEnv("SIE_BASE_URL", "https://env.example.test");
    vi.stubEnv("SIE_API_KEY", "env-key");

    const client = new SIEClient("http://localhost:8080", { apiKey: "explicit-key" });

    expect(client.getBaseUrl()).toBe("http://localhost:8080");
    expect(await encodeAuthorization(client)).toBe("Bearer explicit-key");
  });

  it.each([
    "https://gateway.example.test",
    "https://GATEWAY.example.test:443/v1",
    "https://gateway.example.test/",
  ])("sends SIE_API_KEY to the SIE_BASE_URL origin (%s)", async (baseUrl) => {
    vi.stubEnv("SIE_BASE_URL", "https://gateway.example.test");
    vi.stubEnv("SIE_API_KEY", "env-key");

    expect(await encodeAuthorization(new SIEClient(baseUrl))).toBe("Bearer env-key");
  });

  it.each([
    ["", "https://gateway.example.test"],
    ["https://gateway.example.test", "https://other.example.test"],
    ["https://gateway.example.test", "http://gateway.example.test"],
    ["https://gateway.example.test", "https://gateway.example.test:8443"],
    ["not a url", "https://gateway.example.test"],
  ])(
    "withholds SIE_API_KEY when SIE_BASE_URL is %j and the base URL is %s",
    async (envBaseUrl, baseUrl) => {
      vi.stubEnv("SIE_BASE_URL", envBaseUrl);
      vi.stubEnv("SIE_API_KEY", "env-key");

      expect(await encodeAuthorization(new SIEClient(baseUrl))).toBeUndefined();
    },
  );

  it("requires an explicit key for a controlPlaneUrl on another origin", async () => {
    vi.stubEnv("SIE_BASE_URL", "https://gateway.example.test");
    vi.stubEnv("SIE_API_KEY", "env-key");

    expect(
      () => new SIEClient(undefined, { controlPlaneUrl: "https://control.example.test" }),
    ).toThrow(/controlPlaneUrl/);
    const sameOrigin = new SIEClient(undefined, {
      controlPlaneUrl: "https://gateway.example.test/control",
    });
    expect(await encodeAuthorization(sameOrigin)).toBe("Bearer env-key");
    mockFetch.mockReset();
    const explicit = new SIEClient(undefined, {
      apiKey: "explicit-key",
      controlPlaneUrl: "https://control.example.test",
    });
    expect(await encodeAuthorization(explicit)).toBe("Bearer explicit-key");
  });

  it("treats an unreadable environment as unset", () => {
    const original = globalThis.process.env;
    Object.defineProperty(globalThis.process, "env", {
      configurable: true,
      get() {
        throw new Error("env access denied");
      },
    });
    try {
      expect(() => new SIEClient()).toThrow(/SIE_BASE_URL/);
      expect(new SIEClient("http://localhost:8080").getBaseUrl()).toBe("http://localhost:8080");
    } finally {
      Object.defineProperty(globalThis.process, "env", {
        configurable: true,
        value: original,
        writable: true,
      });
    }
  });

  it("sends no credential when apiKey is an explicit empty string", async () => {
    vi.stubEnv("SIE_BASE_URL", "http://localhost:8080");
    vi.stubEnv("SIE_API_KEY", "env-key");

    const client = new SIEClient("http://localhost:8080", { apiKey: "" });

    expect(await encodeAuthorization(client)).toBeUndefined();
  });

  it("ignores a blank SIE_API_KEY", async () => {
    vi.stubEnv("SIE_BASE_URL", "http://localhost:8080");
    vi.stubEnv("SIE_API_KEY", "   ");

    const client = new SIEClient("http://localhost:8080");

    expect(await encodeAuthorization(client)).toBeUndefined();
  });

  it("requires a base URL when SIE_BASE_URL is unset or blank", () => {
    expect(() => new SIEClient()).toThrow(/SIE_BASE_URL/);
    vi.stubEnv("SIE_BASE_URL", "  ");
    expect(() => new SIEClient()).toThrow(/SIE_BASE_URL/);
  });

  it("validates the base URL read from the environment", () => {
    vi.stubEnv("SIE_BASE_URL", "localhost:8080");

    expect(() => new SIEClient()).toThrow(/must be an absolute http\(s\) URL/);
  });
});
