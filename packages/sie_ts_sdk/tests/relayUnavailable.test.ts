/**
 * SIE Cloud refusals to serve through SIE Relay surface as `RelayUnavailableError`.
 *
 * SIE Cloud answers `403` with `THIRD_PARTY_SERVING_DISABLED`,
 * `THIRD_PARTY_RUNTIME_UNLISTED` or `THIRD_PARTY_REGION_UNAVAILABLE` when it
 * will not serve a request through SIE Relay, its partner capacity. The region
 * refusal also names `key_region` and `deployment_region`. The refusal is
 * terminal: the SDK throws it on the first response. Mirrors the Python SDK's
 * `test_relay_unavailable.py`.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import {
  AccountInactiveError,
  RelayUnavailableError,
  RequestError,
  SIEError,
} from "../src/errors.js";
import { handleError } from "../src/internal/parsing.js";
import { packMessage } from "../src/msgpack.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

const RELAY_CODES = [
  "THIRD_PARTY_SERVING_DISABLED",
  "THIRD_PARTY_RUNTIME_UNLISTED",
  "THIRD_PARTY_REGION_UNAVAILABLE",
] as const;

function refusal(
  code: string,
  fields: Record<string, unknown> = {},
  status = 403,
  headers: Record<string, string> = {},
): Response {
  return new Response(
    JSON.stringify({ detail: { code, message: `refused: ${code}`, ...fields } }),
    {
      status,
      headers: { "Content-Type": "application/json", ...headers },
    },
  );
}

function encodeSuccess(): Response {
  return new Response(packMessage({ items: [{ dense: { values: new Float32Array([0.1]) } }] }), {
    status: 200,
    headers: { "Content-Type": "application/msgpack" },
  });
}

async function thrown(promise: Promise<unknown>): Promise<unknown> {
  return promise.then(
    () => {
      throw new Error("expected a rejection");
    },
    (error: unknown) => error,
  );
}

describe("handleError maps SIE Relay refusals", () => {
  it.each(RELAY_CODES)("maps 403 %s to a typed RequestError", async (code) => {
    const error = await thrown(handleError(refusal(code)));

    expect(error).toBeInstanceOf(RelayUnavailableError);
    expect(error).toBeInstanceOf(RequestError);
    expect(error).toBeInstanceOf(SIEError);
    expect(error).toMatchObject({
      name: "RelayUnavailableError",
      code,
      statusCode: 403,
      message: `refused: ${code}`,
      keyRegion: undefined,
      deploymentRegion: undefined,
    });
  });

  it("carries both regions of a region refusal", async () => {
    const error = await thrown(
      handleError(
        refusal("THIRD_PARTY_REGION_UNAVAILABLE", { key_region: "eu", deployment_region: "us" }),
      ),
    );

    expect(error).toMatchObject({ keyRegion: "eu", deploymentRegion: "us" });
  });

  it("reads a key without a region as undefined", async () => {
    const error = await thrown(
      handleError(
        refusal("THIRD_PARTY_REGION_UNAVAILABLE", { key_region: null, deployment_region: "us" }),
      ),
    );

    expect(error).toMatchObject({ keyRegion: undefined, deploymentRegion: "us" });
  });

  it("drops regions that are not strings", async () => {
    const error = await thrown(
      handleError(
        refusal("THIRD_PARTY_REGION_UNAVAILABLE", { key_region: 7, deployment_region: ["us"] }),
      ),
    );

    expect(error).toMatchObject({ keyRegion: undefined, deploymentRegion: undefined });
  });

  it("identifies a relay refusal in the OpenAI envelope by X-SIE-Error-Code", async () => {
    const response = new Response(
      JSON.stringify({
        error: { message: "refused", type: "invalid_request_error", code: "permission_denied" },
      }),
      {
        status: 403,
        headers: {
          "Content-Type": "application/json",
          "X-SIE-Error-Code": "THIRD_PARTY_SERVING_DISABLED",
          "X-SIE-Request-ID": "req-relay-1",
        },
      },
    );

    const error = await thrown(handleError(response));

    expect(error).toBeInstanceOf(RelayUnavailableError);
    expect(error).toMatchObject({
      code: "THIRD_PARTY_SERVING_DISABLED",
      message: "refused",
      requestId: "req-relay-1",
      request: { id: "req-relay-1" },
    });
  });

  it.each([
    [403, "INVALID_KEY"],
    [403, "ACCOUNT_SUSPENDED"],
    [400, "THIRD_PARTY_SERVING_DISABLED"],
  ] as const)("leaves %s %s alone", async (status, code) => {
    const error = await thrown(handleError(refusal(code, {}, status)));

    expect(error).toBeInstanceOf(RequestError);
    expect(error).not.toBeInstanceOf(RelayUnavailableError);
    expect(error instanceof AccountInactiveError).toBe(code === "ACCOUNT_SUSPENDED");
  });
});

describe("the client throws a relay refusal on the first response", () => {
  beforeEach(() => mockFetch.mockReset());
  afterEach(() => vi.clearAllMocks());

  it.each(RELAY_CODES)("encode: %s", async (code) => {
    const client = new SIEClient("http://localhost:8080");
    mockFetch
      .mockResolvedValueOnce(refusal(code, { deployment_region: "us" }))
      .mockResolvedValueOnce(encodeSuccess());

    const error = await thrown(client.encode("bge-m3", { text: "hi" }));

    expect(error).toBeInstanceOf(RelayUnavailableError);
    expect(error).toMatchObject({ code, deploymentRegion: "us" });
    expect(mockFetch).toHaveBeenCalledTimes(1);
    await client.close();
  });

  it.each(RELAY_CODES)("generate: %s", async (code) => {
    const client = new SIEClient("http://localhost:8080");
    mockFetch.mockResolvedValue(refusal(code, { key_region: "eu", deployment_region: "us" }));

    const error = await thrown(client.generate("m", "hi", { maxNewTokens: 4 }));

    expect(error).toBeInstanceOf(RelayUnavailableError);
    expect(error).toMatchObject({ code, keyRegion: "eu", deploymentRegion: "us" });
    expect(mockFetch).toHaveBeenCalledTimes(1);
    await client.close();
  });
});
