import { afterEach, describe, expect, it, vi } from "vitest";
import { SIEClient } from "../src/client.js";
import { unpackMessage } from "../src/msgpack.js";

const fetchMock = vi.fn();
vi.stubGlobal("fetch", fetchMock);
afterEach(() => fetchMock.mockReset());

function refusal(): Response {
  return new Response(
    JSON.stringify({ detail: { code: "INVALID_INPUT", message: "test transport" } }),
    {
      status: 400,
      headers: { "content-type": "application/json" },
    },
  );
}

function nativeCall(client: SIEClient, kind: string, adapterOptions: Record<string, unknown>) {
  if (kind === "score")
    return client.score("m", { text: "q" }, [{ text: "x" }], { adapterOptions });
  if (kind === "systemOne")
    return client.systemOne("m", "x", { yes: { type: "noul" } }, { adapterOptions });
  if (kind === "extract") return client.extract("m", { text: "x" }, { adapterOptions });
  return client.encode("m", { text: "x" }, { adapterOptions });
}

describe("LoRA selector boundaries", () => {
  for (const kind of ["encode", "extract", "score", "systemOne"]) {
    for (const selected of [undefined, null, "legal"]) {
      it.each(["lora_adapter", "lora", "lora_id"])(
        `${kind} omits undefined %s alongside ${selected}`,
        async (key) => {
          fetchMock.mockResolvedValueOnce(refusal());
          const other = key === "lora_adapter" ? "lora" : "lora_adapter";
          const options = Object.freeze({
            [key]: undefined,
            [other]: selected,
            profile: "served",
            unrelated: undefined,
          });
          const client = new SIEClient("http://localhost:8080");
          await expect(nativeCall(client, kind, options)).rejects.toThrow();
          expect(fetchMock).toHaveBeenCalledOnce();
          const body = unpackMessage(fetchMock.mock.calls[0]?.[1].body) as {
            options: Record<string, unknown>;
            params: { options: Record<string, unknown> };
          };
          const actual = kind === "score" ? body.options : body.params.options;
          expect(actual).not.toHaveProperty(key);
          if (selected === undefined) expect(actual).not.toHaveProperty(other);
          else expect(actual[other]).toBe(selected);
          expect(actual.profile).toBe("served");
          // Only selector omission changes; unrelated MessagePack fields retain their semantics.
          expect(actual.unrelated).toBeNull();
          expect(Object.hasOwn(options, key)).toBe(true);
          expect(options[key]).toBeUndefined();
          expect(options[other]).toBe(selected);
          await client.close();
        },
      );
    }

    it.each(["lora_adapter", "lora", "lora_id"])(
      `${kind} preserves explicit null via %s`,
      async (key) => {
        fetchMock.mockResolvedValueOnce(refusal());
        const options = { [key]: null };
        const client = new SIEClient("http://localhost:8080");
        await expect(nativeCall(client, kind, options)).rejects.toThrow();
        expect(fetchMock).toHaveBeenCalledOnce();
        const body = unpackMessage(fetchMock.mock.calls[0]?.[1].body) as {
          options: Record<string, unknown>;
          params: { options: Record<string, unknown> };
        };
        const actual = kind === "score" ? body.options : body.params.options;
        expect(actual[key]).toBeNull();
        expect(options).toEqual({ [key]: null });
        await client.close();
      },
    );

    it.each([
      { lora_adapter: null, lora: "legal" },
      { lora_adapter: "legal", lora_id: "other" },
      { lora_adapter: false },
      { lora_id: " " },
    ])(`${kind} rejects ambiguity before fetch: %j`, async (options) => {
      const client = new SIEClient("http://localhost:8080");
      await expect(nativeCall(client, kind, options)).rejects.toThrow(TypeError);
      expect(fetchMock).not.toHaveBeenCalled();
      await client.close();
    });
  }

  it.each([undefined, null, "legal"])(
    "generation preserves canonical selector %s",
    async (selection) => {
      fetchMock.mockResolvedValueOnce(refusal());
      const client = new SIEClient("http://localhost:8080");
      await expect(
        client.generate("m", "hi", { maxNewTokens: 1, loraAdapter: selection }),
      ).rejects.toThrow();
      const body = JSON.parse(fetchMock.mock.calls[0]?.[1].body);
      if (selection === undefined) expect(body).not.toHaveProperty("lora_adapter");
      else expect(body.lora_adapter).toBe(selection);
      expect(body).not.toHaveProperty("lora");
      expect(body).not.toHaveProperty("lora_id");
      await client.close();
    },
  );

  it("chat preserves explicit canonical null", async () => {
    fetchMock.mockResolvedValueOnce(refusal());
    const client = new SIEClient("http://localhost:8080");
    await expect(
      client.chatCompletions({ model: "m", messages: [], lora_adapter: null }),
    ).rejects.toThrow();
    expect(JSON.parse(fetchMock.mock.calls[0]?.[1].body).lora_adapter).toBeNull();
    await client.close();
  });
});
