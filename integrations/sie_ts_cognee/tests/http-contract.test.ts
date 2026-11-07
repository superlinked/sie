import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { expect, it } from "vitest";

it("uses the published Cognee runtime with compatible endpoints and stable source provenance", async () => {
  const root = mkdtempSync(join(tmpdir(), "sie-cognee-http-"));
  const requests: { path: string; body: Record<string, unknown>; authorization?: string }[] = [];
  const server = createServer(async (request, response) => {
    let text = "";
    for await (const chunk of request) text += chunk;
    const body = JSON.parse(text || "{}") as Record<string, unknown>;
    requests.push({ path: request.url || "", body, authorization: request.headers.authorization });
    response.setHeader("content-type", "application/json");
    if (request.url?.endsWith("/embeddings")) {
      const inputs = Array.isArray(body.input) ? body.input : [body.input];
      response.end(
        JSON.stringify({
          object: "list",
          model: body.model,
          data: inputs.map((_, index) => ({ object: "embedding", index, embedding: [1, 0, 0, 0] })),
          usage: { prompt_tokens: 1, total_tokens: 1 },
        }),
      );
    } else {
      response.end(
        JSON.stringify({
          id: "fixture",
          object: "chat.completion",
          created: 1,
          model: body.model,
          choices: [
            {
              index: 0,
              message: {
                role: "assistant",
                content: null,
                tool_calls: [
                  {
                    id: "call",
                    type: "function",
                    function: {
                      name: "extract_structured_data",
                      arguments: JSON.stringify({ nodes: [], edges: [] }),
                    },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
      );
    }
  });
  await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("Missing test address");
  const endpoint = `http://127.0.0.1:${address.port}`;
  const entry = pathToFileURL(resolve("dist/index.js")).href;
  const code = `
    const {createCognee,SourceDocument,SourceMemory}=await import(${JSON.stringify(entry)});
    const settings={llm:{baseUrl:${JSON.stringify(endpoint)},apiKey:'generation-test-key'},embedding:{baseUrl:${JSON.stringify(endpoint)},apiKey:'embedding-test-key'},llmModel:'test/model',embeddingModel:'test/embedding',embeddingDimensions:4,storageRoot:${JSON.stringify(root)},concurrency:1};
    const otherSettings={...settings,llm:{baseUrl:settings.llm.baseUrl,apiKey:'second-generation-key'},embedding:{baseUrl:settings.embedding.baseUrl,apiKey:'second-embedding-key'},llmModel:'test/second',embeddingModel:'test/embedding-second',maxCompletionTokens:4096,storageRoot:settings.storageRoot+'/second'};
    const [cognee,other]=await Promise.all([createCognee(settings),createCognee(otherSettings)]);
    if(process.env.LLM_ARGS!==''||process.env.CACHE_BACKEND!=='redis')throw new Error('Environment was not restored');
    for(const [client,expected] of [[cognee,settings],[other,otherSettings]]) {
      const actual=client.config.get();
      if(actual.llmModel!==expected.llmModel||actual.embeddingModel!==expected.embeddingModel||actual.systemRootDirectory!==expected.storageRoot||actual.cacheBackend!=='fs'||actual.dbProvider!=='sqlite'||actual.storageBackend!=='local'||actual.autoFeedback!==false||actual.enableCaching!==true)throw new Error('Handle settings were not isolated');
    }
    try {
      await cognee.warm();
      const memory=new SourceMemory(cognee,{dataset:'evidence',storageRoot:settings.storageRoot});
      const source=new SourceDocument({text:'https://example.org/literal-note is the authored source text.',sourceId:'SOURCE-01',filename:'note.txt'});
      const added=await memory.add(source);
      await memory.build();
      const chunks=await memory.retrieve('What is the authored source?',3);
      if(chunks.length!==1||chunks[0].documentId!==added.documentId||chunks[0].sources[0].sourceId!=='SOURCE-01'||chunks[0].text!==source.text)throw new Error('Source identity mismatch');
      const next=await memory.add(new SourceDocument({text:'A second source says the microphone failed its check.',sourceId:'SOURCE-02',filename:'second.txt'}));
      await memory.build();
      const later=await memory.retrieve('What happened to the microphone?',3);
      if(!later.some(chunk=>chunk.documentId===next.documentId&&chunk.sources[0].sourceId==='SOURCE-02'))throw new Error('Incremental source was not indexed');
      await other.warm();
      const otherMemory=new SourceMemory(other,{dataset:'other-evidence',storageRoot:otherSettings.storageRoot});
      await otherMemory.add(new SourceDocument({text:'A separate memory has its own generation budget.',sourceId:'OTHER',filename:'other.txt'}));
      await otherMemory.build();
      console.log('CONTRACT_OK');
    } finally {(await import('@cognee/cognee-ts')).shutdown();}
  `;
  let output = "";
  try {
    const exit = await new Promise<number | null>((done, reject) => {
      const child = spawn(process.execPath, ["--input-type=module", "-e", code], {
        cwd: process.cwd(),
        env: {
          ...process.env,
          COGNEE_DISABLE_DOTENV: "1",
          TELEMETRY_DISABLED: "1",
          COGNEE_BINDING_SUPPRESS_LOGS: "1",
          MOCK_LLM: "",
          MOCK_EMBEDDING: "",
          LLM_ARGS: "",
          CACHE_BACKEND: "redis",
          CACHING: "false",
          AUTO_FEEDBACK: "true",
          LLM_CASSETTE: "",
          LLM_RECORD_PATH: "",
          LLM_FALLBACK_MODEL: "",
          LLM_FALLBACK_PROVIDER: "",
          LLM_FALLBACK_ENDPOINT: "",
          LLM_FALLBACK_API_KEY: "",
        },
        stdio: ["ignore", "pipe", "pipe"],
      });
      const timeout = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error("Cognee HTTP contract timed out"));
      }, 25_000);
      child.stdout.on("data", (data) => {
        output += data;
      });
      child.stderr.on("data", (data) => {
        output += data;
      });
      child.on("error", (error) => {
        clearTimeout(timeout);
        reject(error);
      });
      child.on("exit", (code) => {
        clearTimeout(timeout);
        done(code);
      });
    });
    expect(output).toContain("CONTRACT_OK");
    expect(exit).toBe(0);
    const embeddings = requests.filter((request) => request.path === "/v1/embeddings");
    const generation = requests.filter((request) => request.path === "/v1/chat/completions");
    expect(embeddings.length).toBeGreaterThan(0);
    expect(generation.length).toBeGreaterThan(0);
    expect(new Set(generation.map((request) => request.body.model))).toEqual(
      new Set(["test/model", "test/second"]),
    );
    expect(
      embeddings.every(
        (request) =>
          (request.authorization === "Bearer embedding-test-key" &&
            request.body.model === "test/embedding") ||
          (request.authorization === "Bearer second-embedding-key" &&
            request.body.model === "test/embedding-second"),
      ),
    ).toBe(true);
    expect(
      generation.every(
        (request) =>
          ((request.authorization === "Bearer generation-test-key" &&
            request.body.model === "test/model" &&
            request.body.max_tokens === 8192) ||
            (request.authorization === "Bearer second-generation-key" &&
              request.body.model === "test/second" &&
              request.body.max_tokens === 4096)) &&
          !Object.hasOwn(request.body, "functions") &&
          !Object.hasOwn(request.body, "think"),
      ),
    ).toBe(true);
  } finally {
    await new Promise<void>((done) => server.close(() => done()));
    rmSync(root, { recursive: true, force: true });
  }
}, 30_000);
