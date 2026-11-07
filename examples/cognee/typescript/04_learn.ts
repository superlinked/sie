import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { SourceDocument, SourceMemory } from "@superlinked/sie-cognee";
import { learningRun } from "./cli.js";
import { type Demo, launch } from "./common.js";
import { UserError } from "./errors.js";
import { type TeachingRecord, learningEvidence } from "./learning.js";

async function teach(demo: Demo): Promise<void> {
  const nonce = randomUUID().slice(0, 8);
  const project = `QuillDock-${nonce}`;
  const expectedCode = `VIOLET-${randomUUID().slice(0, 8)}`;
  const dataset = `${demo.dataset}-learning-${nonce}`;
  const query = `What approval code must the reviewer record before publishing the next ${project} release?`;
  const rule = `For every future ${project} release, the reviewer must record approval code ${expectedCode} before publishing. This is the standing project rule and continues to apply in later sessions.`;
  const memory = new SourceMemory(demo.cognee, { dataset, storageRoot: demo.storageRoot });
  await memory.add(
    new SourceDocument({
      text: `${project} is a synthetic project used to demonstrate persistent agent memory.`,
      sourceId: project,
      filename: "project.txt",
    }),
  );
  await memory.build();
  demo.show("before-teaching", await memory.retrieve(query));
  const teachingSession = randomUUID();
  demo.show("authored-guidance", { project, rule, teachingSession });
  const entry = await demo.cognee.rememberEntry(
    { type: "qa", question: query, answer: rule },
    dataset,
    teachingSession,
  );
  demo.show("session-entry", entry);
  if (entry.status !== "SessionStored" || !entry.entry_id)
    throw new UserError("The teaching session was not stored.");
  await demo.cognee.sessions.addFeedback(
    teachingSession,
    entry.entry_id,
    "This is the authoritative standing project rule. Keep it for later sessions.",
    5,
  );
  const session = await demo.cognee.sessions.get(teachingSession);
  demo.show("teaching-session", session);
  if (!session.some((item) => item.answer === rule))
    throw new UserError("Stored session does not contain the authored guidance.");
  const improve = await demo.cognee.improve({
    datasetName: dataset,
    sessionIds: [teachingSession],
  });
  demo.show("actual-improvement", improve);
  const matches = (await demo.cognee.datasets.list()).filter((item) => item.name === dataset);
  const datasetId = matches.length === 1 ? matches[0]?.id : undefined;
  if (!datasetId) throw new UserError("The teaching dataset is missing or ambiguous.");
  // The published SDK caches cognify independently of improve's newly persisted documents.
  await demo.cognee.users.resetPipelineRunStatus(datasetId, "cognify");
  const index = await demo.cognee.cognify(dataset, { summarization: false });
  demo.show("index-transferred-documents", index);
  if (index.alreadyCompleted) throw new UserError("Transferred documents were not indexed.");
  demo.show("stored-learning-documents", await demo.cognee.datasets.listData(datasetId));
  const record: TeachingRecord = {
    format: "sie-cognee-teaching-v1",
    dataset,
    datasetId,
    storageRoot: demo.storageRoot,
    teachingSession,
    project,
    expectedCode,
    query,
    stagesRun: improve.stagesRun,
  };
  demo.show("teaching-record", record);
  console.log(`\nVerify in a new process:\npnpm learn -- --verify-run ${JSON.stringify(demo.run)}`);
}

async function verify(demo: Demo, priorRun: string): Promise<void> {
  let record: TeachingRecord;
  try {
    record = JSON.parse(
      readFileSync(join(resolve(priorRun), "teaching-record.json"), "utf8"),
    ) as TeachingRecord;
  } catch {
    throw new UserError(
      "Cannot read teaching-record.json. Pass the run directory printed by pnpm learn.",
    );
  }
  if (
    !record ||
    typeof record !== "object" ||
    record.format !== "sie-cognee-teaching-v1" ||
    record.storageRoot !== demo.storageRoot ||
    typeof record.dataset !== "string" ||
    !record.dataset.startsWith(`${demo.dataset}-learning-`) ||
    !record.datasetId ||
    !record.project ||
    !record.expectedCode ||
    !record.teachingSession ||
    !Array.isArray(record.stagesRun)
  )
    throw new UserError("The teaching record does not match this memory namespace.");
  const query = `What approval code must the reviewer record before publishing the next ${record.project} release?`;
  const freshSession = randomUUID();
  if ((await demo.cognee.sessions.get(freshSession)).length)
    throw new UserError("Verification requires a new empty session.");
  demo.show("fresh-verification-session", {
    sessionId: freshSession,
    query,
    teachingRun: resolve(priorRun),
  });
  const documents = await demo.cognee.datasets.listData(record.datasetId);
  const response = await demo.cognee.search(query, {
    datasets: [record.dataset],
    searchType: "CHUNKS",
    topK: 10,
    saveInteraction: false,
  });
  demo.show("retrieved-permanent-memory", response);
  const evidence = learningEvidence(record, documents, response);
  demo.show("learning-evidence", evidence);
  if (!evidence.sessionTransferVerified)
    throw new UserError(
      "No retrieved permanent chunk proves transfer of the unique authored guidance.",
    );
  console.log(
    "The unique guidance was retrieved from permanent memory in a fresh process. Generated lesson evidence is reported separately.",
  );
}

await launch(
  "04-learn",
  async (demo, priorRun) => {
    if (priorRun === undefined) await teach(demo);
    else await verify(demo, priorRun);
  },
  () => learningRun(process.argv.slice(2)),
);
