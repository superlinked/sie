// account-signal-scoring frontend: SSE-driven scoring pipeline + account brief.
// No build step; vanilla JS.

const els = {
  badge: document.getElementById("badge"),
  sieState: document.getElementById("sie-state"),
  briefState: document.getElementById("brief-state"),
  sieUrl: document.getElementById("sie-url"),
  accounts: document.getElementById("accounts"),
  pipeline: document.getElementById("pipeline"),
  pipelineMeta: document.getElementById("pipeline-meta"),
  brief: document.getElementById("brief"),
  briefMeta: document.getElementById("brief-meta"),
  timings: document.getElementById("timings"),
};

let activeAccount = null;

function setBadge(text, cls) {
  els.badge.textContent = text;
  els.badge.className = "badge" + (cls ? " " + cls : "");
}

function fmtMs(ms) {
  if (ms < 1000) return `${ms} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function fmtUsd(n) {
  if (n == null) return "n/a";
  return `$${n.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderAccounts(accounts) {
  if (!accounts?.length) {
    els.accounts.innerHTML = '<p class="hint">no accounts</p>';
    return;
  }
  const groups = { risk: [], opportunity: [] };
  for (const a of accounts) groups[a.direction]?.push(a);

  const card = (a) => `
    <div class="account" data-id="${a.id}">
      <div class="account-score band-${a.band}">${a.score}</div>
      <div>
        <div class="account-head">
          <span class="account-name">${escapeHtml(a.name)}</span>
          <span class="account-arr">${fmtUsd(a.arr)}</span>
        </div>
        <div class="account-reason">${escapeHtml(a.reason)}</div>
        <div class="account-meta">
          <span class="pill ${a.direction}">${a.direction}</span>
          <span class="pill">renewal ${a.renewalDays}d</span>
          <span class="pill">${escapeHtml(a.owner)}</span>
        </div>
      </div>
    </div>`;

  const section = (key, title, rows) =>
    rows.length
      ? `<div class="board-head ${key}">${title} · ${rows.length}</div>${rows.map(card).join("")}`
      : "";

  els.accounts.innerHTML =
    section("risk", "Churn-risk board", groups.risk) +
    section("opportunity", "Expansion board", groups.opportunity);

  for (const node of els.accounts.querySelectorAll(".account")) {
    node.addEventListener("click", () => {
      for (const o of els.accounts.querySelectorAll(".account")) o.classList.remove("active");
      node.classList.add("active");
      const id = node.dataset.id;
      activeAccount = accounts.find((a) => a.id === id);
      runScore(id);
    });
  }
}

function renderPipelineStart() {
  els.pipeline.innerHTML = `
    <div class="risk-stages">
      <div class="stage" data-key="extract"><span class="stage-dot"></span> extract</div>
      <div class="stage" data-key="encode"><span class="stage-dot"></span> encode</div>
      <div class="stage" data-key="score"><span class="stage-dot"></span> score</div>
      <div class="stage" data-key="brief"><span class="stage-dot"></span> brief</div>
    </div>
    <div class="risk-body"></div>`;
  els.pipelineMeta.textContent = "";
  els.timings.textContent = "";
}

function setStage(key, state) {
  const node = els.pipeline.querySelector(`.stage[data-key="${key}"]`);
  if (node) node.dataset.state = state;
}

function renderEntities(entities) {
  if (!entities?.length) return '<p class="hint">no entities extracted</p>';
  return entities
    .map(
      (e) => `
      <span class="entity-group" title="GLiNER confidence: ${e.score.toFixed(3)}">
        <span class="ekey">${escapeHtml(e.label)}</span>
        <span class="eval">${escapeHtml(e.text)}</span>
        <span class="econf">${e.score.toFixed(2)}</span>
      </span>`,
    )
    .join("");
}

function renderBand(signals) {
  const bandWhy = {
    green: "under 10 — healthy. No action required, monitor.",
    amber: "10 - 40 — worth a proactive touch this week.",
    red: "over 40 — act now, this is at the top of the board.",
  }[signals.band];
  return `
    <div class="risk-band band-${signals.band}">
      <div class="band-label">Signal score</div>
      <div class="band-value">${signals.score}</div>
      <div class="band-score">${signals.direction} · ${signals.band.toUpperCase()}</div>
    </div>
    <p class="band-why">${bandWhy}</p>`;
}

function renderHits(hits) {
  return `
    <div class="score-legend">
      Each row is a past-outcome <strong>playbook</strong>. The number on the right is the
      <strong>cross-encoder reranker score</strong> for this account against that playbook.
      The top match (highlighted) drives the recommended play.
    </div>
    <div class="hits">
      ${hits
        .map(
          (h, i) => `
        <div class="hit${i === 0 ? " top" : ""}">
          <div class="hit-head">
            <span class="hit-label">${escapeHtml(h.label)}</span>
            <span class="hit-score" title="BGE-reranker-base relevance score.">${h.score.toFixed(3)}</span>
          </div>
          <div class="hit-summary">${escapeHtml(h.summary)}</div>
          <div class="hit-play">▸ ${escapeHtml(h.play)}</div>
          <div class="hit-meta">
            <span class="pill ${h.direction}">${h.direction}</span>
            <span class="pill">outcome: ${escapeHtml(h.outcome)}</span>
          </div>
        </div>`,
        )
        .join("")}
    </div>`;
}

function renderBrief(data) {
  const isLlm = data.source === "sie-chat";
  els.brief.innerHTML = `
    <div class="brief-card">
      <span class="brief-source ${isLlm ? "llm" : ""}">${isLlm ? "SIE LLM" : "deterministic"}</span>
      <div class="brief-account">${escapeHtml(activeAccount?.name ?? "")}</div>
      <div class="brief-field">
        <div class="k">Summary</div>
        <div class="v">${escapeHtml(data.summary)}</div>
      </div>
      <div class="brief-field">
        <div class="k">Drivers</div>
        <div class="v">${escapeHtml(data.drivers)}</div>
      </div>
      <div class="brief-field brief-play">
        <div class="k">Recommended play</div>
        <div class="v">${escapeHtml(data.recommendedPlay)}</div>
      </div>
    </div>
    <div class="arr-stake">
      <span class="k">ARR at stake</span>
      <span class="v">${fmtUsd(data.arrAtStake)}</span>
    </div>`;
  els.briefMeta.textContent = isLlm ? `llm · ${fmtMs(data.ms)}` : `deterministic · ${fmtMs(data.ms)}`;
}

function runScore(id) {
  setBadge("running", "running");
  renderPipelineStart();
  els.brief.innerHTML = '<p class="hint">Scoring pipeline running...</p>';
  const evt = new EventSource(`/api/run?id=${encodeURIComponent(id)}`);
  const body = els.pipeline.querySelector(".risk-body");
  const ts = { extract: 0, encode: 0, score: 0, brief: 0, total: 0 };

  evt.addEventListener("signals", (e) => {
    const data = JSON.parse(e.data);
    body.insertAdjacentHTML(
      "beforeend",
      `<div class="risk-section">
         <div class="risk-section-head">Signal roll-up (deterministic)</div>
         ${renderBand(data)}
       </div>`,
    );
  });
  evt.addEventListener("extracting", () => setStage("extract", "running"));
  evt.addEventListener("extracted", (e) => {
    setStage("extract", "done");
    const data = JSON.parse(e.data);
    ts.extract = data.ms;
    body.insertAdjacentHTML(
      "beforeend",
      `<div class="risk-section">
         <div class="risk-section-head">Extracted entities · ${fmtMs(data.ms)}</div>
         <div class="hint" style="margin-bottom:8px">
           GLiNER zero-shot NER pulls typed entities out of the account summary.
         </div>
         <div class="entities">${renderEntities(data.entities)}</div>
       </div>`,
    );
  });
  evt.addEventListener("encoding", () => setStage("encode", "running"));
  evt.addEventListener("encoded", (e) => {
    setStage("encode", "done");
    const data = JSON.parse(e.data);
    ts.encode = data.ms;
    body.insertAdjacentHTML(
      "beforeend",
      `<div class="risk-section">
         <div class="risk-section-head">Encoded account context · ${fmtMs(data.ms)}</div>
         <div class="hint">
           MiniLM-L6 turns the account summary into a dense ${data.dim}-dimensional vector.
           Cosine similarity against the pre-encoded playbook corpus picks the top candidates for reranking.
         </div>
       </div>`,
    );
  });
  evt.addEventListener("scoring", () => setStage("score", "running"));
  evt.addEventListener("scored", (e) => {
    setStage("score", "done");
    const data = JSON.parse(e.data);
    ts.score = data.ms;
    body.insertAdjacentHTML(
      "beforeend",
      `<div class="risk-section">
         <div class="risk-section-head">Reranked playbooks · ${fmtMs(data.ms)}</div>
         ${renderHits(data.hits)}
       </div>`,
    );
  });
  evt.addEventListener("briefing", () => setStage("brief", "running"));
  evt.addEventListener("brief", (e) => {
    setStage("brief", "done");
    const data = JSON.parse(e.data);
    ts.brief = data.ms;
    renderBrief(data);
  });
  evt.addEventListener("done", (e) => {
    const data = JSON.parse(e.data);
    ts.total = data.totalMs;
    setBadge("done", "green");
    els.timings.textContent = `extract ${fmtMs(ts.extract)} · encode ${fmtMs(ts.encode)} · score ${fmtMs(ts.score)} · brief ${fmtMs(ts.brief)} · total ${fmtMs(ts.total)}`;
    evt.close();
  });
  evt.addEventListener("error", (e) => {
    setBadge("error", "red");
    let msg = "stream error";
    try {
      const data = JSON.parse(e.data);
      msg = `${data.stage}: ${data.message}`;
    } catch {
      /* network error event has no payload */
    }
    body.insertAdjacentHTML("beforeend", `<div class="error">${escapeHtml(msg)}</div>`);
    evt.close();
  });
}

async function init() {
  els.sieUrl.textContent = "...";
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    els.sieUrl.textContent = j.sieUrl;
    els.sieState.textContent = j.sie
      ? `SIE healthy · ${j.registeredModels} models`
      : "SIE not reachable yet";
    els.briefState.textContent =
      j.brief === "llm" ? `brief: LLM (${j.chatModel})` : "brief: deterministic";
  } catch {
    els.sieState.textContent = "could not reach the local server";
  }
  try {
    const r = await fetch("/api/accounts");
    renderAccounts(await r.json());
  } catch {
    els.accounts.innerHTML = '<p class="hint">failed to load accounts</p>';
  }
}

init();
