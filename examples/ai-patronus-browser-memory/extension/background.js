// Patronus AI - background service worker.
// Talks to the partner APIs directly: Gemini (brain), SLNG (voice), Tavily (eyes),
// Mubit/Minima (picks the cheapest capable model, then we run it). Keys come from
// chrome.storage.local first, then optional config.local.js defaults (gitignored).

let DEF = {};
try { importScripts("config.local.js"); DEF = self.__PATRONUS_DEFAULTS || {}; } catch (e) { /* no local defaults */ }

const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const SLNG_TTS = "https://api.slng.ai/v1/bridges/unmute/tts/slng/deepgram/aura:2-en";
const TAVILY_SEARCH = "https://api.tavily.com/search";
const MINIMA_BASE = "https://api.minima.sh/v1";

// Where "I'm bored, launch a game" sends the user. Override via storage key GAMES.
const DEFAULT_GAMES = ["https://game-factory.tech", "https://tims-arcade.pages.dev"];

async function getCfg() {
  const d = await chrome.storage.local.get([
    "GEMINI_API_KEY", "GEMINI_MODEL", "SLNG_API_KEY", "TAVILY_API_KEY", "MUBIT_API_KEY",
    "N8N_WEBHOOK_URL", "SUPERLINKED_URL", "SUPERLINKED_TOKEN", "GAMES", "muted"
  ]);
  return {
    geminiKey: d.GEMINI_API_KEY || DEF.GEMINI_API_KEY || "",
    geminiModel: d.GEMINI_MODEL || DEF.GEMINI_MODEL || "gemini-2.5-flash",
    slngKey: d.SLNG_API_KEY || DEF.SLNG_API_KEY || "",
    tavilyKey: d.TAVILY_API_KEY || DEF.TAVILY_API_KEY || "",
    mubitKey: d.MUBIT_API_KEY || DEF.MUBIT_API_KEY || "",
    n8nWebhook: d.N8N_WEBHOOK_URL || DEF.N8N_WEBHOOK_URL || "",
    superlinkedUrl: d.SUPERLINKED_URL || DEF.SUPERLINKED_URL || "",
    superlinkedToken: d.SUPERLINKED_TOKEN || DEF.SUPERLINKED_TOKEN || "",
    games: (Array.isArray(d.GAMES) && d.GAMES.length) ? d.GAMES : DEFAULT_GAMES,
    muted: !!d.muted
  };
}

// fetch with a hard timeout (the SW can be torn down on a slow request)
async function fetchT(url, opts, ms = 12000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  try { return await fetch(url, { ...opts, signal: ctrl.signal }); }
  finally { clearTimeout(t); }
}

// ---- sponsor usage log (flight recorder): real request/response per call ------
async function logEvent(e) {
  try {
    const d = await chrome.storage.local.get("sponsorLog");
    const log = Array.isArray(d.sponsorLog) ? d.sponsorLog : [];
    log.push(Object.assign({ ts: Date.now() }, e));
    while (log.length > 150) log.shift();
    await chrome.storage.local.set({ sponsorLog: log });
  } catch (err) {}
}

// ---- Mubit / Minima: pick the cheapest capable model, then we run it ---------
async function minimaPick(taskText, taskType, via) {
  const cfg = await getCfg();
  if (!cfg.mubitKey) return null;
  try {
    const body = { task: { task: (taskText || "").slice(0, 300), task_type: taskType || "other" }, cost_quality_tradeoff: 3 };
    const r = await fetchT(MINIMA_BASE + "/recommend", {
      method: "POST",
      headers: { "authorization": "Bearer " + cfg.mubitKey, "content-type": "application/json" },
      body: JSON.stringify(body)
    }, 6000);
    const j = await r.json();
    if (!r.ok) return null;
    // only pick models we can actually run with the Gemini key
    const runnable = m => m && m.provider === "google" && /^gemini-/.test(m.model_id || "");
    const m = runnable(j.recommended_model) ? j.recommended_model : (j.ranked || []).find(runnable);
    logEvent({ sponsor: "Mubit", op: "recommend", via: via || taskText, request: body, response: { picked: m ? m.model_id : null, est_cost_usd: m ? m.est_cost_usd : null, recommendation_id: j.recommendation_id, ranked: (j.ranked || []).slice(0, 4).map(x => x.model_id) } });
    return { recommendationId: j.recommendation_id, modelId: m ? m.model_id : null, est: m ? m.est_cost_usd : (j.recommended_model && j.recommended_model.est_cost_usd) };
  } catch (e) { return null; }
}
async function minimaFeedback(recommendationId, modelId, usage) {
  const cfg = await getCfg();
  if (!cfg.mubitKey || !recommendationId) return;
  try {
    await fetchT(MINIMA_BASE + "/feedback", {
      method: "POST",
      headers: { "authorization": "Bearer " + cfg.mubitKey, "content-type": "application/json" },
      body: JSON.stringify({ recommendation_id: recommendationId, chosen_model_id: modelId, outcome: "success", input_tokens: (usage && usage.in) || 0, output_tokens: (usage && usage.out) || 0 })
    }, 5000);
  } catch (e) {}
}

// ---- Gemini: a soul-driven reply (model chosen by Minima) -------------------
async function callGemini(model, key, sys, message, history) {
  const url = `${GEMINI_BASE}/${model}:generateContent?key=${encodeURIComponent(key)}`;
  const contents = (Array.isArray(history) ? history : []).map(h => ({ role: h.role === "user" ? "user" : "model", parts: [{ text: String(h.text || "").slice(0, 600) }] }));
  contents.push({ role: "user", parts: [{ text: message }] });
  const r = await fetchT(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: sys }] },
      contents,
      generationConfig: { temperature: 0.9, maxOutputTokens: 220 }
    })
  });
  const j = await r.json();
  if (!r.ok) return { ok: false, status: r.status, errMsg: (j && j.error && j.error.message) || ("http " + r.status) };
  const text = (j?.candidates?.[0]?.content?.parts || []).map(p => p.text).join("").trim() || "...";
  const u = j.usageMetadata || {};
  return { ok: true, text, usage: { in: u.promptTokenCount || 0, out: u.candidatesTokenCount || 0 } };
}

async function geminiReply({ soul, message, context, taskType, history, via }) {
  const cfg = await getCfg();
  if (!cfg.geminiKey) return { error: "no_gemini_key", text: "(Add a Gemini key so I can think.)" };
  const sys = (soul || "You are a friendly browser companion.") + (context ? `\n\n## Current page context (may help)\n${context.slice(0, 8000)}` : "");
  const pick = await minimaPick(message, taskType || "creative", via);
  let model = (pick && pick.modelId) || cfg.geminiModel;
  try {
    let res = await callGemini(model, cfg.geminiKey, sys, message, history);
    if (!res.ok && model !== cfg.geminiModel) { model = cfg.geminiModel; res = await callGemini(model, cfg.geminiKey, sys, message, history); } // fallback
    if (!res.ok) { logEvent({ sponsor: "Gemini", op: taskType || "chat", via: via || message, ok: false, request: { model, system: (sys || "").slice(0, 500), user: message }, response: { error: res.errMsg } }); return { error: "gemini_" + (res.status || "x"), text: res.errMsg || "Gemini hiccup." }; }
    if (pick && pick.recommendationId) minimaFeedback(pick.recommendationId, model, res.usage); // fire and forget
    logEvent({ sponsor: "Gemini", op: taskType || "chat", via: via || message, request: { model, system: (sys || "").slice(0, 500), user: message }, response: { text: res.text }, meta: { model, in: res.usage.in, out: res.usage.out } });
    return { text: res.text, model, minima: pick ? model : null, est: pick ? pick.est : null };
  } catch (e) { return { error: "gemini_fetch", text: "(Couldn't reach Gemini.)" }; }
}

// ---- SLNG: text -> spoken audio (WAV) --------------------------------------
function abToBase64(buf) {
  const bytes = new Uint8Array(buf); let bin = ""; const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  return btoa(bin);
}
async function slngSpeak({ text, voice, via }) {
  const cfg = await getCfg();
  if (cfg.muted) return { skipped: "muted" };
  if (!cfg.slngKey) return { error: "no_slng_key" };
  try {
    const body = { text }; if (voice) body.voice = voice;
    const r = await fetchT(SLNG_TTS, {
      method: "POST",
      headers: { "Authorization": "Bearer " + cfg.slngKey, "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    if (!r.ok) { logEvent({ sponsor: "SLNG", op: "tts", via: via || text, ok: false, request: { endpoint: SLNG_TTS, text: (text || "").slice(0, 200), voice }, response: { status: r.status } }); return { error: "slng_http_" + r.status }; }
    const buf = await r.arrayBuffer();
    logEvent({ sponsor: "SLNG", op: "tts", via: via || text, request: { endpoint: SLNG_TTS, text: (text || "").slice(0, 200), voice }, response: { status: r.status, audio_bytes: buf.byteLength } });
    return { audio: "data:audio/wav;base64," + abToBase64(buf) };
  } catch (e) { return { error: "slng_fetch" }; }
}

// ---- Tavily: research the whole web ----------------------------------------
async function tavilyResearch({ query, via }) {
  const cfg = await getCfg();
  if (!cfg.tavilyKey) return { error: "no_tavily_key" };
  try {
    const reqBody = { query, max_results: 5, include_answer: "advanced", search_depth: "advanced" };
    const r = await fetchT(TAVILY_SEARCH, {
      method: "POST",
      headers: { "Authorization": "Bearer " + cfg.tavilyKey, "Content-Type": "application/json" },
      body: JSON.stringify(reqBody)
    });
    const j = await r.json();
    if (!r.ok) { logEvent({ sponsor: "Tavily", op: "search", via: via || query, ok: false, request: reqBody, response: { status: r.status } }); return { error: "tavily_http_" + r.status }; }
    logEvent({ sponsor: "Tavily", op: "search", via: via || query, request: reqBody, response: { answer: (j.answer || "").slice(0, 400), results: (j.results || []).map(x => ({ title: x.title, url: x.url })).slice(0, 5) } });
    return { answer: j.answer || "", results: (j.results || []).map(x => ({ title: x.title, url: x.url })) };
  } catch (e) { return { error: "tavily_fetch" }; }
}

async function openGame() {
  const cfg = await getCfg();
  const url = cfg.games[Math.floor(Math.random() * cfg.games.length)];
  await chrome.tabs.create({ url });
  return { opened: url };
}

async function openUrl({ url }) {
  if (!url || !/^https?:\/\//.test(url)) return { error: "bad_url" };
  await chrome.tabs.create({ url });
  return { opened: url };
}

// Hand the search to the OPENED TAB's content script, which performs it VISIBLY
// (the character walks to the search bar, types, submits, then suggests).
async function siteSearch({ site, query }) {
  let s = String(site || "").trim().replace(/^https?:\/\//, "").replace(/\/.*$/, "");
  if (!s) return { error: "bad_site" };
  if (!s.includes(".")) s += ".com";   // "harrods" -> "harrods.com"
  const url = "https://" + s;
  let q = String(query || ""); try { if (/%[0-9a-f]{2}/i.test(q)) q = decodeURIComponent(q); } catch (e) {}  // un-encode if the model encoded it
  await chrome.storage.local.set({ pendingSearch: { host: s, query: q, ts: Date.now() } });
  await chrome.tabs.create({ url, active: true });
  return { opened: url, searched: query };
}

// ---- n8n: fire a workflow via its webhook (automations / reminders) ----------
async function n8nRun({ task }) {
  const cfg = await getCfg();
  if (!cfg.n8nWebhook) return { error: "no_n8n" };
  try {
    const r = await fetchT(cfg.n8nWebhook, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, source: "patronus-ai" })
    }, 9000);
    return { ok: r.ok, status: r.status };
  } catch (e) { return { error: "n8n_fetch" }; }
}

// ---- Superlinked: remember a page / recall it semantically -------------------
function slHeaders(cfg) {
  return { "Content-Type": "application/json", ...(cfg.superlinkedToken ? { "Authorization": "Bearer " + cfg.superlinkedToken } : {}) };
}
async function slRemember({ text, url, title }) {
  const cfg = await getCfg();
  if (!cfg.superlinkedUrl) return { error: "no_superlinked" };
  try {
    const r = await fetchT(cfg.superlinkedUrl.replace(/\/$/, "") + "/ingest", { method: "POST", headers: slHeaders(cfg), body: JSON.stringify({ text, url, title }) }, 9000);
    const j = await r.json().catch(() => ({}));
    logEvent({ sponsor: "Superlinked", op: "remember page", via: title || url || "page memory", request: { title, url, text_chars: String(text || "").length }, response: { ok: r.ok, id: j.id, count: j.count } });
    return { ok: r.ok, id: j.id, count: j.count };
  } catch (e) { return { error: "sl_fetch" }; }
}
async function slRecall({ query }) {
  const cfg = await getCfg();
  if (!cfg.superlinkedUrl) return { error: "no_superlinked" };
  try {
    const r = await fetchT(cfg.superlinkedUrl.replace(/\/$/, "") + "/query", { method: "POST", headers: slHeaders(cfg), body: JSON.stringify({ query, limit: 5 }) }, 9000);
    const j = await r.json().catch(() => ({}));
    logEvent({ sponsor: "Superlinked", op: "semantic recall", via: query, request: { query, limit: 5 }, response: { ok: r.ok, results: (j.results || j.hits || []).slice(0, 5).map(x => ({ title: x.title, url: x.url, score: x.score })) } });
    return { ok: r.ok, results: j.results || j.hits || [] };
  } catch (e) { return { error: "sl_fetch" }; }
}
// Superlinked SIE: open-model (Qwen) generation via the OpenAI-compatible gateway.
// Short timeout: when the cluster model is warm we use it; when cold we return null
// and the caller falls back to Gemini so the demo never hangs.
const SIE_MODEL = "Qwen/Qwen3.5-4B";
async function sieGenerate({ prompt, system, via }) {
  const cfg = await getCfg();
  if (!cfg.superlinkedUrl) return { ok: false };
  try {
    const base = cfg.superlinkedUrl.replace(/\/$/, "");
    const messages = []; if (system) messages.push({ role: "system", content: String(system).slice(0, 1500) });
    messages.push({ role: "user", content: prompt });
    const r = await fetchT(base + "/v1/chat/completions", {
      method: "POST",
      headers: slHeaders(cfg),
      body: JSON.stringify({ model: SIE_MODEL, messages, max_tokens: 180 })
    }, 6000);
    const j = await r.json().catch(() => ({}));
    const text = j?.choices?.[0]?.message?.content;
    if (!r.ok || !text) return { ok: false };   // cold/loading -> caller falls back
    logEvent({ sponsor: "Superlinked", op: "generate (open Qwen)", via: via || prompt, request: { endpoint: base + "/v1/chat/completions", model: SIE_MODEL, user: prompt.slice(0, 220) }, response: { text } });
    return { ok: true, text: text.trim() };
  } catch (e) { return { ok: false }; }
}
// Superlinked SIE: semantic ranking via open embeddings (reliable; the gen model is
// often cold). One batched /v1/embeddings call ranks the products by the query.
async function sieRank({ query, items, via }) {
  const cfg = await getCfg();
  if (!cfg.superlinkedUrl || !Array.isArray(items) || !items.length) return { ok: false };
  try {
    const base = cfg.superlinkedUrl.replace(/\/$/, "");
    const list = items.slice(0, 10);
    const input = [query, ...list];
    const r = await fetchT(base + "/v1/embeddings", {
      method: "POST",
      headers: slHeaders(cfg),
      body: JSON.stringify({ model: "sentence-transformers/all-MiniLM-L6-v2", input })
    }, 12000);
    const j = await r.json().catch(() => ({}));
    const vecs = (j.data || []).map(d => d.embedding);
    if (!r.ok || vecs.length !== input.length) return { ok: false };
    const q = vecs[0], cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return s / (Math.sqrt(na * nb) || 1); };
    const order = list.map((it, i) => ({ i, score: cos(q, vecs[i + 1]) })).sort((a, b) => b.score - a.score);
    logEvent({ sponsor: "Superlinked", op: "embeddings rank (SIE)", via: via || query, request: { endpoint: base + "/v1/embeddings", model: "all-MiniLM-L6-v2", query, items: list.length }, response: { top: order.slice(0, 3).map(o => ({ item: list[o.i].slice(0, 40), score: +o.score.toFixed(3) })) } });
    return { ok: true, order: order.map(o => o.i) };
  } catch (e) { return { ok: false }; }
}

// ---- agentic router: turn "find flip flops on harrods" into a real action ----
async function routeIntent({ soul, message, name, history }) {
  const cfg = await getCfg();
  if (!cfg.geminiKey) return { action: "answer" };
  const recent = (Array.isArray(history) ? history : []).slice(-6).map(h => `${h.role === "user" ? "User" : (name || "You")}: ${String(h.text || "").slice(0, 160)}`).join("\n");
  const sys =
    `You are the intent router for a browser guardian character${name ? " named " + name : ""}. ` +
    `Personality (for the spoken line only):\n${(soul || "").slice(0, 800)}\n\n` +
    (recent ? `Recent conversation (memory/context):\n${recent}\n\n` : "") +
    `STRONGLY prefer a real ACTION over just answering whenever the user says find / get / show / open / go / buy / where / play / watch / research / remind. ` +
    `Reply ONLY with minified JSON: {"action":"site_search"|"navigate"|"page_qa"|"web_research"|"play_game"|"perform"|"remember"|"recall"|"answer","site":"","query":"","url":"","say":""}. Rules: ` +
    `- SHOPPING for a product ("find / buy / get / cheapest / show me <product>", with or WITHOUT a store): action="site_search". If a store/brand is named use its domain (e.g. "harrods.com", "nike.com", "amazon.co.uk"); if NO store is named, default site="amazon.co.uk". query=the product terms. Do NOT guess a search URL. ` +
    `- A shop/brand/place + a location, or "near me"/"in <city>": action="navigate", url="https://www.google.com/maps/search/QUERY". ` +
    `- Videos: action="navigate", url="https://www.youtube.com/results?search_query=QUERY". ` +
    `- General web lookups/facts: action="navigate", url="https://www.google.com/search?q=QUERY". ` +
    `- "summarize/explain/what's on THIS page/article": action="page_qa". ` +
    `- "research X / latest on X / dig into X": action="web_research", query=topic. ` +
    `- "I'm bored / play a game / play the brainrot game / brainrot 2048 / meme game / play <name>": action="play_game", query=the game name if mentioned. ` +
    `- "fly around / dance / spin / go crazy / do a trick / show off / animate yourself / draw stuff": action="perform" (a fun on-screen animation). ` +
    `- Save memory ("remember this page", "save this article for later"): action="remember". ` +
    `- Recall ("what did I see about...", "that page I saved about..."): action="recall", query=topic. ` +
    `- Only pure conversation: action="answer". ` +
    `URL-encode queries ONLY inside navigate URLs; the site_search "query" must be PLAIN text (no %20). "say" = ONE short in-character spoken line under 22 words.`;
  try {
    const r = await fetchT(`${GEMINI_BASE}/${cfg.geminiModel}:generateContent?key=${encodeURIComponent(cfg.geminiKey)}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: sys }] },
        contents: [{ role: "user", parts: [{ text: message }] }],
        generationConfig: { temperature: 0.3, maxOutputTokens: 240, responseMimeType: "application/json" }
      })
    }, 9000);
    const j = await r.json();
    if (!r.ok) return { action: "answer" };
    let t = (j?.candidates?.[0]?.content?.parts || []).map(p => p.text).join("").trim().replace(/^```json/i, "").replace(/```$/, "").trim();
    const o = JSON.parse(t);
    logEvent({ sponsor: "Gemini", op: "route", via: message, request: { model: cfg.geminiModel, user: message }, response: { action: o.action, target: o.url || o.site || o.query || "", raw: t.slice(0, 300) } });
    if (o.action === "site_search" && o.site && o.query) return { action: "site_search", site: o.site, query: o.query, say: o.say || "On it!" };
    if (o.action === "remember") return { action: "remember", say: o.say || "Saved for later." };
    if (o.action === "recall" && o.query) return { action: "recall", query: o.query, say: o.say || "Let me remember..." };
    if (o.action === "perform") return { action: "perform", say: o.say || "wheee!" };
    if (o.action === "page_qa") return { action: "page_qa", say: o.say || "Reading this page..." };
    if (o.action === "web_research") return { action: "web_research", query: o.query || message, say: o.say || "Searching the web..." };
    if (o.action === "play_game") return { action: "play_game", say: o.say || "Let's play!" };
    if (o.action === "navigate" && /^https?:\/\//.test(o.url || "")) return { action: "navigate", url: o.url, say: o.say || "On it!" };
    return { action: "answer", say: o.say || "" };
  } catch (e) { return { action: "answer" }; }
}

// ---- message router ---------------------------------------------------------
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      switch (msg && msg.type) {
        case "CHAT": sendResponse(await geminiReply({ soul: msg.soul, message: msg.message, context: msg.context, taskType: "creative", history: msg.history, via: msg.via })); break;
        case "PAGE_QA": sendResponse(await geminiReply({ soul: msg.soul, message: msg.question, context: msg.text, taskType: "qa", history: msg.history, via: msg.via })); break;
        case "SPEAK": sendResponse(await slngSpeak(msg)); break;
        case "RESEARCH": sendResponse(await tavilyResearch(msg)); break;
        case "OPEN_GAME": sendResponse(await openGame()); break;
        case "GAME_URL": { const qq = String(msg.query || "").toLowerCase(); let url; if (/brainrot|meme|2048/.test(qq)) { url = "https://game-factory.tech/games/brainrot_2048/"; } else { const c = await getCfg(); url = c.games[Math.floor(Math.random() * c.games.length)]; } sendResponse({ url }); break; }
        case "ACT": sendResponse(await routeIntent(msg)); break;
        case "OPEN_URL": sendResponse(await openUrl(msg)); break;
        case "SITE_SEARCH": sendResponse(await siteSearch(msg)); break;
        case "N8N_RUN": sendResponse(await n8nRun(msg)); break;
        case "SL_REMEMBER": sendResponse(await slRemember(msg)); break;
        case "SL_RECALL": sendResponse(await slRecall(msg)); break;
        case "SIE_GEN": sendResponse(await sieGenerate(msg)); break;
        case "SIE_RANK": sendResponse(await sieRank(msg)); break;
        case "GET_POWERS": { const c = await getCfg(); sendResponse({ gemini: !!c.geminiKey, slng: !!c.slngKey, tavily: !!c.tavilyKey, mubit: !!c.mubitKey, n8n: !!c.n8nWebhook, superlinked: !!c.superlinkedUrl }); break; }
        default: sendResponse({ error: "unknown_message" });
      }
    } catch (e) { sendResponse({ error: "handler_crash", detail: String(e) }); }
  })();
  return true; // async
});

// Re-inject the content script into already-open tabs on install/update/reload,
// so a reloaded extension takes over stale tabs without a manual page refresh.
chrome.runtime.onInstalled.addListener(reinjectAll);
chrome.runtime.onStartup && chrome.runtime.onStartup.addListener(reinjectAll);
async function reinjectAll() {
  try {
    const tabs = await chrome.tabs.query({ url: ["http://*/*", "https://*/*"] });
    for (const t of tabs) {
      try { await chrome.scripting.executeScript({ target: { tabId: t.id }, files: ["content.js"] }); } catch (e) {}
    }
  } catch (e) {}
}
