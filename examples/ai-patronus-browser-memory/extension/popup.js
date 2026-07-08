// Patronus AI popup: your souls, power status, and optional API settings.
const REAL = [
  { id: "grandma",  e: "👵", n: "Surf Granny" },
  { id: "tungtung", e: "🪵", n: "Tung Tung" },
  { id: "sixseven", e: "✋", n: "Six Seven" },
  { id: "cat",      e: "🐱", n: "Mochi" },
  { id: "dog",      e: "🐶", n: "Biscuit" },
  { id: "pupa",     e: "🐛", n: "Pupa" },
  { id: "wizard",   e: "🧙", n: "Harry" }
];
const TEASER = { e: "✨", n: "soon" };
const $ = id => document.getElementById(id);
let active = "grandma";
const KEY_FIELDS = [
  ["GEMINI_API_KEY", "keyGemini"],
  ["GEMINI_MODEL", "keyGeminiModel"],
  ["SLNG_API_KEY", "keySlng"],
  ["TAVILY_API_KEY", "keyTavily"],
  ["MUBIT_API_KEY", "keyMubit"],
  ["SUPERLINKED_URL", "keySuperlinkedUrl"],
  ["SUPERLINKED_TOKEN", "keySuperlinkedToken"]
];

function load() {
  chrome.storage.local.get(["activeChar"], d => {
    active = (d.activeChar && REAL.some(c => c.id === d.activeChar)) ? d.activeChar : "grandma";
    chrome.storage.local.set({ activeChar: active, unlocked: REAL.map(c => c.id) });
    render(); loadPowers(); loadKeys();
  });
}
function setActive(id) { if (!REAL.some(c => c.id === id)) return; active = id; chrome.storage.local.set({ activeChar: active }); render(); }

function render() {
  const a = REAL.find(c => c.id === active) || REAL[0];
  $("heroE").innerHTML = ""; const hi = document.createElement("img"); hi.src = `chars/${a.id}.png`; hi.alt = "";
  hi.onerror = () => { $("heroE").textContent = a.e; }; $("heroE").appendChild(hi);
  $("heroN").textContent = a.n; $("streak").textContent = "";

  const tiles = REAL.map(c =>
    `<div class="tile${c.id === active ? " active" : ""}" data-id="${c.id}"><img class="te" src="chars/${c.id}.png" data-e="${c.e}" alt=""><div class="t">${c.n}</div></div>`
  ).concat([`<div class="tile locked"><div class="e">${TEASER.e}</div><div class="t">${TEASER.n}</div></div>`]).join("");
  $("grid").innerHTML = tiles;
  $("grid").querySelectorAll("img.te").forEach(img => img.onerror = () => { const d = document.createElement("div"); d.className = "e"; d.textContent = img.dataset.e; img.replaceWith(d); });
  document.querySelectorAll(".tile[data-id]").forEach(t => t.addEventListener("click", () => setActive(t.dataset.id)));

  $("unlockBox").innerHTML = `<div class="unlock wait">✨ One more soul coming soon - check back tomorrow.</div>`;
}

// read-only powers strip - shows which partner powers are wired (no key values)
function loadPowers() {
  const labels = { gemini: "Gemini", slng: "SLNG voice", tavily: "Tavily", mubit: "Mubit", superlinked: "Superlinked" };
  chrome.runtime.sendMessage({ type: "GET_POWERS" }, p => {
    p = p || {};
    $("powers").innerHTML = Object.keys(labels).map(k =>
      `<span class="pw ${p[k] ? "" : "off"}"><span class="dot"></span>${labels[k]}</span>`).join("");
  });
}

function loadKeys() {
  chrome.storage.local.get(KEY_FIELDS.map(([key]) => key), d => {
    KEY_FIELDS.forEach(([key, id]) => {
      const el = $(id);
      if (el) el.value = d[key] || "";
    });
  });
}

function setStatus(text) {
  $("keyStatus").textContent = text;
  setTimeout(() => { $("keyStatus").textContent = ""; }, 1800);
}

document.getElementById("saveKeys").addEventListener("click", () => {
  const data = {};
  KEY_FIELDS.forEach(([key, id]) => {
    data[key] = ($(id).value || "").trim();
  });
  chrome.storage.local.set(data, () => {
    setStatus("saved");
    loadPowers();
  });
});

document.getElementById("clearKeys").addEventListener("click", () => {
  chrome.storage.local.remove(KEY_FIELDS.map(([key]) => key), () => {
    loadKeys();
    loadPowers();
    setStatus("cleared");
  });
});

document.getElementById("logBtn").addEventListener("click", () => chrome.tabs.create({ url: chrome.runtime.getURL("log.html") }));
document.getElementById("resetBtn").addEventListener("click", () => {
  chrome.storage.local.get(null, all => {
    const keys = Object.keys(all).filter(k => k.startsWith("chat_") || k === "sponsorLog" || k === "pendingSearch" || k === "pendingSuggest");
    chrome.storage.local.remove(keys, () => { const b = document.getElementById("resetBtn"); b.textContent = "✓ cleared"; setTimeout(() => b.textContent = "🧹 Reset for demo", 1500); });
  });
});

load();
