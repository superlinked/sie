// Renders the integration usage log from chrome.storage.local, grouped by request.
const SPON = { Gemini: "#1a73e8", SLNG: "#e0564f", Tavily: "#6d40d6", Mubit: "#1f9d57", Superlinked: "#b8860b" };
const esc = s => String(s == null ? "" : s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const $ = s => document.querySelector(s);

function render() {
  chrome.storage.local.get("sponsorLog", d => {
    const log = (d.sponsorLog || []).slice().reverse();
    if (!log.length) { $("#log").innerHTML = '<div class="empty">No integration calls yet.<br>Try "remember this page", then hit Refresh.</div>'; return; }
    const counts = {}; log.forEach(e => counts[e.sponsor] = (counts[e.sponsor] || 0) + 1);
    const summary = Object.keys(counts).map(s => `<span class="badge" style="background:${SPON[s] || "#555"}">${esc(s)} &times;${counts[s]}</span>`).join(" ");
    // group consecutive calls by their triggering request ("via")
    const groups = []; let cur = null;
    for (const e of log) { const via = e.via || "(misc)"; if (!cur || cur.via !== via) { cur = { via, items: [] }; groups.push(cur); } cur.items.push(e); }
    $("#log").innerHTML = `<div class="summary">${summary}</div>` + groups.map(g => `
      <div class="grp"><div class="via">▶ ${esc(g.via)}</div>
        ${g.items.map(e => `<details class="call"><summary>
            <span class="badge" style="background:${SPON[e.sponsor] || "#555"}">${esc(e.sponsor)}</span>
            <b>${esc(e.op)}</b> ${e.ok === false ? '<span class="err">failed</span>' : ""}
            <span class="t">${new Date(e.ts).toLocaleTimeString()}</span></summary>
          <div class="kv">REQUEST</div><pre>${esc(JSON.stringify(e.request, null, 2))}</pre>
          <div class="kv">RESPONSE</div><pre>${esc(JSON.stringify(e.response, null, 2))}</pre>
          ${e.meta ? `<div class="kv">META</div><pre>${esc(JSON.stringify(e.meta, null, 2))}</pre>` : ""}
        </details>`).join("")}
      </div>`).join("");
  });
}
$("#refresh").addEventListener("click", render);
$("#clear").addEventListener("click", () => chrome.storage.local.set({ sponsorLog: [] }, render));
render();
