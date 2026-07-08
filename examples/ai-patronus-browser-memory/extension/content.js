// Patronus AI - content script. Your guardian, living on every page.
// Shadow DOM isolates it; supports hot re-injection on extension reload.
(function () {
  const old = document.getElementById("patronus-host");
  if (old) { try { old.remove(); } catch (e) {} }   // a newer version takes over
  window.__patronusLoaded = true;

  const CHARS = {
    grandma:  { name: "Surf Granny", emoji: "👵", accent: "#8ecbff", voice: "aura-2-theia-en" },
    tungtung: { name: "Tung Tung",   emoji: "🪵", accent: "#c79a5b", voice: "aura-2-orion-en" },
    sixseven: { name: "Six Seven",   emoji: "✋", accent: "#7fc7f5", voice: "aura-2-thalia-en" },
    cat:      { name: "Mochi",       emoji: "🐱", accent: "#ff9bb3", voice: "aura-2-thalia-en" },
    dog:      { name: "Biscuit",     emoji: "🐶", accent: "#ffce7a", voice: "aura-2-orion-en" },
    pupa:     { name: "Pupa",        emoji: "🐛", accent: "#9be08a", voice: "aura-2-luna-en" },
    wizard:   { name: "Harry",       emoji: "🧙", accent: "#9b8cff", voice: "aura-2-orion-en" }
  };
  const DEFAULT_CHAR = "grandma";
  const GREETING = {
    grandma: "cowabunga, sweetie - what are we doing?", tungtung: "tung tung tung! what's the mission?",
    sixseven: "ayy six seven - whatcha need?", cat: "mrrp. what do you want, human?",
    dog: "HI!! what are we finding today??", pupa: "hi. let's become something today.",
    wizard: "expecto... assistance! what do you need?"
  };

  let charId = DEFAULT_CHAR, muted = false, soulCache = {}, history = [], lastVia = "";
  let bubbleTimer = null, idleTimer = null, wanderTimer = null, recog = null, audio = null;

  const host = document.createElement("div");
  host.id = "patronus-host";
  host.style.cssText = "position:fixed;inset:0;z-index:2147483646;pointer-events:none;";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
    <style>
      .pet{position:fixed;right:26px;bottom:26px;width:84px;height:84px;cursor:grab;pointer-events:auto;
        user-select:none;display:flex;align-items:center;justify-content:center;font-size:60px;line-height:1;
        filter:drop-shadow(0 8px 14px rgba(0,0,0,.28));animation:bob 2.6s ease-in-out infinite alternate;z-index:3}
      .pet:active{cursor:grabbing}
      .pet img{width:100%;height:100%;object-fit:contain;display:none;-webkit-user-drag:none;pointer-events:none}
      .pet video{width:100%;height:100%;object-fit:cover;border-radius:16px;display:none;pointer-events:none}
      .pet.hasimg #emoji{display:none} .pet.hasimg img{display:block}
      .pet.hasvid #emoji,.pet.hasvid img{display:none} .pet.hasvid video{display:block}
      @keyframes bob{to{transform:translateY(-8px)}}
      .pet.nap{animation:none;transform:rotate(8deg) scale(.92);filter:drop-shadow(0 4px 8px rgba(0,0,0,.2)) grayscale(.2)}
      .zzz{position:fixed;font-size:18px;pointer-events:none;color:#7c8aa0;font-weight:800;z-index:3}
      .bubble{position:fixed;max-width:230px;background:#fff;color:#1b2330;border:1px solid #e7eaf0;padding:9px 13px;
        border-radius:16px;font:600 13.5px/1.45 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;
        box-shadow:0 10px 26px rgba(20,30,55,.18);pointer-events:none;opacity:0;transform:translateY(6px) scale(.96);
        transform-origin:bottom right;transition:opacity .18s,transform .18s;z-index:4}
      .bubble::after{content:"";position:absolute;right:26px;bottom:-7px;width:13px;height:13px;background:#fff;
        border-right:1px solid #e7eaf0;border-bottom:1px solid #e7eaf0;transform:rotate(45deg)}
      .bubble.show{opacity:1;transform:none}
      .bubble .nm{display:block;font-size:10px;font-weight:800;letter-spacing:.04em;text-transform:uppercase;color:var(--ac);margin-bottom:2px}
      .panel{position:fixed;right:26px;bottom:120px;width:320px;background:#f7f8fb;border:1px solid #e7eaf0;border-radius:18px;
        box-shadow:0 18px 44px rgba(20,30,55,.24);pointer-events:auto;z-index:5;overflow:hidden;display:none;
        font:13px/1.5 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;color:#1b2330}
      .panel.open{display:block;animation:rise .2s both}
      @keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
      .phead{display:flex;align-items:center;gap:8px;padding:10px 12px;background:#fff;border-bottom:1px solid #eef1f6}
      .phead .av{font-size:22px}.phead b{font-size:13px}
      .phead .ctl{margin-left:auto;display:flex;gap:10px;align-items:center}
      .phead .ic{cursor:pointer;color:#9aa4b4;font-size:14px;font-weight:800}
      .thread{max-height:300px;overflow:auto;padding:12px 12px 4px;display:flex;flex-direction:column;gap:8px}
      .msg{max-width:88%;padding:8px 11px;border-radius:13px;font-size:13px;white-space:pre-wrap;word-wrap:break-word}
      .msg.you{align-self:flex-end;background:var(--ac);color:#fff;border-bottom-right-radius:4px}
      .msg.bot{align-self:flex-start;background:#fff;border:1px solid #eef1f6;color:#2c3647;border-bottom-left-radius:4px}
      .msg .tag{display:block;margin-top:5px;font-size:10.5px;color:#94a0b2}
      .msg a{display:block;margin-top:5px;color:var(--ac);font-size:12px;font-weight:600;text-decoration:none;
        border:1px solid #eef1f6;border-radius:9px;padding:6px 9px;background:#fbfcfe}
      .empty{color:#aeb6c4;font-size:12px;text-align:center;padding:16px 8px}
      .ask{display:flex;gap:6px;padding:10px 12px;border-top:1px solid #eef1f6;background:#fff}
      .ask input{flex:1;border:1px solid #e2e6ee;border-radius:10px;padding:9px 11px;font-size:13px;outline:none}
      .ask input:focus{border-color:var(--ac)}
      .ask button{border:none;border-radius:10px;width:38px;cursor:pointer;font-size:15px}
      .ask .mic{background:#eef1f6;color:#3a4256}.ask .mic.on{background:#ffe0e0;color:#e0564f}
      .ask .go{background:var(--ac);color:#fff}
      .gov{position:fixed;inset:0;z-index:7;pointer-events:auto;display:flex;align-items:center;justify-content:center}
      .govbg{position:absolute;inset:0;background:rgba(10,12,20,.55)}
      .govbox{position:relative;width:min(440px,92vw);height:min(780px,86vh);background:#0e141d;border-radius:16px;overflow:hidden;box-shadow:0 24px 70px rgba(0,0,0,.5);display:flex;flex-direction:column}
      .govbar{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:#161d28;color:#fff;font:700 12px -apple-system,system-ui,sans-serif}
      .govx{cursor:pointer;font-weight:800;font-size:14px}
      .govbox iframe{flex:1;width:100%;border:0;background:#000}
      .spin{position:fixed;inset:0;z-index:8;pointer-events:auto;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px}
      .spinbg{position:absolute;inset:0;background:radial-gradient(circle at 50% 44%,rgba(22,30,48,.66),rgba(8,10,16,.88))}
      .spinwrap{position:relative;width:min(66vw,330px);height:min(66vw,330px);touch-action:none;cursor:grab;filter:drop-shadow(0 18px 42px rgba(0,0,0,.55))}
      .spinwrap:active{cursor:grabbing}
      .spinwrap svg{width:100%;height:100%;display:block;will-change:transform;transform-origin:50% 50%}
      .spincap{position:relative;color:#fff;font:800 19px -apple-system,system-ui,sans-serif;text-shadow:0 2px 10px rgba(0,0,0,.6);text-align:center}
      .spinrpm{position:relative;color:#dcebfb;font:800 14px ui-monospace,Menlo,monospace;opacity:.95;letter-spacing:.04em}
      .spinhint{position:relative;color:#aebfd2;font:600 12.5px -apple-system,system-ui,sans-serif;opacity:.85}
      .spinx{position:absolute;top:16px;right:18px;color:#fff;font:800 16px sans-serif;cursor:pointer;background:rgba(0,0,0,.4);width:34px;height:34px;border-radius:50%;display:flex;align-items:center;justify-content:center}
    </style>
    <div class="pet" id="pet"><video id="petvid" muted loop autoplay playsinline></video><img id="petimg" alt="" draggable="false"><span id="emoji">👵</span></div>
    <div class="bubble" id="bubble"></div>
    <div class="panel" id="panel">
      <div class="phead"><span class="av" id="pav">👵</span><b id="pnm">Surf Granny</b>
        <span class="ctl"><span class="ic" id="pmute">🔊</span><span class="ic" id="px">✕</span></span></div>
      <div class="thread" id="thread"></div>
      <div class="ask">
        <input id="inp" placeholder="ask me to find or do anything...">
        <button class="mic" id="mic" title="speak">🎤</button>
        <button class="go" id="go" title="send">↑</button>
      </div>
    </div>`;
  (document.documentElement || document.body).appendChild(host);

  const $ = id => root.getElementById(id);
  const pet = $("pet"), emojiEl = $("emoji"), petimg = $("petimg"), petvid = $("petvid"), bubble = $("bubble");
  const panel = $("panel"), thread = $("thread"), inp = $("inp");
  const send = msg => new Promise(res => { try { chrome.runtime.sendMessage(msg, res); } catch (e) { res({ error: "no_bg" }); } });
  // when the extension is reloaded, this OLD content script's context dies. detect it and
  // self-destruct cleanly instead of throwing "Extension context invalidated" from stale timers.
  const alive = () => { try { return !!(chrome.runtime && chrome.runtime.id); } catch (e) { return false; } };
  function teardown() { clearTimeout(bubbleTimer); clearTimeout(idleTimer); clearTimeout(wanderTimer); try { if (audio) audio.pause(); } catch (e) {} try { if (recog) recog.stop(); } catch (e) {} try { host.remove(); } catch (e) {} }
  const store = { get: (k, cb) => { try { chrome.storage.local.get(k, cb); } catch (e) {} }, set: o => { try { chrome.storage.local.set(o); } catch (e) {} }, remove: k => { try { chrome.storage.local.remove(k); } catch (e) {} } };
  const esc = s => (s || "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const escA = s => (s || "").replace(/"/g, "%22");
  const cur = () => CHARS[charId] || CHARS[DEFAULT_CHAR];

  function applyChar() {
    const c = cur();
    [host, pet, panel, bubble].forEach(el => el.style.setProperty("--ac", c.accent));
    emojiEl.textContent = c.emoji;
    pet.classList.remove("hasimg", "hasvid");
    petimg.onload = () => { if (!pet.classList.contains("hasvid")) pet.classList.add("hasimg"); };
    petimg.onerror = () => pet.classList.remove("hasimg");
    try { petimg.src = chrome.runtime.getURL("chars/" + charId + ".png"); } catch (e) {}
    petvid.onloadeddata = () => { pet.classList.add("hasvid"); petvid.play().catch(() => {}); };
    petvid.onerror = () => pet.classList.remove("hasvid");
    try { petvid.src = chrome.runtime.getURL("chars/" + charId + ".mp4"); petvid.load(); } catch (e) {}
    $("pav").textContent = c.emoji; $("pnm").textContent = c.name;
  }

  // ---- conversation history (persisted -> she remembers you across sessions) --
  function renderThread() {
    if (!history.length) { thread.innerHTML = `<div class="empty">say hi, or ask me to find something.</div>`; return; }
    thread.innerHTML = history.map(m => {
      const links = (m.links || []).map(l => `<a href="${escA(l.url)}" target="_blank" rel="noopener">${esc(l.label || l.url)} ↗</a>`).join("");
      const tag = m.tag ? `<span class="tag">${esc(m.tag)}</span>` : "";
      return `<div class="msg ${m.role === "you" ? "you" : "bot"}">${esc(m.text)}${links}${tag}</div>`;
    }).join("");
    thread.scrollTop = thread.scrollHeight;
  }
  function pushMsg(role, text, extra) { history.push(Object.assign({ role, text }, extra || {})); if (history.length > 40) history = history.slice(-40); renderThread(); store.set({ ["chat_" + charId]: history }); }
  function loadHistory() { store.get(["chat_" + charId], d => { history = Array.isArray(d["chat_" + charId]) ? d["chat_" + charId] : []; renderThread(); }); }
  function recentContext() { return history.slice(-8).map(m => ({ role: m.role === "you" ? "user" : "model", text: m.text })); }

  async function getSoul() {
    if (soulCache[charId]) return soulCache[charId];
    try { const r = await fetch(chrome.runtime.getURL(`souls/${charId}.md`)); soulCache[charId] = await r.text(); return soulCache[charId]; }
    catch (e) { return `You are ${cur().name}, a friendly browser guardian.`; }
  }

  // place the speech bubble ABOVE the pet, right-aligned, never overlapping it
  function positionNear() {
    const r = pet.getBoundingClientRect();
    const bw = bubble.offsetWidth || 220, bh = bubble.offsetHeight || 56;
    let left = Math.max(10, Math.min(r.right - bw, window.innerWidth - bw - 10));
    let top = r.top - bh - 14;
    if (top < 10) top = Math.min(window.innerHeight - bh - 10, r.bottom + 14); // flip below if no room
    bubble.style.left = left + "px"; bubble.style.top = top + "px";
  }
  function say(text, { speak = true } = {}) {
    if (!text) return;
    if (!panel.classList.contains("open")) {           // panel open -> the thread shows it; skip the floating bubble
      bubble.innerHTML = `<span class="nm">${esc(cur().name)}</span>${esc(text)}`;
      bubble.classList.add("show");
      positionNear();                                  // measure AFTER content is set
      clearTimeout(bubbleTimer);
      bubbleTimer = setTimeout(() => bubble.classList.remove("show"), Math.min(9000, 2600 + text.length * 45));
    }
    if (speak && !muted) voice(text);
  }
  async function voice(text) {
    const r = await send({ type: "SPEAK", text: text.slice(0, 300), voice: cur().voice, via: lastVia });
    if (r && r.audio) { try { if (audio) audio.pause(); audio = new Audio(r.audio); audio.play().catch(() => browserVoice(text)); return; } catch (e) {} }
    browserVoice(text);
  }
  function browserVoice(text) {
    try {
      if (!window.speechSynthesis) return;
      const u = new SpeechSynthesisUtterance(text);
      const map = { grandma: { rate: .9, pitch: .8 }, cat: { rate: 1.05, pitch: 1.5 }, dog: { rate: 1.25, pitch: 1.3 }, pupa: { rate: .95, pitch: 1.1 }, tungtung: { rate: 1.1, pitch: .7 }, sixseven: { rate: 1.15, pitch: 1.2 }, skibidi: { rate: 1.0, pitch: 1.0 } };
      Object.assign(u, map[charId] || {}); speechSynthesis.cancel(); speechSynthesis.speak(u);
    } catch (e) {}
  }

  // a character reply lands in the thread, the bubble, and the voice
  function botReply(text, extra) { if (!text) return; pushMsg("bot", text, extra); say(text); }

  async function rememberCurrentPage() {
    const text = document.body ? document.body.innerText.slice(0, 14000) : "";
    if (!text.trim()) { botReply("I could not find readable text on this page."); return; }
    const r = await send({ type: "SL_REMEMBER", title: document.title, url: location.href, text });
    if (r && r.ok) botReply("Saved this page to memory.", { tag: "Superlinked memory" });
    else botReply("Connect the Superlinked memory bridge first.", { tag: "Set Superlinked URL in API settings" });
  }

  async function recallMemories(query) {
    const r = await send({ type: "SL_RECALL", query });
    const links = (r && r.results || []).slice(0, 5).map(x => ({ url: x.url || "#", label: x.title || x.url || "Saved page" }));
    if (links.length) botReply("Here's what I remembered:", { links, tag: "Superlinked semantic recall" });
    else botReply("I do not have matching saved pages yet.", { tag: "Try: remember this page" });
  }

  // ---- the one agentic input: she decides what to do --------------------------
  async function submit() {
    const q = inp.value.trim(); if (!q) return;
    inp.value = ""; resetIdle(); lastVia = q;
    pushMsg("you", q);
    // hardcoded: fidget spinner (Six Seven's bit). ONLY on an imperative spin command,
    // never on a question that merely mentions spinners ("research fidget spinners").
    const spinWord = /\b(spin|spinner|fidget)\b/i.test(q);
    const askingAbout = /\b(research|history|what|whats|who|whose|when|where|why|how|best|top|review|reviews|about|explain|tell|find|search|buy|price|cheap|origin|invent|meaning|info)\b/i.test(q) || q.includes("?");
    if (spinWord && !askingAbout) {
      botReply(charId === "sixseven" ? "six seven!! 🌀 watch this spinnnn!!" : "okok - flick it and watch!");
      showSpinner(); return;
    }
    if (/\b(remember|save)\b.*\b(this|page|article|site)\b/i.test(q)) {
      botReply("Saving this page...");
      await rememberCurrentPage();
      return;
    }
    if (/\b(recall|remembered|what did i save|what did i see|that page i saved)\b/i.test(q)) {
      await recallMemories(q);
      return;
    }
    const soul = await getSoul(); const ctx = recentContext();
    const route = await send({ type: "ACT", soul, message: q, name: cur().name, history: ctx, via: q });
    const act = route && route.action;
    try {
      if (act === "navigate" && route.url) {
        botReply(route.say || "On it!", { links: [{ url: route.url, label: route.url }] });
        confetti(); await send({ type: "OPEN_URL", url: route.url });
      } else if (act === "site_search" && route.site) {
        botReply(route.say || "On it!", { tag: `exploring ${route.site} for "${route.query}"…` });
        confetti(); await send({ type: "SITE_SEARCH", site: route.site, query: route.query });
      } else if (act === "recall") {
        const r = await send({ type: "SL_RECALL", query: route.query });
        const links = (r && r.results || []).slice(0, 5).map(x => ({ url: x.url || "#", label: x.title || x.url }));
        botReply(route.say || "Here's what I remembered:", links.length ? { links } : { tag: "(connect Superlinked to recall saved pages)" });
      } else if (act === "remember") {
        await rememberCurrentPage();
      } else if (act === "play_game") {
        botReply(route.say || "let's play - right here!"); confetti();
        const g = await send({ type: "GAME_URL", query: route.query || q });
        showGameOverlay((g && g.url) || "https://game-factory.tech");
      } else if (act === "perform") {
        botReply(route.say || "wheee! watch this!"); flyAround();
      } else if (act === "page_qa") {
        const text = document.body ? document.body.innerText.slice(0, 14000) : "";
        const r = await send({ type: "PAGE_QA", soul, question: q, text, history: ctx, via: q });
        botReply(r.text || "(couldn't read this page)", r.minima ? { tag: `🧠 ${r.minima}` } : {});
      } else if (act === "web_research") {
        const r = await send({ type: "RESEARCH", query: route.query || q, via: q });
        if (r && r.results) { botReply(r.answer || "here's what I found.", { links: r.results.slice(0, 5).map(x => ({ url: x.url, label: x.title || x.url })), tag: "✦ Tavily web search · Gemini" }); }
        else botReply("connect Tavily and I'll search the whole web.", {});
      } else {
        const r = await send({ type: "CHAT", soul, message: q, context: document.title, history: ctx, via: q });
        botReply(r.text || "(hmm, no answer)", r.minima ? { tag: `🧠 ${r.minima}` } : {});
      }
    } catch (e) { botReply("oops, something hiccuped - try again?"); }
  }

  // ---- behaviors: nap when idle, wander across the page -----------------------
  function resetIdle() { pet.classList.remove("nap"); pet.style.animation = ""; clearTimeout(idleTimer); idleTimer = setTimeout(() => { if (!alive()) return teardown(); pet.classList.add("nap"); puff("💤"); }, 45000); }
  function puff(txt) { const r = pet.getBoundingClientRect(); const z = document.createElement("div"); z.className = "zzz"; z.textContent = txt; z.style.left = (r.right - 18) + "px"; z.style.top = (r.top - 6) + "px"; root.appendChild(z); z.animate([{ opacity: 0 }, { opacity: 1, offset: .3 }, { opacity: 0, transform: "translateY(-34px)" }], { duration: 1800, easing: "ease-out" }).onfinish = () => z.remove(); }
  // confetti burst from the pet (celebrate actions / tricks)
  function confetti(cx, cy) {
    const r = pet.getBoundingClientRect();
    cx = cx ?? (r.left + 42); cy = cy ?? (r.top + 18);
    const cols = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#c780ff", "#ff9bb3"];
    for (let i = 0; i < 28; i++) {
      const p = document.createElement("div"); const sz = 6 + Math.random() * 6;
      p.style.cssText = `position:fixed;left:${cx}px;top:${cy}px;width:${sz}px;height:${sz}px;background:${cols[i % cols.length]};border-radius:${Math.random() < .5 ? "50%" : "2px"};z-index:6;pointer-events:none`;
      root.appendChild(p);
      const ang = Math.random() * Math.PI * 2, dist = 60 + Math.random() * 120;
      p.animate([{ transform: "translate(0,0) rotate(0)", opacity: 1 }, { transform: `translate(${Math.cos(ang) * dist}px,${Math.sin(ang) * dist + 170}px) rotate(${Math.random() * 720}deg)`, opacity: 0 }],
        { duration: 1100 + Math.random() * 700, easing: "cubic-bezier(.2,.6,.4,1)" }).onfinish = () => p.remove();
    }
  }
  function spark(x, y, ch) { const s = document.createElement("div"); s.textContent = ch; s.style.cssText = `position:fixed;left:${x}px;top:${y}px;font-size:18px;z-index:3;pointer-events:none`; root.appendChild(s); s.animate([{ opacity: .9, transform: "scale(1)" }, { opacity: 0, transform: "scale(.5) translateY(20px)" }], { duration: 900 }).onfinish = () => s.remove(); }

  function specialMove() {
    if (!alive()) return teardown();
    if (panel.classList.contains("open")) return scheduleWander();
    const vw = window.innerWidth;
    pet.style.transition = "transform 2.4s cubic-bezier(.4,0,.4,1)";
    pet.style.transform = `translateX(${-(vw - 220)}px)`;
    let n = 0; const trail = setInterval(() => { const r = pet.getBoundingClientRect(); spark(r.right - 12, r.top + 28 + Math.random() * 24, ["✨", "💫", "🌊", "⭐"][n % 4]); if (++n > 14) clearInterval(trail); }, 150);
    setTimeout(() => { pet.style.transform = "translateX(0)"; }, 2600);
    setTimeout(() => { clearInterval(trail); pet.style.transition = ""; confetti(); scheduleWander(); }, 5400);
  }
  function scheduleWander() { clearTimeout(wanderTimer); wanderTimer = setTimeout(specialMove, 24000 + Math.random() * 26000); }

  // "fly around the screen, crazy animate" -> zoom to random spots with confetti + spins
  function flyAround() {
    clearTimeout(wanderTimer); resetIdle();
    let i = 0; const moves = 7;
    pet.style.transition = "left .5s cubic-bezier(.34,1.4,.5,1), top .5s cubic-bezier(.34,1.4,.5,1), transform .5s";
    (function step() {
      if (!alive()) return teardown();
      if (i++ >= moves) { pet.style.transition = ""; pet.style.transform = ""; confetti(); scheduleWander(); return; }
      const x = 30 + Math.random() * Math.max(60, window.innerWidth - 150);
      const y = 30 + Math.random() * Math.max(60, window.innerHeight - 150);
      pet.style.right = "auto"; pet.style.bottom = "auto"; pet.style.left = x + "px"; pet.style.top = y + "px";
      pet.style.transform = `rotate(${Math.random() * 80 - 40}deg) scale(1.12)`;
      confetti(x + 42, y + 20);
      setTimeout(step, 520);
    })();
  }

  function togglePanel(open) { panel.classList.toggle("open", open ?? !panel.classList.contains("open")); if (panel.classList.contains("open")) { resetIdle(); renderThread(); inp.focus(); } }
  let dragMoved = false;
  pet.addEventListener("click", () => { if (!dragMoved) togglePanel(); });
  $("px").addEventListener("click", () => togglePanel(false));
  $("pmute").addEventListener("click", () => { muted = !muted; store.set({ muted }); $("pmute").textContent = muted ? "🔇" : "🔊"; });
  $("go").addEventListener("click", submit);
  inp.addEventListener("keydown", e => { if (e.key === "Enter") submit(); });

  // voice input (push-to-talk)
  $("mic").addEventListener("click", () => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { say("voice input isn't supported in this browser, sweetie.", { speak: false }); return; }
    if (recog) { try { recog.stop(); } catch (e) {} return; }
    recog = new SR(); recog.lang = "en-US"; recog.interimResults = false; recog.maxAlternatives = 1;
    $("mic").classList.add("on"); say("i'm listening...", { speak: false });
    recog.onresult = e => { inp.value = (e.results[0][0].transcript || "").trim(); };
    recog.onend = () => { recog = null; $("mic").classList.remove("on"); if (inp.value.trim()) submit(); };
    recog.onerror = () => { recog = null; $("mic").classList.remove("on"); };
    try { recog.start(); } catch (e) { recog = null; $("mic").classList.remove("on"); }
  });

  // drag
  let dragging = false, sx, sy, ox, oy;
  pet.addEventListener("pointerdown", e => { dragging = true; dragMoved = false; sx = e.clientX; sy = e.clientY; const r = pet.getBoundingClientRect(); ox = r.left; oy = r.top; pet.setPointerCapture(e.pointerId); });
  pet.addEventListener("pointermove", e => { if (!dragging) return; const dx = e.clientX - sx, dy = e.clientY - sy; if (Math.abs(dx) + Math.abs(dy) > 5) dragMoved = true; pet.style.right = "auto"; pet.style.bottom = "auto"; pet.style.left = (ox + dx) + "px"; pet.style.top = (oy + dy) + "px"; });
  pet.addEventListener("pointerup", () => { dragging = false; resetIdle(); });

  // ---- visible site search: the character walks to the search bar and types -----
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const visEl = e => e && e.offsetParent !== null && e.getClientRects().length;
  function hostMatch(h) { try { const base = String(h || "").replace(/^www\./, "").split(".")[0]; return base && location.hostname.includes(base); } catch (e) { return false; } }

  function checkPending() {
    store.get(["pendingSearch", "pendingSuggest"], d => {
      const now = Date.now();
      const ps = d.pendingSearch;
      if (ps && now - (ps.ts || 0) < 60000 && hostMatch(ps.host)) {
        // keep pendingSearch until the search actually finds the box, so a redirect
        // (e.g. adidas.com -> adidas.co.uk) lets the next page-load retry it.
        setTimeout(() => runVisibleSearch(ps.query), 1400);
        return;
      }
      const sg = d.pendingSuggest;
      if (sg && now - (sg.ts || 0) < 90000 && hostMatch(sg.host) && /[?&/](q|k|query|search|searchterm)=|\/search|\/s\b|results/i.test(location.href)) {
        store.remove("pendingSuggest");
        setTimeout(() => suggestFromResults(sg.query), 2600);
      }
    });
  }
  function dismissCookies() {
    return new Promise(res => {
      try {
        const btns = [...document.querySelectorAll('button,[role="button"],a')];
        const b = btns.find(x => /reject all|only necessary|necessary only|essential only/i.test(x.textContent || "")) ||
          btns.find(x => /accept all|accept cookies|i agree|got it|allow all/i.test(x.textContent || ""));
        if (b) b.click();
      } catch (e) {}
      setTimeout(res, 800);
    });
  }
  function findSearchInput() {
    return new Promise(res => {
      let tries = 0;
      const sel = 'input[type="search"],input[name="q"],input[name*="search" i],input[name*="term" i],input[placeholder*="search" i],input[aria-label*="search" i],input[id*="search" i]';
      (function tick() {
        let inp = [...document.querySelectorAll(sel)].find(visEl);
        if (!inp) {
          const togs = [...document.querySelectorAll('button,[role="button"],a')].filter(b => /search/i.test((b.getAttribute("aria-label") || "") + " " + (b.className || "") + " " + (b.id || "") + " " + (b.getAttribute("data-auto-id") || "")));
          const t = togs.find(visEl); if (t) { try { t.click(); } catch (e) {} }
          inp = [...document.querySelectorAll(sel)].find(visEl);
        }
        if (inp && visEl(inp)) return res(inp);
        if (++tries > 16) return res(null);
        setTimeout(tick, 500);
      })();
    });
  }
  function movePetTo(el) {
    return new Promise(res => {
      const r = el.getBoundingClientRect();
      pet.style.transition = "left .6s cubic-bezier(.34,1.3,.5,1),top .6s cubic-bezier(.34,1.3,.5,1)";
      pet.style.right = "auto"; pet.style.bottom = "auto";
      pet.style.left = Math.max(8, Math.min(window.innerWidth - 92, r.left + r.width / 2 - 42)) + "px";
      pet.style.top = Math.max(8, r.bottom + 8) + "px";
      setTimeout(() => { pet.style.transition = ""; res(); }, 720);
    });
  }
  async function typeVisibly(inp, q) {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    inp.focus();
    for (let i = 1; i <= q.length; i++) {
      setter.call(inp, q.slice(0, i));
      inp.dispatchEvent(new Event("input", { bubbles: true }));
      try { inp.dispatchEvent(new InputEvent("input", { bubbles: true, data: q[i - 1], inputType: "insertText" })); } catch (e) {}
      await sleep(75);
    }
    inp.dispatchEvent(new Event("change", { bubbles: true }));
  }
  function submitSearch(inp) {
    const form = inp.closest("form");
    if (form) { try { form.requestSubmit ? form.requestSubmit() : form.submit(); return; } catch (e) {} }
    const sb = [...document.querySelectorAll('button[type="submit"],button[aria-label*="search" i],[role="button"][aria-label*="search" i]')].find(visEl);
    if (sb) { sb.click(); return; }
    ["keydown", "keypress", "keyup"].forEach(t => inp.dispatchEvent(new KeyboardEvent(t, { key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true })));
    const before = location.href, val = inp.value;
    setTimeout(() => { if (location.href === before) location.href = location.origin + "/search?q=" + encodeURIComponent(val); }, 2600); // URL fallback
  }
  async function runVisibleSearch(query) {
    await dismissCookies();
    const inp = await findSearchInput();
    if (!inp) return;   // not found (maybe mid-redirect) - leave pendingSearch so the next page-load retries
    store.remove("pendingSearch");            // committed on THIS page
    store.set({ pendingSuggest: { host: location.hostname, query, ts: Date.now() } });
    togglePanel(true);
    lastVia = `find "${query}"`;
    botReply(`found the search bar - looking for "${query}"...`);
    await movePetTo(inp);
    try { spark(inp.getBoundingClientRect().left + 20, inp.getBoundingClientRect().top - 6, "✨"); } catch (e) {}
    await typeVisibly(inp, query);
    await sleep(450);
    submitSearch(inp);
  }
  // scrape REAL product cards (image + short text + product link) off a results page
  function extractProducts() {
    const out = [], seen = new Set(), priceRe = /(?:£|\$|€)\s?\d[\d.,]*/;
    const bad = /below|under|up to|from\s*[£$€]|trending|view all|discover|new in|categor|explore|shop all/i;
    const conts = [...document.querySelectorAll('li,article,[class*="product" i],[class*="card" i],[data-auto-id*="product" i]')];
    for (const c of conts) {
      if (c.querySelector('li,article')) continue;                 // leaf-ish cards only
      const txt = (c.textContent || "").replace(/\s+/g, " ").trim();
      if (txt.length > 220) continue;                              // product cards are short
      const pm = txt.match(priceRe); if (!pm || bad.test(txt)) continue;   // skip promo/category tiles
      const link = c.querySelector('a[href]'); if (!link || !/^https?:/.test(link.href) || seen.has(link.href)) continue;
      if (!c.querySelector('img')) continue;                       // real products have an image
      let name = ((c.querySelector('h1,h2,h3,h4,[class*="title" i],[class*="name" i]') || {}).textContent || "").trim() ||
        ((c.querySelector("img") || {}).alt || "").trim() || (link.getAttribute("aria-label") || "").trim() || (link.textContent || "").trim();
      name = name.replace(priceRe, "").replace(/\s+/g, " ").trim().slice(0, 70); if (!name) continue;
      seen.add(link.href); out.push({ name, price: pm[0].replace(/\s/g, ""), url: link.href });
      if (out.length >= 12) break;
    }
    return out;
  }
  async function suggestFromResults(query) {
    togglePanel(true);
    lastVia = `find "${query}"`;
    const soul = await getSoul();
    const products = extractProducts();
    if (products.length) {
      // Superlinked SIE: semantically rank the products by the query (open embeddings)
      let ranked = products, slBit = "";
      const rk = await send({ type: "SIE_RANK", query, items: products.map(p => p.name), via: lastVia });
      if (rk && rk.ok && Array.isArray(rk.order)) { ranked = rk.order.map(i => products[i]).filter(Boolean); slBit = " · Superlinked semantic rank"; }
      const list = ranked.slice(0, 10).map((p, i) => `${i + 1}. ${p.name} - ${p.price}`).join("\n");
      const prompt = `I searched "${query}". Products (ranked by relevance):\n${list}\n\nPick the 2-3 best for me - call out the CHEAPEST explicitly. For EACH, name + ONE short reason it fits (price, value, use or style). Specific and helpful, in your character's voice. Under 60 words.`;
      const r = await send({ type: "CHAT", soul, context: document.title, history: recentContext(), via: lastVia, message: prompt });
      const minimaBit = (r && r.minima) ? ` · Mubit ${r.minima}` : "";
      const tag = `✦ site search${slBit} · Gemini reasoning${minimaBit}` + (muted ? "" : " · SLNG voice");
      botReply((r && r.text) || "here's what I'd pick for you!", { links: ranked.slice(0, 4).map(p => ({ url: p.url, label: `${p.name} - ${p.price}` })), tag });
    } else {
      const text = document.body ? document.body.innerText.slice(0, 9000) : "";
      const r = await send({ type: "PAGE_QA", soul, question: `I searched "${query}". Recommend 2-3 results that fit me and why, briefly.`, text, via: lastVia });
      if (r && (r.text || !r.error)) botReply(r.text || "here's what I found!", { tag: "✦ site search · Gemini reasoning" });
    }
  }

  // play an embedded game from the factory as an in-browser overlay
  function showGameOverlay(url) {
    let ov = root.getElementById("gov"); if (ov) ov.remove();
    ov = document.createElement("div"); ov.id = "gov"; ov.className = "gov";
    ov.innerHTML = `<div class="govbg"></div><div class="govbox"><div class="govbar"><span>🎮 Patronus Arcade</span><span class="govx" id="govx">✕</span></div><iframe id="goviframe" allow="autoplay; fullscreen; gamepad; microphone" referrerpolicy="no-referrer"></iframe></div>`;
    root.appendChild(ov);
    root.getElementById("govx").addEventListener("click", () => ov.remove());
    ov.querySelector(".govbg").addEventListener("click", () => ov.remove());
    root.getElementById("goviframe").src = url;
  }

  // ---- fidget spinner: Six Seven's signature. flick it, watch it spin down -----
  function spinnerSVG(color) {
    const arms = [0, 120, 240].map(a => { const r = a * Math.PI / 180, cx = 100 + 58 * Math.cos(r), cy = 100 + 58 * Math.sin(r); return `<line x1="100" y1="100" x2="${cx.toFixed(1)}" y2="${cy.toFixed(1)}" stroke="${color}" stroke-width="46" stroke-linecap="round"/>`; }).join("");
    const lobes = [0, 120, 240].map(a => { const r = a * Math.PI / 180, cx = 100 + 58 * Math.cos(r), cy = 100 + 58 * Math.sin(r); return `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="33" fill="${color}"/><circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="17" fill="#0e141d"/><circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="9" fill="#fff" opacity=".85"/>`; }).join("");
    return `<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg"><g>${arms}${lobes}<circle cx="100" cy="100" r="40" fill="${color}"/><circle cx="100" cy="100" r="23" fill="#0e141d"/><circle cx="100" cy="100" r="13" fill="#fff" opacity=".9"/><circle cx="100" cy="100" r="6" fill="${color}"/></g></svg>`;
  }
  let spinRAF = 0;
  function closeSpinner() { cancelAnimationFrame(spinRAF); spinRAF = 0; const o = root.getElementById("spin"); if (o) o.remove(); }
  function showSpinner() {
    closeSpinner();
    const c = cur();
    const sixseven = charId === "sixseven";
    const ov = document.createElement("div"); ov.id = "spin"; ov.className = "spin";
    ov.innerHTML = `<div class="spinbg"></div><span class="spinx" id="spinx">✕</span>
      <div class="spincap">${esc(sixseven ? "six seven!! 🌀" : c.name + " says: spinnnn!")}</div>
      <div class="spinwrap" id="spinwrap" title="flick to spin">${spinnerSVG(c.accent)}</div>
      <div class="spinrpm" id="spinrpm">0 RPM</div>
      <div class="spinhint">flick or drag the spinner</div>`;
    root.appendChild(ov);
    const wrap = root.getElementById("spinwrap"), g = wrap.querySelector("svg"), rpmEl = root.getElementById("spinrpm");
    root.getElementById("spinx").addEventListener("click", closeSpinner);
    ov.querySelector(".spinbg").addEventListener("click", closeSpinner);

    let angle = 0, vel = 26, dragging = false, lastA = 0;   // open with a big flick
    const ptrAngle = e => { const r = wrap.getBoundingClientRect(); return Math.atan2(e.clientY - (r.top + r.height / 2), e.clientX - (r.left + r.width / 2)) * 180 / Math.PI; };
    wrap.addEventListener("pointerdown", e => { dragging = true; vel = 0; lastA = ptrAngle(e); try { wrap.setPointerCapture(e.pointerId); } catch (x) {} });
    wrap.addEventListener("pointermove", e => { if (!dragging) return; const a = ptrAngle(e); let d = a - lastA; if (d > 180) d -= 360; if (d < -180) d += 360; angle += d; vel = d; lastA = a; });
    const release = () => { if (!dragging) return; dragging = false; if (Math.abs(vel) < 3) vel = 20 + Math.random() * 10; confetti(window.innerWidth / 2, window.innerHeight / 2); };
    wrap.addEventListener("pointerup", release);
    wrap.addEventListener("pointercancel", release);

    let lastT = 0;
    (function frame(t) {
      if (!alive()) return closeSpinner();
      if (!root.getElementById("spin")) return;
      const dt = lastT ? Math.min(48, t - lastT) : 16; lastT = t;
      if (!dragging) { angle += vel * dt / 16; vel *= 0.986; if (Math.abs(vel) < 0.03) vel = 0; }
      g.style.transform = `rotate(${angle}deg)`;
      rpmEl.textContent = Math.round(Math.abs(vel) * 10.4) + " RPM";
      spinRAF = requestAnimationFrame(frame);
    })(0);
    confetti(window.innerWidth / 2, window.innerHeight / 2);
  }

  // boot
  store.get(["activeChar", "muted"], d => {
    if (d.activeChar && CHARS[d.activeChar]) charId = d.activeChar;
    muted = !!d.muted; $("pmute").textContent = muted ? "🔇" : "🔊";
    applyChar(); loadHistory();                         // per-character chat
    setTimeout(() => say(GREETING[charId] || "hi!"), 1200);
    resetIdle(); scheduleWander();
    checkPending();                                     // visible site search / suggest
  });
  chrome.storage.onChanged.addListener(ch => {
    if (ch.activeChar && CHARS[ch.activeChar.newValue]) { charId = ch.activeChar.newValue; applyChar(); loadHistory(); say(GREETING[charId] || "hi!"); }
    if (ch.muted) { muted = !!ch.muted.newValue; $("pmute").textContent = muted ? "🔇" : "🔊"; }
  });
})();
