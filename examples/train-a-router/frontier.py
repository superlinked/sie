"""Offline, dependency-free view of recorded calibration cost–quality candidates."""

import json
import math
from pathlib import Path
import shlex


METHODS = {
    "base": "Unchanged Laya",
    "lora": "LoRA Laya",
    "embedding": "Qwen embeddings + heads",
    "topic": "Topic control",
    "always_strong": "Always Sol",
    "always_cheap": "Always Luna",
    "random": "Random routing (expected)",
}
COSTS = {"answer": "answer_usd_per_10000", "total": "total_usd_per_10000"}


def pareto_ids(points, cost_key):
    """Exact recorded-point dominance; keep metric ties and negative quality gaps."""
    ordered = sorted(
        (p for p in points if p.get(cost_key) is not None and not p.get("analytic_random", False)),
        key=lambda p: (p[cost_key], p["quality_gap_pp"]),
    )
    kept, best_gap, index = set(), math.inf, 0
    while index < len(ordered):
        end = index + 1
        cost, gap = ordered[index][cost_key], ordered[index]["quality_gap_pp"]
        while end < len(ordered) and ordered[end][cost_key] == cost:
            end += 1
        if gap < best_gap:
            kept.update(p["point_id"] for p in ordered[index:end] if p["quality_gap_pp"] == gap)
        best_gap = min(best_gap, gap)
        index = end
    return kept


def _points(items):
    result, seen = [], set()
    for item in items:
        if item.get("method") not in METHODS:  # Diagnostic models are not deployment candidates.
            continue
        p = dict(item)
        p.pop("live_command", None)
        identity = p.get("point_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Frontier points require unique, nonempty point_id strings")
        seen.add(identity)
        candidate = p.get("candidate_id")
        if candidate is not None and type(candidate) is not int:
            raise ValueError("candidate_id must be an integer or null")
        for field in (
            "answer_accuracy_percent",
            "quality_gap_pp",
            "answer_usd_per_10000",
            "strong_share",
            "router_usd_per_10000",
            "total_usd_per_10000",
        ):
            value = p.get(field)
            if value is None and field in ("router_usd_per_10000", "total_usd_per_10000"):
                continue
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"{identity}: {field} must be finite")
            if field.endswith("usd_per_10000") and value < 0:
                raise ValueError(f"{identity}: cost cannot be negative")
        result.append(p)
    return result


def _indexes(points):
    indexes = {}
    for basis, key in COSTS.items():
        individual = set()
        for method in METHODS:
            individual.update(pareto_ids([p for p in points if p["method"] == method], key))
        indexes[basis] = {"per_method": sorted(individual), "combined": sorted(pareto_ids(points, key))}
    return indexes


def _command(run, point):
    if point["method"] not in ("base", "lora", "embedding") or point["candidate_id"] is None:
        return None
    args = ["uv", "run", "--frozen", "--group", "live"]
    if point["method"] in ("base", "lora"):
        args += ["--group", "training"]
    args += [
        "python",
        "router.py",
        "answer",
        "--run",
        str(run),
        "--method",
        point["method"],
        "--candidate-id",
        str(point["candidate_id"]),
        "--question",
        "What is 2 + 2?",
        "--answer-type",
        "integer",
        "--output",
        "answers/example",
    ]
    return "# Try this calibration candidate on a new question; use a fresh output path.\n" + shlex.join(args)


def _payload(report, run):
    calibration = _points(report.get("calibration", []))
    if not calibration:
        raise ValueError("A frontier view requires calibration candidates")
    test = _points(report.get("test", []))
    by_id = {p["point_id"]: p for p in calibration}
    for point in test:
        original = by_id.get(point["point_id"])
        if original is None or (original["method"], original["candidate_id"]) != (
            point["method"],
            point["candidate_id"],
        ):
            raise ValueError("Every test point must map to the same calibration candidate")
    for point in calibration:
        point["live_command"] = _command(run, point)
    return {
        "format": report.get("format"),
        "methods": METHODS,
        "run": str(run),
        "calibration": calibration,
        "test": test,
        "frontiers": {"calibration": _indexes(calibration), "test": _indexes(test)},
        "provenance": report.get("provenance", {}),
        "interpretation": report.get("interpretation", {}),
    }


def write_view(report, output, run):
    """Write one standalone HTML file; the caller owns the adjacent source JSON."""
    payload = _payload(report, run)
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    # A script element's HTML parser recognizes </script> even inside JSON strings.
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_HTML.replace("__FRONTIER_DATA__", encoded))
    return {
        "format": "train-a-router-frontier-view-v1",
        "html": str(path),
        "calibration_points": len(payload["calibration"]),
        "test_points": len(payload["test"]),
    }


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline';
style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Routing cost and quality</title>
<style>
:root{color-scheme:light;--ink:#172c3d;--muted:#536575;--line:#d8e1e7;--accent:#176a62;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#f4f7f8;color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:1500px;padding:32px 28px 48px;margin:auto}h1{font-size:clamp(28px,3.2vw,44px);line-height:1.15;
letter-spacing:-.035em;margin:10px 0 14px}h2{font-size:21px;margin:0 0 12px}h3{font-size:17px;margin:18px 0 8px}
p{margin:8px 0 14px}.eyebrow{color:var(--accent);font-size:12px;font-weight:750;letter-spacing:.13em;text-transform:uppercase}
.intro{max-width:1000px;color:var(--muted)}.toolbar,.card{border:1px solid var(--line);background:var(--paper);border-radius:12px}
.toolbar{padding:16px 20px;margin:22px 0 14px;display:flex;flex-wrap:wrap;gap:16px 24px;align-items:start}
label.field{display:grid;gap:5px;font-size:13px;font-weight:700}select,input,button{font:inherit}
select,input[type=number]{color:var(--ink);background:white;border:1px solid #a7b6c1;border-radius:5px;padding:8px}
button{border:1px solid #a7b6c1;border-radius:5px;padding:8px 12px;background:white;color:var(--ink);cursor:pointer}
button:hover{background:#eaf2f3}button:disabled{opacity:.5;cursor:default}input[type=checkbox]{accent-color:var(--accent)}
:focus-visible{outline:3px solid #087dcb;outline-offset:3px}.layout{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px}
.chart-card{padding:18px 20px 12px;min-width:0}.card.side{padding:20px}.legend{display:flex;flex-wrap:wrap;gap:8px 16px;margin:6px 0 12px}
.legend label{display:flex;align-items:center;gap:5px;font-size:13px;cursor:pointer}.swatch{width:10px;height:10px;border-radius:50%}
.muted,.hint{color:var(--muted)}.hint{font-size:13px}.notice{border-left:4px solid #a46915;background:#fff5e2;padding:10px 14px;
font-size:14px;margin:0 0 14px}.status{font-size:13px;min-height:20px;margin:8px 0}.plot{width:100%;height:auto;display:block}
.plot text{font-family:system-ui,sans-serif;font-size:12px;fill:var(--muted)}.plot .axis-label{font-size:14px;fill:var(--ink)}
.plot .point{cursor:pointer}.plot .point:focus{outline:none;stroke:#000;stroke-width:3px}.plot .point:hover{stroke:#000;stroke-width:2px}
dl{display:grid;grid-template-columns:1fr auto;gap:7px 14px;font-size:14px}dt{color:var(--muted)}dd{margin:0;text-align:right;
font-variant-numeric:tabular-nums}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f7;border-radius:6px;
padding:12px;font:12px/1.5 ui-monospace,monospace;max-height:280px;overflow:auto}.rule{font:12px/1.5 ui-monospace,monospace}
.table-card{margin-top:16px;padding:20px}.table-wrap{overflow:auto;max-height:430px}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:9px 12px;border-bottom:1px solid #e2e8ec;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{background:#f6f8fa;position:sticky;top:0}td button{padding:3px 8px;font-size:12px}.pager{display:flex;align-items:center;gap:12px;margin:12px 0}
.candidate-form{display:flex;gap:8px;align-items:end;flex-wrap:wrap}.candidate-form input{width:100px}.table-head{display:flex;justify-content:space-between;
align-items:center;gap:15px;flex-wrap:wrap}details{margin:18px 0}summary{cursor:pointer;font-weight:650}.facts{max-width:1000px;font-size:14px}
[hidden]{display:none!important}@media(max-width:1020px){.layout{grid-template-columns:1fr}.side{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:640px){main{padding:20px 12px}.toolbar{padding:12px}.chart-card{padding:12px 8px}.side{display:block!important}
.table-card{padding:12px}.plot text{font-size:13px}.toolbar .field{width:100%}.toolbar select{width:100%}}
</style>
</head>
<body><main>
<header><div class="eyebrow">Choose an operating point</div><h1>What does better routing buy?</h1>
<p class="intro">Compare cost with answer quality across every recorded cutoff. Move toward the upper left:
lower cost and higher accuracy. A frontier point has no alternative that is at least as good on both axes and better on one.</p>
<p class="hint">Answer models: <strong>GPT-6 Luna / GPT-6.1 Sol</strong> via OpenAI.
The embedding router uses <strong>Qwen3-Embedding-4B</strong> via SIE; it does not generate the answers.</p></header>
<div class="toolbar">
<label class="field">Population<select id="role"><option value="calibration">Calibration — choose a candidate</option>
<option value="test" id="test-option">Test — retrospective inspection only</option></select></label>
<label class="field">Cost basis<select id="basis"><option value="answer">Answer models only — all methods</option>
<option value="total">Priced total — known router fees included</option></select></label>
<label class="field">Candidates shown<select id="scope"><option value="per_method">Each method's calibration frontier</option>
<option value="combined">Combined calibration frontier</option><option value="all">Every recorded candidate</option>
<option value="hindsight" id="hindsight-option" hidden disabled>Test frontier — hindsight only</option></select></label>
</div>
<p class="notice" id="role-note">Choose using calibration. These empirical tradeoffs are estimates, not guarantees on future questions.</p>
<div class="layout"><section class="card chart-card" aria-labelledby="chart-heading">
<h2 id="chart-heading">Calibration tradeoffs</h2><div class="legend" id="legend" aria-label="Visible methods"></div>
<p class="hint" id="cost-note"></p><svg class="plot" id="plot" viewBox="0 0 900 520" role="group"
aria-label="Cost versus answer accuracy. The candidate table provides the same points and keyboard controls."></svg>
<p class="status" id="status" aria-live="polite"></p>
<p class="hint">Dots are recorded candidates, not interpolated policies. The dashed line is the expected result of
randomly choosing Luna or Sol independently of the question. It is not a deterministic frontier.
Coincident points and exact ties remain available in the table.</p>
</section><aside class="card side" aria-label="Candidate details">
<div><h2 id="details-title">Inspect a candidate</h2><p id="details-hint" class="hint">Hover over a dot for a preview.
Click a dot or use the table to inspect its metrics.</p><div id="metrics"></div><div id="rule" class="rule"></div></div>
<div><h3>Try a calibration choice</h3><p id="command-note" class="hint">Select a concrete calibration candidate to see a command.
No candidate is selected automatically.</p><pre id="command" hidden></pre>
<p class="hint" id="command-scope" hidden>The command makes paid answer-model calls. Embedding routing also needs SIE Cloud
or your own encoder. Native routing needs downloaded weights and the training dependency group; its default device is CPU.
Replace the sample question and use a new output path.</p></div></aside></div>
<section class="card table-card" aria-labelledby="table-heading"><div class="table-head"><div><h2 id="table-heading">Candidate table</h2>
<p class="hint">The table follows the chart filters and retains exact ties. No quality target is preferred.</p></div>
<div class="candidate-form"><label class="field">Find method<select id="find-method"><option value="base">Unchanged Laya</option>
<option value="lora">LoRA Laya</option><option value="embedding">Qwen embeddings + heads</option><option value="topic">Topic control</option></select></label>
<label class="field">Candidate ID<input id="find-id" type="number" step="1" aria-label="Candidate ID"></label>
<button id="find">Inspect ID</button></div></div><p id="find-status" class="hint" aria-live="polite"></p>
<div class="table-wrap"><table><thead><tr><th scope="col">Method / candidate</th><th scope="col">Accuracy</th>
<th scope="col">Gap vs Sol</th><th scope="col">Answers / 10k</th><th scope="col">Router / 10k</th>
<th scope="col">Priced total / 10k</th><th scope="col">Sol share</th><th scope="col">Details</th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="pager"><button id="previous">Previous</button><span id="page" class="hint"></span><button id="next">Next</button></div></section>
<details class="facts"><summary>How to read the frontier and its limits</summary>
<p>The frontier uses exact recorded point estimates: another candidate must cost no more and have no larger quality loss,
with at least one strict improvement. Ties and negative losses are retained. Filtering methods does not recalculate the
combined frontier. Random expectation points are shown separately and do not remove deterministic candidates.</p>
<p>Calibration-frontier membership is recalculated for each cost basis. In the test view, those same candidate IDs are
shown at their test outcomes; no cutoff is refitted. The optional test frontier is hindsight only and cannot establish
the performance of a newly selected deployment policy. Published calibration data are also exposed: a new claim needs fresh evaluation.</p>
<p>Answer-only cost omits all routing overhead. Priced total includes only costs present in the source report, not a complete
deployment invoice. Unknown native or topic hosting is never zero. Always-arm rules may bypass routing and have a known total.
The test view cannot create a live command.</p>
<p>There are no invented confidence bands. Point estimates across a sweep are correlated; only explicitly supplied intervals
are displayed in details. The original report contains the uncertainty analysis for its predeclared settings.</p>
</details><details><summary>Source and interpretation</summary><pre id="provenance"></pre></details>
<noscript><p class="notice">Enable JavaScript to explore this offline view. The adjacent JSON file contains every source point.</p></noscript>
</main>
<script id="frontier-data" type="application/json">__FRONTIER_DATA__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("frontier-data").textContent);
const $=id=>document.getElementById(id), NS="http://www.w3.org/2000/svg";
const colors={base:"#687b8c",lora:"#ad4a20",embedding:"#117a6c",topic:"#7a50a3",
always_strong:"#213c60",always_cheap:"#7b5d12",random:"#526571"};
const labels=data.methods, cal=new Map(data.calibration.map(p=>[p.point_id,p]));
const enabled=new Set(Object.keys(labels));
const state={role:"calibration",basis:"answer",scope:"per_method",page:0,inspected:null};
const number=(v,d=2)=>Number(v).toFixed(d), money=v=>v===null||v===undefined?"Unpriced":"$"+number(v,3);
const cost=p=>p[state.basis==="answer"?"answer_usd_per_10000":"total_usd_per_10000"];
const name=p=>labels[p.method]+(p.candidate_id===null?
  (p.method==="random"?" · "+number(100*p.strong_share,1)+"% Sol ["+p.point_id+"]":""):" #"+p.candidate_id);
const text=(tag,value,parent,attrs={})=>{const el=document.createElement(tag);el.textContent=value;
Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));parent.append(el);return el;};
const svg=(tag,attrs,parent,value)=>{const el=document.createElementNS(NS,tag);
Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(value!==undefined)el.textContent=value;parent.append(el);return el;};
const always=p=>p.method==="always_strong"||p.method==="always_cheap"||p.method==="random";
function candidates(){
  const f=data.frontiers[state.scope==="hindsight"?"test":"calibration"][state.basis];
  const ids=new Set(f[state.scope==="per_method"?"per_method":"combined"]);
  return data[state.role].filter(p=>enabled.has(p.method)&&cost(p)!==null&&cost(p)!==undefined&&
    (state.scope==="all"||always(p)||ids.has(p.point_id)));
}
function title(p){
  const c=cal.get(p.point_id);
  return name(p)+"\n"+(state.role==="test"?"Test (retrospective)":"Calibration")+": "+
    number(p.answer_accuracy_percent)+"% correct; gap "+number(p.quality_gap_pp)+" pp; "+money(cost(p))+" / 10k"+
    (state.role==="test"?"\nCalibration: "+number(c.answer_accuracy_percent)+"% correct; gap "+
      number(c.quality_gap_pp)+" pp; "+money(cost(c))+" / 10k":"")+
    "\nSol share "+number(100*p.strong_share)+"%";
}
function metricBlock(p,label){
  text("h3",label,$("metrics"));const dl=document.createElement("dl");$("metrics").append(dl);
  const values=[["Answer accuracy",number(p.answer_accuracy_percent,3)+"%"],["Quality loss vs Sol",number(p.quality_gap_pp,3)+" pp"],
    ["Answer models / 10k",money(p.answer_usd_per_10000)],["Router / 10k",money(p.router_usd_per_10000)],
    ["Priced total / 10k",money(p.total_usd_per_10000)],["Sol share",number(100*p.strong_share,2)+"%"]];
  values.forEach(([k,v])=>{text("dt",k,dl);text("dd",v,dl);});
  if(p.selected_targets_pp?.length)text("p","Original comparison setting(s): "+p.selected_targets_pp.join(", ")+" pp. These are not preferred targets.",$("metrics"),{class:"hint"});
  if(p.intervals&&Object.keys(p.intervals).length){text("p","Supplied intervals (source labels retained):",$("metrics"),{class:"hint"});
    text("pre",JSON.stringify(p.intervals,null,2),$("metrics"));}
}
function inspect(p){state.inspected=p.point_id;renderDetails(p);draw(candidates());}
function renderDetails(p){
  $("metrics").replaceChildren();$("rule").textContent="";$("command").hidden=true;$("command").textContent="";
  $("command-scope").hidden=true;
  if(!p){$("details-title").textContent="Inspect a candidate";$("details-hint").hidden=false;
    $("command-note").textContent="Select a concrete calibration candidate to see a command. No candidate is selected automatically.";return;}
  $("details-title").textContent=name(p);$("details-hint").hidden=true;
  metricBlock(cal.get(p.point_id),"Calibration");
  if(state.role==="test")metricBlock(p,"Test — retrospective");
  $("rule").textContent="Rule: "+(typeof p.rule==="string"?p.rule:JSON.stringify(p.rule));
  if(state.role!=="calibration"){
    $("command-note").textContent="Inspection only. Return to calibration to make an explicit deployment choice; test results cannot generate a command.";
  }else if(p.live_command){
    $("command-note").textContent="This is your selected calibration candidate, not a recommendation or a future quality guarantee.";
    $("command").textContent=p.live_command;$("command").hidden=false;$("command-scope").hidden=false;
  }else{$("command-note").textContent="This control is for comparison. The live CLI supports concrete unchanged Laya, LoRA and embedding candidates.";}
}
function draw(points){
  const plot=$("plot");plot.replaceChildren();const W=900,H=520,L=76,R=22,T=28,B=72;
  if(!points.length){svg("text",{x:W/2,y:H/2,"text-anchor":"middle"},plot,"No priced candidates match these filters.");return;}
  // Keep scales fixed while filtering so hiding a method does not visually exaggerate gains.
  const domain=data[state.role].filter(p=>cost(p)!==null&&cost(p)!==undefined);
  let loX=Infinity,hiX=-Infinity,loY=Infinity,hiY=-Infinity;
  domain.forEach(p=>{loX=Math.min(loX,cost(p));hiX=Math.max(hiX,cost(p));
    loY=Math.min(loY,p.answer_accuracy_percent);hiY=Math.max(hiY,p.answer_accuracy_percent);});
  const padX=Math.max((hiX-loX)*.045,.05),padY=Math.max((hiY-loY)*.08,.15);
  loX=Math.max(0,loX-padX);hiX+=padX;loY-=padY;hiY+=padY;
  const x=v=>L+(v-loX)/(hiX-loX)*(W-L-R),y=v=>H-B-(v-loY)/(hiY-loY)*(H-T-B);
  for(let i=0;i<=5;i++){
    const xv=loX+(hiX-loX)*i/5,yv=loY+(hiY-loY)*i/5;
    svg("line",{x1:x(xv),y1:T,x2:x(xv),y2:H-B,stroke:"#e4eaee"},plot);
    svg("text",{x:x(xv),y:H-B+25,"text-anchor":"middle"},plot,"$"+number(xv));
    svg("line",{x1:L,y1:y(yv),x2:W-R,y2:y(yv),stroke:"#e4eaee"},plot);
    svg("text",{x:L-12,y:y(yv)+4,"text-anchor":"end"},plot,number(yv,1)+"%");
  }
  svg("text",{x:(L+W-R)/2,y:H-18,"text-anchor":"middle",class:"axis-label"},plot,
    (state.basis==="answer"?"Answer-model cost":"Priced total")+" per 10,000 requests (USD)");
  svg("text",{x:18,y:(T+H-B)/2,transform:"rotate(-90 18 "+((T+H-B)/2)+")",
    "text-anchor":"middle",class:"axis-label"},plot,"Answer accuracy (%) · higher is better");
  const strong=domain.find(p=>p.method==="always_strong"),cheap=domain.find(p=>p.method==="always_cheap");
  if(strong){svg("line",{x1:L,y1:y(strong.answer_accuracy_percent),x2:W-R,y2:y(strong.answer_accuracy_percent),
    stroke:"#a1afba","stroke-dasharray":"2 6"},plot);}
  if(enabled.has("random")&&strong&&cheap){const line=svg("line",{x1:x(cost(cheap)),y1:y(cheap.answer_accuracy_percent),
    x2:x(cost(strong)),y2:y(strong.answer_accuracy_percent),stroke:colors.random,"stroke-width":2,"stroke-dasharray":"7 5"},plot);
    svg("title",{},line,"Expected random mixture of Luna and Sol; no question-dependent selection.");}
  const layer=p=>p.method==="random"?0:always(p)?2:1;
  const ordered=[...points].sort((a,b)=>layer(a)-layer(b));
  ordered.forEach(p=>{
    const cx=x(cost(p)),cy=y(p.answer_accuracy_percent),chosen=p.point_id===state.inspected;
    const g=svg("g",{class:"point",tabindex:0,role:"button","aria-label":title(p),
      "data-point-id":p.point_id,fill:colors[p.method],stroke:chosen?"#172c3d":"#fff","stroke-width":chosen?2.5:.7},plot);
    const size=always(p)?5:3.4;
    if(p.method==="lora")svg("path",{d:`M ${cx} ${cy-size-1} L ${cx-size-1} ${cy+size} L ${cx+size+1} ${cy+size} Z`},g);
    else if(p.method==="embedding"||p.method==="always_cheap")svg("rect",{x:cx-size,y:cy-size,width:size*2,height:size*2},g);
    else if(p.method==="topic"||p.method==="always_strong")svg("path",{d:`M ${cx} ${cy-size-1} L ${cx+size+1} ${cy} L ${cx} ${cy+size+1} L ${cx-size-1} ${cy} Z`},g);
    else svg("circle",{cx,cy,r:size},g);
    svg("title",{},g,title(p));g.addEventListener("click",()=>inspect(p));
    g.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();inspect(p);}});
    if(chosen)svg("circle",{cx,cy,r:10,fill:"none",stroke:colors[p.method],"stroke-width":2,"pointer-events":"none"},plot);
  });
}
function table(points){
  const ordered=[...points].sort((a,b)=>cost(a)-cost(b)||a.quality_gap_pp-b.quality_gap_pp||a.point_id.localeCompare(b.point_id));
  const pages=Math.max(1,Math.ceil(ordered.length/25));state.page=Math.min(state.page,pages-1);$("rows").replaceChildren();
  ordered.slice(state.page*25,state.page*25+25).forEach(p=>{
    const row=document.createElement("tr");$("rows").append(row);
    [name(p),number(p.answer_accuracy_percent,3)+"%",number(p.quality_gap_pp,3)+" pp",money(p.answer_usd_per_10000),
      money(p.router_usd_per_10000),money(p.total_usd_per_10000),number(100*p.strong_share,2)+"%"].forEach(v=>text("td",v,row));
    const cell=document.createElement("td");row.append(cell);const button=text("button","Inspect",cell,{"aria-label":"Inspect "+name(p)});
    button.addEventListener("click",()=>inspect(p));
  });
  $("page").textContent="Page "+(state.page+1)+" of "+pages+" · "+ordered.length+" points";
  $("previous").disabled=state.page===0;$("next").disabled=state.page+1===pages;
}
function render(){
  const points=candidates();draw(points);table(points);
  $("chart-heading").textContent=state.role==="test"?"Same candidates, test outcomes":"Calibration tradeoffs";
  $("role-note").textContent=state.role==="test"?
    "Retrospective inspection only. These test outcomes are exposed. Do not use them to choose a new policy and claim held-out performance.":
    "Choose using calibration. These empirical tradeoffs are estimates, not guarantees on future questions.";
  $("cost-note").textContent=state.basis==="answer"?
    "Answer-model costs only. All router inference and hosting overhead is omitted on this axis.":
    "Only candidates with a known total are shown. Unpriced native/topic hosting is excluded, not treated as free. Always-arm bypasses may remain.";
  const missing=data[state.role].filter(p=>enabled.has(p.method)&&(cost(p)===null||cost(p)===undefined)).length;
  $("status").textContent=points.length+" displayed · "+missing+" excluded for unknown cost"+
    (state.scope==="hindsight"?" · Hindsight dominance uses test outcomes; it cannot select a live policy.":"");
  $("hindsight-option").hidden=state.role!=="test";
  $("hindsight-option").disabled=state.role!=="test";
  const point=data[state.role].find(p=>p.point_id===state.inspected);renderDetails(point);
}
Object.keys(labels).forEach(method=>{
  const label=document.createElement("label"),input=document.createElement("input");input.type="checkbox";input.checked=true;
  input.setAttribute("aria-label",labels[method]);label.append(input);const mark=text("span","",label,{class:"swatch"});
  mark.style.backgroundColor=colors[method];label.append(document.createTextNode(labels[method]));$("legend").append(label);
  input.addEventListener("change",()=>{input.checked?enabled.add(method):enabled.delete(method);state.page=0;render();});
});
$("test-option").disabled=!data.test.length;
$("role").addEventListener("change",()=>{
  state.role=$("role").value;state.page=0;state.inspected=null;
  if(state.role==="calibration"&&state.scope==="hindsight"){state.scope="per_method";$("scope").value=state.scope;}render();
});
$("basis").addEventListener("change",()=>{state.basis=$("basis").value;state.page=0;render();});
$("scope").addEventListener("change",()=>{state.scope=$("scope").value;state.page=0;render();});
$("previous").addEventListener("click",()=>{state.page--;table(candidates());});
$("next").addEventListener("click",()=>{state.page++;table(candidates());});
$("find").addEventListener("click",()=>{
  const value=$("find-id").value,id=Number(value),method=$("find-method").value;
  const p=value!==""&&Number.isInteger(id)?data[state.role].find(p=>p.method===method&&p.candidate_id===id):null;
  $("find-status").textContent=p?"Inspecting "+name(p)+". Chart filters are unchanged.":"No matching candidate in this population.";
  if(p)inspect(p);
});
$("provenance").textContent=JSON.stringify({format:data.format,run:data.run,provenance:data.provenance,
  interpretation:data.interpretation},null,2);
render();
</script></body></html>
"""
