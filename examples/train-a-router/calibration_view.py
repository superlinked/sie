"""Inspect calibration policies before opening any final-test results."""

import html
import json
from pathlib import Path

from embedding import recipes as core

LABELS = {
    "base": "Laya", "lora": "Laya + LoRA", "knn": "Embedding neighbors",
    "embedding": "Embedding classifier", "text_length": "Text + length control",
    "text_topic": "Predicted task control", "task_metadata": "Known task (metadata-assisted control)",
}


def label(name):
    key = name.removesuffix("__textcost")
    return LABELS.get(key, key) + (" · predicted cost" if name.endswith("__textcost") else " · average cost")


def render(calibration_path, output, *, reproduction=False):
    document = core.read(calibration_path)
    core.require(document["format"] == "bbeh-calibration-frontiers-v1"
        and document["final_outcomes_accessed"] is False, "Expected calibration-only results")
    core.require(not Path(output).exists(), "Preserve previous frontier views")
    methods = []
    for name, method in document["methods"].items():
        core.require(method["no_preferred_quality_target"] is True, "Expected the full frontier")
        candidates = {row["candidate_id"]: row for row in method["candidates"]}
        rows = []
        selected = {row["candidate_id"]: row for row in method["frontier"]}
        selected.update({cid: row for cid, row in candidates.items() if row["policy"]["mode"].startswith("always_")})
        for row in sorted(selected.values(), key=lambda r: r["calibration"]["answer_cost_usd_per10000"]):
            core.require(candidates[row["candidate_id"]] == row, "Frontier policy changed")
            value = row["calibration"]
            rows.append({"id": row["candidate_id"], "policy": row["policy"],
                "accuracy": value["answer_accuracy_percent"], "gap": value["gap_vs_strong_pp"],
                "answer": value["answer_cost_usd_per10000"], "total": value["total_cost_usd_per10000"],
                "share": 1 - value["cheap_share"], "baseline": row["policy"]["mode"].startswith("always_")})
        methods.append({"name": name, "label": label(name), "assisted": method["metadata_assisted"], "points": rows})
    data = json.dumps(methods, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    mode = "Published-data reproduction. " if reproduction else ""
    text = TEMPLATE.replace("__DATA__", data).replace("__COUNT__", str(len(document["ids"])))
    text = text.replace("__MODE__", html.escape(mode)).replace("__SHA__", core.sha(calibration_path))
    Path(output).write_text(text, encoding="utf-8")
    return {"open": str(output), "calibration_questions": len(document["ids"]), "final_answers_read": False}


TEMPLATE = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'">
<title>Choose a routing policy</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;color:#172b40;background:#f5f7fa;margin:auto;max-width:1100px;padding:24px}
h1{line-height:1.2}select,button{font:inherit;padding:8px;border:1px solid #789;border-radius:5px;background:white}
.controls{display:flex;flex-wrap:wrap;gap:18px;align-items:end}label{display:block}svg{width:100%;background:white;border:1px solid #ccd5df;margin:20px 0}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #dde2e9;text-align:right}th:first-child,td:first-child{text-align:left}
.scroll{overflow:auto;max-height:400px}.note{color:#41576d}button:focus-visible,select:focus-visible{outline:3px solid #c67a00}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e9eef5;padding:14px}
</style>
<h1>Choose a routing policy</h1>
<p>__MODE__Compare cost and answer accuracy on <b>__COUNT__ calibration questions</b>. This page never reads final-test answers. Choose the tradeoff that fits your application, then evaluate the same policy on the test set.</p>
<div class="controls"><div><label for="method">Router and cost estimate</label><select id="method"></select></div>
<div><label for="basis">Cost shown</label><select id="basis"><option value="answer">Answer models only</option><option value="total">Answers + known Cloud encoder fee</option></select></div></div>
<p class="note" id="scope"></p><svg viewBox="0 0 1000 430" id="plot" role="img" aria-label="Calibration cost and accuracy curve"></svg>
<p id="choice">Select a point or table row to copy its method and candidate ID.</p><pre id="command"></pre>
<div class="scroll"><table><thead><tr><th>Candidate</th><th>Cost / 10,000</th><th>Accuracy</th><th>Accuracy loss</th><th>Sent to Sol</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="note">The blue curve contains policies for which no other calibration policy is both cheaper and more accurate. The dashed line is the expected result of choosing Luna or Sol at random, at different spending levels. A negative accuracy loss means this router beat always-Sol on these questions. A one-point loss means, for example, 80% accuracy becomes 79%. Training, idle capacity and application hosting are separate costs.</p>
<details><summary>Source</summary><p>Calibration artifact SHA-256: __SHA__</p></details>
<script>
const data=__DATA__, $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const node=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n};
const svg=(tag,attrs,text)=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n};
for(const m of data){const o=node('option',m.label);o.value=m.name;$('method').append(o)}
function choose(m,p){$('choice').textContent=m.label+' — candidate '+p.id;$('command').textContent=(m.assisted?'Diagnostic only: this control needs the benchmark task label. It is unavailable for live questions.':'--method '+m.name+' --candidate-id '+p.id)+'\nPolicy: '+JSON.stringify(p.policy)}
function draw(){const m=data.find(x=>x.name===$('method').value),basis=$('basis').value,ps=m.points.filter(p=>p[basis]!==null);$('plot').replaceChildren();$('rows').replaceChildren();$('command').textContent='';
 $('scope').textContent=(m.assisted?'This control uses task metadata that a live request may not supply. ':'')+(basis==='total'?'Cloud encoder fees are included where measured; methods with unpriced hosting are omitted.':'Router hosting is not included in these answer-model costs.');
 if(!ps.length){$('choice').textContent='No measured total serving cost for this method. Choose answer-model cost or add your hosting costs.';return}
 const maxX=Math.max(...ps.map(p=>p[basis]))*1.06,minY=Math.max(0,Math.floor(Math.min(...ps.map(p=>p.accuracy))/5)*5-5),maxY=Math.min(100,Math.ceil(Math.max(...ps.map(p=>p.accuracy))/5)*5+5),X=v=>75+v/maxX*880,Y=v=>365-(v-minY)/(maxY-minY)*320;
 for(let i=0;i<=5;i++){let x=maxX*i/5,y=minY+(maxY-minY)*i/5;$('plot').append(svg('line',{x1:75,y1:Y(y),x2:955,y2:Y(y),stroke:'#e1e6ed'}),svg('text',{x:60,y:Y(y)+5,'text-anchor':'end','font-size':13},y.toFixed(0)+'%'),svg('text',{x:X(x),y:390,'text-anchor':'middle','font-size':13},'$'+x.toFixed(1)))}
 $('plot').append(svg('text',{x:515,y:418,'text-anchor':'middle'},'Cost per 10,000 requests'),svg('text',{x:75,y:25},'Answer accuracy'));
 const line=ps.filter(p=>!p.baseline).map(p=>X(p[basis])+','+Y(p.accuracy)).join(' ');$('plot').append(svg('polyline',{points:line,fill:'none',stroke:'#2769aa','stroke-width':2}));
 const ends=ps.filter(p=>p.baseline);if(ends.length===2)$('plot').append(svg('line',{x1:X(ends[0][basis]),y1:Y(ends[0].accuracy),x2:X(ends[1][basis]),y2:Y(ends[1].accuracy),stroke:'#657587','stroke-dasharray':'7 5'}));
 for(const p of ps){const c=svg('circle',{cx:X(p[basis]),cy:Y(p.accuracy),r:4,fill:'#2769aa',tabindex:0,role:'button','aria-label':'Candidate '+p.id+', '+p.accuracy.toFixed(2)+'% accuracy'});c.append(svg('title',{},'Candidate '+p.id));c.addEventListener('click',()=>choose(m,p));c.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(m,p)}});$('plot').append(c);
 const tr=node('tr'),td=node('td'),b=node('button','Select '+p.id);b.addEventListener('click',()=>choose(m,p));td.append(b);tr.append(td,node('td','$'+p[basis].toFixed(2)),node('td',p.accuracy.toFixed(2)+'%'),node('td',p.gap.toFixed(2)+' points'),node('td',(100*p.share).toFixed(1)+'%'));$('rows').append(tr)}
 choose(m,ps[0]);}
$('method').addEventListener('change',draw);$('basis').addEventListener('change',draw);draw();
</script></html>"""
