"""Render frozen calibration policies on final data as an offline HTML explorer.

This module validates report lineage and formats saved metrics. It does not fit,
grade, select policies, recompute a frontier, or require third-party packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


METHOD_LABELS = {
    "base": "Laya (unchanged weights)",
    "lora": "Laya + LoRA",
    "embedding": "Embedding + logistic heads",
    "knn": "Embedding nearest neighbors",
    "text_length": "Text + length",
    "text_topic": "Predicted task from text",
    "task_metadata": "True task metadata (assisted control)",
}


def method_label(name):
    original = name.removesuffix("__textcost")
    label = METHOD_LABELS.get(original, original)
    return label + " · predicted cost" if original != name else label


METRICS = (
    "answer_accuracy_percent",
    "quality_gap_pp",
    "answer_usd_per_10000",
    "router_usd_per_10000",
    "total_usd_per_10000",
    "strong_share",
    "answer_savings_percent",
    "net_savings_percent",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, name, *, nullable=False):
    if value is None and nullable:
        return
    require(
        type(value) in (int, float) and math.isfinite(value),
        f"Invalid finite metric: {name}",
    )


def unique_map(rows, key, context):
    require(isinstance(rows, list), f"Expected a list: {context}")
    result = {}
    for row in rows:
        value = row[key]
        require(
            type(value) is int and value >= 0 and value not in result,
            f"Invalid/duplicate ID: {context}",
        )
        result[value] = row
    return result


def checked_data(calibration_path, frozen_path, final_path):
    """Return presentation data after verifying the three saved artifact links."""
    paths = {
        "calibration": Path(calibration_path),
        "frozen": Path(frozen_path),
        "final_report": Path(final_path),
    }
    docs = {key: json.loads(path.read_text()) for key, path in paths.items()}
    hashes = {key: sha(path) for key, path in paths.items()}
    cal, frozen, final = (
        docs[key] for key in ("calibration", "frozen", "final_report")
    )
    for doc, expected in (
        (cal, "bbeh-calibration-frontiers-v1"),
        (frozen, "bbeh-frozen-frontiers-v1"),
        (final, "bbeh-frontier-final-report-v1"),
    ):
        require(doc["format"] == expected, "Unsupported report format")
    require(
        cal["protocol"] == frozen["protocol"] == final["protocol"], "Protocol mismatch"
    )
    require(
        cal["final_outcomes_accessed"] is False
        and frozen["final_outcomes_accessed"] is False
        and frozen["frozen_before_final_outcomes"] is True
        and final["no_final_selection"] is True,
        "Missing calibration-before-final selection boundary",
    )
    require(
        frozen["calibration_sha256"] == hashes["calibration"],
        "Calibration file hash mismatch",
    )
    require(final["frozen_sha256"] == hashes["frozen"], "Frozen file hash mismatch")
    for key in ("source_plan_sha256", "pair_sha256", "cost_scope"):
        require(
            cal[key] == frozen[key] == final[key], f"Artifact binding mismatch: {key}"
        )
    for key in ("plan_sha256", "fees_sha256", "implementation_sha256"):
        require(cal[key] == frozen[key], f"Calibration/freeze mismatch: {key}")
    require(
        cal["grader_sha256"] == frozen["calibration_grader_sha256"],
        "Calibration grader mismatch",
    )
    source = frozen["source_rows"]
    require(
        len(source) == final["n"] > 0
        and len({row["family"] for row in source}) == final["families"]
        and len(set(frozen["ids"])) == len(source)
        and frozen["ids"] == [row["id"] for row in source]
        and all(row["role"] == "final" for row in source),
        "Final population mismatch",
    )
    require(
        set(cal["methods"]) == set(frozen["methods"]) == set(final["methods"]),
        "Method set mismatch",
    )
    random = final["random"]
    require(
        random["analytic_expectation"] is True
        and random["strong_probability_range"] == [0, 1],
        "Expected the analytic always-arm mixture",
    )
    for key in ("accuracy_endpoints_percent", "answer_usd_per_10000_endpoints"):
        require(len(random[key]) == 2, "Invalid random baseline endpoints")
        for value in random[key]:
            number(value, key)
    require(
        all(0 <= v <= 100 for v in random["accuracy_endpoints_percent"]),
        "Invalid baseline accuracy",
    )
    require(
        all(v >= 0 for v in random["answer_usd_per_10000_endpoints"]),
        "Negative baseline cost",
    )

    points, methods, baselines = [], [], {}
    order = {name: index for index, name in enumerate(METHOD_LABELS)}
    for name in sorted(
        cal["methods"], key=lambda item: (order.get(item.removesuffix("__textcost"), len(order)), item)
    ):
        prior, saved, result = (
            cal["methods"][name],
            frozen["methods"][name],
            final["methods"][name],
        )
        candidates = unique_map(prior["candidates"], "candidate_id", name)
        final_points = unique_map(result["points"], "candidate_id", name)
        frontier = unique_map(prior["frontier"], "candidate_id", name + " frontier")
        require(
            list(frontier)
            == saved["calibration_frontier_candidate_ids"]
            == result["calibration_frontier_candidate_ids"]
            and prior["candidates"] == saved["candidates"]
            and set(candidates) == set(final_points)
            and all(candidates[cid] == row for cid, row in frontier.items()),
            f"Saved candidate/frontier mismatch: {name}",
        )
        require(
            prior["no_preferred_quality_target"] is True,
            "Expected target-independent calibration",
        )
        require(
            prior["model_sha256"] == saved["model_sha256"],
            "Model changed after calibration",
        )
        kind, assisted = result["router_kind"], result["metadata_assisted"]
        require(
            kind in {"embedding", "native", "cpu"}
            and type(assisted) is bool
            and prior["router_kind"] == saved["scores"]["router_kind"] == kind
            and prior["metadata_assisted"]
            == saved["scores"]["metadata_assisted"]
            == assisted,
            f"Method identity changed: {name}",
        )
        require(
            result["native_or_cpu_hosting_priced"] is False,
            "This report schema has no bound native/CPU hosting-price receipt",
        )
        label = method_label(name)
        if assisted and name.removesuffix("__textcost") != "task_metadata":
            label += " (true task metadata; assisted control)"
        methods.append(
            {"name": name, "label": label, "kind": kind, "assisted": assisted}
        )
        for cid, point in final_points.items():
            candidate = candidates[cid]
            require(
                point["rule"] == candidate["policy"],
                f"Final rule differs from calibration: {name}/{cid}",
            )
            for metric in METRICS:
                number(
                    point[metric],
                    metric,
                    nullable=metric
                    in {
                        "router_usd_per_10000",
                        "total_usd_per_10000",
                        "answer_savings_percent",
                        "net_savings_percent",
                    },
                )
            require(
                0 <= point["answer_accuracy_percent"] <= 100
                and 0 <= point["strong_share"] <= 1,
                "Accuracy/share outside its range",
            )
            require(point["answer_usd_per_10000"] >= 0, "Negative answer cost")
            bypass = point["rule"]["mode"] in {"always_cheap", "always_strong"}
            if kind != "embedding" and not bypass:
                require(
                    point["router_usd_per_10000"] is None
                    and point["total_usd_per_10000"] is None
                    and point["net_savings_percent"] is None,
                    "Unpriced hosting must remain unknown",
                )
            else:
                require(
                    point["router_usd_per_10000"] is not None
                    and point["total_usd_per_10000"] is not None,
                    "Missing known fee/bypass cost",
                )
                require(
                    point["router_usd_per_10000"] >= 0
                    and math.isclose(
                        point["total_usd_per_10000"],
                        point["answer_usd_per_10000"] + point["router_usd_per_10000"],
                        rel_tol=1e-10,
                        abs_tol=1e-10,
                    ),
                    "Answer plus router cost mismatch",
                )
                require(
                    not bypass or point["router_usd_per_10000"] == 0,
                    "Static baseline charged a router fee",
                )
            if cid in frontier:
                require(point.get("paired_ci95"), "Missing frontier uncertainty")
                require(
                    point.get("calibration_matched_random_strong_probability")
                    == candidate["calibration_matched_random_strong_probability"],
                    "Random comparator changed",
                )
                probability = point["calibration_matched_random_strong_probability"]
                number(probability, "random strong probability", nullable=True)
                require(
                    probability is None or 0 <= probability <= 1,
                    "Random probability outside [0, 1]",
                )
            for metric, interval in point.get("paired_ci95", {}).items():
                lower, upper = interval["lower"], interval["upper"]
                number(lower, metric, nullable=True)
                number(upper, metric, nullable=True)
                defined, total = interval["defined_draws"], interval["total_draws"]
                require(
                    type(defined) is int
                    and type(total) is int
                    and 0 <= defined <= total
                    and total == final["bootstrap_draws"],
                    "Invalid interval draw counts",
                )
                require(
                    (defined == 0 and lower is None and upper is None)
                    or (
                        defined > 0
                        and lower is not None
                        and upper is not None
                        and lower <= upper
                    ),
                    "Invalid uncertainty bounds",
                )
            item = {
                **point,
                "key": f"{name}:{cid}",
                "method": name,
                "label": label,
                "kind": kind,
                "assisted": assisted,
                "calibration": candidate["calibration"],
                "on_calibration_frontier": cid in frontier,
                "baseline": False,
            }
            if bypass:
                arm = int(point["rule"]["mode"] == "always_strong")
                require(
                    math.isclose(
                        point["answer_accuracy_percent"],
                        random["accuracy_endpoints_percent"][arm],
                        abs_tol=1e-10,
                    )
                    and math.isclose(
                        point["answer_usd_per_10000"],
                        random["answer_usd_per_10000_endpoints"][arm],
                        abs_tol=1e-10,
                    ),
                    "Baseline disagrees with analytic endpoints",
                )
                previous = baselines.get(arm)
                if previous is None or (
                    not previous.get("paired_ci95") and point.get("paired_ci95")
                ):
                    baselines[arm] = {
                        **item,
                        "key": f"baseline:{arm}",
                        "method": "baseline",
                        "baseline": True,
                        "label": "Always Sol" if arm else "Always Luna",
                        "assisted": False,
                        "candidate_id": None,
                    }
            elif cid in frontier:
                points.append(item)
    require(set(baselines) == {0, 1}, "Missing always-arm baselines")
    # Order within each method is the saved calibration frontier order. Never
    # reorder or remove policies using final accuracy, final cost, or a new hull.
    rank = {
        (name, cid): i
        for name in cal["methods"]
        for i, cid in enumerate(
            frozen["methods"][name]["calibration_frontier_candidate_ids"]
        )
    }
    method_order = {method["name"]: i for i, method in enumerate(methods)}
    points.sort(
        key=lambda row: (
            method_order[row["method"]],
            rank[row["method"], row["candidate_id"]],
        )
    )
    return {
        "format": "bbeh-cost-quality-explorer-v1",
        "n": final["n"],
        "families": final["families"],
        "tasks": len({row["task"] for row in source}),
        "calibration_n": len(cal["ids"]),
        "methods": methods,
        "points": [baselines[0], baselines[1], *points],
        "random": random,
        "bootstrap_draws": final["bootstrap_draws"],
        "protocol": final["protocol"],
        "cost_scope": final["cost_scope"],
        "router_cost_scope": final["router_cost_scope"],
        "latency_note": final["latency_note"],
        "input_sha256": hashes,
        "source_plan_sha256": final["source_plan_sha256"],
        "pair_sha256": final["pair_sha256"],
    }


CSS = r"""
:root{color-scheme:light;--ink:#17263b;--muted:#46586c;--line:#d8e0e8;--paper:#fff;--accent:#1551a8}
*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:1420px;margin:auto;padding:28px}h1{font-size:clamp(1.8rem,4vw,2.5rem);line-height:1.2;margin:0 0 12px}
h2{font-size:1.18rem;margin:0 0 12px}h3{font-size:1rem}p{max-width:92ch;margin:10px 0}.muted,caption{color:var(--muted)}
.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}
.controls,.legend,.pager,.stats{display:flex;gap:12px;flex-wrap:wrap;align-items:center}.controls{align-items:flex-start}
.controls>*{flex:1;min-width:240px}fieldset{border:1px solid var(--line);border-radius:8px;padding:10px 14px}
legend{font-weight:650}label{cursor:pointer}fieldset label{display:block;margin:5px 0}input{accent-color:var(--accent)}
input[type=search]{font:inherit;padding:8px;width:100%;border:1px solid #72849a;border-radius:6px}
button{font:inherit;background:white;border:1px solid #72849a;border-radius:6px;padding:6px 12px;color:var(--ink);cursor:pointer}
button:hover{background:#edf3fb}button:disabled{opacity:.45;cursor:default}:focus-visible{outline:3px solid #d26c00;outline-offset:3px}
.legend label{font-size:.9rem;display:flex;gap:6px;align-items:center}.mark{font-weight:800;font-size:1.2rem}
.plot{width:100%;height:auto;display:block;min-height:240px}svg text{font:13px system-ui,sans-serif;fill:var(--ink)}
.grid{stroke:#e4e9ef;stroke-width:1}.axis{stroke:#6c7c8c;stroke-width:1.2}.point{cursor:pointer}.point:hover{stroke-width:3}
.point:focus{outline:none;stroke:#111;stroke-width:4}.selected{stroke:#111;stroke-width:3}.random-line{stroke:#4d5965;stroke-width:2;stroke-dasharray:7 5}
.split{display:grid;grid-template-columns:1fr 1fr;gap:18px}.stats div{background:#f4f7fa;padding:8px 12px;border-radius:6px}
.stats strong{font-size:1.25rem;display:block}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:.9rem}
caption{text-align:left;margin:5px 0 10px}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line);vertical-align:top}
th{background:#f4f7fa}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tr.active{background:#edf3fb}
.pager{justify-content:flex-end;margin-top:12px}code,pre{font:13px/1.4 ui-monospace,monospace;overflow-wrap:anywhere}
pre{white-space:pre-wrap;background:#f4f7fa;padding:12px;border-radius:6px}details{margin:12px 0}summary{cursor:pointer;font-weight:650}
.badge{font-size:.8rem;border:1px solid #916400;border-radius:4px;padding:1px 5px;color:#624200}.notice{border-left:4px solid #bd7a00;padding-left:12px}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:800px){main{padding:16px}.card{padding:14px}.split{grid-template-columns:1fr}.plot{min-height:200px}}
@media print{body{background:white}main{max-width:none}.controls,.pager,.legend input,button{display:none}.card{break-inside:avoid}.scroll{overflow:visible}}
"""


JS = r"""
'use strict';
const data=JSON.parse(document.getElementById('report-data').textContent);
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const colors=['#1358a4','#a83459','#007970','#8b5300','#7250a3','#586a16','#ab4500'];
const symbols=['●','■','▲','◆','▼','⬟','✚'];
const enabled=new Set(data.methods.map(m=>m.name));
let mode='gross',page=0,selected=data.points[0].key;
const perPage=20;
const money=v=>v===null||v===undefined?'Not priced':'$'+v.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:4});
const num=(v,d=2)=>v===null||v===undefined?'Unavailable':v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const pct=v=>num(v)+'%';
const cost=p=>mode==='gross'?p.answer_usd_per_10000:p.total_usd_per_10000;
const methodIndex=p=>data.methods.findIndex(m=>m.name===p.method);
const color=p=>p.baseline?'#17263b':colors[methodIndex(p)%colors.length];
function node(tag,text,attrs={}){const n=document.createElement(tag);if(text!==null)n.textContent=text;for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);return n;}
function svgNode(tag,attrs,text){const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function visible(){const q=$('search').value.trim().toLowerCase();return data.points.filter(p=>(p.baseline||enabled.has(p.method))&&cost(p)!==null&&(!q||(p.label+' '+p.candidate_id+' '+p.rule.mode).toLowerCase().includes(q)));}
function ruleName(p){if(p.baseline)return p.label;return 'Policy '+p.candidate_id+' · '+p.rule.mode.replaceAll('_',' ');}
function choose(key){const focus=document.activeElement,chartFocus=focus?.closest('#chart'),tableFocus=focus?.closest('#policies');selected=key;render();$('selection-status').textContent='Selected '+describe(data.points.find(p=>p.key===key));if(chartFocus)$('chart').querySelector('.selected')?.focus();else if(tableFocus)$('policies').querySelector('button[aria-pressed="true"]')?.focus();}
function describe(p){return p.label+', '+ruleName(p)+', '+money(cost(p))+' per 10,000 requests, '+pct(p.answer_accuracy_percent)+' answer accuracy';}
function marker(p,x,y){const index=p.baseline?-1:methodIndex(p)%7;let n;
 if(index===1)n=svgNode('rect',{x:x-5,y:y-5,width:10,height:10});
 else if(index===2||index===4){const sign=index===2?1:-1;n=svgNode('polygon',{points:`${x},${y-6*sign} ${x-6},${y+5*sign} ${x+6},${y+5*sign}`});}
 else if(index===3)n=svgNode('polygon',{points:`${x},${y-7} ${x-6},${y} ${x},${y+7} ${x+6},${y}`});
 else if(index===5)n=svgNode('polygon',{points:[0,1,2,3,4,5].map(i=>`${x+6*Math.cos(i*Math.PI/3)},${y+6*Math.sin(i*Math.PI/3)}`).join(' ')});
 else if(index===6)n=svgNode('path',{d:`M ${x-6} ${y-2} H ${x-2} V ${y-6} H ${x+2} V ${y-2} H ${x+6} V ${y+2} H ${x+2} V ${y+6} H ${x-2} V ${y+2} H ${x-6} Z`});
 else n=svgNode('circle',{cx:x,cy:y,r:p.baseline?6:5});
 n.setAttribute('fill',color(p));n.setAttribute('stroke',p.key===selected?'#111':'white');
 n.setAttribute('class','point'+(p.key===selected?' selected':''));n.setAttribute('tabindex','0');n.setAttribute('role','button');
 n.setAttribute('aria-label',describe(p));n.setAttribute('aria-pressed',String(p.key===selected));n.append(svgNode('title',{},describe(p)));
 n.addEventListener('click',()=>choose(p.key));n.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose(p.key);}});return n;
}
function draw(points){const svg=$('chart');svg.replaceChildren(svgNode('title',{id:'chart-title'},'Final answer accuracy and cost of policies chosen on calibration'),svgNode('desc',{id:'chart-desc'},'Each symbol is one saved policy. The dashed segment is analytic random routing between always Luna and always Sol. Use the table below for the same values.'));
 const chosen=points.find(p=>p.key===selected),ci=chosen?.paired_ci95||{},costKey=mode==='gross'?'answer_usd_per_10000':'total_usd_per_10000';
 const xs=[...points.map(cost),...data.random.answer_usd_per_10000_endpoints];
 const ys=[...points.map(p=>p.answer_accuracy_percent),...data.random.accuracy_endpoints_percent];
 if(ci[costKey]?.lower!==null&&ci[costKey]?.lower!==undefined)xs.push(ci[costKey].lower,ci[costKey].upper);
 if(ci.accuracy_percent?.lower!==null&&ci.accuracy_percent?.lower!==undefined)ys.push(ci.accuracy_percent.lower,ci.accuracy_percent.upper);
 const xmin=0,xmax=Math.max(...xs,0.01)*1.08,spread=Math.max(4,Math.max(...ys)-Math.min(...ys));
 const ymin=Math.max(0,Math.min(...ys)-spread*.12),ymax=Math.min(100,Math.max(...ys)+spread*.12);
 const x=v=>84+v/xmax*886,y=v=>440-(v-ymin)/(ymax-ymin||1)*386;
 for(let i=0;i<=5;i++){const xv=xmax*i/5,yv=ymin+(ymax-ymin)*i/5;
  svg.append(svgNode('line',{x1:x(xv),x2:x(xv),y1:54,y2:440,class:'grid'}),svgNode('text',{x:x(xv),y:464,'text-anchor':'middle'},money(xv)));
  svg.append(svgNode('line',{x1:84,x2:970,y1:y(yv),y2:y(yv),class:'grid'}),svgNode('text',{x:72,y:y(yv)+4,'text-anchor':'end'},num(yv,1)+'%'));
 }
 svg.append(svgNode('line',{x1:84,x2:970,y1:440,y2:440,class:'axis'}),svgNode('line',{x1:84,x2:84,y1:54,y2:440,class:'axis'}));
 svg.append(svgNode('text',{x:527,y:505,'text-anchor':'middle'},(mode==='gross'?'Answer model cost':'Answer + known router fees')+' · USD per 10,000 requests'));
 svg.append(svgNode('text',{transform:'translate(22 247) rotate(-90)','text-anchor':'middle'},'Answer accuracy (%)'));
 const[cheap,strong]=data.random.answer_usd_per_10000_endpoints,[qa,qb]=data.random.accuracy_endpoints_percent;
 svg.append(svgNode('line',{x1:x(cheap),y1:y(qa),x2:x(strong),y2:y(qb),class:'random-line'}));
 if(chosen){const c=ci[costKey],a=ci.accuracy_percent,attrs={stroke:color(chosen),'stroke-width':2};
  if(c?.lower!==null&&c?.lower!==undefined){const yy=y(chosen.answer_accuracy_percent);svg.append(svgNode('line',{x1:x(c.lower),x2:x(c.upper),y1:yy,y2:yy,...attrs}));for(const v of[c.lower,c.upper])svg.append(svgNode('line',{x1:x(v),x2:x(v),y1:yy-5,y2:yy+5,...attrs}));}
  if(a?.lower!==null&&a?.lower!==undefined){const xx=x(cost(chosen));svg.append(svgNode('line',{x1:xx,x2:xx,y1:y(a.lower),y2:y(a.upper),...attrs}));for(const v of[a.lower,a.upper])svg.append(svgNode('line',{x1:xx-5,x2:xx+5,y1:y(v),y2:y(v),...attrs}));}
 }
 for(const p of points.filter(p=>p.key!==selected))svg.append(marker(p,x(cost(p)),y(p.answer_accuracy_percent)));
 if(chosen)svg.append(marker(chosen,x(cost(chosen)),y(chosen.answer_accuracy_percent)));
}
const ciNames={accuracy_percent:'Answer accuracy (%)',quality_gap_pp:'Accuracy loss versus always Sol (pp)',accuracy_gain_over_luna_pp:'Accuracy gain versus always Luna (pp)',answer_usd_per_10000:'Answer cost ($/10,000)',answer_savings_percent:'Answer cost savings versus Sol (%)',total_usd_per_10000:'Answer + known router fees ($/10,000)',net_savings_percent:'Savings after known router fees (%)',accuracy_gain_over_calibration_budget_random_pp:'Accuracy gain versus calibration-budget random (pp)',cost_difference_from_calibration_budget_random_usd_per_10000:'Cost minus calibration-budget random ($/10,000)'};
function estimate(p,key){if(key==='accuracy_percent')return p.answer_accuracy_percent;if(key==='accuracy_gain_over_luna_pp')return p.answer_accuracy_percent-data.random.accuracy_endpoints_percent[0];
 const rp=p.calibration_matched_random_strong_probability;
 if(key==='accuracy_gain_over_calibration_budget_random_pp')return p.answer_accuracy_percent-((1-rp)*data.random.accuracy_endpoints_percent[0]+rp*data.random.accuracy_endpoints_percent[1]);
 if(key==='cost_difference_from_calibration_budget_random_usd_per_10000')return(p.total_usd_per_10000??p.answer_usd_per_10000)-((1-rp)*data.random.answer_usd_per_10000_endpoints[0]+rp*data.random.answer_usd_per_10000_endpoints[1]);
 return p[key];
}
function details(p){const box=$('detail');box.replaceChildren();if(!p){box.append(node('p','No policy matches these filters.'));return;}
 box.append(node('h3',p.label+' · '+ruleName(p)));if(p.assisted)box.append(node('p','Assisted control: uses the true source task label. It is not an ordinary question-only router.',{class:'notice'}));
 const stats=node('div',null,{class:'stats'});for(const[label,value]of[['Final accuracy',pct(p.answer_accuracy_percent)],['Final cost / 10,000',money(cost(p))],['Requests sent to Sol',pct(100*p.strong_share)]]){const d=node('div',label);d.append(node('strong',value));stats.append(d);}box.append(stats);
 box.append(node('p','Answer model cost: '+money(p.answer_usd_per_10000)+' / 10,000. Known router fee: '+money(p.router_usd_per_10000)+' / 10,000. '+(p.baseline?'Static baseline bypasses the router.':p.kind==='embedding'?'Encoder API fee included in the priced view; local head hosting remains unpriced.':'Router hosting is unpriced; this policy appears in the answer-cost view only.')));
 box.append(node('p','Calibration: '+pct(p.calibration.answer_accuracy_percent)+' accuracy, '+money(p.calibration.selection_cost_usd_per10000)+' / 10,000 under its declared cost basis. '+(p.baseline?'Always-arm comparator.':'Policy belongs to the calibration frontier; its final result did not determine inclusion.')));
 const intervals=p.paired_ci95||{},table=node('table');table.append(node('caption','95% pointwise paired bootstrap intervals; '+data.bootstrap_draws.toLocaleString()+' draws. Negative accuracy loss means higher accuracy than always Sol.'));
 const head=node('tr');for(const label of['Measure','Estimate','95% interval','Defined draws'])head.append(node('th',label,{scope:'col'}));const thead=node('thead');thead.append(head);table.append(thead);const body=node('tbody');
 for(const[key,value]of Object.entries(intervals)){const row=node('tr');row.append(node('th',ciNames[key]||key,{scope:'row'}),node('td',num(estimate(p,key),4),{class:'num'}),node('td',value.lower===null?'Unavailable':num(value.lower,4)+' to '+num(value.upper,4),{class:'num'}),node('td',value.defined_draws+'/'+value.total_draws,{class:'num'}));body.append(row);}table.append(body);
 if(Object.keys(intervals).length){const scroll=node('div',null,{class:'scroll'});scroll.append(table);box.append(scroll);}else box.append(node('p','No uncertainty interval was supplied for this comparator; none is fabricated.'));
 const rp=p.calibration_matched_random_strong_probability;if(rp!==null&&rp!==undefined)box.append(node('p','The random comparator sends '+pct(100*rp)+' of requests to Sol. This probability matched the policy’s calibration budget and was frozen there; its final cost may differ. The cost difference uses '+(p.total_usd_per_10000===null?'answer cost only.':'answer cost plus known router fees.')));
 const raw=node('details');raw.append(node('summary','Exact saved rule'),node('pre',JSON.stringify(p.rule,null,2)));box.append(raw);
}
function table(points){const body=$('policies');body.replaceChildren();const pages=Math.max(1,Math.ceil(points.length/perPage));page=Math.min(page,pages-1);
 for(const p of points.slice(page*perPage,(page+1)*perPage)){const tr=node('tr',null,{class:p.key===selected?'active':''});const pick=node('button',p.baseline?'Select baseline':'Select '+p.candidate_id,{type:'button','aria-label':'Select '+describe(p),'aria-pressed':String(p.key===selected)});pick.addEventListener('click',()=>choose(p.key));const first=node('td');first.append(pick);tr.append(first,node('th',p.label,{scope:'row'}),node('td',money(cost(p)),{class:'num'}),node('td',pct(p.answer_accuracy_percent),{class:'num'}),node('td',pct(100*p.strong_share),{class:'num'}),node('td',pct(p.calibration.answer_accuracy_percent),{class:'num'}),node('td',money(p.calibration.selection_cost_usd_per10000),{class:'num'}));body.append(tr);}
 $('page-info').textContent='Page '+(page+1)+' of '+pages+' · '+points.length+' policies and baselines';$('previous').disabled=page===0;$('next').disabled=page===pages-1;
}
function render(){const points=visible();if(!points.some(p=>p.key===selected))selected=points[0]?.key??null;
 const unpriced=data.points.filter(p=>!p.baseline&&enabled.has(p.method)&&p.total_usd_per_10000===null).length;
 $('cost-note').textContent=mode==='gross'?'All methods use answer model cost only. Router serving, training, and acquisition overhead are excluded.':'Embedding policies include the known encoder API fee. '+unpriced+' policies with unpriced native/CPU hosting are omitted. Local head hosting, training, and acquisition overhead remain excluded; this is not a complete operating-cost estimate.';
 $('count').textContent=points.length+' policies and baselines shown. The policy order is fixed by calibration, not final performance.';
 draw(points);details(points.find(p=>p.key===selected));table(points);
}
for(const[index,m]of data.methods.entries()){const label=node('label'),input=node('input',null,{type:'checkbox',checked:'','aria-label':'Show '+m.label});input.addEventListener('change',()=>{input.checked?enabled.add(m.name):enabled.delete(m.name);page=0;render();});label.append(input,node('span',symbols[index%symbols.length],{class:'mark',style:'color:'+colors[index%colors.length],'aria-hidden':'true'}),node('span',m.label));$('legend').append(label);}
document.querySelectorAll('input[name=cost]').forEach(input=>input.addEventListener('change',()=>{mode=input.value;page=0;render();}));
$('search').addEventListener('input',()=>{page=0;render();});$('previous').addEventListener('click',()=>{page--;table(visible());});$('next').addEventListener('click',()=>{page++;table(visible());});
$('population').textContent=data.n.toLocaleString()+' final questions · '+data.families.toLocaleString()+' families · '+data.tasks+' tasks. Calibration used '+data.calibration_n.toLocaleString()+' questions.';
$('scope').textContent=data.cost_scope;$('latency').textContent=data.latency_note;
for(const[key,value]of Object.entries(data.input_sha256)){const p=node('p',key+': ');p.append(node('code',value));$('hashes').append(p);}
render();
"""


def render_html(data):
    banner = (
        '<p class="notice" role="note">Replay on published test answers; not a new blind evaluation.</p>'
        if data.get("reproduction") else ""
    )
    # JSON is inert text, but escape HTML delimiters even inside script[type=json].
    payload = json.dumps(
        data, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    for char, escaped in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        payload = payload.replace(char, escaped)
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>BBEH routing: cost and answer quality</title><style>"""
        + CSS
        + """</style></head>
<body><main><header><h1>BBEH routing: cost and answer quality</h1>
"""
        + banner
        + """
<p>Explore the final performance of policies chosen on calibration. Each symbol is a distinct saved policy; there is no new selection or frontier fitted to final answers.</p>
<p id="population" class="muted"></p></header>
<noscript><p class="notice">Enable JavaScript to use this offline explorer. All data and code are embedded in this file; no network access is needed.</p></noscript>
<section class="card" aria-labelledby="explore-title"><h2 id="explore-title">Cost and accuracy</h2>
<div class="controls"><fieldset><legend>Cost basis</legend>
<label><input type="radio" name="cost" value="gross" checked> Answer model cost · all methods</label>
<label><input type="radio" name="cost" value="priced"> Answer + known router fees</label></fieldset>
<div><label for="search">Filter methods or policy IDs</label><input id="search" type="search" placeholder="For example: LoRA or 42"></div></div>
<p id="cost-note" class="muted"></p><div id="legend" class="legend" role="group" aria-label="Visible methods"></div>
<p class="muted">Black circles: always Luna / always Sol. Dashed line: analytic expected random routing between those arms, with zero router inference fee.</p>
<svg id="chart" class="plot" viewBox="0 0 1020 530" role="group" aria-labelledby="chart-title chart-desc"></svg>
<p id="count" class="muted" aria-live="polite"></p>
<p class="muted">Select a symbol or a table row for its uncertainty. Axis bounds may zoom to the data. Error bars are pointwise 95% intervals for the selected policy, not a confidence band for the curve.</p></section>
<section class="card" aria-labelledby="detail-title"><h2 id="detail-title">Selected policy</h2><div id="detail"></div>
<p class="muted">Intervals resample whole families within each task, retaining the observed task weights. They are conditional on these tasks and do not provide simultaneous coverage across policies or a guarantee on future requests.</p>
<div id="selection-status" class="sr-only" role="status" aria-live="polite"></div></section>
<section class="card" aria-labelledby="table-title"><h2 id="table-title">All calibration-frontier policies</h2>
<div class="scroll"><table><caption>Final measurements alongside the calibration values used to include each policy. Always-arm baselines are shown once. Calibration cost includes encoder fees for embedding methods and excludes unpriced hosting for native/CPU methods.</caption>
<thead><tr><th scope="col">Choose</th><th scope="col">Method</th><th scope="col" class="num">Final $ / 10,000</th><th scope="col" class="num">Final accuracy</th><th scope="col" class="num">Sol share</th><th scope="col" class="num">Calibration accuracy</th><th scope="col" class="num">Calibration $ / 10,000</th></tr></thead>
<tbody id="policies"></tbody></table></div><div class="pager"><span id="page-info" aria-live="polite"></span><button id="previous" type="button">Previous</button><button id="next" type="button">Next</button></div></section>
<section class="card" aria-labelledby="scope-title"><h2 id="scope-title">Interpretation and provenance</h2>
<p>True task metadata is an assisted control: it uses the source task label, which an ordinary question-only router does not receive. Predicted task from text is a separate method.</p>
<p>The random line is an expected mixture, not an observed random run. A policy’s calibration-budget random comparator keeps its calibration mixing probability; its final spend need not match the policy.</p>
<p id="scope"></p><p id="latency"></p><details><summary>Exact input file hashes</summary><div id="hashes"></div></details>
<p class="muted">This file is self-contained and makes no network requests. It only formats the supplied reports; it performs no grading, training, or policy selection.</p></section>
</main><script id="report-data" type="application/json">"""
        + payload
        + """</script><script>"""
        + JS
        + """</script></body></html>
"""
    )


def render(calibration_path, frozen_path, final_path, output, *, reproduction=False):
    data = checked_data(calibration_path, frozen_path, final_path)
    if reproduction:
        data["reproduction"] = True
        data["new_blind_evaluation"] = False
    output = Path(output)
    require(
        not output.exists(), "Output directory already exists; use a fresh directory"
    )
    document = render_html(data)
    hashes = "\n".join(
        f"- {name}: `{value}`" for name, value in data["input_sha256"].items()
    )
    disclosure = "Replay on published test answers; not a new blind evaluation.\n\n" if reproduction else ""
    replay_flag = " --reproduction" if reproduction else ""
    readme = f"""# BBEH cost–quality explorer

{disclosure}Open [index.html](index.html) in a browser. It is self-contained and works offline.

Each point is a policy from the calibration frontier, replayed unchanged on {data["n"]:,} final questions ({data["families"]:,} families). Always-Luna and always-Sol baselines are included. No final winner or new final frontier is selected.

The default view compares answer-model dollars per 10,000 requests with answer accuracy. The second view adds known encoder API fees and omits native/CPU policies whose hosting is unpriced. Local head hosting, training, and data acquisition are not included. The true-task-metadata method is an assisted control.

Select a point or table row for every supplied 95% paired family-bootstrap interval. These are pointwise, conditional on the observed tasks, and not a guarantee for future requests. The random line is an analytic expectation. A calibration-budget random comparator may have different spend on final data.

Regenerate from the three exact saved artifacts, using a new output directory:

```bash
# Render saved calibration policies and their frozen final evaluation.
python -m evaluation.report --calibration calibration.json --frozen frozen.json --final-report final-report.json --output explorer{replay_flag}
```

The renderer verifies artifact hashes, policy identities, population metadata, and the calibration-frontier membership before writing either output. No model, grading, or network call occurs.

{hashes}
"""
    output.mkdir(parents=True)
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {
        "format": data["format"],
        "policies_and_baselines": len(data["points"]),
        "input_sha256": data["input_sha256"],
        "html_sha256": sha(output / "index.html"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproduction", action="store_true", help="Label replay on already published test answers")
    args = parser.parse_args()
    print(
        json.dumps(
            render(args.calibration, args.frozen, args.final_report, args.output, reproduction=args.reproduction),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
