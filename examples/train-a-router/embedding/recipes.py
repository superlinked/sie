"""Small frozen-encoder recipes; role-separated fitting, tuning and scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ARMS = ("cheap", "strong")
ROLES = ("train", "development", "calibration", "final")
MODEL = "Qwen/Qwen3-Embedding-4B"
REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
INSTRUCTION = "Represent the knowledge and reasoning required to answer this question correctly."
PROTOCOL = {
    "format": "bbeh-embedding-recipes-v1",
    "seed": 89,
    "neighbors": [8, 32, 128],
    "neighbor_prior_rows": 2,
    "penalties": [0.01, 0.1, 1.0, 10.0],
    "selection": "Minimum summed development Bernoulli NLL; ties larger k/penalty",
    "refit": "One selected refit on train+development; calibration/final never fit",
    "cost": "Four fit-only arm/correctness means; absent class uses arm mean",
    "task_prior_rows": 20,
    "text": {"ngram_range": [1, 2], "min_df": 2, "max_features": 20000},
    "text_penalty": 1.0,
    "optimizer": {"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7, "maxls": 40},
    "versions": {"numpy": "2.4.6", "scipy": "1.17.0", "sklearn": "1.6.1"},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require_hash(value):
    require(isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef"), "Invalid SHA256")


def prepare(source, pair_path, run):
    source, run = Path(source), Path(run)
    require(not run.exists(), "Preserve prior prepared runs")
    source_plan, pair = read(source / "source-plan.json"), read(pair_path)
    require(pair["models"] == {"cheap": "gpt-6-luna", "strong": "gpt-6.1-sol"}, "Wrong selected pair")
    require(bool(pair["cost_scope"]), "Explicit selected-response cost scope required")
    paths = ["meta.json", *[f"question-only/{role}-inputs.json" for role in ROLES]]
    for name in paths:
        require(sha(source / name) == source_plan["files_sha256"][name], "Source file hash changed: " + name)
    metadata = read(source / "meta.json")
    by_id = {row["id"]: row for row in metadata}
    require(len(by_id) == len(metadata), "Duplicate source ID")
    families, roles, seen = {}, {}, set()
    for role in ROLES:
        items = read(source / f"question-only/{role}-inputs.json")
        require(isinstance(items, list) and items, "Empty fixed role")
        rows = []
        for item in items:
            require(set(item) == {"id", "request", "response_contract"}, "Unexpected question-only fields")
            meta = by_id[item["id"]]
            require(meta["retained"] is True and meta["role"] == role, "Source role/eligibility changed")
            require(
                item["id"] not in seen and isinstance(item["request"], str) and item["request"], "Duplicate/empty input"
            )
            seen.add(item["id"])
            require(families.setdefault(meta["family"], role) == role, "Family crosses source roles")
            rows.append({**item, "role": role, "task": meta["task"], "family": meta["family"], "dataset": "bbeh"})
        roles[role] = rows
    require(seen == {row["id"] for row in metadata if row["retained"]}, "Source retained population mismatch")
    run.mkdir(parents=True)
    for role, rows in roles.items():
        write(run / f"rows-{role}.json", rows)
    (run / "pair.json").write_bytes(Path(pair_path).read_bytes())
    plan = {
        "format": "bbeh-recipe-plan-v1",
        "protocol": PROTOCOL,
        "source_plan_sha256": sha(source / "source-plan.json"),
        "pair_sha256": sha(pair_path),
        "pair": pair,
        "recipe_source_sha256": sha(__file__),
        "roles": {
            role: {
                "n": len(rows),
                "rows_sha256": sha(run / f"rows-{role}.json"),
                "inputs_sha256": sha(source / f"question-only/{role}-inputs.json"),
            }
            for role, rows in roles.items()
        },
    }
    write(run / "plan.json", plan)
    return plan


def checked_plan(run):
    run = Path(run)
    plan = read(run / "plan.json")
    require(plan["format"] == "bbeh-recipe-plan-v1" and plan["protocol"] == PROTOCOL, "Recipe protocol changed")
    require(plan["recipe_source_sha256"] == sha(__file__), "Recipe source changed")
    require(plan["pair_sha256"] == sha(run / "pair.json") and plan["pair"] == read(run / "pair.json"), "Pair changed")
    return plan


def role_rows(run, plan, role):
    require(role in ROLES, "Unknown role")
    path = Path(run) / f"rows-{role}.json"
    require(sha(path) == plan["roles"][role]["rows_sha256"], "Role metadata hash changed")
    rows = read(path)
    require(len(rows) == plan["roles"][role]["n"] and all(row["role"] == role for row in rows), "Wrong role metadata")
    return rows


def grades(plan, rows, path, role):
    data = read(path)
    require(data["format"] == "bbeh-paired-grades-v1" and data["role"] == role, "Wrong grade role/format")
    require(
        data["source_plan_sha256"] == plan["source_plan_sha256"] and data["pair_sha256"] == plan["pair_sha256"],
        "Grade source/pair changed",
    )
    require(
        data["models"] == plan["pair"]["models"] and data["cost_scope"] == plan["pair"]["cost_scope"],
        "Grade models/pricing changed",
    )
    for key in ("grader_sha256", "acquisition_protocol_sha256"):
        require_hash(data[key])
    require([row["id"] for row in data["rows"]] == [row["id"] for row in rows], "Exact role grade population required")
    for row in data["rows"]:
        for arm in ARMS:
            require(type(row[f"{arm}_correct"]) in (int, bool) and row[f"{arm}_correct"] in (0, 1), "Nonbinary label")
            require_hash(row[f"{arm}_record_sha256"])
            value = row[f"{arm}_cost_usd"]
            require(
                value is not None and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0,
                "Unknown costs cannot be dropped or imputed free",
            )
    y = np.array([[row[f"{arm}_correct"] for arm in ARMS] for row in data["rows"]], dtype=float)
    c = np.array([[float(row[f"{arm}_cost_usd"]) for arm in ARMS] for row in data["rows"]])
    return y, c, data


def features(plan, rows, root, role):
    path = Path(root) / role
    info = read(path / "features.json")
    require(info["inputs_sha256"] == plan["roles"][role]["inputs_sha256"], "Features use different role inputs")
    require(
        (info["model"], info["hf_revision"], info["instruction"], info["dimensions"])
        == (MODEL, REVISION, INSTRUCTION, 2560),
        "Encoder identity changed",
    )
    require(
        sha(path / "ids.json") == info["ids_sha256"] and sha(path / "vectors.npy") == info["vectors_sha256"],
        "Feature bytes changed",
    )
    require(read(path / "ids.json") == [row["id"] for row in rows], "Feature order changed")
    audit = read(path / "token-audit.json")
    require(
        sha(path / "token-audit.json") == info["token_audit_sha256"]
        and audit["inputs_sha256"] == info["inputs_sha256"]
        and audit["model"] == MODEL
        and audit["hf_revision"] == REVISION
        and audit["instruction"] == INSTRUCTION
        and audit["truncated_ids"] == []
        and audit["overflow_ids"] == [],
        "Exact no-truncation encoder audit required",
    )
    matrix = np.load(path / "vectors.npy", mmap_mode="r", allow_pickle=False)
    require(
        matrix.dtype == np.float32
        and matrix.shape == (len(rows), 2560)
        and np.isfinite(matrix).all()
        and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-6, rtol=0),
        "Invalid normalized features",
    )
    return matrix, sha(path / "features.json")


def conditional_costs(y, costs):
    means, counts = [], []
    for a in range(2):
        means.append(
            [
                float(costs[y[:, a] == label, a].mean()) if (y[:, a] == label).any() else float(costs[:, a].mean())
                for label in (0, 1)
            ]
        )
        counts.append([int((y[:, a] == label).sum()) for label in (0, 1)])
    return {"means": means, "counts": counts, "missing_class": "arm-wide mean", "unit": "USD"}


def expected_cost(probabilities, stats):
    means = np.array(stats["means"])
    return (1 - probabilities) * means[:, 0] + probabilities * means[:, 1]


def nll(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p), axis=0).sum())


def fit_logistic(matrix, target, penalty):
    from scipy.optimize import minimize
    from scipy.special import expit
    from scipy import sparse

    matrix = sparse.csr_matrix(matrix, dtype=float) if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    require(matrix.ndim == 2 and target.shape == (matrix.shape[0],) and np.isin(target, (0, 1)).all(), "Misaligned fit")
    require(np.isfinite(matrix.data if sparse.issparse(matrix) else matrix).all(), "Nonfinite features")
    mean = np.asarray(matrix.mean(axis=0)).reshape(-1)
    frequency = float(target.mean())
    if frequency in (0, 1):
        return {
            "weights": [0.0] * matrix.shape[1],
            "bias": float(np.log((target.sum() + 0.5) / (len(target) - target.sum() + 0.5))),
            "constant_class": True,
        }
    initial = np.zeros(matrix.shape[1] + 1)
    initial[-1] = np.log(frequency / (1 - frequency))

    def loss(theta):
        w, b = theta[:-1], theta[-1]
        logits = np.asarray(matrix @ w).reshape(-1) - mean @ w + b
        residual = expit(logits) - target
        value = np.logaddexp(0, (1 - 2 * target) * logits).sum() + 0.5 * penalty * (w @ w)
        gradient = np.asarray(matrix.T @ residual).reshape(-1) - mean * residual.sum() + penalty * w
        return float(value), np.r_[gradient, residual.sum()]

    result = minimize(loss, initial, method="L-BFGS-B", jac=True, options=PROTOCOL["optimizer"])
    require(result.success and np.isfinite(result.x).all(), "Optimizer failed: " + str(result.message))
    return {
        "weights": result.x[:-1].tolist(),
        "bias": float(result.x[-1] - mean @ result.x[:-1]),
        "iterations": int(result.nit),
    }


def logistic_probability(matrix, head):
    logits = np.asarray(matrix @ np.asarray(head["weights"])).reshape(-1) + head["bias"]
    return np.exp(-np.logaddexp(0, -logits))


def two_heads(matrix, y, penalty):
    return [fit_logistic(matrix, y[:, arm], penalty) for arm in range(2)]


def predict_heads(matrix, heads):
    return np.column_stack([logistic_probability(matrix, head) for head in heads])


def nearest(matrix, reference, labels, k, *, batch=128):
    require(type(k) is int and 0 < k <= len(reference), "Invalid fixed neighbor count")
    prior = np.asarray(labels, dtype=float).mean(axis=0)
    result = []
    reference = np.asarray(reference, dtype=float)
    for start in range(0, len(matrix), batch):
        similarity = np.asarray(matrix[start : start + batch], dtype=float) @ reference.T
        indices = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
        result.append((labels[indices].sum(axis=1) + 2 * prior) / (k + 2))
    return np.concatenate(result)


def lengths(rows):
    values = []
    for row in rows:
        text = row["request"]
        values.append(
            [
                np.log1p(len(text)),
                np.log1p(len(text.split())),
                np.log1p(text.count("\n") + 1),
                sum(c.isdigit() for c in text) / max(len(text), 1),
                sum(c in "[](){}" for c in text) / max(len(text), 1),
            ]
        )
    return np.array(values)


def text_matrix(rows, saved=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy import sparse

    spec = {**PROTOCOL["text"], "ngram_range": tuple(PROTOCOL["text"]["ngram_range"])}
    vectorizer = TfidfVectorizer(**spec, vocabulary=None if saved is None else saved["vocabulary"])
    if saved is None:
        matrix = vectorizer.fit_transform([row["request"] for row in rows])
        raw = lengths(rows)
        scale = np.maximum(raw.std(axis=0), 1e-6)
        saved = {
            "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
            "idf": vectorizer.idf_.tolist(),
            "length_scale": scale.tolist(),
        }
    else:
        vectorizer.idf_ = np.array(saved["idf"])
        matrix = vectorizer.transform([row["request"] for row in rows])
    return matrix, sparse.hstack((matrix, lengths(rows) / np.array(saved["length_scale"])), format="csr"), saved


def task_statistics(rows, y, costs):
    names = sorted({row["task"] for row in rows})
    return {
        "names": names,
        "probabilities": [
            (
                (y[[r["task"] == name for r in rows]].sum(axis=0) + 20 * y.mean(axis=0))
                / (sum(r["task"] == name for r in rows) + 20)
            ).tolist()
            for name in names
        ],
        "costs": [
            (
                (costs[[r["task"] == name for r in rows]].sum(axis=0) + 20 * costs.mean(axis=0))
                / (sum(r["task"] == name for r in rows) + 20)
            ).tolist()
            for name in names
        ],
        "global_probabilities": y.mean(axis=0).tolist(),
        "global_costs": costs.mean(axis=0).tolist(),
    }


def fit_controls(rows, y, costs):
    from sklearn.naive_bayes import MultinomialNB

    word, text, text_info = text_matrix(rows)
    task = task_statistics(rows, y, costs)
    nb = MultinomialNB(alpha=1).fit(word, [row["task"] for row in rows])
    require(nb.classes_.tolist() == task["names"], "Task class alignment changed")
    return {
        "text": text_info,
        "text_heads": two_heads(text, y, PROTOCOL["text_penalty"]),
        "task": task,
        "topic_nb": {
            "feature_log_prob": nb.feature_log_prob_.tolist(),
            "class_log_prior": nb.class_log_prior_.tolist(),
        },
    }


def predict_controls(model, rows):
    from scipy.special import softmax

    word, text, _ = text_matrix(rows, model["text"])
    nb, task = model["topic_nb"], model["task"]
    weights = softmax(word @ np.array(nb["feature_log_prob"]).T + np.array(nb["class_log_prior"]), axis=1)
    by_task = {name: i for i, name in enumerate(task["names"])}
    return {
        "text_length": {"probabilities": predict_heads(text, model["text_heads"])},
        "text_topic": {
            "probabilities": weights @ np.array(task["probabilities"]),
            "expected_usd": weights @ np.array(task["costs"]),
        },
        "task_metadata": {
            "probabilities": np.array(
                [
                    task["probabilities"][by_task[r["task"]]] if r["task"] in by_task else task["global_probabilities"]
                    for r in rows
                ]
            ),
            "expected_usd": np.array(
                [task["costs"][by_task[r["task"]]] if r["task"] in by_task else task["global_costs"] for r in rows]
            ),
        },
    }


def fit_arrays(train_x, train_rows, train_y, train_cost, dev_x, dev_rows, dev_y, dev_cost):
    require(
        all(r["role"] == "train" for r in train_rows) and all(r["role"] == "development" for r in dev_rows),
        "Only train/development may fit or tune",
    )
    require(not ({r["family"] for r in train_rows} & {r["family"] for r in dev_rows}), "Families cross fitting/tuning")
    require(
        len(train_x) == len(train_rows) == len(train_y) == len(train_cost)
        and len(dev_x) == len(dev_rows) == len(dev_y) == len(dev_cost),
        "Fit arrays misaligned",
    )
    tuning, train_means = [], conditional_costs(train_y, train_cost)
    started = time.perf_counter()
    for k in PROTOCOL["neighbors"]:
        p = nearest(dev_x, train_x, train_y, k)
        tuning.append(
            {
                "method": "knn",
                "setting": k,
                "nll": nll(dev_y, p),
                "probabilities": p.tolist(),
                "expected_usd": expected_cost(p, train_means).tolist(),
            }
        )
    for penalty in PROTOCOL["penalties"]:
        heads = two_heads(train_x, train_y, penalty)
        p = predict_heads(dev_x, heads)
        tuning.append(
            {
                "method": "embedding",
                "setting": penalty,
                "nll": nll(dev_y, p),
                "probabilities": p.tolist(),
                "expected_usd": expected_cost(p, train_means).tolist(),
            }
        )
    train_controls = fit_controls(train_rows, train_y, train_cost)
    for name, values in predict_controls(train_controls, dev_rows).items():
        p = values["probabilities"]
        c = values.get("expected_usd", expected_cost(p, train_means))
        tuning.append(
            {
                "method": name,
                "setting": "fixed",
                "nll": nll(dev_y, p),
                "probabilities": p.tolist(),
                "expected_usd": c.tolist(),
                "metadata_assisted": name == "task_metadata",
            }
        )
    selected = {
        name: min([r for r in tuning if r["method"] == name], key=lambda r: (r["nll"], -r["setting"]))["setting"]
        for name in ("knn", "embedding")
    }
    tuning_seconds = time.perf_counter() - started
    x, y, cost = (
        np.concatenate((train_x, dev_x)),
        np.concatenate((train_y, dev_y)),
        np.concatenate((train_cost, dev_cost)),
    )
    rows = train_rows + dev_rows
    started = time.perf_counter()
    heads = two_heads(x, y, selected["embedding"])
    controls = fit_controls(rows, y, cost)
    result = {
        "selected": selected,
        "tuning": tuning,
        "conditional_cost_usd": conditional_costs(y, cost),
        "heads": heads,
        "labels": y.tolist(),
        "reference_ids": [r["id"] for r in rows],
        **controls,
        "timing": {"tuning_seconds": tuning_seconds, "refit_seconds": time.perf_counter() - started},
        "knn_storage": {
            "rows": len(x),
            "dimensions": x.shape[1],
            "float32_bytes": len(x) * x.shape[1] * 4,
            "similarity_multiply_adds_per_request": len(x) * x.shape[1],
        },
    }
    return result, x.astype(np.float32)


def predict_arrays(model, reference, matrix, rows):
    p = {
        "knn": nearest(matrix, reference, np.array(model["labels"]), model["selected"]["knn"]),
        "embedding": predict_heads(matrix, model["heads"]),
    }
    out = {
        name: {
            "probabilities": value.tolist(),
            "expected_usd": expected_cost(value, model["conditional_cost_usd"]).tolist(),
            "router_kind": "embedding" if name in {"knn", "embedding"} else "cpu",
            "metadata_assisted": False,
        }
        for name, value in p.items()
    }
    for name, values in predict_controls(model, rows).items():
        out[name] = {
            "probabilities": values["probabilities"].tolist(),
            "expected_usd": values.get(
                "expected_usd", expected_cost(values["probabilities"], model["conditional_cost_usd"])
            ).tolist(),
            "router_kind": "cpu",
            "metadata_assisted": name == "task_metadata",
        }
    return out


def fit(run, train_grades, development_grades, feature_root):
    run = Path(run)
    require(not (run / "model.json").exists(), "Preserve fitted models")
    plan = checked_plan(run)
    rows = [role_rows(run, plan, role) for role in ("train", "development")]
    g = [
        grades(plan, r, path, role)
        for r, path, role in zip(rows, (train_grades, development_grades), ("train", "development"), strict=True)
    ]
    require(g[0][2]["grader_sha256"] == g[1][2]["grader_sha256"], "Tuning grading semantics changed")
    x = [features(plan, r, feature_root, role) for r, role in zip(rows, ("train", "development"), strict=True)]
    model, reference = fit_arrays(x[0][0], rows[0], g[0][0], g[0][1], x[1][0], rows[1], g[1][0], g[1][1])
    with (run / "reference.npy").open("xb") as stream:
        np.save(stream, reference, allow_pickle=False)
    model.update(
        {
            "format": "bbeh-embedding-model-v1",
            "plan_sha256": sha(run / "plan.json"),
            "source_plan_sha256": plan["source_plan_sha256"],
            "pair_sha256": plan["pair_sha256"],
            "train_grades_sha256": sha(train_grades),
            "development_grades_sha256": sha(development_grades),
            "features_sha256": {"train": x[0][1], "development": x[1][1]},
            "reference_sha256": sha(run / "reference.npy"),
            "calibration_or_final_outcomes_accessed": False,
        }
    )
    write(run / "model.json", model)
    return model


def score(run, feature_root, role, output):
    run = Path(run)
    require(
        role in {"calibration", "final"},
        "Selected refit scores only calibration/final; tuning predictions are saved separately",
    )
    plan, model = checked_plan(run), read(run / "model.json")
    require(
        model["plan_sha256"] == sha(run / "plan.json") and model["reference_sha256"] == sha(run / "reference.npy"),
        "Fitted model/reference changed",
    )
    rows = role_rows(run, plan, role)
    matrix, feature_sha = features(plan, rows, feature_root, role)
    reference = np.load(run / "reference.npy", mmap_mode="r", allow_pickle=False)
    started = time.perf_counter()
    methods = predict_arrays(model, reference, matrix, rows)
    result = {
        "format": "bbeh-input-only-scores-v1",
        "role": role,
        "ids": [r["id"] for r in rows],
        "source_plan_sha256": plan["source_plan_sha256"],
        "pair_sha256": plan["pair_sha256"],
        "model_sha256": sha(run / "model.json"),
        "features_sha256": feature_sha,
        "methods": methods,
        "outcomes_accessed": False,
        "batch_scoring_seconds": time.perf_counter() - started,
        "latency_scope": "Batch scoring throughput; not a single-request serving measurement",
    }
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--run", required=True)
    p = commands.add_parser("fit")
    p.add_argument("--run", required=True)
    p.add_argument("--train-grades", required=True)
    p.add_argument("--development-grades", required=True)
    p.add_argument("--features", required=True)
    p = commands.add_parser("score")
    p.add_argument("--run", required=True)
    p.add_argument("--split", choices=("calibration", "final"), required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.pair, args.run)
    elif args.command == "fit":
        fit(args.run, args.train_grades, args.development_grades, args.features)
    else:
        score(args.run, args.features, args.split, args.output)


if __name__ == "__main__":
    main()
