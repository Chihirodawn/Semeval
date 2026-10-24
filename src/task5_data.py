"""Task5 EN NN data preparation and metrics from public definitions.

Context variance uses population SD (ddof=0), including singleton SD=0.
This is a local implementation; official scorer equivalence is unverified.
"""

import csv
import hashlib
import json
import math
import random
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len({row["ContextID"] for row in rows}) == len(rows)
    for row in rows:
        for name in ("ModAvg", "HeadAvg"):
            assert math.isfinite(float(row[name]))
    return rows


def grouped_split(rows, seed=2027, val_fraction=0.2):
    # Targets connected by an identical normalized sentence also stay together.
    targets = sorted({normalized(row["Compound"]) for row in rows})
    parent = {target: target for target in targets}

    def find(target):
        while parent[target] != target:
            parent[target] = parent[parent[target]]
            target = parent[target]
        return target

    by_context = {}
    for row in rows:
        target, context = normalized(row["Compound"]), normalized(row["Context"])
        if context in by_context:
            a, b = find(target), find(by_context[context])
            parent[max(a, b)] = min(a, b)
        by_context[context] = target
    components = defaultdict(list)
    for target in targets:
        components[find(target)].append(target)
    groups = sorted(components.values())
    random.Random(seed).shuffle(groups)
    wanted = round(len(targets) * val_fraction)
    val_targets = set()
    for group in groups:
        if len(val_targets) < wanted:
            val_targets.update(group)
    train = [row for row in rows if normalized(row["Compound"]) not in val_targets]
    val = [row for row in rows if normalized(row["Compound"]) in val_targets]
    assert train and val
    for column in ("Compound", "Context"):
        assert not ({normalized(r[column]) for r in train} & {normalized(r[column]) for r in val})
    return train, val, {
        "seed": seed,
        "requested_val_fraction": val_fraction,
        "train_rows": len(train),
        "val_rows": len(val),
        "train_targets": len(targets) - len(val_targets),
        "val_targets": len(val_targets),
        "connected_components": len(components),
        "target_overlap": 0,
        "normalized_context_overlap": 0,
    }


def ranks(values):
    indices = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(indices):
        end = start + 1
        while end < len(indices) and values[indices[end]] == values[indices[start]]:
            end += 1
        for pos in range(start, end):
            result[indices[pos]] = (start + 1 + end) / 2.0
        start = end
    return result


def spearman(gold, pred):
    if len(gold) < 2 or max(gold) - min(gold) < 1e-12 or max(pred) - min(pred) < 1e-12:
        return None
    a, b = ranks(gold), ranks(pred)
    am, bm = statistics.mean(a), statistics.mean(b)
    denom = math.sqrt(sum((x - am) ** 2 for x in a) * sum((y - bm) ** 2 for y in b))
    return sum((x - am) * (y - bm) for x, y in zip(a, b)) / denom


def metrics(rows, predictions):
    assert len(rows) == len(predictions) and len(rows) > 0
    assert all(len(p) == 2 and all(math.isfinite(float(v)) for v in p) for p in predictions)
    result = {"rows": len(rows), "metric_implementation": "local_public_definition_ddof0_singletons0"}
    for col, label in enumerate(("ModAvg", "HeadAvg")):
        gold = [float(row[label]) for row in rows]
        pred = [float(p[col]) for p in predictions]
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[normalized(row["Compound"])].append(i)
        gold_std = [statistics.pstdev(gold[i] for i in indices) for indices in groups.values()]
        pred_std = [statistics.pstdev(pred[i] for i in indices) for indices in groups.values()]
        result[label] = {
            "spearman": spearman(gold, pred),
            "rmse": math.sqrt(statistics.mean((g - p) ** 2 for g, p in zip(gold, pred))),
            "context_variance_spearman": spearman(gold_std, pred_std),
            "targets": len(groups),
            "singleton_targets": sum(len(indices) == 1 for indices in groups.values()),
        }
    rho = [result[key]["spearman"] for key in ("ModAvg", "HeadAvg")]
    result["mean_spearman"] = statistics.mean(rho) if all(x is not None for x in rho) else None
    result["mean_rmse"] = statistics.mean(result[key]["rmse"] for key in ("ModAvg", "HeadAvg"))
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root):
    root = Path(root)
    source = root / "train/en-nn-train.tsv"
    train, val, summary = grouped_split(read_rows(source))
    destination = root / "data/en_nn_seed2027"
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = destination / (name + ".jsonl")
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        if path.exists():
            assert path.read_text(encoding="utf-8") == content, "refuse to change frozen split"
        else:
            path.write_text(content, encoding="utf-8")
        summary[name + "_sha256"] = sha256(path)
        summary[name + "_ids"] = [row["ContextID"] for row in rows]
    summary["source_sha256"] = sha256(source)
    summary["scope"] = "Internal Val from official Train only; no official Dev/Test"
    (destination / "split.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_ids")}, indent=2))


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
