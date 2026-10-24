"""Post-hoc diagnostics of frozen Task5 Internal Val outputs; no model execution."""
import collections
import itertools
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from task5_data import metrics, normalized, ranks, sha256, spearman

OUT = ROOT / "reports/20261022/loss_diagnostics"
NAMES = {
    "last": "qwen4b_context_b8_seed2027", "mean": "qwen4b_mean_seed2027",
    "rail": "qwen4b_rail_seed2027", "ce": "qwen4b_ce_seed2027",
    "raft": "qwen4b_raft_seed2027", "tract1": "qwen4b_tract_s1_seed2027",
    "tract2": "qwen4b_tract_s2_seed2027", "real": "qwen4b_real_seed2027",
}
SCORE = re.compile(r"\bScore\s*:\s*([0-5](?:\.\d+)?)(?![\d.])", re.I)
INPUTS = set()


def load(path):
    INPUTS.add(path)
    return json.loads(path.read_text())


def lines(path):
    INPUTS.add(path)
    return [json.loads(x) for x in path.read_text().splitlines()]


def mean(values):
    return sum(values) / len(values)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def fast_rho(a, b):
    a, b = ranks(a), ranks(b)
    am, bm = mean(a), mean(b)
    numerator = sum((x - am) * (y - bm) for x, y in zip(a, b))
    denominator = math.sqrt(sum((x - am) ** 2 for x in a) * sum((y - bm) ** 2 for y in b))
    return numerator / denominator if denominator else None


def subset(gold, pred, indices):
    errors = [pred[i] - gold[i] for i in indices]
    return {"n": len(indices), "gold_mean": mean([gold[i] for i in indices]),
            "prediction_mean": mean([pred[i] for i in indices]), "bias": mean(errors),
            "mae": mean([abs(e) for e in errors]), "rmse": math.sqrt(mean([e * e for e in errors])),
            "sse": sum(e * e for e in errors)}


def component_stats(rows, gold, pred, groups, component):
    result = subset(gold, pred, list(range(len(gold))))
    result["rho"] = spearman(gold, pred)
    result["gold_sd"] = statistics.pstdev(gold)
    result["prediction_sd"] = statistics.pstdev(pred)
    bins = {}
    for lo in range(5):
        ids = [i for i, y in enumerate(gold) if lo <= y and (y < lo + 1 or lo == 4 and y <= 5)]
        if ids:
            bins[str(lo)] = subset(gold, pred, ids)
            bins[str(lo)]["sse_share"] = bins[str(lo)]["sse"] / result["sse"]
    assert sum(x["n"] for x in bins.values()) == len(gold)
    assert math.isclose(sum(x["sse"] for x in bins.values()), result["sse"], abs_tol=1e-9)
    result["gold_bins"] = bins
    centered_gold, centered_pred, targets = [], [], []
    between_sse = within_sse = correct = tied = comparable = 0
    for target, ids in groups.items():
        g, p = [gold[i] for i in ids], [pred[i] for i in ids]
        gm, pm = mean(g), mean(p)
        between_sse += len(ids) * (pm - gm) ** 2
        within_sse += sum(((pred[i] - gold[i]) - (pm - gm)) ** 2 for i in ids)
        centered_gold.extend(x - gm for x in g)
        centered_pred.extend(x - pm for x in p)
        targets.append({"target": target, **subset(gold, pred, ids), "gold_sd": statistics.pstdev(g),
                        "prediction_sd": statistics.pstdev(p), "rho": spearman(g, p)})
        for i, j in itertools.combinations(ids, 2):
            dg, dp = gold[i] - gold[j], pred[i] - pred[j]
            if abs(dg) <= 1e-12:
                continue
            comparable += 1
            if abs(dp) <= 1e-12:
                tied += 1
            elif dg * dp > 0:
                correct += 1
    assert math.isclose(between_sse + within_sse, result["sse"], abs_tol=1e-9)
    result.update(target_mean_error_sse_share=between_sse / result["sse"],
                  within_target_error_sse_share=within_sse / result["sse"],
                  target_mean_rho=spearman([x["gold_mean"] for x in targets], [x["prediction_mean"] for x in targets]),
                  within_target_centered_rho=spearman(centered_gold, centered_pred),
                  within_target_pair_accuracy=(correct + 0.5 * tied) / comparable,
                  within_target_comparable_pairs=comparable,
                  mean_target_gold_sd=mean([x["gold_sd"] for x in targets]),
                  mean_target_prediction_sd=mean([x["prediction_sd"] for x in targets]))
    result["largest_target_sse"] = sorted(targets, key=lambda x: x["sse"], reverse=True)[:12]
    worst = sorted(range(len(gold)), key=lambda i: abs(pred[i] - gold[i]), reverse=True)[:12]
    result["worst_examples"] = [{"ContextID": rows[i]["ContextID"], "Compound": rows[i]["Compound"],
                                 "Context": rows[i]["Context"], "gold": gold[i], "prediction": pred[i],
                                 "error": pred[i] - gold[i]} for i in worst]
    disagreement = [float(x[("ModStd", "HeadStd")[component]]) for x in rows]
    result["annotator_sd_vs_absolute_error_rho"] = spearman(disagreement, [abs(p - g) for p, g in zip(pred, gold)])
    return result


def reasoning_stats(path):
    rows = lines(path)
    found = [(r, float(scores[-1])) for r in rows if (scores := SCORE.findall(r["cot_text"]))]
    result = {"rows": len(rows), "stop_reasons": dict(collections.Counter(r["stop_reason"] for r in rows)),
              "mean_cot_tokens": mean([len(r["cot_ids"]) for r in rows]),
              "embedded_score_rows": len(found), "embedded_score_fraction": len(found) / len(rows),
              "regex": SCORE.pattern, "interpretation": "generated teacher scores, NOT leaked human gold"}
    if found:
        result.update(embedded_mean=mean([p for r, p in found]),
                      gold_mean_on_detected_subset=mean([r["gold"] for r, p in found]),
                      embedded_bias=mean([p - r["gold"] for r, p in found]),
                      embedded_mae=mean([abs(p - r["gold"]) for r, p in found]),
                      embedded_fraction_5=mean([p == 5 for r, p in found]),
                      embedded_error_at_least_1=mean([abs(p - r["gold"]) >= 1 for r, p in found]))
        result["examples"] = [{k: r[k] for k in ["ContextID", "component", "gold", "cot_text", "stop_reason"]}
                              for r, score in found[:5]]
    return result


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    train = lines(ROOT / "data/en_nn_seed2027/train.jsonl")
    val = lines(ROOT / "data/en_nn_seed2027/val.jsonl")
    split = load(ROOT / "data/en_nn_seed2027/split.json")
    for name in ("train", "val"):
        assert sha256(ROOT / f"data/en_nn_seed2027/{name}.jsonl") == split[name + "_sha256"]
    groups = collections.defaultdict(list)
    for i, row in enumerate(val):
        groups[normalized(row["Compound"])].append(i)
    contexts = collections.defaultdict(set)
    for row in val:
        contexts[normalized(row["Context"])].add(normalized(row["Compound"]))
    assert all(len(targets) == 1 for targets in contexts.values()), "bootstrap needs shared-context target clusters"
    gold = [[float(r[label]) for r in val] for label in ("ModAvg", "HeadAvg")]
    entropy = []
    for row in train:
        for label in ("ModAvg", "HeadAvg"):
            fraction = float(row[label]) % 1
            entropy.append(-sum(p * math.log(p) for p in (fraction, 1 - fraction) if p))
    result = {"scope": "Post-hoc diagnostics of fixed seed2027 Internal Val, no Test, no model calls",
              "rows": len(val), "targets": len(groups), "soft_label_train_entropy_floor_nats": mean(entropy),
              "models": {}, "reasoning": {}, "comparisons": {}}
    predictions = {}
    for short, name in NAMES.items():
        folder = ROOT / "runs" / name
        cfg, done = load(folder / "config.json"), load(folder / "completion.json")
        saved = lines(folder / "val_predictions.jsonl")
        assert len(saved) == len(val)
        assert [x["ContextID"] for x in saved] == [x["ContextID"] for x in val]
        assert [x["gold"] for x in saved] == list(map(list, zip(*gold)))
        assert sha256(folder / "val_predictions.jsonl") == done["prediction_sha256"]
        for source, expected in cfg["source_sha256"].items():
            assert sha256(ROOT / "src" / source) == expected, source
        pred = [[row["clipped"][j] for row in saved] for j in (0, 1)]
        predictions[short] = pred
        m = metrics(val, [row["clipped"] for row in saved])
        stored = load(folder / "metrics.json")["clipped"]
        for key in ("mean_spearman", "mean_rmse"):
            assert abs(m[key] - stored[key]) < 1e-12
        for j, label in enumerate(("ModAvg", "HeadAvg")):
            assert abs(fast_rho(gold[j], pred[j]) - m[label]["spearman"]) < 1e-12
        history = load(folder / "history.json")
        curve = []
        for h in history:
            metric = h["metrics"]["clipped"]
            val_mse = mean([metric[label]["rmse"] ** 2 for label in ("ModAvg", "HeadAvg")])
            entry = {"epoch": h["epoch"], "train_mse_online": h.get("train_mse"),
                     "train_loss_online": h.get("train_loss", h.get("train_mse")),
                     "val_mean_rho": metric["mean_spearman"], "val_mean_rmse": metric["mean_rmse"],
                     "val_mse_pooled_components": val_mse}
            if short in ("tract1", "tract2", "real"):
                entry["cot_ce" if short.startswith("tract") else "policy_loss_inferred"] = h["train_loss"] - h["train_mse"]
            if short == "ce":
                entry["soft_ce_above_entropy_floor"] = h["train_loss"] - result["soft_label_train_entropy_floor_nats"]
            curve.append(entry)
        model = {"run": name, "config": {k: cfg.get(k) for k in ("loss", "seed", "batch_size", "grad_accum", "lr", "source_revision")},
                 "primary_metrics": {k: m[k] for k in ("mean_spearman", "mean_rmse")}, "history": curve,
                 "components": {label: component_stats(val, gold[j], pred[j], groups, j)
                                for j, label in enumerate(("ModAvg", "HeadAvg"))},
                 "elapsed_seconds": done["elapsed_seconds"]}
        log = OUT / "logs" / (name + ".log")
        if log.exists():
            INPUTS.add(log)
            states = []
            for line in log.read_text().splitlines():
                try:
                    x = json.loads(line)
                except ValueError:
                    continue
                if isinstance(x, dict) and x.get("stage") == "TRAINING":
                    states.append(x)
            norms = [x["gradient_norm"] for x in states]
            model["sampled_training_log"] = {"logged_updates": len(norms), "norm_before_clipping_median": statistics.median(norms),
                                               "fraction_norm_over_clip1": mean([x > 1 for x in norms]),
                                               "min_batch_loss": min(x["loss"] for x in states),
                                               "max_batch_loss": max(x["loss"] for x in states),
                                               "limit": "sampled logged updates only; no per-term gradients or raw reward/advantage logs"}
        diagnostic = folder / "numeric_diagnostics.json"
        if diagnostic.exists():
            model["numeric_mass"] = load(diagnostic)
        result["models"][short] = model
    for name, filename in [("qwen4b_cot_base_seed2027", "reasoning.jsonl"), ("qwen4b_cot_s1_seed2027", "reasoning.jsonl"),
                           ("qwen4b_tract_s1_seed2027", "best_validation_inputs.jsonl"),
                           ("qwen4b_tract_s2_seed2027", "best_validation_inputs.jsonl"),
                           ("qwen4b_real_seed2027", "best_validation_inputs.jsonl")]:
        path = ROOT / "runs" / name / filename
        done = load(path.parent / "completion.json")
        assert sha256(path) == done["trace_sha256" if filename == "reasoning.jsonl" else "validation_trace_sha256"]
        result["reasoning"][name] = reasoning_stats(path)
    for a, b in [("ce", "raft"), ("real", "tract2"), ("ce", "real"), ("ce", "last")]:
        differences = []
        for i, row in enumerate(val):
            for j, label in enumerate(("ModAvg", "HeadAvg")):
                ae, be = predictions[a][j][i] - gold[j][i], predictions[b][j][i] - gold[j][i]
                differences.append({"ContextID": row["ContextID"], "Compound": row["Compound"],
                                    "component": label, "gold": gold[j][i], a: predictions[a][j][i], b: predictions[b][j][i],
                                    "squared_error_difference": ae * ae - be * be})
        result["comparisons"][a + "_minus_" + b] = {
            "squared_error_win_fraction": mean([x["squared_error_difference"] < 0 for x in differences]),
            "mean_squared_error_difference": mean([x["squared_error_difference"] for x in differences]),
            "largest_improvements": sorted(differences, key=lambda x: x["squared_error_difference"])[:10],
            "largest_regressions": sorted(differences, key=lambda x: x["squared_error_difference"], reverse=True)[:10]}
    # Resample entire targets, preserving their contexts and both components.
    rng, clusters, boot = random.Random(20261022), list(groups.values()), collections.defaultdict(list)
    for _ in range(2000):
        ids = [i for group in rng.choices(clusters, k=len(clusters)) for i in group]
        gs = [[g[i] for i in ids] for g in gold]
        scores = {n: mean([fast_rho(gs[j], [predictions[n][j][i] for i in ids]) for j in (0, 1)])
                  for n in ("ce", "raft", "tract2", "real", "last")}
        for a, b in [("ce", "raft"), ("real", "tract2"), ("ce", "real"), ("ce", "last")]:
            boot[a + "_minus_" + b].append(scores[a] - scores[b])
    result["cluster_bootstrap"] = {"replicates": 2000, "seed": 20261022,
                                    "unit": "89 entire target clusters, with replacement",
                                    "limit": "fixed trained models and stored CoTs; no training-seed or fresh-CoT uncertainty",
                                    "rho_difference_95_percentile_intervals": {
                                        key: [percentile(values, .025), percentile(values, .975)] for key, values in boot.items()}}
    result["source_sha256"] = sha256(Path(__file__))
    result["input_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(INPUTS)}
    (OUT / "diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "models": len(NAMES), "input_files": len(INPUTS),
                      "soft_label_entropy_floor": result["soft_label_train_entropy_floor_nats"],
                      "bootstrap": result["cluster_bootstrap"], "output": str(OUT / "diagnostics.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
