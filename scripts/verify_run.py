"""Independent CPU checks of a completed run, including SciPy rank metrics."""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(run):
    config = json.loads((run / "config.json").read_text())
    completion = json.loads((run / "completion.json").read_text())
    scores = json.loads((run / "metrics.json").read_text())
    rows = [json.loads(line) for line in (run / "val_predictions.jsonl").read_text().splitlines()]
    source = Path(config["data"]) / "val.jsonl"
    gold = [json.loads(line) for line in source.read_text().splitlines()]
    if config["smoke"]:
        gold = gold[:32]
    assert [row["ContextID"] for row in rows] == [row["ContextID"] for row in gold]
    assert len({row["ContextID"] for row in rows}) == len(rows)
    assert [row["gold"] for row in rows] == [[float(row["ModAvg"]), float(row["HeadAvg"])] for row in gold]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["Compound"]].append(i)
    canonical = json.loads(json.dumps(scores))
    secondary_diagnostics = []
    for form in ("raw", "clipped"):
        for col, label in enumerate(("ModAvg", "HeadAvg")):
            target = np.asarray([row["gold"][col] for row in rows], dtype=np.float64)
            pred = np.asarray([row[form][col] for row in rows], dtype=np.float64)
            assert np.isfinite(pred).all()
            if form == "clipped":
                assert ((pred >= 0) & (pred <= 5)).all()
            independent = {
                "rmse": float(np.sqrt(np.mean((target - pred) ** 2))),
                "spearman": float(spearmanr(target, pred).statistic),
            }
            for name, value in independent.items():
                observed = scores[form][label][name]
                if math.isnan(value):
                    assert observed is None
                else:
                    assert observed is not None and abs(value - observed) < 1e-12, (form, label, name, value, observed)
            # Ranks are discontinuous at ties. SDs differing only in floating
            # roundoff can produce different rho despite identical statistics.
            # Verify SD vectors independently, and publish a separate, explicit
            # 12-decimal tie convention. Preserve the original metrics.json.
            gold_np = np.asarray([float(np.std(target[indices], ddof=0)) for indices in groups.values()])
            pred_np = np.asarray([float(np.std(pred[indices], ddof=0)) for indices in groups.values()])
            gold_ref = np.asarray([statistics.pstdev(float(target[i]) for i in indices) for indices in groups.values()])
            pred_ref = np.asarray([statistics.pstdev(float(pred[i]) for i in indices) for indices in groups.values()])
            np.testing.assert_allclose(gold_np, gold_ref, rtol=0, atol=1e-12)
            np.testing.assert_allclose(pred_np, pred_ref, rtol=0, atol=1e-12)
            rounded = [[round(float(x), 12) for x in a] for a in (gold_np, pred_np, gold_ref, pred_ref)]
            assert rounded[0] == rounded[2] and rounded[1] == rounded[3]
            value = float(spearmanr(rounded[0], rounded[1]).statistic)
            canonical[form][label]["context_variance_spearman"] = None if math.isnan(value) else value
            raw_numpy = float(spearmanr(gold_np, pred_np).statistic)
            secondary_diagnostics.append({"form": form, "label": label,
                "original_stored_rho": scores[form][label]["context_variance_spearman"],
                "unrounded_numpy_rho": None if math.isnan(raw_numpy) else raw_numpy,
                "canonical_round12_rho": None if math.isnan(value) else value})
        canonical[form]["metric_implementation"] = "local_public_definition_ddof0_singletons0_std_round12"
        assert abs(np.mean([scores[form][label]["spearman"] for label in ("ModAvg", "HeadAvg")]) - scores[form]["mean_spearman"]) < 1e-12
        assert abs(np.mean([scores[form][label]["rmse"] for label in ("ModAvg", "HeadAvg")]) - scores[form]["mean_rmse"]) < 1e-12
    for name, sha in completion["checkpoint_sha256"].items():
        assert digest(run / "best" / name) == sha
    assert digest(run / "val_predictions.jsonl") == completion["prediction_sha256"]
    assert completion["reload_max_abs_diff"] < 1e-6
    assert completion["repeat_max_abs_diff"] < 1e-6
    (run / "metrics_std_round12.json").write_text(json.dumps(canonical, indent=2, allow_nan=False) + "\n")
    proof = {"status": "PASS", "rows": len(rows), "primary_metrics_match_scipy_numpy": True,
             "context_std_vectors_match_atol1e12_and_canonical_ranks_match": True,
             "id_order_gold_finite_range_checkpoint_hashes": True,
             "secondary_diagnostics": secondary_diagnostics,
             "note": "Official scorer not obtained. ddof=0, singleton SD=0. Original metrics preserved; secondary SD rounded to 12 decimals in metrics_std_round12.json; model selection and predictions unchanged."}
    (run / "INDEPENDENT_VERIFIED.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof), flush=True)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run)
