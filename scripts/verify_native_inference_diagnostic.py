"""Independent SciPy metric and raw-trace checks for the inference diagnostic."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from task5_data import sha256
from thinking_protocol import check_trace,identity,inputs,sample_seed,validate_reasoning


def main():
    from transformers import AutoTokenizer
    folder=Path(sys.argv[1]);cfg=json.loads((folder/"config.json").read_text())
    done=json.loads((folder/"completion.json").read_text());report=json.loads((folder/"metrics.json").read_text())
    assert done["status"]=="COMPLETED" and done["optimizer_updates"]==0
    for name,digest in {**cfg["source_sha256"],**cfg["data_sha256"]}.items():assert sha256(ROOT/name)==digest
    assert sha256(folder/"predictions.jsonl")==done["prediction_sha256"]
    assert sha256(folder/"thinking_traces.jsonl")==done["thinking_trace_sha256"]
    rows=[json.loads(x) for x in (folder/"predictions.jsonl").read_text().splitlines()]
    traces=[json.loads(x) for x in (folder/"thinking_traces.jsonl").read_text().splitlines()]
    original=[json.loads(x) for x in (ROOT/"data/en_nn_seed2027/val.jsonl").read_text().splitlines()][:32]
    assert len(rows)==32 and len(traces)==64
    assert [r["ContextID"] for r in rows]==cfg["validation_ids"]
    assert [r["gold"] for r in rows]==[[float(r["ModAvg"]),float(r["HeadAvg"])] for r in original]
    assert [identity(r) for r in traces]==[str(r["ContextID"])+":"+str(c) for r in rows for c in (0,1)]
    tok=AutoTokenizer.from_pretrained(ROOT/cfg["model"],local_files_only=True)
    expected=inputs(original,tok,"free");direct=inputs(original,tok,"direct")
    base_done=json.loads((ROOT/cfg["baseline"]/"completion.json").read_text())
    for name,digest in base_done["checkpoint_sha256"].items():assert sha256(ROOT/cfg["baseline"]/"best"/name)==digest
    for i,r in enumerate(traces):
        assert all(k not in r for k in ("gold","ModAvg","HeadAvg"));check_trace(r,tok)
        assert r["prompt_ids"]==(direct[i] if r["fallback"] else expected[i])["prompt_ids"]
        for attempt,a in enumerate(r["attempts"]):
            assert tok.decode(a["cot_ids"],skip_special_tokens=False)==a["cot_text"]
            assert a["generation_seed"]==sample_seed(2027,r,attempt)
            assert a["error"]==validate_reasoning(a["cot_text"],"free",r["Context"],a["stop_reason"])
        if r["fallback"]:assert rows[i//2]["thinking"][i%2]==rows[i//2]["direct"][i%2]
    y=np.asarray([r["gold"] for r in rows])
    for key,field in (("direct","direct"),("thinking_with_explicit_direct_fallback","thinking")):
        pred=np.asarray([r[field] for r in rows]);assert np.isfinite(pred).all()
        rho=[float(spearmanr(y[:,c],pred[:,c]).statistic) for c in (0,1)]
        rmses=np.sqrt(((pred-y)**2).mean(0))
        expected=report[key]
        if all(np.isfinite(rho)):assert abs(expected["mean_spearman"]-float(np.mean(rho)))<1e-12
        else:assert expected["mean_spearman"] is None
        assert abs(expected["mean_rmse"]-float(rmses.mean()))<1e-12
    assert report["fallback_queries"]==sum(r["fallback"] for r in traces)
    assert report["attempt_errors"]==dict(Counter(a["error"] or "valid" for r in traces for a in r["attempts"]))
    for c,name in enumerate(("Modifier","Head")):
        selected=[i for i in range(32) if not traces[2*i+c]["fallback"]]
        group=report["valid_thinking_only_matched_components"][name]
        assert group["valid_queries"]==len(selected)
        for key in ("direct","thinking"):
            if not selected:assert group[key] is None;continue
            pred=np.asarray([rows[i][key][c] for i in selected]);gold=y[selected,c]
            assert abs(group[key]["rmse"]-float(np.sqrt(((pred-gold)**2).mean())))<1e-12
            value=float(spearmanr(gold,pred).statistic) if len(selected)>1 else float("nan")
            if np.isfinite(value):assert abs(group[key]["spearman"]-value)<1e-12
            else:assert group[key]["spearman"] is None
    result={"status":"PASS","scope":"Inference-only diagnostic, not A1/A2 training or promotion", "rows":32,"queries":64,
            "independent_scipy_metrics":True,"source_data_and_trace_hashes":True,"fallback_and_success_masks_verified":True}
    (folder/"INDEPENDENT_VERIFIED.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()
