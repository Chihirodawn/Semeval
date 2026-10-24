"""Matched 32-row inference-only diagnostic. Does not admit A1/A2 training."""
import argparse
import collections
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from task5_data import metrics, sha256, spearman
from thinking_protocol import MAX_SEQUENCE, SAMPLING, SCORE_CUE, attach_gold, check_trace, identity, inputs


def comparison(rows, direct, thinking, traces):
    assert len(rows) == len(direct) == len(thinking) and len(traces) == 2 * len(rows)
    assert all(math.isfinite(v) and 0 <= v <= 5 for pair in direct + thinking for v in pair)
    component_results = {}
    for c, name in enumerate(("Modifier", "Head")):
        selected = [i for i in range(len(rows)) if not traces[2*i+c]["fallback"]]
        gold = [float(rows[i][("ModAvg", "HeadAvg")[c]]) for i in selected]
        result = {"valid_queries": len(selected), "direct": None, "thinking": None}
        if selected:
            for key, predictions in (("direct", direct), ("thinking", thinking)):
                pred = [predictions[i][c] for i in selected]
                result[key] = {"spearman": spearman(gold, pred),
                               "rmse": math.sqrt(sum((a-b)**2 for a,b in zip(gold,pred))/len(gold))}
        component_results[name] = result
    attempts = [a for r in traces for a in r["attempts"]]
    return {"rows": len(rows), "queries": len(traces),
            "direct": metrics(rows, direct), "thinking_with_explicit_direct_fallback": metrics(rows, thinking),
            "valid_thinking_only_matched_components": component_results,
            "fallback_queries": sum(r["fallback"] for r in traces),
            "first_attempt_valid_queries": sum(r["attempts"][0]["error"] is None for r in traces),
            "attempt_errors": dict(collections.Counter(a["error"] or "valid" for a in attempts)),
            "limitations": ["Inference-only on a direct-CE-trained checkpoint; no thinking training",
                            "Fixed first32 Internal Val rows, not full704 or Test; no checkpoint/prompt selection",
                            "Thinking uses frozen v6 free-style semantic input; comparison is the full inference flow",
                            "Overall thinking score includes explicitly disclosed direct fallback",
                            "Successful-only component subsets are selected subsets, not unbiased overall estimates",
                            "This diagnostic cannot pass or override original Train pilot/Smoke/quality gates"]}


def dump(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    output = ROOT / plan["output"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def status(**fields):
        dump(output/"status.json", dict(fields, pid=os.getpid(), elapsed_seconds=time.monotonic()-started))
    status(stage="CPU_CHECKS")
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_thinking_suite import append_record
    cfg = {**plan, "plan_sha256": sha256(plan_path), "optimizer_updates": 0}
    dump(output/"config.json", cfg)
    try:
        for name, digest in {**plan["source_sha256"], **plan["data_sha256"]}.items():
            assert sha256(ROOT/name) == digest, name
        proof = json.loads((ROOT/plan["previous_cpu_proof"]).read_text())
        assert proof["status"] == "PASS"
        for name, digest in proof["source_sha256"].items():
            assert sha256(ROOT/name) == digest, ("frozen v6 dependency changed", name)
        for test in ("tests/thinking_protocol_contract.py", "tests/native_inference_diagnostic_contract.py"):
            subprocess.run([sys.executable, test], cwd=ROOT, check=True, timeout=60)
        baseline = ROOT/plan["baseline"]
        completion = json.loads((baseline/"completion.json").read_text())
        assert completion["status"] == "COMPLETED"
        assert json.loads((baseline/"INDEPENDENT_VERIFIED.json").read_text())["status"] == "PASS"
        for name, digest in completion["checkpoint_sha256"].items():
            assert sha256(baseline/"best"/name) == digest, name
        assert sha256(baseline/"val_predictions.jsonl") == completion["prediction_sha256"]
        # Initialize numpy before torch, matching the existing server runtime.
        import numpy as np
        import torch
        from transformers import AutoTokenizer
        from thinking_model import ThinkingPolicy
        tokenizer = AutoTokenizer.from_pretrained(ROOT/plan["model"], local_files_only=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        all_rows = [json.loads(s) for s in (ROOT/"data/en_nn_seed2027/val.jsonl").read_text().splitlines()]
        rows = all_rows[:32]
        assert [r["ContextID"] for r in rows] == plan["validation_ids"]
        direct_rows = inputs(rows, tokenizer, "direct")
        thinking_rows = inputs(rows, tokenizer, "free")
        changed_gold = [dict(r, ModAvg=-999, HeadAvg=999) for r in rows]
        assert inputs(changed_gold, tokenizer, "free") == thinking_rows
        old_inputs = [json.loads(s) for s in (baseline/"best_validation_inputs.jsonl").read_text().splitlines()][:64]
        assert [r["prompt_ids"] for r in old_inputs] == [r["prompt_ids"] for r in direct_rows]
        for row in thinking_rows:
            assert tokenizer.decode(row["prompt_ids"][-8:], skip_special_tokens=False).endswith("<think>\n")
            assert len(row["prompt_ids"])+plan["budget"]+len(tokenizer.encode(SCORE_CUE, add_special_tokens=False)) <= MAX_SEQUENCE
        contract = {"status": "PASS", "scope": "CPU/source/checkpoint/input checks, not quality",
                    "rows": 32, "queries": 64, "native_thinking": True,
                    "gold_independent_generation_input": True, "matched_direct_prompt_tokens": True}
        dump(output/"INPUT_CONTRACT_VERIFIED.json", contract)
        append_record(output.name+":INPUT_CONTRACT", "Inference diagnostic CPU checks", contract, cfg)
        assets = json.loads((ROOT/plan["model"]/"DOWNLOAD_VERIFIED.json").read_text())
        assert assets["revision"] == plan["model_revision"]
        for name, info in assets["files"].items():
            assert sha256(ROOT/plan["model"]/name) == info["sha256"], name
        with (ROOT/"gpu0_task5.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                            "--format=csv,noheader,nounits"], text=True)
            assert not any(s.startswith(plan["gpu_uuid"]) for s in apps.splitlines()), "GPU0 occupied"
            assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["gpu_uuid"]
            assert shutil.disk_usage(ROOT).free > 5*1024**3
            torch.set_num_threads(4)
            torch.cuda.set_per_process_memory_fraction(.9, 0)
            assert torch.cuda.mem_get_info()[0] > 21*1024**3
            model = ThinkingPolicy(ROOT/plan["model"],
                    [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(6)],
                    adapter=baseline/"best", trainable=False, native_check=True).to("cuda:0").eval()
            assert not any(p.requires_grad for p in model.parameters())
            direct_records = attach_gold([model.reason(r,r,tokenizer,0,2027) for r in direct_rows], rows)
            def score(records):
                pred = []
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i in range(0,len(records),2):
                        pred.extend(max(0.,min(5.,v)) for v in
                                    model.batch(records[i:i+2],tokenizer,supervise=False)["prediction"].tolist())
                return pred
            direct_flat = score(direct_records)
            direct = [direct_flat[i:i+2] for i in range(0,64,2)]
            old_pred = [json.loads(s)["clipped"] for s in (baseline/"val_predictions.jsonl").read_text().splitlines()][:32]
            difference = max(abs(a-b) for x,y in zip(direct,old_pred) for a,b in zip(x,y))
            assert difference < 1e-6, ("direct baseline did not reproduce", difference)
            dump(output/"DIRECT_REPRODUCED.json", {"max_abs_diff":difference, "metrics":metrics(rows,direct)})
            traces, thinking = [], []
            with (output/"thinking_traces.jsonl").open("x") as stream, (output/"predictions.jsonl").open("x") as preds:
                for i, row in enumerate(thinking_rows):
                    attempts = []
                    for attempt in range(3):
                        assert time.monotonic()-started < plan["max_hours"]*3600, "diagnostic time limit"
                        status(stage="GENERATING_NATIVE_THINKING", completed_queries=i, total_queries=64,
                               current_id=identity(row), attempt=attempt+1, max_new_tokens=plan["budget"])
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            trace = model.generate_one(row,tokenizer,plan["budget"],2027,attempt)
                        attempts.append(trace)
                        dump(output/"current_attempts.json", {"id":identity(row), "attempts":attempts})
                        if trace["error"] is None:
                            record = dict(row, **{k:v for k,v in trace.items() if k!="error"},
                                          fallback=False, attempts=attempts, sampling=SAMPLING)
                            break
                    else:
                        record = dict(row, prompt_ids=direct_rows[i]["prompt_ids"],
                                      direct_prompt_ids=direct_rows[i]["prompt_ids"], cot_ids=[], cot_text="",
                                      stop_reason="fallback_direct", fallback=True, attempts=attempts, sampling=SAMPLING)
                    check_trace(record,tokenizer)
                    traces.append(record)
                    stream.write(json.dumps(record,ensure_ascii=False)+"\n");stream.flush()
                    if i%2 == 1:
                        pair = score(attach_gold(traces[-2:],rows[i//2:i//2+1]))
                        # Explicit fallback means exactly the matched direct
                        # score, avoiding a padding-shape change in that score.
                        for c in (0,1):
                            if traces[-2+c]["fallback"]:pair[c] = direct[i//2][c]
                        thinking.append(pair)
                        preds.write(json.dumps({"ContextID":rows[i//2]["ContextID"],
                            "gold":[float(rows[i//2]["ModAvg"]),float(rows[i//2]["HeadAvg"])],
                            "direct":direct[i//2],"thinking":pair},ensure_ascii=False)+"\n");preds.flush()
                    status(stage="GENERATING_NATIVE_THINKING", completed_queries=i+1, total_queries=64,
                           fallback_queries=sum(r["fallback"] for r in traces))
            result = comparison(rows,direct,thinking,traces)
            dump(output/"metrics.json", result)
            done = {"status":"COMPLETED", "scope":plan["scope"], "optimizer_updates":0,
                    "elapsed_seconds":time.monotonic()-started,
                    "max_gpu_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
                    "max_gpu_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,
                    "fallback_queries":result["fallback_queries"],
                    "prediction_sha256":sha256(output/"predictions.jsonl"),
                    "thinking_trace_sha256":sha256(output/"thinking_traces.jsonl")}
            dump(output/"completion.json",done)
        status(stage="VERIFYING")
        subprocess.run([sys.executable,"scripts/verify_native_inference_diagnostic.py",str(output)],
                       cwd=ROOT,check=True,timeout=600)
        append_record(output.name+":COMPLETED", "32-row Internal Val inference-only diagnostic", result, cfg)
        status(stage="COMPLETED",completion=done)
    except Exception as error:
        status(stage="FAILED",error=repr(error))
        (output/"failure.txt").write_text(traceback.format_exc())
        append_record(output.name+":FAILED", "Inference diagnostic FAILED", json.loads((output/"status.json").read_text()),cfg)
        raise


if __name__ == "__main__":
    main()
