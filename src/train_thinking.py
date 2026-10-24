"""Independent staged native-thinking worker: pilot/cache/train/resample."""
import argparse
import collections
import gc
import importlib.metadata
import json
import math
import os
import random
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ce_model_compare_v3 import REVISIONS
from regression_methods.train import save_trace
from task5_data import sha256
from train_encoder import load_rows, scores_for, write_json
from thinking_protocol import (MAX_SEQUENCE, SAMPLING, attach_gold, check_trace, fallback_rate,
                               identity, inputs, pairing)
from thinking_model import ThinkingPolicy, objective

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ["src/train_thinking.py", "src/thinking_model.py", "src/thinking_protocol.py",
          "src/ce_model_compare_v3.py", "src/regression_methods/core.py",
          "src/regression_methods/train.py", "src/train_encoder.py", "src/task5_data.py"]


def configure(args):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == args.gpu
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader,nounits"], text=True)
    assert not any(line.startswith(args.gpu) for line in apps.splitlines()), "selected GPU occupied"
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    assert torch.cuda.mem_get_info()[0] > 21 * 1024**3
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    proof = json.loads((args.model / "DOWNLOAD_VERIFIED.json").read_text())
    assert REVISIONS[proof["repo"]] == proof["revision"]
    for name, entry in proof["files"].items():
        assert sha256(args.model / name) == entry["sha256"], name
    split = json.loads((args.data / "split.json").read_text())
    for part in ("train", "val"):
        assert sha256(args.data / (part+".jsonl")) == split[part+"_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    digit_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(6)]
    assert all(len(x) == 1 for x in digit_ids) and len({x[0] for x in digit_ids}) == 6
    cfg = {**{k:str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items()},
           "scope": "Train-only length pilot/cache" if args.action in ("pilot", "cache") else
                    "Smoke" if args.smoke else "EN NN Internal Val; no official Dev/Test",
           "repo":proof["repo"], "source_revision":proof["revision"],
           "source_sha256":{p[4:]:sha256(ROOT / p) for p in SOURCE},
           "train_sha256":split["train_sha256"], "val_sha256":split["val_sha256"],
           "smoke":args.smoke, "batch_size":2, "grad_accum":8,
           "effective_scalar_batch":16, "lr":5e-5, "epochs":1 if args.smoke else 3,
           "cot_ce_weight":0.0 if args.style == "direct" else 0.1,
           "rank_weight":args.rank_weight, "huber_weight":args.huber_weight,
           "huber_beta":1.0,"pair_tie_threshold":0.25,"max_sequence_tokens":MAX_SEQUENCE,
           "sampling":SAMPLING,"enable_thinking":args.style != "direct",
           "expectation":"sum y*p_full_vocab(y), no digit-only renormalization",
           "reload_scope":"same saved sampled thinking trajectories; fresh resampling checked separately",
           "selection":"max clipped mean Spearman; tie lower mean component RMSE",
           "adaptations":["TRACT-inspired CoT CE plus soft score CE, not full TRACT",
                          "local frozen 4B Train teacher, no gold input",
                          "new native-thinking generation and strict admission; no legacy64-token decoder",
                          "native LM head frozen, LoRA rank16 alpha32 dropout0.05",
                          "new shared double-newline score cue and micro2 paired ordering for all new arms"],
           "runtime":{x:importlib.metadata.version(x) for x in ("torch","transformers","peft","numpy")}}
    write_json(args.output / "config.json",cfg)
    return tokenizer, [x[0] for x in digit_ids], cfg


def status(args, **fields):
    write_json(args.output / "status.json", fields)
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def summary(traces):
    attempts = [t for r in traces for t in r.get("attempts", [])]
    return {"queries":len(traces), "fallback_count":sum(r.get("fallback",False) for r in traces),
            "fallback_fraction":fallback_rate(traces),
            "first_attempt_valid_count":sum(bool(r.get("attempts")) and r["attempts"][0]["error"] is None for r in traces),
            "retried_queries":sum(len(r.get("attempts",[]))>1 for r in traces),
            "stops":dict(collections.Counter(t["stop_reason"] for t in attempts)),
            "errors":dict(collections.Counter(t["error"] or "valid" for t in attempts)),
            "tokens_generated_including_retries":sum(len(t["cot_ids"]) for t in attempts),
            "generation_seconds":sum(t["generation_seconds"] for t in attempts)}


def run_generation(args, tokenizer, model):
    assert args.style != "direct"
    # No gold is returned by inputs or present in records passed to the model.
    source = load_rows(args.data / "train.jsonl")
    if args.action == "pilot":
        source = random.Random(2027).sample(source,64)
    elif args.smoke:
        source = source[:64]
    data, direct = inputs(source,tokenizer,args.style), inputs(source,tokenizer,"direct")
    direct = {identity(r):r for r in direct}
    budgets = (512,1024,2048) if args.action == "pilot" else (args.budget,)
    results=[]
    for budget in budgets:
        traces=[]
        path=args.output / f"traces_{budget}.jsonl"
        with path.open("x") as stream:
            for i,row in enumerate(data):
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    trace=model.reason(row,direct[identity(row)],tokenizer,budget,2027,retry=True)
                assert "gold" not in trace and "ModAvg" not in trace and "HeadAvg" not in trace
                check_trace(trace,tokenizer)
                traces.append(trace)
                stream.write(json.dumps(trace,ensure_ascii=False)+"\n");stream.flush()
                if i % 8 == 0:
                    status(args,stage="LENGTH_PILOT" if args.action=="pilot" else "GENERATING_CACHE",
                           budget=budget, completed=i+1,total=len(data),diagnostics=summary(traces))
                # Two failures among128 already make99% completion impossible.
                # Reject only a failed budget early; a passing budget MUST run
                # the entire fixed pilot, never report a prefix as128 samples.
                if args.action=="pilot" and sum(r.get("fallback",False) for r in traces) > .01*len(data):
                    break
        result={"budget":budget,**summary(traces),"requested_queries":len(data),
                "early_rejection":len(traces)<len(data),"trace_sha256":sha256(path)}
        results.append(result)
        write_json(args.output / "generation_results.json",results)
        if args.action=="cache":
            assert result["fallback_fraction"]<=0.01, "cache admission failed: over1% fallback"
            save_trace(args.output / "reasoning.jsonl",traces)
            return {"status":"COMPLETED","scope":"Train-only teacher cache; no gold input",
                    "budget":budget,"diagnostics":result,"trace_sha256":sha256(args.output / "reasoning.jsonl")}
        if len(traces)==len(data) and result["fallback_fraction"] <=0.01:
            return {"status":"COMPLETED","scope":"Train-only length pilot, no quality claim",
                    "selected_budget":budget,"results":results}
    raise RuntimeError("No pilot budget reached99% valid completion; stop full training")


def eval_records(args, model, tokenizer, bare, direct, rows, budget, seed, trace_file=None, fixed=None):
    model.eval()
    records=[]
    if fixed is None:
        handle = trace_file.open("x") if trace_file else None
        try:
            for i,row in enumerate(bare):
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    trace=model.reason(row,direct[identity(row)],tokenizer,budget,seed)
                records.append(trace)
                if handle:
                    handle.write(json.dumps(trace,ensure_ascii=False)+"\n");handle.flush()
                if i % 32==0:
                    status(args,stage="VALIDATING_THINKING",completed=i+1,total=len(bare),generation_seed=seed)
        finally:
            if handle:handle.close()
    else:
        records=fixed
    assert len(records)==len(bare) and [identity(r) for r in records]==[identity(r) for r in bare]
    assert fallback_rate(records)<=0.01, "validation admission failed: over1% fallback"
    labeled=attach_gold(records,rows)
    predictions,masses=[],[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for i in range(0,len(labeled),2):
            output=model.batch(labeled[i:i+2],tokenizer,supervise=False)
            predictions.extend(output["prediction"].tolist())
            masses.extend(output["digit_mass"].tolist())
    assert all(math.isfinite(x) and 0 <= x <= 5.000001 for x in predictions)
    raw=[predictions[i:i+2] for i in range(0,len(predictions),2)]
    metrics, clipped=scores_for(rows,raw)
    return raw, clipped, metrics, records, {**summary(records),"mean_numeric_mass":sum(masses)/len(masses)}


def verify_cache(path, records, tokenizer, args):
    cfg=json.loads((path/"config.json").read_text())
    done=json.loads((path/"completion.json").read_text())
    assert done["status"]=="COMPLETED" and cfg["action"]=="cache"
    assert cfg["repo"]=="Qwen/Qwen3.5-4B" and cfg["style"]==args.style
    assert cfg["train_sha256"]==sha256(args.data/"train.jsonl")
    assert done["budget"]==args.budget
    assert sha256(path/"reasoning.jsonl")==done["trace_sha256"]
    for name,digest in cfg["source_sha256"].items():
        assert sha256(ROOT/"src"/name)==digest,"teacher cache code changed"
    rows=load_rows(path/"reasoning.jsonl")
    annotations={identity(r):r for r in rows}
    assert len(annotations)==len(rows)
    result=[]
    for row in records:
        trace=annotations[identity(row)]
        assert "gold" not in trace and "ModAvg" not in trace and "HeadAvg" not in trace
        if not trace.get("fallback"):
            assert row["prompt_ids"]==trace["prompt_ids"]
        check_trace(trace,tokenizer)
        result.append(trace)
    assert fallback_rate(result)<=0.01
    return result


def train(args, tokenizer, digits, cfg, model):
    train_rows=load_rows(args.data/"train.jsonl")
    val_rows=load_rows(args.data/"val.jsonl")
    if args.smoke:train_rows,val_rows=train_rows[:64],val_rows[:32]
    bare=inputs(train_rows,tokenizer,args.style)
    if args.style=="direct":
        direct={identity(r):r for r in bare}
        records=[model.reason(r,direct[identity(r)],tokenizer,0,2027) for r in bare]
    else:
        assert args.cache
        records=verify_cache(args.cache,bare,tokenizer,args)
        cfg["annotation_sha256"]=sha256(args.cache/"reasoning.jsonl")
    data=attach_gold(records,train_rows)
    val=inputs(val_rows,tokenizer,args.style)
    direct={identity(r):r for r in inputs(val_rows,tokenizer,"direct")}
    cfg.update(used_rows={"train":len(train_rows),"val":len(val_rows)},scalar_train_queries=len(data))
    write_json(args.output/"config.json",cfg)
    params=[p for p in model.parameters() if p.requires_grad]
    assert params and not model.lm_head.weight.requires_grad
    assert all(not p.requires_grad for n,p in model.backbone.named_parameters() if "lora_" not in n)
    optimizer=torch.optim.AdamW(params,lr=5e-5,weight_decay=0.01)
    epochs=cfg["epochs"]
    planned=[pairing(data,args.seed,e) for e in range(1,epochs+1)]
    total_steps=sum(math.ceil(len(batches)/8) for batches,_ in planned)
    scheduler=get_linear_schedule_with_warmup(optimizer,max(1,round(total_steps*.06)),total_steps)
    write_json(args.output/"pair_schedules.json",[stats for _,stats in planned])
    step,history,best=0,[],None
    for epoch,(batches,pair_stats) in enumerate(planned,1):
        epoch_started=time.monotonic()
        model.train();optimizer.zero_grad(set_to_none=True)
        totals=collections.Counter()
        for i,group in enumerate(batches):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                output=model.batch(group,tokenizer,supervise=args.style!="direct")
                loss,parts=objective(output,group,args.rank_weight,args.huber_weight)
            assert torch.isfinite(loss)
            first=i//8*8;divisor=min(8,len(batches)-first)
            (loss/divisor).backward()
            for name,value in parts.items():totals[name]+=float(value)*len(group)
            if (i+1)%8==0 or i+1==len(batches):
                norm=torch.nn.utils.clip_grad_norm_(params,1.0)
                assert torch.isfinite(norm)
                if step==0:
                    energy=sum(float(p.grad.float().square().sum()) for p in params if p.grad is not None)
                    assert energy>0
                    write_json(args.output/"FIRST_BACKWARD_VERIFIED.json",
                               {"lora_gradient_squared":energy,"native_head_frozen":True,
                                "cot_target_count":sum(output["cot_target_count"]),"parts":{k:float(v) for k,v in parts.items()}})
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                if step==1 or step%10==0:
                    status(args,stage="TRAINING",epoch=epoch,global_step=step,total_steps=total_steps,
                           loss=float(loss.detach()),parts={k:float(v) for k,v in parts.items()})
        train_seconds=time.monotonic()-epoch_started
        validation_started=time.monotonic()
        raw,_,metrics,traces,diagnostics=eval_records(
            args,model,tokenizer,val,direct,val_rows,args.budget,2027,
            args.output/f"epoch{epoch}_validation_thinking.jsonl")
        entry={"epoch":epoch,"step":step,"metrics":metrics,"diagnostics":diagnostics,
               "pairing":pair_stats,"train_seconds":train_seconds,
               "validation_seconds":time.monotonic()-validation_started,
               "train_objectives":{k:v/len(data) for k,v in totals.items()}}
        history.append(entry);write_json(args.output/"history.json",history)
        current=(metrics["clipped"]["mean_spearman"],-metrics["clipped"]["mean_rmse"])
        if best is None or current>best:
            best=current
            folder=args.output/"best";folder.mkdir(exist_ok=True)
            model.save(folder);tokenizer.save_pretrained(folder)
            write_json(folder/"method_state.json",{"method":"native_thinking","style":args.style,
                       "reasoning":args.style!="direct","budget":args.budget,"digit_ids":digits})
            write_json(args.output/"best_selection.json",entry)
            write_json(args.output/"best_reference_predictions.json",raw)
            save_trace(args.output/"best_validation_inputs.jsonl",traces)
    del optimizer,scheduler,params,output,loss,parts
    model.to("cpu");gc.collect();torch.cuda.empty_cache()
    model=ThinkingPolicy(args.model,digits,args.output/"best",trainable=False).to("cuda:0")
    fixed=load_rows(args.output/"best_validation_inputs.jsonl")
    raw,clipped,metrics,_,diagnostics=eval_records(args,model,tokenizer,val,direct,val_rows,args.budget,2027,fixed=fixed)
    reference=json.loads((args.output/"best_reference_predictions.json").read_text())
    diff=max(abs(x-y) for a,b in zip(raw,reference) for x,y in zip(a,b))
    repeated,*_=eval_records(args,model,tokenizer,val,direct,val_rows,args.budget,2027,fixed=fixed)
    repeat=max(abs(x-y) for a,b in zip(raw,repeated) for x,y in zip(a,b))
    assert diff<1e-6 and repeat<1e-6,(diff,repeat)
    resamples=[]
    resampling_started=time.monotonic()
    for seed in (2027,2028,2029):
        rs_raw,_,rs_metrics,_,rs_diag=eval_records(args,model,tokenizer,val,direct,val_rows,args.budget,seed,
                                                args.output/f"best_resample_{seed}.jsonl")
        if seed==2027:
            regenerated=max(abs(x-y) for a,b in zip(raw,rs_raw) for x,y in zip(a,b))
            assert regenerated<1e-6,("fresh generation is not reproducible",regenerated)
        resamples.append({"generation_seed":seed,"metrics":rs_metrics,"diagnostics":rs_diag})
    write_json(args.output/"resampling.json",{"runs":resamples,
               "mean_spearman":float(np.mean([r["metrics"]["clipped"]["mean_spearman"] for r in resamples])),
               "std_spearman":float(np.std([r["metrics"]["clipped"]["mean_spearman"] for r in resamples]))})
    with (args.output/"val_predictions.jsonl").open("w") as handle:
        for row,p,c in zip(val_rows,raw,clipped):
            handle.write(json.dumps({"ContextID":row["ContextID"],"Compound":row["Compound"],
                         "gold":[float(row["ModAvg"]),float(row["HeadAvg"])],"raw":p,"clipped":c})+"\n")
    with (args.output/"en-nn-internal-val-pred.tsv").open("w") as handle:
        for row,p in zip(val_rows,clipped):handle.write(f"{row['ContextID']}\t{p[0]:.8f}\t{p[1]:.8f}\n")
    write_json(args.output/"metrics.json",metrics)
    write_json(args.output/"numeric_diagnostics.json",diagnostics)
    selected=json.loads((args.output/"best_selection.json").read_text())
    assert metrics["clipped"]["mean_spearman"]==selected["metrics"]["clipped"]["mean_spearman"]
    return {"status":"COMPLETED","steps":step,"best_epoch":selected["epoch"],
            "reload_max_abs_diff":diff,"repeat_max_abs_diff":repeat,"regenerated_max_abs_diff":regenerated,
            "timing":{"train_seconds":sum(r["train_seconds"] for r in history),
                      "validation_seconds":sum(r["validation_seconds"] for r in history),
                      "resampling_and_export_seconds":time.monotonic()-resampling_started},
            "checkpoint_sha256":{p.name:sha256(p) for p in (args.output/"best").iterdir() if p.is_file()},
            "prediction_sha256":sha256(args.output/"val_predictions.jsonl"),
            "validation_trace_sha256":sha256(args.output/"best_validation_inputs.jsonl")}


def main():
    parser=argparse.ArgumentParser()
    for name in ("model","data","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--gpu",required=True)
    parser.add_argument("--action",choices=("pilot","cache","train"),required=True)
    parser.add_argument("--style",choices=("direct","free","evidence"),required=True)
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--budget",type=int,choices=(512,1024,2048),default=512)
    parser.add_argument("--seed",type=int,default=2027)
    parser.add_argument("--rank-weight",type=float,choices=(0.0,0.2),default=0.0)
    parser.add_argument("--huber-weight",type=float,choices=(0.0,0.5),default=0.0)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    status(args,stage="STARTING")
    try:
        tokenizer,digits,cfg=configure(args)
        model=ThinkingPolicy(args.model,digits,trainable=args.action=="train",native_check=True).to("cuda:0")
        if args.action=="train":result=train(args,tokenizer,digits,cfg,model)
        else:result=run_generation(args,tokenizer,model)
        result.update(scope=cfg["scope"],elapsed_seconds=time.monotonic()-started,
                      max_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                      max_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        write_json(args.output/"completion.json",result)
        status(args,stage="COMPLETED",completion=result)
    except Exception as error:
        resource={"elapsed_seconds":time.monotonic()-started}
        if torch.cuda.is_initialized():
            resource.update(max_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                            max_gpu_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        status(args,stage="FAILED",error=repr(error),**resource)
        (args.output/"failure.txt").write_text(traceback.format_exc())
        raise


if __name__=="__main__":main()
