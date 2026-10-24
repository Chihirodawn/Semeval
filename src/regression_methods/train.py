"""Fixed Task5 experiments: RAIL, soft-CE, RAFT, two-stage TRACT, REAL.

Adaptations are recorded explicitly: two scalar component queries, grid0..5,
local teacher instead of GPT4, bounded CoTs, LoRA, and a one-epoch REAL pilot.
"""
import argparse
import collections
import gc
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from train_encoder import load_rows, scores_for, write_json
from task5_data import sha256
from regression_methods.core import NumericPolicy, examples, key, real_loss, CUE

ROOT = Path(__file__).resolve().parents[2]
GPU = "GPU-f3ed3c37-a52e-c126-79e9-e0a0783c584d"


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def guard(args):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    assert not any(x.startswith(GPU) for x in apps.splitlines()), "GPU1 occupied"
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    assert torch.cuda.mem_get_info()[0] > 18 * 1024**3
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    proof = json.loads((args.model / "DOWNLOAD_VERIFIED.json").read_text())
    assert proof["revision"] == "cdbee75f17c01a7cc42f958dc650907174af0554"
    for name, info in proof["files"].items():
        assert sha256(args.model / name) == info["sha256"], name
    split = json.loads((args.data / "split.json").read_text())
    for name in ("train", "val"):
        assert sha256(args.data / (name + ".jsonl")) == split[name + "_sha256"]
    return proof, split


def load_annotations(path, train_examples):
    rows = load_rows(path / "reasoning.jsonl")
    allowed = {key(r): r for r in train_examples}
    selected = {key(r): r for r in rows if key(r) in allowed}
    assert len(selected) == len(allowed), "every train component needs one trace"
    for ident, original in allowed.items():
        annotation = selected[ident]
        assert original["prompt_ids"] == annotation["prompt_ids"]
        assert original["gold"] == annotation["gold"], "never substitute generated scores for gold"
    return [dict(e, cot_ids=selected[key(e)]["cot_ids"]) for e in train_examples]


@torch.inference_mode()
def evaluate(model, records, tokenizer, size, reasoning, max_tokens, fixed=None, seed=2027):
    model.eval()
    used, predictions, modes, masses, full_digits = [], [], [], [], []
    # Validation sampling must not change subsequent training's RNG stream.
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        for group in chunks(fixed if fixed is not None else records, size):
            if reasoning and fixed is None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    group = model.generate_reasoning(group, tokenizer, max_tokens, sample=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.batch(group, tokenizer, reasoning=reasoning)
            predictions += output["prediction"].tolist()
            modes += output["mode"].tolist()
            masses += output["digit_mass"].tolist()
            full_digits += output["full_top1_is_digit"].tolist()
            used += group
    assert all(math.isfinite(x) and 0 <= x <= 5 + 1e-6 for x in predictions)
    pairs = [predictions[i:i+2] for i in range(0, len(predictions), 2)]
    mode_pairs = [modes[i:i+2] for i in range(0, len(modes), 2)]
    return pairs, used, {"mean_numeric_mass": float(np.mean(masses)), "min_numeric_mass": min(masses),
                         "full_vocab_top1_digit_fraction": float(np.mean(full_digits)), "digit_constrained_mode_pairs": mode_pairs}


def save_trace(path, records):
    with path.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def run(args):
    started = time.monotonic()
    proof, split = guard(args)
    train_rows = load_rows(args.data / "train.jsonl")
    val_rows = [] if args.mode == "generate" else load_rows(args.data / "val.jsonl")
    if args.smoke:
        train_rows, val_rows = train_rows[:64], val_rows[:32]
    reasoning = args.mode in ("generate", "tract", "real")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    digit_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(6)]
    assert all(len(x) == 1 for x in digit_ids) and len({x[0] for x in digit_ids}) == 6
    digit_ids = [x[0] for x in digit_ids]
    train_data, val_data = examples(train_rows, tokenizer, reasoning), examples(val_rows, tokenizer, reasoning)
    epochs = 0 if args.mode in ("generate", "rail") else 1 if args.smoke else args.epochs
    source_files = [Path(__file__), Path(__file__).with_name("core.py"), ROOT / "src/task5_data.py", ROOT / "src/train_encoder.py"]
    cfg = {**{k: str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items()},
           "scope": "Train annotations only" if args.mode == "generate" else "Smoke" if args.smoke else "EN NN Internal Val; no official Dev/Test",
           "used_rows": {"train": len(train_rows), "val": 0 if args.mode == "generate" else len(val_rows)},
           "scalar_train_queries": len(train_data), "effective_scalar_batch": args.batch_size * args.grad_accum,
           "source_revision": proof["revision"], "train_sha256": split["train_sha256"], "val_sha256": split["val_sha256"],
           "source_sha256": {str(p.relative_to(ROOT / "src")): sha256(p) for p in source_files},
           "grid": list(range(6)), "digit_token_ids": digit_ids, "expectation": "sum y*p_full_vocab(y), no digit-only renormalization",
           "projection": "frozen pretrained tied LM projection, not an added regression head",
           "loss": {"rail":"none", "ce":"adjacent-bin soft CE preserving continuous label mean", "raft":"MSE on numeric expectation",
                    "tract":"MSE + token-mean CE on annotated reasoning and fixed score delimiter; numeric score masked from CE",
                    "real":"on-policy K2 RLOO trajectory sum-logprob + beta1 prediction MSE; lambda0", "generate":"none"}[args.mode],
           "selection": "max clipped mean Spearman; tie lower mean component RMSE", "actual_epochs": epochs,
           "adaptations": ["independent modifier/head scalar prompts", "continuous labels, grid0..5", "BF16 LoRA attention r16 alpha32",
                           "local Qwen teacher instead of GPT4", "bounded64-token CoTs", "REAL one-epoch pilot, no auxiliary score CE"],
           "initial_adapter_sha256": {p.name:sha256(p) for p in args.init_adapter.glob('*') if p.is_file()} if args.init_adapter else None,
           "annotation_sha256": sha256(args.reasoning_data / "reasoning.jsonl") if args.reasoning_data else None,
           "reload_scope": "same stored sampled validation CoTs, if applicable; stochastic resampling is not claimed identical"}
    write_json(args.output / "config.json", cfg)
    write_json(args.output / "environment.json", {x:importlib.metadata.version(x) for x in ("torch","transformers","peft","numpy","scipy")})
    model = NumericPolicy(args.model, digit_ids, args.init_adapter, trainable=epochs > 0).to("cuda:0")
    if args.mode == "real":
        model.disable_dropout()
    trainables = [p for p in model.parameters() if p.requires_grad]
    assert all(not p.requires_grad for n,p in model.backbone.named_parameters() if "lora_" not in n)
    write_json(args.output / "trainable_parameters.json", {"numel":sum(p.numel() for p in trainables), "base_frozen": True})
    if args.mode == "generate":
        generated = []
        for group in chunks(train_data, args.batch_size):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                generated += model.generate_reasoning(group, tokenizer, args.max_cot_tokens, sample=True)
            # Persist partial progress for interruption diagnosis, never use as a complete cache.
            write_json(args.output / "status.json", {"stage":"GENERATING", "completed":len(generated),"total":len(train_data),"elapsed_seconds":time.monotonic()-started})
        assert len({key(r) for r in generated}) == len(train_data)
        save_trace(args.output / "reasoning.jsonl", generated)
        write_json(args.output / "completion.json", {"status":"COMPLETED", "scope":cfg["scope"], "generated_records":len(generated),
                   "stop_reasons":dict(collections.Counter(r["stop_reason"] for r in generated)), "gold_never_in_prompt":True,
                   "trace_sha256":sha256(args.output / "reasoning.jsonl"), "elapsed_seconds":time.monotonic()-started,
                   "max_gpu_allocated_gib":torch.cuda.max_memory_allocated()/1024**3})
        write_json(args.output / "status.json", {"stage":"COMPLETED"})
        return
    if args.mode == "tract":
        assert args.reasoning_data
        train_data = load_annotations(args.reasoning_data, train_data)
    # Remaining optimization and export share exactly the same scoring/selection path.
    optimize_and_export(args, cfg, model, tokenizer, digit_ids, train_data, val_data, val_rows, trainables, epochs, reasoning, started)


def train_epochs(args, model, tokenizer, train_data, val_data, val_rows, trainables, epochs, reasoning, started):
    optimizer = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=0.01) if epochs else None
    total_steps = math.ceil(math.ceil(len(train_data) / args.batch_size) / args.grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, round(total_steps * 0.06)), total_steps) if epochs else None
    history, step, best_key = [], 0, None
    for epoch in range(1, epochs + 1) if epochs else [0]:
        total_loss = total_mse = 0.0
        if epochs:
            order = list(train_data)
            random.Random(args.seed + epoch).shuffle(order)
            batches = list(chunks(order, args.batch_size))
            model.train(); optimizer.zero_grad(set_to_none=True)
            for i, group in enumerate(batches):
                if args.mode == "real":
                    expanded = [dict(r) for r in group for _ in range(2)]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        group_for_model = model.generate_reasoning(expanded, tokenizer, args.max_cot_tokens, sample=True)
                    model.train()
                else:
                    group_for_model = group
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model.batch(group_for_model, tokenizer, reasoning=reasoning,
                                         cot_ce=args.mode == "tract", policy=args.mode == "real")
                    mse = (result["prediction"] - result["gold"]).square().mean()
                    if args.mode == "ce":
                        loss = result["score_ce"].mean()
                    elif args.mode == "tract":
                        loss = mse + result["cot_ce"].mean()
                    elif args.mode == "real":
                        loss, _ = real_loss(result["prediction"], result["gold"], result["trajectory_logprob"])
                    else:
                        loss = mse
                assert torch.isfinite(loss)
                group_start = i // args.grad_accum * args.grad_accum
                divisor = min(args.grad_accum, len(batches) - group_start)
                (loss / divisor).backward()
                total_loss += float(loss.detach()) * len(group)
                total_mse += float(mse.detach()) * len(group)
                if (i + 1) % args.grad_accum == 0 or i + 1 == len(batches):
                    norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0)
                    assert torch.isfinite(norm)
                    if step == 0:
                        energy = sum(float(p.grad.float().square().sum()) for p in trainables if p.grad is not None)
                        assert energy > 0
                        write_json(args.output / "FIRST_BACKWARD_VERIFIED.json", {"lora_gradient_squared":energy,
                                   "base_frozen":True,"objective":args.mode,"no_added_regression_head":True})
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if step == 1 or step % 10 == 0:
                        state = {"stage":"TRAINING","epoch":epoch,"global_step":step,"total_steps":total_steps,
                                 "loss":float(loss.detach()),"gradient_norm":float(norm),"elapsed_seconds":time.monotonic()-started}
                        write_json(args.output / "status.json", state); print(json.dumps(state), flush=True)
        raw, used, diagnostics = evaluate(model, val_data, tokenizer, args.batch_size, reasoning, args.max_cot_tokens, seed=args.seed)
        scores, _ = scores_for(val_rows, raw)
        mode_scores, _ = scores_for(val_rows, diagnostics.pop("digit_constrained_mode_pairs"))
        entry = {"epoch":epoch,"step":step,"train_loss":total_loss / len(train_data),"train_mse":total_mse / len(train_data),
                 "metrics":scores,"diagnostics":diagnostics,"digit_constrained_mode_metrics":mode_scores,
                 "elapsed_seconds":time.monotonic()-started}
        history.append(entry); write_json(args.output / "history.json",history)
        print(json.dumps(entry),flush=True)
        rho = scores["clipped"]["mean_spearman"]
        current_key = (rho if rho is not None else -float("inf"), -scores["clipped"]["mean_rmse"])
        if best_key is None or current_key > best_key:
            best_key = current_key
            best = args.output / "best"; best.mkdir(exist_ok=True)
            model.save(best); tokenizer.save_pretrained(best)
            write_json(best / "method_state.json", {"method":args.mode,"grid":list(range(6)),"reasoning":reasoning,"digit_ids":model.digit_ids})
            write_json(args.output / "best_selection.json",entry)
            write_json(args.output / "best_reference_predictions.json",raw)
            save_trace(args.output / "best_validation_inputs.jsonl",used)
    return step


def optimize_and_export(args, cfg, model, tokenizer, digit_ids, train_data, val_data, val_rows, trainables, epochs, reasoning, started):
    steps = train_epochs(args, model, tokenizer, train_data, val_data, val_rows, trainables, epochs, reasoning, started)
    # Release optimizer/graphs by leaving train_epochs. Move the old model off
    # GPU even though the caller still holds its Python reference.
    model.to("cpu"); gc.collect(); torch.cuda.empty_cache()
    reloaded = NumericPolicy(args.model, digit_ids, args.output / "best", trainable=False).to("cuda:0")
    fixed = load_rows(args.output / "best_validation_inputs.jsonl")
    raw, _, diagnostics = evaluate(reloaded, val_data, tokenizer, args.batch_size, reasoning, args.max_cot_tokens, fixed=fixed)
    reference = json.loads((args.output / "best_reference_predictions.json").read_text())
    reload_diff = max(abs(x-y) for a,b in zip(raw,reference) for x,y in zip(a,b))
    assert reload_diff < 1e-6, ("fresh numeric adapter reload",reload_diff)
    repeated, _, _ = evaluate(reloaded, val_data, tokenizer, args.batch_size, reasoning, args.max_cot_tokens, fixed=fixed)
    repeat_diff = max(abs(x-y) for a,b in zip(raw,repeated) for x,y in zip(a,b))
    assert repeat_diff < 1e-6
    scores, clipped = scores_for(val_rows,raw)
    selected = json.loads((args.output / "best_selection.json").read_text())
    assert scores["clipped"]["mean_spearman"] == selected["metrics"]["clipped"]["mean_spearman"]
    with (args.output / "val_predictions.jsonl").open("w") as handle:
        for row, prediction, final in zip(val_rows,raw,clipped):
            handle.write(json.dumps({"ContextID":row["ContextID"],"Compound":row["Compound"],
                         "gold":[float(row["ModAvg"]),float(row["HeadAvg"])],"raw":prediction,"clipped":final})+"\n")
    with (args.output / "en-nn-internal-val-pred.tsv").open("w") as handle:
        for row, values in zip(val_rows,clipped):handle.write(f"{row['ContextID']}\t{values[0]:.8f}\t{values[1]:.8f}\n")
    write_json(args.output / "metrics.json",scores)
    diagnostics.pop("digit_constrained_mode_pairs")
    write_json(args.output / "numeric_diagnostics.json",diagnostics)
    completion = {"status":"COMPLETED","scope":cfg["scope"],"steps":steps,"best_epoch":selected["epoch"],
                  "elapsed_seconds":time.monotonic()-started,"reload_max_abs_diff":reload_diff,"repeat_max_abs_diff":repeat_diff,
                  "reload_scope":cfg["reload_scope"],"max_gpu_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
                  "max_gpu_reserved_gib":torch.cuda.max_memory_reserved()/1024**3,
                  "checkpoint_sha256":{p.name:sha256(p) for p in (args.output / "best").iterdir() if p.is_file()},
                  "prediction_sha256":sha256(args.output / "val_predictions.jsonl"),
                  "validation_trace_sha256":sha256(args.output / "best_validation_inputs.jsonl")}
    write_json(args.output / "completion.json",completion)
    write_json(args.output / "status.json",{"stage":"COMPLETED","metrics":scores,"best_epoch":selected["epoch"]})
    print(json.dumps(completion),flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("model","data","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--mode",choices=["rail","ce","raft","generate","tract","real"],required=True)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--reasoning-data",type=Path)
    parser.add_argument("--init-adapter",type=Path)
    for name,value in (("epochs",3),("batch-size",8),("grad-accum",2),("seed",2027),("max-cot-tokens",64)):
        parser.add_argument("--"+name,type=int,default=value)
    parser.add_argument("--lr",type=float,default=5e-5)
    args = parser.parse_args()
    args.model,args.data,args.output = args.model.resolve(),args.data.resolve(),args.output.resolve()
    assert args.mode != "real" or args.init_adapter, "REAL starts from a trained TRACT adapter"
    assert 1 <= args.max_cot_tokens <= 64
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output / "status.json",{"stage":"STARTING"})
    try:
        run(args)
    except Exception as error:
        write_json(args.output / "status.json",{"stage":"FAILED","error_type":type(error).__name__,"error":str(error)})
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
