"""Qwen3-4B BF16 LoRA + two numeric regression heads, EN NN only."""

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from task5_data import sha256
from train_encoder import collator, load_rows, scores_for, write_json

GPU = "GPU-f3ed3c37-a52e-c126-79e9-e0a0783c584d"


class QwenRows(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.items, lengths = [], []
        for row in rows:
            messages = [
                {"role": "system", "content": "Assess noun compound compositionality in the given sentence. Rate how much the modifier and head retain their literal meanings. Scores range from 0 (non-compositional) to 5 (compositional). Represent both scores for regression."},
                {"role": "user", "content": f"Context: {row['Context']}\nCompound: {row['Compound']}\nModifier: {row['Mod']}\nHead: {row['Head']}"},
            ]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            assert len(ids) <= max_length, "prompt exceeds frozen length budget; no silent truncation"
            lengths.append(len(ids))
            self.items.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                               "labels": [float(row["ModAvg"]), float(row["HeadAvg"])]})
        self.token_stats = {"max_untruncated": max(lengths), "p95_untruncated": float(np.percentile(lengths, 95)), "truncated_rows": 0}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class QwenRegressor(nn.Module):
    def __init__(self, model_path, means=None, adapter=None):
        super().__init__()
        base = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True)
        base.config.use_cache = False
        if adapter is None:
            config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.05,
                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")
            self.backbone = get_peft_model(base, config)
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
        else:
            self.backbone = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(base.config.hidden_size, 1), nn.Linear(base.config.hidden_size, 1)])
        if means is not None:
            for head, mean in zip(self.heads, means):
                nn.init.normal_(head.weight, std=0.02)
                nn.init.constant_(head.bias, float(mean))

    def forward(self, **inputs):
        hidden = self.backbone(**inputs, return_dict=True).last_hidden_state
        positions = inputs["attention_mask"].sum(dim=1) - 1
        assert torch.all(positions >= 0)
        # Keep continuous scores in FP32; outer BF16 autocast would otherwise
        # quantize the two linear outputs and introduce avoidable rank ties.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            pooled = self.dropout(hidden[torch.arange(hidden.shape[0], device=hidden.device), positions].float())
            return torch.cat([head(pooled) for head in self.heads], dim=-1)


@torch.inference_mode()
def predict(model, loader):
    model.eval()
    values = []
    for encoded, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(**{k: v.to("cuda:0") for k, v in encoded.items()})
        assert torch.isfinite(prediction).all()
        values.extend(prediction.cpu().tolist())
    return values


def run(args):
    started = time.monotonic()
    torch.set_num_threads(4)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    assert not any(line.startswith(GPU) for line in busy.splitlines()), "GPU1 has an existing compute process"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    assert torch.cuda.mem_get_info()[0] > 18 * 1024 ** 3
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    proof = json.loads((args.model / "DOWNLOAD_VERIFIED.json").read_text())
    for name, item in proof["files"].items():
        assert sha256(args.model / name) == item["sha256"], name
    split = json.loads((args.data / "split.json").read_text())
    for name in ("train", "val"):
        assert sha256(args.data / (name + ".jsonl")) == split[name + "_sha256"]
    train_rows, val_rows = load_rows(args.data / "train.jsonl"), load_rows(args.data / "val.jsonl")
    if args.smoke:
        train_rows, val_rows = train_rows[:64], val_rows[:32]
    cfg = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "scope": "Smoke" if args.smoke else "EN NN Internal Val; no official Dev/Test",
           "used_rows": {"train": len(train_rows), "val": len(val_rows)}, "source_revision": proof["revision"],
           "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
           "train_sha256": split["train_sha256"], "val_sha256": split["val_sha256"],
           "selection": "max clipped mean Spearman; tie lower clipped mean RMSE",
           "pooling": "last non-pad token after full context, target fields and assistant prefix; right padding",
           "loss": "two-score mean MSE; no language model cross entropy",
           "lora": {"rank": 16, "alpha": 32, "dropout": 0.05, "modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
           "base_dtype": "BF16, no quantization", "regression_head_compute_dtype": "FP32 outside autocast",
           "gradient_checkpointing": "non-reentrant",
           "output": "linear raw; clip[0,5] for selection/export; both metrics saved"}
    write_json(args.output / "config.json", cfg)
    environment = {"python": platform.python_version(), "visible_devices": GPU, "gpu": torch.cuda.get_device_name(0)}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors", "numpy", "scipy"):
        environment[name] = importlib.metadata.version(name)
    write_json(args.output / "environment.json", environment)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data, val_data = QwenRows(train_rows, tokenizer, args.max_length), QwenRows(val_rows, tokenizer, args.max_length)
    write_json(args.output / "token_stats.json", {"train": train_data.token_stats, "val": val_data.token_stats})
    collect = collator(tokenizer)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), collate_fn=collect, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collect, num_workers=0)
    means = [sum(float(row[key]) for row in train_rows) / len(train_rows) for key in ("ModAvg", "HeadAvg")]
    model = QwenRegressor(args.model, means).to("cuda:0")
    base_frozen = [p for n, p in model.backbone.named_parameters() if "lora_" not in n]
    assert base_frozen and not any(p.requires_grad for p in base_frozen)
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert any("lora_" in name for name, p in trainable)
    write_json(args.output / "trainable_parameters.json", {"names": [name for name, p in trainable], "numel": sum(p.numel() for _, p in trainable)})
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
        {"params": model.heads.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    epochs = 1 if args.smoke else args.epochs
    updates = math.ceil(len(train_loader) / args.grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, round(updates * 0.06)), updates)
    history, global_step, best_key = [], 0, (-float("inf"), -float("inf"))
    checkpoint = args.output / "best"
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for i, (encoded, labels) in enumerate(train_loader):
            encoded, labels = {k: v.to("cuda:0") for k, v in encoded.items()}, labels.to("cuda:0")
            group_start = i // args.grad_accum * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(**encoded)
                loss = nn.functional.mse_loss(prediction, labels)
            assert torch.isfinite(loss)
            (loss / group_size).backward()
            total_loss += loss.item() * labels.shape[0]
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(train_loader):
                norm = nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                assert torch.isfinite(norm)
                if global_step == 0:
                    lora_grad = sum(float(p.grad.float().square().sum()) for n, p in trainable if "lora_" in n and p.grad is not None)
                    head_grad = sum(float(p.grad.float().square().sum()) for n, p in trainable if n.startswith("heads") and p.grad is not None)
                    assert lora_grad > 0 and head_grad > 0
                    write_json(args.output / "FIRST_BACKWARD_VERIFIED.json", {"lora_grad_squared": lora_grad, "head_grad_squared": head_grad, "base_parameters_frozen": True})
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step == 1 or global_step % 10 == 0:
                    state = {"stage": "TRAINING", "epoch": epoch, "global_step": global_step, "total_steps": updates, "loss": loss.item(), "gradient_norm": norm.item(), "elapsed_seconds": time.monotonic() - started}
                    write_json(args.output / "status.json", state)
                    print(json.dumps(state), flush=True)
        raw = predict(model, val_loader)
        scores, _ = scores_for(val_rows, raw)
        rho = scores["clipped"]["mean_spearman"]
        assert rho is not None
        entry = {"epoch": epoch, "step": global_step, "train_mse": total_loss / len(train_rows), "metrics": scores, "elapsed_seconds": time.monotonic() - started}
        history.append(entry)
        write_json(args.output / "history.json", history)
        print(json.dumps(entry), flush=True)
        key = (rho, -scores["clipped"]["mean_rmse"])
        if key > best_key:
            best_key = key
            checkpoint.mkdir(exist_ok=True)
            model.backbone.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            save_file({k: v.detach().cpu().contiguous() for k, v in model.heads.state_dict().items()}, str(checkpoint / "regression_heads.safetensors"))
            write_json(args.output / "best_selection.json", entry)
            write_json(args.output / "best_reference_predictions.json", raw)
    del model, optimizer, scheduler, trainable, base_frozen, prediction, loss, norm, encoded, labels
    gc.collect()
    torch.cuda.empty_cache()
    model = QwenRegressor(args.model, adapter=checkpoint).to("cuda:0")
    model.heads.load_state_dict(load_file(str(checkpoint / "regression_heads.safetensors")))
    raw = predict(model, val_loader)
    reference = json.loads((args.output / "best_reference_predictions.json").read_text())
    reload_diff = max(abs(x - y) for a, b in zip(raw, reference) for x, y in zip(a, b))
    assert reload_diff < 1e-5, "fresh adapter reload prediction mismatch"
    repeat = predict(model, val_loader)
    repeat_diff = max(abs(x - y) for a, b in zip(raw, repeat) for x, y in zip(a, b))
    assert repeat_diff < 1e-5
    scores, clipped = scores_for(val_rows, raw)
    selected = json.loads((args.output / "best_selection.json").read_text())
    assert abs(scores["clipped"]["mean_spearman"] - selected["metrics"]["clipped"]["mean_spearman"]) < 1e-12
    with (args.output / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, original, final in zip(val_rows, raw, clipped):
            handle.write(json.dumps({"ContextID": row["ContextID"], "Compound": row["Compound"], "gold": [float(row["ModAvg"]), float(row["HeadAvg"])], "raw": original, "clipped": final}, ensure_ascii=False) + "\n")
    with (args.output / "en-nn-internal-val-pred.tsv").open("w", encoding="utf-8") as handle:
        for row, final in zip(val_rows, clipped):
            handle.write(f"{row['ContextID']}\t{final[0]:.8f}\t{final[1]:.8f}\n")
    write_json(args.output / "metrics.json", scores)
    write_json(args.output / "completion.json", {"status": "COMPLETED", "scope": cfg["scope"], "steps": global_step,
        "best_epoch": selected["epoch"], "elapsed_seconds": time.monotonic() - started,
        "max_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "max_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
        "reload_max_abs_diff": reload_diff, "repeat_max_abs_diff": repeat_diff,
        "checkpoint_sha256": {p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()},
        "prediction_sha256": sha256(args.output / "val_predictions.jsonl")})
    write_json(args.output / "status.json", {"stage": "COMPLETED", "best_epoch": selected["epoch"], "metrics": scores})
    print("COMPLETED " + str(args.output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2027)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    write_json(arguments.output / "status.json", {"stage": "STARTING"})
    try:
        run(arguments)
    except Exception as error:
        write_json(arguments.output / "status.json", {"stage": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        (arguments.output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
