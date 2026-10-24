"""First EN NN encoder baseline. Uses only frozen Train/Internal Val files."""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from task5_data import metrics, sha256


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class EncodedRows(Dataset):
    def __init__(self, rows, tokenizer, mode, max_length):
        self.rows = rows
        self.items = []
        lengths = []
        for row in rows:
            target = f"Compound: {row['Compound']}; Modifier: {row['Mod']}; Head: {row['Head']}"
            if mode == "context":
                full = tokenizer(target, row["Context"], truncation=False)
                item = tokenizer(target, row["Context"], truncation="only_second", max_length=max_length)
            else:
                full = tokenizer(target, truncation=False)
                item = tokenizer(target, truncation=True, max_length=max_length)
            lengths.append(len(full["input_ids"]))
            item["labels"] = [float(row["ModAvg"]), float(row["HeadAvg"])]
            self.items.append(item)
        self.token_stats = {
            "max_untruncated": max(lengths),
            "p95_untruncated": float(np.percentile(lengths, 95)),
            "truncated_rows": sum(length > max_length for length in lengths),
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class Regressor(nn.Module):
    def __init__(self, model_path, means=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_path, local_files_only=True)
        width = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(width, 1), nn.Linear(width, 1)])
        if means is not None:
            for head, mean in zip(self.heads, means):
                nn.init.normal_(head.weight, std=0.02)
                nn.init.constant_(head.bias, float(mean))

    def forward(self, **inputs):
        hidden = self.backbone(**inputs).last_hidden_state
        # Compute masked pooling in FP32, including both target and context.
        weights = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden.float() * weights).sum(1) / weights.sum(1)
        pooled = self.dropout(pooled)
        return torch.cat([head(pooled) for head in self.heads], dim=-1).float()


def collator(tokenizer):
    pad = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def collect(items):
        labels = torch.tensor([item["labels"] for item in items], dtype=torch.float32)
        encoded = pad([{k: v for k, v in item.items() if k != "labels"} for item in items])
        return encoded, labels

    return collect


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    outputs = []
    for encoded, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(**{k: v.to(device) for k, v in encoded.items()})
        assert torch.isfinite(scores).all()
        outputs.extend(scores.cpu().tolist())
    return outputs


def scores_for(rows, raw):
    clipped = [[max(0.0, min(5.0, value)) for value in pair] for pair in raw]
    return {"raw": metrics(rows, raw), "clipped": metrics(rows, clipped)}, clipped


def run(args, output):
    started = time.monotonic()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-f3ed3c37-a52e-c126-79e9-e0a0783c584d", "must use physical GPU1 UUID"
    free, total = torch.cuda.mem_get_info()
    assert free > 18 * 1024 ** 3, "GPU1 no longer sufficiently idle"
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    device = torch.device("cuda:0")
    train_rows, val_rows = load_rows(args.data / "train.jsonl"), load_rows(args.data / "val.jsonl")
    full_counts = {"train": len(train_rows), "val": len(val_rows)}
    if args.smoke:
        train_rows, val_rows = train_rows[:64], val_rows[:32]
    proof = json.loads((args.model / "DOWNLOAD_VERIFIED.json").read_text())
    for name, item in proof["files"].items():
        assert sha256(args.model / name) == item["sha256"], name
    split = json.loads((args.data / "split.json").read_text())
    for name in ("train", "val"):
        assert sha256(args.data / (name + ".jsonl")) == split[name + "_sha256"]
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scope": "Smoke" if args.smoke else "Internal Val, EN NN, no official Dev/Test",
        "full_split_rows": full_counts,
        "used_rows": {"train": len(train_rows), "val": len(val_rows)},
        "selection": "max clipped mean Spearman; tie lower clipped mean RMSE",
        "source_revision": proof["revision"],
        "pooling": "FP32 attention-mask mean, including special tokens",
        "head": "two independent Linear(hidden,1), dropout=0.1",
        "output": "linear raw, clipped [0,5] for selection/export; both metrics saved",
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "train_sha256": split["train_sha256"],
        "val_sha256": split["val_sha256"],
    }
    write_json(output / "config.json", config)
    environment = {"python": platform.python_version(), "gpu": torch.cuda.get_device_name(0), "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    for name in ("torch", "transformers", "numpy", "scipy", "sentencepiece", "safetensors"):
        environment[name] = importlib.metadata.version(name)
    write_json(output / "environment.json", environment)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=True)
    train_data = EncodedRows(train_rows, tokenizer, args.mode, args.max_length)
    val_data = EncodedRows(val_rows, tokenizer, args.mode, args.max_length)
    write_json(output / "token_stats.json", {"train": train_data.token_stats, "val": val_data.token_stats})
    collect = collator(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collect, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, collate_fn=collect, num_workers=0)
    # Identical target-only inputs are evaluated once, avoiding padding-related
    # numerical noise being misreported as sensitivity to different contexts.
    if args.mode == "target_only":
        unique, indices, mapping = {}, [], []
        for i, item in enumerate(val_data.items):
            key = tuple(item["input_ids"])
            if key not in unique:
                unique[key] = len(indices)
                indices.append(i)
            mapping.append(unique[key])
        val_loader = DataLoader(torch.utils.data.Subset(val_data, indices), batch_size=args.batch_size, shuffle=False, collate_fn=collect, num_workers=0)

    def val_predictions(net):
        values = predict(net, val_loader, device)
        return [values[index] for index in mapping] if args.mode == "target_only" else values

    means = [sum(float(row[key]) for row in train_rows) / len(train_rows) for key in ("ModAvg", "HeadAvg")]
    baseline, _ = scores_for(val_rows, [means] * len(val_rows))
    write_json(output / "train_mean_reference.json", baseline)
    model = Regressor(args.model, means).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr},
        {"params": model.heads.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01)
    epochs = 1 if args.smoke else args.epochs
    updates = math.ceil(len(train_loader) / args.grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, round(updates * 0.06)), updates)
    history, global_step = [], 0
    best_key = (-float("inf"), -float("inf"))
    checkpoint = output / "best"
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch_index, (encoded, labels) in enumerate(train_loader):
            encoded, labels = {k: v.to(device) for k, v in encoded.items()}, labels.to(device)
            group_start = (batch_index // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, len(train_loader) - group_start)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(**encoded)
                loss = nn.functional.mse_loss(pred, labels)
            assert torch.isfinite(loss)
            (loss / group_size).backward()
            total_loss += loss.item() * labels.shape[0]
            if (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == len(train_loader):
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                assert torch.isfinite(norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step == 1 or global_step % 25 == 0:
                    state = {"stage": "TRAINING", "epoch": epoch, "global_step": global_step, "total_steps": updates, "loss": loss.item(), "gradient_norm": norm.item(), "elapsed_seconds": time.monotonic() - started}
                    write_json(output / "status.json", state)
                    print(json.dumps(state), flush=True)
        raw = val_predictions(model)
        score, _ = scores_for(val_rows, raw)
        rho = score["clipped"]["mean_spearman"]
        assert rho is not None
        item = {"epoch": epoch, "step": global_step, "train_mse": total_loss / len(train_rows), "metrics": score, "elapsed_seconds": time.monotonic() - started}
        history.append(item)
        write_json(output / "history.json", history)
        print(json.dumps(item), flush=True)
        key = (rho, -score["clipped"]["mean_rmse"])
        if key > best_key:
            best_key = key
            checkpoint.mkdir(exist_ok=True)
            model.backbone.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            save_file({k: v.detach().cpu().contiguous() for k, v in model.heads.state_dict().items()}, str(checkpoint / "regression_heads.safetensors"))
            write_json(output / "best_selection.json", item)
            write_json(output / "best_reference_predictions.json", raw)
    # Run an actual fresh load of the selected checkpoint and compare all values.
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    best_model = Regressor(checkpoint).to(device)
    best_model.heads.load_state_dict(load_file(str(checkpoint / "regression_heads.safetensors")))
    raw = val_predictions(best_model)
    score, clipped = scores_for(val_rows, raw)
    reference = json.loads((output / "best_reference_predictions.json").read_text())
    reload_difference = max(abs(a - b) for x, y in zip(raw, reference) for a, b in zip(x, y))
    assert reload_difference < 1e-6, "reload prediction mismatch"
    selected = json.loads((output / "best_selection.json").read_text())
    for form in ("raw", "clipped"):
        for key in ("mean_spearman", "mean_rmse"):
            assert abs(score[form][key] - selected["metrics"][form][key]) < 1e-6, "reload metric mismatch"
    repeat = val_predictions(best_model)
    max_difference = max(abs(a - b) for x, y in zip(raw, repeat) for a, b in zip(x, y))
    assert max_difference < 1e-6, "nondeterministic inference"
    with (output / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row, pred, final in zip(val_rows, raw, clipped):
            handle.write(json.dumps({"ContextID": row["ContextID"], "Compound": row["Compound"], "gold": [float(row["ModAvg"]), float(row["HeadAvg"])], "raw": pred, "clipped": final}, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "en-nn-internal-val-pred.tsv").open("w", encoding="utf-8") as handle:
        for row, values in zip(val_rows, clipped):
            handle.write(f"{row['ContextID']}\t{values[0]:.8f}\t{values[1]:.8f}\n")
    write_json(output / "metrics.json", score)
    write_json(output / "completion.json", {
        "status": "COMPLETED", "scope": config["scope"], "steps": global_step,
        "best_epoch": selected["epoch"], "elapsed_seconds": time.monotonic() - started,
        "max_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "reload_metrics_match": True, "repeat_max_abs_diff": max_difference,
        "reload_max_abs_diff": reload_difference,
        "checkpoint_sha256": {p.name: sha256(p) for p in checkpoint.iterdir() if p.is_file()},
        "prediction_sha256": sha256(output / "val_predictions.jsonl"),
    })
    write_json(output / "status.json", {"stage": "COMPLETED", "best_epoch": selected["epoch"], "metrics": score})
    print("COMPLETED " + str(output), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("context", "target_only"), default="context")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2027)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    write_json(arguments.output / "status.json", {"stage": "STARTING"})
    try:
        run(arguments, arguments.output)
    except Exception as error:
        write_json(arguments.output / "status.json", {"stage": "FAILED", "error_type": type(error).__name__, "error": str(error)})
        (arguments.output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
