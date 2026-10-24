"""Frozen soft-CE recipe on Qwen3.5 text backbones and native LM projections.

The old trainers stay unchanged. Reuse their exact prompts, loss, train loop,
selection rule and export format, with an explicit non-thinking chat template.
"""
import argparse
import functools
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from regression_methods.core import NumericPolicy, examples
from regression_methods.train import evaluate, save_trace, train_epochs
from task5_data import sha256
from train_encoder import load_rows, scores_for, write_json

ROOT = Path(__file__).resolve().parents[1]
GPU = "GPU-f3ed3c37-a52e-c126-79e9-e0a0783c584d"
REVISIONS = {"Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
             "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a"}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
           "in_proj_b", "in_proj_a", "out_proj"]


class NativeNumericCE(NumericPolicy):
    def __init__(self, base_path, digit_ids, adapter=None, trainable=True, native_check=False):
        nn.Module.__init__(self)
        config = json.loads((Path(base_path) / "config.json").read_text())
        loader = AutoModelForImageTextToText if config["model_type"] == "qwen3_5" else AutoModelForCausalLM
        full = loader.from_pretrained(base_path, dtype=torch.bfloat16, local_files_only=True,
                                      attn_implementation="sdpa")
        full.eval()
        base = full.model.language_model if config["model_type"] == "qwen3_5" else full.model
        head = full.get_output_embeddings()
        self.native_proof = {"native_architecture": type(full).__name__, "text_architecture": type(base).__name__,
                             "lm_head_tied": head.weight.data_ptr() == base.get_input_embeddings().weight.data_ptr(),
                             "native_pretrained_head": True, "visual_removed": config["model_type"] == "qwen3_5"}
        if native_check:
            # Match the native LM head GEMM shape: projecting 3 tokens vs 1 can
            # round differently in BF16 even when backbone and head are identical.
            ids = torch.tensor([[digit_ids[1], digit_ids[2], digit_ids[3]]])
            with torch.inference_mode():
                expected = full(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                logits_to_keep=1).logits[:, -1].float()
                hidden = base(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[:, -1]
                actual = head(hidden).float()
            difference = (expected - actual).abs().max().item()
            assert difference == 0, ("native full-model projection", difference)
            self.native_proof["native_cpu_logits_max_abs_diff"] = difference
            self.native_proof["projection_shape"] = "one final token on both native and extracted paths"
            del expected, hidden, actual, ids
        if config["model_type"] == "qwen3_5":
            full.model.language_model = None
        else:
            full.model = None
        full.set_output_embeddings(None)
        del full
        gc.collect()
        base.config.use_cache = False
        base.config._name_or_path = str(base_path)
        base.requires_grad_(False)
        head.requires_grad_(False)
        self.lm_head = head
        if adapter:
            self.backbone = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
        else:
            lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32,
                              lora_dropout=0.05, target_modules=TARGETS, bias="none")
            self.backbone = get_peft_model(base, lora)
        if trainable:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
        else:
            self.requires_grad_(False)
        self.digit_ids = list(digit_ids)

    def logits(self, hidden):
        return self.lm_head(hidden).float()


def execute(args):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == GPU
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    assert not any(x.startswith(GPU) for x in apps.splitlines()), "GPU1 occupied"
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    assert torch.cuda.mem_get_info()[0] > 21 * 1024**3
    torch.cuda.set_per_process_memory_fraction(0.9, 0)
    started = time.monotonic()
    proof = json.loads((args.model / "DOWNLOAD_VERIFIED.json").read_text())
    assert REVISIONS[proof["repo"]] == proof["revision"]
    for filename, info in proof["files"].items():
        assert sha256(args.model / filename) == info["sha256"], filename
    split = json.loads((args.data / "split.json").read_text())
    for name in ("train", "val"):
        assert sha256(args.data / (name + ".jsonl")) == split[name + "_sha256"]
    train_rows, val_rows = load_rows(args.data / "train.jsonl"), load_rows(args.data / "val.jsonl")
    if args.smoke:
        train_rows, val_rows = train_rows[:64], val_rows[:32]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    # Keep the old prompt strings verbatim; only specify the new template flag.
    prompt_tokenizer = SimpleNamespace(apply_chat_template=functools.partial(
        tokenizer.apply_chat_template, enable_thinking=False, return_dict=False))
    train_data, val_data = examples(train_rows, prompt_tokenizer), examples(val_rows, prompt_tokenizer)
    assert all(isinstance(x["prompt_ids"], list) and len(x["prompt_ids"]) > 20
               and all(isinstance(i, int) for i in x["prompt_ids"]) for x in train_data + val_data)
    digit_ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(6)]
    assert all(len(x) == 1 for x in digit_ids) and len({x[0] for x in digit_ids}) == 6
    digit_ids = [x[0] for x in digit_ids]
    args.mode, args.max_cot_tokens = "ce", 64
    files = [Path(__file__), ROOT / "src/regression_methods/core.py", ROOT / "src/regression_methods/train.py",
             ROOT / "src/task5_data.py", ROOT / "src/train_encoder.py"]
    cfg = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "scope": "Smoke" if args.smoke else "EN NN Internal Val; no official Dev/Test",
           "used_rows": {"train": len(train_rows), "val": len(val_rows)},
           "scalar_train_queries": len(train_data), "effective_scalar_batch": args.batch_size * args.grad_accum,
           "repo": proof["repo"], "source_revision": proof["revision"],
           "train_sha256": split["train_sha256"], "val_sha256": split["val_sha256"],
           "source_sha256": {str(p.relative_to(ROOT / "src")): sha256(p) for p in files},
           "grid": list(range(6)), "digit_token_ids": digit_ids,
           "loss": "adjacent-bin soft CE preserving continuous label mean",
           "expectation": "sum y*p_full_vocab(y), no digit-only renormalization",
           "projection": "actual frozen pretrained lm_head; tied and untied checkpoints both supported",
           "enable_thinking": False, "cot": False, "actual_epochs": 1 if args.smoke else args.epochs,
           "lora": {"rank": 16, "alpha": 32, "dropout": .05, "target_modules": TARGETS},
           "selection": "max clipped mean Spearman; tie lower mean component RMSE",
           "adaptations": ["Qwen3.5 text backbone, vision discarded", "include linear-attention projections in attention-only LoRA",
                           "new isolated Transformers/PEFT runtime", "native non-thinking chat template; same prompt strings",
                           "large-model microbatch may differ; effective batch16 unchanged"],
           "reload_scope": "deterministic direct scalar prompts; no sampled reasoning"}
    write_json(args.output / "config.json", cfg)
    write_json(args.output / "environment.json", {p: importlib.metadata.version(p) for p in
               ("torch", "transformers", "peft", "accelerate", "numpy", "scipy", "safetensors")})
    model = NativeNumericCE(args.model, digit_ids, native_check=args.smoke).to("cuda:0")
    write_json(args.output / "NATIVE_PROJECTION_VERIFIED.json", model.native_proof)
    frozen_head_sha = hashlib.sha256(model.lm_head.weight.detach().view(torch.uint8).cpu().numpy()).hexdigest()
    assert not model.lm_head.weight.requires_grad
    assert all(not p.requires_grad for n, p in model.backbone.named_parameters() if "lora_" not in n)
    trainables = [p for p in model.parameters() if p.requires_grad]
    write_json(args.output / "trainable_parameters.json", {"numel": sum(p.numel() for p in trainables),
               "base_frozen": True, "lm_head_frozen": True,
               "lora_modules": [n for n, p in model.named_parameters() if p.requires_grad]})
    steps = train_epochs(args, model, tokenizer, train_data, val_data, val_rows, trainables,
                         1 if args.smoke else args.epochs, False, started)
    assert hashlib.sha256(model.lm_head.weight.detach().view(torch.uint8).cpu().numpy()).hexdigest() == frozen_head_sha
    write_json(args.output / "FROZEN_LM_HEAD_VERIFIED.json", {"status": "PASS", "tensor_sha256": frozen_head_sha})
    model.to("cpu"); gc.collect(); torch.cuda.empty_cache()
    reloaded = NativeNumericCE(args.model, digit_ids, args.output / "best", trainable=False).to("cuda:0")
    raw, _, diagnostic = evaluate(reloaded, val_data, tokenizer, args.batch_size, False, 64)
    reference = json.loads((args.output / "best_reference_predictions.json").read_text())
    reload_diff = max(abs(a - b) for x, y in zip(raw, reference) for a, b in zip(x, y))
    assert reload_diff < 1e-6, reload_diff
    repeated, _, _ = evaluate(reloaded, val_data, tokenizer, args.batch_size, False, 64)
    repeat_diff = max(abs(a - b) for x, y in zip(raw, repeated) for a, b in zip(x, y))
    assert repeat_diff < 1e-6
    assert hashlib.sha256(reloaded.lm_head.weight.detach().view(torch.uint8).cpu().numpy()).hexdigest() == frozen_head_sha
    scores, clipped = scores_for(val_rows, raw)
    selected = json.loads((args.output / "best_selection.json").read_text())
    assert scores["clipped"]["mean_spearman"] == selected["metrics"]["clipped"]["mean_spearman"]
    with (args.output / "val_predictions.jsonl").open("w") as handle:
        for row, p, final in zip(val_rows, raw, clipped):
            handle.write(json.dumps({"ContextID": row["ContextID"], "Compound": row["Compound"],
                         "gold": [float(row["ModAvg"]), float(row["HeadAvg"])], "raw": p, "clipped": final}) + "\n")
    with (args.output / "en-nn-internal-val-pred.tsv").open("w") as handle:
        for row, p in zip(val_rows, clipped):
            handle.write(f"{row['ContextID']}\t{p[0]:.8f}\t{p[1]:.8f}\n")
    write_json(args.output / "metrics.json", scores)
    diagnostic.pop("digit_constrained_mode_pairs")
    write_json(args.output / "numeric_diagnostics.json", diagnostic)
    save_trace(args.output / "best_validation_inputs.jsonl", val_data)
    completion = {"status": "COMPLETED", "scope": cfg["scope"], "steps": steps,
                  "best_epoch": selected["epoch"], "elapsed_seconds": time.monotonic() - started,
                  "reload_max_abs_diff": reload_diff, "repeat_max_abs_diff": repeat_diff,
                  "reload_scope": cfg["reload_scope"], "max_gpu_allocated_gib": torch.cuda.max_memory_allocated()/1024**3,
                  "max_gpu_reserved_gib": torch.cuda.max_memory_reserved()/1024**3,
                  "checkpoint_sha256": {p.name: sha256(p) for p in (args.output / "best").iterdir() if p.is_file()},
                  "prediction_sha256": sha256(args.output / "val_predictions.jsonl"),
                  "validation_trace_sha256": sha256(args.output / "best_validation_inputs.jsonl")}
    write_json(args.output / "completion.json", completion)
    write_json(args.output / "status.json", {"stage": "COMPLETED", "best_epoch": selected["epoch"], "metrics": scores})
    print(json.dumps(completion), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    args.model, args.data, args.output = args.model.resolve(), args.data.resolve(), args.output.resolve()
    assert args.batch_size * args.grad_accum == 16
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "status.json", {"stage": "STARTING"})
    try:
        execute(args)
    except Exception as error:
        write_json(args.output / "status.json", {"stage": "FAILED", "error": str(error), "type": type(error).__name__})
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
