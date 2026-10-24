"""CPU-only tiny-model contract; not a trained Task5 quality experiment."""

import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config, Qwen3Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_qwen import QwenRegressor

torch.set_num_threads(2)
torch.manual_seed(2027)
with tempfile.TemporaryDirectory(prefix="task5_qwen_contract_") as directory:
    root = Path(directory)
    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                      pad_token_id=0)
    Qwen3Model(cfg).save_pretrained(root / "base")
    model = QwenRegressor(root / "base", means=[2.5, 2.5])
    sample = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    padded = {"input_ids": torch.tensor([[1, 2, 3, 0, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 0, 0]])}
    prediction = model(**sample)
    loss = prediction.square().mean()
    loss.backward()
    lora = [(n, p) for n, p in model.backbone.named_parameters() if "lora_" in n]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in lora)
    assert all(not p.requires_grad for n, p in model.backbone.named_parameters() if "lora_" not in n)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        before, with_padding = model(**sample), model(**padded)
    assert torch.allclose(before, with_padding, atol=1e-5, rtol=0), (before, with_padding)
    model.backbone.save_pretrained(root / "adapter", safe_serialization=True)
    save_file({k: v.detach().contiguous() for k, v in model.heads.state_dict().items()}, str(root / "heads.safetensors"))
    reloaded = QwenRegressor(root / "base", adapter=root / "adapter")
    reloaded.heads.load_state_dict(load_file(str(root / "heads.safetensors")))
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(**sample)
    assert torch.allclose(before, after, atol=1e-5, rtol=0), (before, after)
    print(json.dumps({"status": "PASS", "scope": "CPU tiny random Qwen3 contract only", "shape": list(after.shape),
                      "padding_max_abs_diff": float((before - with_padding).abs().max()),
                      "adapter_reload_max_abs_diff": float((before - after).abs().max()),
                      "lora_backward_nonzero_base_frozen": True}))
