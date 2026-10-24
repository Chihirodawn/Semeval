"""Tiny native Qwen3.5 fixtures: tied/untied heads, CE gradients and reload."""
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5VisionConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ce_model_compare_v3 import NativeNumericCE


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 31

    def encode(self, text, add_special_tokens=False):
        return [7, 8]


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2027)
    for tied in (True, False):
        text = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
                max_position_embeddings=64, tie_word_embeddings=tied,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                 "mrope_section": [1, 1, 2], "partial_rotary_factor": 1.0})
        vision = Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                                     out_hidden_size=32, patch_size=2, temporal_patch_size=1,
                                     spatial_merge_size=1, num_position_embeddings=16)
        config = Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=tied,
                               image_token_id=28, video_token_id=29, vision_start_token_id=30, vision_end_token_id=31)
        with tempfile.TemporaryDirectory(prefix="ce_models_tiny_") as temporary:
            folder = Path(temporary)
            native = Qwen3_5ForConditionalGeneration(config)
            native.save_pretrained(folder)
            del native
            model = NativeNumericCE(folder, list(range(6)), native_check=True)
            assert model.native_proof["native_cpu_logits_max_abs_diff"] == 0
            assert model.native_proof["lm_head_tied"] == tied
            row = {"ContextID": "tiny", "Compound": "x y", "component": 0,
                   "gold": 2.25, "prompt_ids": [10, 11, 12]}
            model.train()
            result = model.batch([row], TinyTokenizer())
            loss = result["score_ce"].mean()
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            assert grads and all(torch.isfinite(x).all() for x in grads)
            assert sum(float(x.float().square().sum()) for x in grads) > 0
            assert model.lm_head.weight.grad is None
            assert all(p.grad is None for n, p in model.backbone.named_parameters() if "lora_" not in n)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            model.eval()
            with torch.inference_mode():
                reference = model.batch([row], TinyTokenizer())["prediction"]
                changed = model.batch([dict(row, gold=5.0)], TinyTokenizer())["prediction"]
            assert torch.equal(reference, changed)
            model.save(folder / "best")
            reloaded = NativeNumericCE(folder, list(range(6)), folder / "best", trainable=False)
            reloaded.eval()
            with torch.inference_mode():
                actual = reloaded.batch([row], TinyTokenizer())["prediction"]
            difference = (reference - actual).abs().max().item()
            assert difference == 0, difference
            print(json.dumps({"fixture": "tiny random Qwen3.5", "head_tied": tied, "native_logits_diff": 0,
                              "soft_ce_lora_gradients": "PASS", "base_and_lm_head_frozen": True,
                              "gold_not_in_prediction": True, "reload_max_abs_diff": difference}), flush=True)


if __name__ == "__main__":
    main()
