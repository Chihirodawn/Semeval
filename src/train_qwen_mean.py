"""Isolated mean-pooling ablation; keep the completed last-token trainer frozen."""
import argparse
import traceback
from pathlib import Path
import torch
import train_qwen as original


class MeanRegressor(original.QwenRegressor):
    def forward(self, **inputs):
        hidden = self.backbone(**inputs, return_dict=True).last_hidden_state
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = self.dropout((hidden.float() * mask).sum(1) / mask.sum(1))
            return torch.cat([head(pooled) for head in self.heads], -1)


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    for name, default in (("epochs", 3), ("batch-size", 8), ("grad-accum", 2), ("max-length", 512), ("seed", 2027)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write = original.write_json

    def annotated_write(path, value):
        if path.name == "config.json":
            value["pooling"] = "masked mean of all non-pad final hidden states; FP32 heads"
            value["ablation"] = "only pooling changed from frozen last-token BF16 LoRA baseline"
        write(path, value)

    original.QwenRegressor = MeanRegressor
    original.write_json = annotated_write
    try:
        original.run(args)
    except Exception as error:
        write(args.output / "status.json", {"stage": "FAILED", "error": str(error)})
        (args.output / "failure.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
