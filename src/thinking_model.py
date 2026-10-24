"""New Qwen3.5 native-thinking path; historical trainers stay unchanged."""
import math
import time

import torch
from torch import nn
from ce_model_compare_v3 import NativeNumericCE
from regression_methods.core import soft_digit_ce
from thinking_protocol import (MAX_SEQUENCE, SAMPLING, SCORE_CUE, check_trace,
                               sample_seed, validate_reasoning)


def objective(output, records, rank_weight=0.0, huber_weight=0.0):
    pred, gold = output["prediction"], output["gold"]
    rank = pred.sum() * 0.0
    eligible = (len(records) == 2 and records[0]["component"] == records[1]["component"]
                and abs(float(gold[0] - gold[1])) >= 0.25)
    if eligible:
        rank = nn.functional.softplus(-torch.sign(gold[0] - gold[1]) * (pred[0] - pred[1]))
    huber = nn.functional.smooth_l1_loss(pred, gold, beta=1.0)
    score = output["score_ce"].mean()
    cot = output["cot_ce"].mean()
    loss = score + 0.1 * cot + rank_weight * rank + huber_weight * huber
    return loss, {"score_ce": score, "cot_ce": cot, "pair": rank, "huber": huber,
                  "mse": (pred-gold).square().mean(), "eligible_pair": int(eligible)}


class ThinkingPolicy(NativeNumericCE):
    def batch(self, records, tokenizer, supervise=True):
        sequences, cot_positions = [], []
        cue = tokenizer.encode(SCORE_CUE, add_special_tokens=False)
        for row in records:
            check_trace(row, tokenizer)
            prompt, cot = row["prompt_ids"], row.get("cot_ids", [])
            seq = prompt + cot + cue
            assert len(seq) <= MAX_SEQUENCE
            sequences.append(seq)
            # Predict only generated thinking tokens (including </think>).
            # No prompt tokens, score cue or synthetic numeric target labels.
            cot_positions.append(list(range(len(prompt)-1, len(prompt)+len(cot)-1)))
        device = next(self.parameters()).device
        length = math.ceil(max(map(len, sequences)) / 8) * 8
        ids = torch.full((len(records), length), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, :len(seq)] = torch.tensor(seq, device=device)
            mask[i, :len(seq)] = 1
        hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False,
                               return_dict=True).last_hidden_state
        logits = self.logits(hidden[torch.arange(len(records), device=device), mask.sum(1)-1])
        logp = logits.log_softmax(-1)
        digit_probs = logp[:, self.digit_ids].exp()
        pred = (digit_probs * torch.arange(6, device=device)).sum(-1)
        gold = torch.tensor([r.get("gold", 0.0) for r in records], dtype=torch.float32, device=device)
        cot_losses = []
        for i, positions in enumerate(cot_positions):
            total = hidden[i, 0, 0].float() * 0.0
            if supervise and positions:
                # Checkpoint the vocabulary projection: retain hidden states,
                # not sequence_length x vocabulary FP32 logits for every chunk.
                from torch.utils.checkpoint import checkpoint
                labels = torch.tensor(records[i]["cot_ids"], device=device)
                for start in range(0, len(positions), 32):
                    selected = hidden[i, positions[start:start+32]]
                    target = labels[start:start+32]
                    def chunk_loss(h, target):
                        return nn.functional.cross_entropy(self.logits(h), target, reduction="sum")
                    total = total + checkpoint(chunk_loss, selected, target, use_reentrant=False)
                total = total / len(positions)
            cot_losses.append(total)
        return {"prediction": pred, "gold": gold, "score_ce": soft_digit_ce(logp, gold, self.digit_ids),
                "cot_ce": torch.stack(cot_losses), "digit_mass": digit_probs.sum(-1),
                "cot_target_count": [len(p) if supervise else 0 for p in cot_positions]}

    @torch.inference_mode()
    def generate_one(self, row, tokenizer, budget, seed, attempt):
        """Single native assistant trajectory, isolated RNG, KV-cached decoding.

        No padded batches, gold labels, legacy Score stops or forced </think>.
        Native </think> is the only successful termination event.
        """
        self.eval()
        device = next(self.parameters()).device
        assert len(row["prompt_ids"]) + budget + len(tokenizer.encode(SCORE_CUE)) <= MAX_SEQUENCE
        ids = torch.tensor([row["prompt_ids"]], device=device)
        attention = torch.ones_like(ids)
        end = tokenizer.encode("</think>", add_special_tokens=False)
        raw, cache, stop = [], None, "token_limit"
        started = time.monotonic()
        devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(sample_seed(seed, row, attempt))
            for _ in range(budget):
                output = self.backbone(input_ids=ids if cache is None else ids[:, -1:],
                                       attention_mask=attention, past_key_values=cache,
                                       use_cache=True, return_dict=True)
                cache = output.past_key_values
                logits = self.logits(output.last_hidden_state[:, -1]).squeeze(0) / SAMPLING["temperature"]
                values, indices = logits.topk(min(SAMPLING["top_k"], logits.numel()))
                probabilities = values.softmax(-1)
                remove = probabilities.cumsum(-1) - probabilities >= SAMPLING["top_p"]
                values.masked_fill_(remove, -float("inf"))
                token = int(indices[torch.multinomial(values.softmax(-1), 1)])
                raw.append(token)
                if raw[-len(end):] == end:
                    stop = "think_end"
                    break
                if token == tokenizer.eos_token_id:
                    stop = "eos"
                    break
                ids = torch.cat((ids, torch.tensor([[token]], device=device)), dim=1)
                attention = torch.cat((attention, attention.new_ones((1, 1))), dim=1)
        text = tokenizer.decode(raw, skip_special_tokens=False)
        return {"cot_ids": raw, "cot_text": text, "stop_reason": stop,
                "max_new_tokens": budget, "generation_seed": sample_seed(seed, row, attempt),
                "generation_seconds": time.monotonic()-started,
                "error": validate_reasoning(text, row["style"], row["Context"], stop,
                                            row.get("reasoning_prefix", ""))}

    @torch.inference_mode()
    def reason(self, row, direct, tokenizer, budget, seed, retry=True):
        if row["style"] == "direct":
            return dict(row, cot_ids=[], cot_text="", stop_reason="direct", fallback=False, attempts=[])
        attempts = []
        for attempt in range(3 if retry else 1):
            trace = self.generate_one(row, tokenizer, budget, seed, attempt)
            attempts.append(trace)
            if trace["error"] is None:
                return dict(row, **{k:v for k,v in trace.items() if k != "error"}, fallback=False,
                            attempts=attempts, sampling=SAMPLING)
        return dict(row, prompt_ids=direct["prompt_ids"], direct_prompt_ids=direct["prompt_ids"],
                    cot_ids=[], cot_text="", stop_reason="fallback_direct", fallback=True,
                    attempts=attempts, sampling=SAMPLING)
