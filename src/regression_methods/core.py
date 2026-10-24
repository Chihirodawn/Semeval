"""Full-vocabulary numeric expectation, CoT supervision, and REAL/RLOO.

The pretrained tied language-model projection is retained. This is not a new
numeric regression head. Digits are NOT renormalized to sum to one.
"""
import math
from pathlib import Path

import torch
from torch import nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel

CUE = "\nScore: "
GRID = list(range(6))


def key(example):
    return str(example["ContextID"]) + ":" + str(example["component"])


def examples(rows, tokenizer, reasoning=False):
    result = []
    for row in rows:
        for component, label in enumerate(("ModAvg", "HeadAvg")):
            focus = "modifier" if component == 0 else "head"
            system = ("Assess noun compound compositionality in context. Score the specified component from 0 "
                      "(no literal meaning retained) to 5 (fully literal meaning retained). ")
            if reasoning:
                system += "Explain briefly in at most 40 words, then end with '\\nScore: '. Do not write the numerical score yet."
            else:
                system += "Return one integer from 0 through 5 without explanation."
            user = (f"Context: {row['Context']}\nCompound: {row['Compound']}\n"
                    f"Modifier: {row['Mod']}\nHead: {row['Head']}\nAssess the {focus} component only.")
            ids = tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                                tokenize=True, add_generation_prompt=True)
            result.append({"ContextID": row["ContextID"], "Compound": row["Compound"], "component": component,
                           "gold": float(row[label]), "prompt_ids": ids})
    return result


def soft_digit_ce(log_probs, targets, digit_ids):
    """Adjacent-bin soft labels preserve each continuous target's exact mean."""
    low = targets.floor().long().clamp(0, 5)
    high = (low + 1).clamp(max=5)
    fraction = targets - low
    indices = torch.as_tensor(digit_ids, device=log_probs.device)
    return -(1 - fraction) * log_probs.gather(1, indices[low, None]).squeeze(1) - fraction * log_probs.gather(1, indices[high, None]).squeeze(1)


def rloo_advantages(reward):
    assert reward.ndim == 2 and reward.shape[1] > 1
    others = (reward.sum(1, keepdim=True) - reward) / (reward.shape[1] - 1)
    advantage = reward - others
    return (advantage / (advantage.std(1, unbiased=False, keepdim=True) + 1e-8)).clamp(-1, 1)


def real_loss(prediction, gold, trajectory_logprob, group_size=2):
    """REAL Eq6-8, lambda=0/beta=1; sum log pi(c|x), NOT token mean."""
    error = (prediction - gold).square()
    advantage = rloo_advantages((-error).detach().reshape(-1, group_size)).reshape(-1)
    policy_loss = -(advantage.detach() * trajectory_logprob).mean()
    refinement = error.mean()
    return policy_loss + refinement, {"policy_loss": policy_loss.detach(), "prediction_refinement": refinement.detach(),
                                      "advantage_abs_mean": advantage.abs().mean().detach()}


class NumericPolicy(nn.Module):
    def __init__(self, base_path, digit_ids, adapter=None, trainable=True):
        super().__init__()
        base = AutoModel.from_pretrained(base_path, torch_dtype=torch.bfloat16, local_files_only=True,
                                         attn_implementation="sdpa", low_cpu_mem_usage=True)
        assert base.config.tie_word_embeddings, "must use the actual tied pretrained LM projection"
        base.config.use_cache = False
        if adapter:
            self.backbone = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
        else:
            config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.05,
                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none")
            self.backbone = get_peft_model(base, config)
        if trainable:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
        else:
            self.requires_grad_(False)
        self.digit_ids = list(digit_ids)

    def logits(self, hidden):
        # This frozen embedding matrix is exactly Qwen's tied pretrained lm_head.
        return nn.functional.linear(hidden, self.backbone.get_input_embeddings().weight).float()

    def scores(self, logits):
        logp = logits.float().log_softmax(-1)
        probabilities = logp.exp()[:, self.digit_ids]
        numeric = (probabilities * torch.arange(6, device=logits.device)).sum(-1)
        return numeric, probabilities.sum(-1), probabilities.argmax(-1).float(), logp

    def batch(self, records, tokenizer, reasoning=False, cot_ce=False, policy=False):
        cue = tokenizer.encode(CUE, add_special_tokens=False)
        sequences, trajectories, starts, lm_tokens = [], [], [], []
        for example in records:
            prompt = example["prompt_ids"]
            raw = list(example.get("cot_ids", []))
            content = raw[:-1] if raw and raw[-1] == tokenizer.eos_token_id else raw
            suffix = content if content[-len(cue):] == cue else content + cue
            if not reasoning:
                assert not raw
                suffix = cue
            sequence = prompt + suffix
            assert len(sequence) <= 512, "no silent truncation"
            sequences.append(sequence)
            starts.append(len(prompt))
            trajectories.append(raw)
            lm_tokens.append(suffix)
        device = next(self.parameters()).device
        length = math.ceil(max(map(len, sequences)) / 8) * 8
        ids = torch.full((len(records), length), tokenizer.pad_token_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, :len(sequence)] = torch.tensor(sequence, device=device)
            mask[i, :len(sequence)] = 1
        hidden = self.backbone(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        positions = mask.sum(1) - 1
        score_logits = self.logits(hidden[torch.arange(len(records), device=device), positions])
        prediction, mass, mode, logp = self.scores(score_logits)
        gold = torch.tensor([r["gold"] for r in records], device=device, dtype=torch.float32)
        result = {"prediction": prediction, "digit_mass": mass, "mode": mode, "gold": gold,
                  "score_ce": soft_digit_ce(logp, gold, self.digit_ids),
                  "full_top1_is_digit": torch.isin(score_logits.argmax(-1), torch.tensor(self.digit_ids, device=device)).float()}
        if cot_ce or policy:
            owners, positions, labels = [], [], []
            for i, tokens in enumerate(trajectories if policy else lm_tokens):
                for j, token in enumerate(tokens):
                    owners.append(i); positions.append(starts[i] - 1 + j); labels.append(token)
            assert labels, "reasoning objective requires actual sampled/annotated tokens"
            owner = torch.tensor(owners, device=device)
            selected = hidden[owner, torch.tensor(positions, device=device)]
            token_loss = nn.functional.cross_entropy(self.logits(selected), torch.tensor(labels, device=device), reduction="none")
            sums = torch.zeros(len(records), device=device).scatter_add(0, owner, token_loss)
            counts = torch.bincount(owner, minlength=len(records)).clamp(min=1)
            result["trajectory_logprob" if policy else "cot_ce"] = -sums if policy else sums / counts
        return result

    @torch.inference_mode()
    def generate_reasoning(self, records, tokenizer, max_new_tokens=64, sample=False):
        """On-policy temperature1/top_p1; keep exact IDs and termination event."""
        self.eval()
        device = next(self.parameters()).device
        cue = tokenizer.encode(CUE, add_special_tokens=False)
        prompts = [r["prompt_ids"] for r in records]
        length = max(map(len, prompts))
        ids = torch.full((len(records), length), tokenizer.pad_token_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, prompt in enumerate(prompts):
            ids[i, -len(prompt):] = torch.tensor(prompt, device=device)
            mask[i, -len(prompt):] = 1
        raw, stops, totals = [[] for _ in records], [None] * len(records), [0.0] * len(records)
        cache = None
        for step in range(max_new_tokens):
            position = mask.cumsum(-1) - 1
            position.masked_fill_(mask == 0, 0)
            output = self.backbone(input_ids=ids if cache is None else ids[:, -1:], attention_mask=mask,
                                   position_ids=position if cache is None else position[:, -1:], past_key_values=cache,
                                   use_cache=True, return_dict=True)
            cache = output.past_key_values
            logp = self.logits(output.last_hidden_state[:, -1]).log_softmax(-1)
            chosen = torch.multinomial(logp.exp(), 1).squeeze(1) if sample else logp.argmax(-1)
            tokens, selected = chosen.tolist(), logp.gather(1, chosen[:, None]).squeeze(1).tolist()
            active = [stop is None for stop in stops]
            for i, token in enumerate(tokens):
                if not active[i]:
                    tokens[i] = tokenizer.pad_token_id
                    continue
                raw[i].append(token); totals[i] += selected[i]
                if token == tokenizer.eos_token_id:
                    stops[i] = "eos"
                elif raw[i][-len(cue):] == cue:
                    stops[i] = "score_delimiter"
            if all(x is not None for x in stops):
                break
            ids = torch.cat([ids, torch.tensor(tokens, device=device)[:, None]], 1)
            mask = torch.cat([mask, torch.tensor(active, device=device, dtype=mask.dtype)[:, None]], 1)
        return [dict(r, cot_ids=tokens, cot_text=tokenizer.decode(tokens, skip_special_tokens=False),
                     stop_reason=stop or "token_limit", sampled_logprob=total, sampling=sample,
                     max_new_tokens=max_new_tokens) for r, tokens, stop, total in zip(records, raw, stops, totals)]

    def disable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0

    def save(self, destination):
        self.backbone.save_pretrained(destination, safe_serialization=True)
