"""Gold-free native-thinking inputs, strict traces, pairing and fixed gates.

This module uses only the standard library so cache admission and scheduling
can be checked independently of the GPU/runtime.
"""
import hashlib
import json
import random
import re
from collections import defaultdict

SCORE_CUE = "\n\nScore: "
MAX_SEQUENCE = 4096
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
STYLES = ("direct", "free", "evidence")


def identity(row):
    return str(row["ContextID"]) + ":" + str(row["component"])


def sample_seed(seed, row, attempt=0):
    payload = f"{seed}:{identity(row)}:{attempt}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:15], 16)


def inputs(rows, tokenizer, style):
    """Deliberately project away all gold columns BEFORE constructing prompts."""
    assert style in STYLES
    result = []
    for original in rows:
        row = {k: original[k] for k in ("ContextID", "Context", "Compound", "Mod", "Head")}
        for component in (0, 1):
            system = ("Assess noun compound compositionality in context. Score the specified component from 0 "
                      "(no literal meaning retained) to 5 (fully literal meaning retained). ")
            if style == "direct":
                system += "Return one integer from 0 through 5 without explanation."
            else:
                # A numeric scoring rubric in the thinking prompt induced
                # repeated 0--5 scale discussion and premature teacher scores.
                # This stage supplies semantics only; the score CE defines the
                # final numeric task after the model closes native thinking.
                # Role labels such as "Component: head" confused native
                # reasoning in the v5 Train pilot. Specify the actual word in
                # ordinary language, without asking it to interpret metadata.
                system = ("Analyze word meaning in context. Briefly explain the specified word's meaning "
                          "in the given expression and how much of its ordinary literal meaning it retains. "
                          "Use qualitative descriptions rather than numerical ratings. ")
                if style == "evidence":
                    system += ("Organize the analysis into three concise notes:\n"
                               "Meaning: the expression's meaning in this sentence.\n"
                               "Literal: the specified word's ordinary meaning and what is retained.\n"
                               'Evidence: "an exact short quotation from the sentence". ')
            focus = "modifier" if component == 0 else "head"
            user = (f"Context: {row['Context']}\nCompound: {row['Compound']}\n"
                    f"Modifier: {row['Mod']}\nHead: {row['Head']}\nAssess the {focus} component only.")
            if style != "direct":
                word = row["Mod"] if component == 0 else row["Head"]
                user = (f"Sentence: {row['Context']}\nExpression: {row['Compound']}\n"
                        f'Explain the word "{word}" as it is used here.')
            tokens = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=True, add_generation_prompt=True, enable_thinking=style != "direct", return_dict=False)
            # Use the native assistant thinking opening exactly. No partial
            # sentence is supplied inside the model's generated reasoning.
            prefix = ""
            assert isinstance(tokens, list) and tokens and all(type(t) is int for t in tokens)
            assert len(tokens) < MAX_SEQUENCE - 2048 - 16, "input cannot fit maximum reasoning budget"
            result.append(dict(row, component=component, style=style, prompt_ids=tokens, reasoning_prefix=prefix))
    assert len({identity(r) for r in result}) == len(result)
    return result


def attach_gold(records, rows):
    gold = {str(r["ContextID"]): (float(r["ModAvg"]), float(r["HeadAvg"])) for r in rows}
    return [dict(r, gold=gold[str(r["ContextID"])][r["component"]]) for r in records]


def validate_reasoning(text, style, context, stop, prefix=""):
    if stop != "think_end" or not text.endswith("</think>"):
        return "incomplete_thinking"
    generated = text[:-len("</think>")].strip()
    if not generated:
        return "invalid_thinking_format"
    body = (prefix + text[:-len("</think>")]).strip()
    if not body or "<think>" in body or "</think>" in body or "<|im_" in body:
        return "invalid_thinking_format"
    # Quoted source numbers are not pseudo-labels. Exclude only exact context
    # quotations before inspecting explicit numeric rating statements.
    body_for_scores = re.sub(r'"([^"\n]+)"',
                             lambda m: "" if m[1] in context else m[0], body)
    numeric = r"(?:[0-5](?:\.\d+)?|zero|one|two|three|four|five)"
    rating = r"(?:score|rating|rated|rate|assign|give|compositionality|literalness)"
    if re.search(rf"\b{rating}\b[^\n.!?]{{0,40}}?\b{numeric}\b", body_for_scores, re.I):
        return "premature_numeric_rating"
    if re.search(r"\b[0-5](?:\.\d+)?\s*(?:/\s*5|out of\s*(?:5|five))", body_for_scores, re.I):
        return "premature_numeric_rating"
    if re.search(r"^\s*Score\s*:", body, re.I | re.M):
        return "premature_score_field"
    if style == "evidence":
        for field in ("Meaning", "Literal", "Evidence"):
            if not re.search(rf"^\s*{field}:\s*\S", body, re.M):
                return "missing_" + field.lower()
        evidence = re.search(r'^\s*Evidence:\s*"([^"\n]+)"\s*$', body, re.M)
        if evidence is None or evidence[1] not in context:
            return "unsupported_quote"
    return None


def check_trace(record, tokenizer):
    assert record["style"] in STYLES
    if record["style"] == "direct":
        assert not record.get("cot_ids") and not record.get("fallback", False)
    elif record.get("fallback"):
        assert not record.get("cot_ids") and record.get("attempts")
        assert record["prompt_ids"] == record["direct_prompt_ids"]
    else:
        assert record.get("cot_ids") and not record.get("fallback", False)
        text = tokenizer.decode(record["cot_ids"], skip_special_tokens=False)
        assert text == record["cot_text"]
        assert validate_reasoning(text, record["style"], record["Context"], record["stop_reason"],
                                  record.get("reasoning_prefix", "")) is None
    suffix = record.get("cot_ids", []) + tokenizer.encode(SCORE_CUE, add_special_tokens=False)
    assert len(record["prompt_ids"]) + len(suffix) <= MAX_SEQUENCE, "no silent truncation"


def pairing(records, seed, epoch):
    """Partition each component exactly once; seek half within-compound pairs.

    No label enters the pairing. Eligibility for the ranking loss is separate.
    Cross-component pairs never occur, including odd small smoke fixtures.
    """
    rng = random.Random(seed + epoch)
    batches, same_count = [], 0
    for component in (0, 1):
        groups = defaultdict(list)
        for r in records:
            if r["component"] == component:
                groups[r["Compound"].casefold().strip()].append(r)
        candidates = []
        for name in sorted(groups):
            rng.shuffle(groups[name])
            candidates.extend((name, k) for k in range(len(groups[name]) // 2))
        rng.shuffle(candidates)
        count = sum(map(len, groups.values()))
        quota = min(len(candidates), (count // 2) // 2)
        for name, _ in candidates[:quota]:
            batches.append([groups[name].pop(), groups[name].pop()])
            same_count += 1
        remaining = [r for name in sorted(groups) for r in groups[name]]
        rng.shuffle(remaining)
        while remaining:
            first = remaining.pop()
            index = next((i for i in range(len(remaining)-1, -1, -1)
                          if remaining[i]["Compound"] != first["Compound"]), None)
            if not remaining:
                batches.append([first])
            else:
                second = remaining.pop(-1 if index is None else index)
                same_count += first["Compound"] == second["Compound"]
                batches.append([first, second])
    rng.shuffle(batches)
    flat = [identity(r) for b in batches for r in b]
    assert len(flat) == len(records) and set(flat) == {identity(r) for r in records}
    assert all(len({r["component"] for r in b}) == 1 for b in batches)
    pairs = sum(len(b) == 2 for b in batches)
    return batches, {"pairs": pairs, "within_compound_pairs": same_count,
                     "within_fraction": same_count / pairs if pairs else None,
                     "schedule_sha256": hashlib.sha256(json.dumps([[identity(r) for r in b] for b in batches]).encode()).hexdigest()}


def fallback_rate(records):
    return sum(bool(r.get("fallback")) for r in records) / len(records)


def better(a, b):
    return (a["mean_spearman"], -a["mean_rmse"]) > (b["mean_spearman"], -b["mean_rmse"])


def promotion(candidate, baseline):
    return (candidate["mean_spearman"] - baseline["mean_spearman"] >= 0.005
            and candidate["mean_rmse"] - baseline["mean_rmse"] <= 0.01)
