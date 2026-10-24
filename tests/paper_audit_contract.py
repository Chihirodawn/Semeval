"""CPU checks for paper objectives and a probe of the known legacy CoT boundary bug.

PASS means the audit reproduced expected behavior, not that legacy CoT is safe.
No pretrained checkpoint or GPU is loaded.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from regression_methods.core import NumericPolicy, rloo_advantages, real_loss, soft_digit_ce

torch.set_num_threads(2)
# Exact adjacent-bin labels, including endpoints, without rounding human means.
logp = torch.arange(24, dtype=torch.float32).reshape(3, 8).log_softmax(-1)
y = torch.tensor([0., 2.25, 5.])
ce = soft_digit_ce(logp, y, list(range(6)))
expected = torch.stack([-logp[0, 0], -.75*logp[1, 2]-.25*logp[1, 3], -logp[2, 5]])
assert torch.allclose(ce, expected)
# The finite-grid predictor is the unrenormalized sum in RAFT Eq10.
probe = object.__new__(NumericPolicy)
probe.digit_ids = list(range(6))
mu, mass, _, _ = NumericPolicy.scores(probe, torch.zeros(1, 8))
assert torch.allclose(mu, torch.tensor([15/8]))
assert torch.allclose(mass, torch.tensor([6/8]))
# RLOO uses within-group leave-one-out; ties provide no exploration gradient.
a = rloo_advantages(torch.tensor([[-1., -4.], [-3., -3.]]))
assert torch.equal(a, torch.tensor([[1., -1.], [0., 0.]]))
pred = torch.tensor([1., 4.], requires_grad=True)
lp = torch.tensor([-2., -3.], requires_grad=True)
loss, parts = real_loss(pred, torch.tensor([3., 3.]), lp)
g_pred, g_lp = torch.autograd.grad(loss, (pred, lp))
assert torch.equal(g_pred, torch.tensor([-2., 1.]))
assert torch.equal(g_lp, torch.tensor([.5, -.5]))
assert set(parts) == {"policy_loss", "prediction_refinement", "advantage_abs_mean"}

class Tokens:
    pad_token_id = 0
    eos_token_id = 2
    def encode(self, text, add_special_tokens=False):
        return [7, 8]

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 4)
    def forward(self, input_ids, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))

class LegacyProbe(NumericPolicy):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = Backbone()
        self.head = nn.Linear(4, 64)
        self.digit_ids = list(range(10, 16))
    def logits(self, hidden):
        return self.head(hidden).float()

# Actual legacy batch path retains an internal teacher digit in text CE.
# 15 stands for generated "5"; the final artificial gold digit is never appended.
model = LegacyProbe()
records = [{"ContextID":"audit", "component":0, "gold":1.3,
            "prompt_ids":[1, 20], "cot_ids":[30, 7, 8, 15, 2]}]
original = nn.functional.cross_entropy
with patch("torch.nn.functional.cross_entropy", wraps=original) as ce_call:
    output = model.batch(records, Tokens(), reasoning=True, cot_ce=True)
    supervised = ce_call.call_args.args[1].tolist()
assert supervised == [30, 7, 8, 15, 7, 8]
assert 15 in supervised
changed = [dict(records[0], gold=4.7)]
with torch.no_grad():
    other = model.batch(changed, Tokens(), reasoning=True, cot_ce=True)
assert torch.equal(output["prediction"], other["prediction"])
print(json.dumps({"status":"PASS", "scope":"CPU analytical objective tests and synthetic legacy trace; no training/quality evaluation",
    "soft_ce_continuous_labels":True, "raft_full_vocabulary_expectation":True,
    "rloo_grouping_and_two_real_gradient_terms":True,
    "known_legacy_embedded_score_bug":"REPRODUCED",
    "generated_score_token_supervised_in_text_ce":True,
    "gold_never_changes_forward_prediction":True}))
