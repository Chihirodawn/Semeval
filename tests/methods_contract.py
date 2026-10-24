"""CPU tiny-model contracts for actual LM projection, gradients and REAL math."""
import json
import sys
import tempfile
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoModelForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
from regression_methods.core import NumericPolicy, rloo_advantages, real_loss, soft_digit_ce


class Tokens:
    pad_token_id=0
    eos_token_id=2
    def encode(self,text,add_special_tokens=False): return [7,8]
    def decode(self,ids,skip_special_tokens=False): return str(ids)


torch.set_num_threads(2)
torch.manual_seed(2027)
advantages=rloo_advantages(torch.tensor([[-1.,-4.],[-2.,-2.]]))
assert torch.equal(advantages,torch.tensor([[1.,-1.],[0.,0.]]))
pred=torch.tensor([1.,4.],requires_grad=True)
lp=torch.tensor([-2.,-3.],requires_grad=True)
loss,_=real_loss(pred,torch.tensor([3.,3.]),lp)
gp,gl=torch.autograd.grad(loss,(pred,lp))
assert torch.equal(gl,torch.tensor([.5,-.5])) and gp.abs().sum()>0
digits=list(range(10,16))
probs=torch.randn(3,128).log_softmax(-1)
targets=torch.tensor([0.,2.25,5.])
ce=soft_digit_ce(probs,targets,digits)
assert torch.allclose(ce,torch.stack([-probs[0,10],-.75*probs[1,12]-.25*probs[1,13],-probs[2,15]]))

with tempfile.TemporaryDirectory(prefix="task5_numeric_contract_") as temp:
    root=Path(temp)
    cfg=Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                    num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=256,
                    tie_word_embeddings=True,use_sliding_window=False,sliding_window=None,pad_token_id=0,eos_token_id=2)
    cfg._attn_implementation="sdpa"
    native=Qwen3ForCausalLM(cfg)
    native.save_pretrained(root / "base")
    # Match actual from_pretrained loading: .to(BF16) would also cast RoPE's
    # FP32 buffers, while the real BF16 loader correctly preserves them.
    native=AutoModelForCausalLM.from_pretrained(root / "base",torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa",local_files_only=True).eval()
    policy=NumericPolicy(root / "base",digits).eval()
    ids=torch.tensor([[1,30,31]])
    with torch.no_grad():
        expected=native(input_ids=ids).logits[:,-1].float()
        hidden=policy.backbone(input_ids=ids).last_hidden_state[:,-1]
        actual=policy.logits(hidden)
    assert torch.equal(expected,actual),float((expected-actual).abs().max())
    logits=torch.zeros(1,128)
    numeric,mass,_,_=policy.scores(logits)
    assert abs(float(numeric)-15/128)<1e-6 and abs(float(mass)-6/128)<1e-6
    records=[{"ContextID":"x","component":0,"gold":2.3,"prompt_ids":[1,30,31],"cot_ids":[20,21,2]}]
    tokenizer=Tokens()
    policy.train()
    out=policy.batch(records,tokenizer,reasoning=True,cot_ce=True)
    total=(out['prediction']-out['gold']).square().mean()+out['cot_ce'].mean()
    total.backward()
    parameters=[p for p in policy.parameters() if p.requires_grad]
    assert sum(float(p.grad.float().square().sum()) for p in parameters if p.grad is not None)>0
    assert all(not p.requires_grad for n,p in policy.backbone.named_parameters() if 'lora_' not in n)
    torch.optim.SGD(parameters,lr=1e-3).step()
    policy.eval()
    with torch.no_grad():before=policy.batch(records,tokenizer,reasoning=True)['prediction']
    changed=[dict(records[0],gold=4.9)]
    with torch.no_grad():no_leak=policy.batch(changed,tokenizer,reasoning=True)['prediction']
    assert torch.equal(before,no_leak)
    policy.save(root / 'adapter')
    fresh=NumericPolicy(root/'base',digits,root/'adapter',trainable=False).eval()
    with torch.no_grad():after=fresh.batch(records,tokenizer,reasoning=True)['prediction']
    assert torch.equal(before,after)
    generated=policy.generate_reasoning([dict(records[0],cot_ids=[])],tokenizer,max_new_tokens=4,sample=True)
    with torch.no_grad():scored=policy.batch(generated,tokenizer,reasoning=True,policy=True)
    delta=abs(float(scored['trajectory_logprob'][0])-generated[0]['sampled_logprob'])
    assert delta<0.02,delta
    print(json.dumps({'status':'PASS','scope':'CPU tiny random Qwen3, not quality/GPU score',
          'native_lm_projection_exact':True,'full_vocab_no_digit_renormalization':True,
          'continuous_soft_ce_endpoints':True,'rloo_and_both_real_gradient_terms':True,
          'raft_cot_gradient_nonzero_base_frozen':True,'labels_do_not_change_prediction':True,
          'fresh_adapter_reload_diff':float((before-after).abs().max()),'cached_vs_teacher_trajectory_logprob_diff':delta}))
