"""Actual tiny Qwen3.5 cache/gradient/mask/reload tests, CPU only."""
import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration,Qwen3_5TextConfig,Qwen3_5VisionConfig

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from thinking_model import ThinkingPolicy,objective


class Tokenizer:
    pad_token_id=0
    eos_token_id=31
    def encode(self,text,add_special_tokens=False):
        return [24] if text=="</think>" else [7,8]
    def decode(self,tokens,skip_special_tokens=False):
        return "Useful analysis.</think>" if tokens==[23,24] else "incomplete"


def fixture(tied):
    text=Qwen3_5TextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,
       num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
       linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8,
       linear_conv_kernel_dim=4,layer_types=["linear_attention"]*3+["full_attention"],
       max_position_embeddings=4096,tie_word_embeddings=tied,
       rope_parameters={"rope_type":"default","rope_theta":10000.,"mrope_section":[1,1,2],"partial_rotary_factor":1.0})
    vision=Qwen3_5VisionConfig(depth=1,hidden_size=16,intermediate_size=32,num_heads=2,
       out_hidden_size=32,patch_size=2,temporal_patch_size=1,spatial_merge_size=1,num_position_embeddings=16)
    return Qwen3_5Config(text_config=text,vision_config=vision,tie_word_embeddings=tied,
       image_token_id=28,video_token_id=29,vision_start_token_id=30,vision_end_token_id=31)


def main():
    torch.set_num_threads(2);torch.manual_seed(2027)
    proofs=[]
    for tied in (True,False):
        with tempfile.TemporaryDirectory(prefix="thinking_contract_") as folder:
            folder=Path(folder)
            native=Qwen3_5ForConditionalGeneration(fixture(tied)).to(torch.bfloat16).eval()
            native.save_pretrained(folder)
            ids=torch.tensor([[10,11,12,13]])
            with torch.no_grad():
                whole=native(input_ids=ids,use_cache=False,logits_to_keep=1).logits[:,-1].float()
                prefix=native.model.language_model(input_ids=ids[:,:3],use_cache=True)
                step=native.model.language_model(input_ids=ids[:,3:],past_key_values=prefix.past_key_values,
                           attention_mask=torch.ones_like(ids),use_cache=True)
                cached=native.lm_head(step.last_hidden_state[:,-1]).float()
                delta=float((whole-cached).abs().max())
                # BF16 cache and non-cache matrix shapes can round differently.
                # Compare at BF16 precision; also require identical decoding argmax.
                assert delta<=.01 and torch.equal(whole.argmax(-1),cached.argmax(-1)),delta
            del native
            model=ThinkingPolicy(folder,list(range(6)),native_check=True)
            rows=[{"ContextID":"a","Compound":"x y","Context":"x y","component":0,
                   "style":"free","gold":1.,"prompt_ids":[10,11,12],"cot_ids":[23,24],
                   "cot_text":"Useful analysis.</think>","stop_reason":"think_end","fallback":False},
                  {"ContextID":"b","Compound":"x y","Context":"x y","component":0,
                   "style":"free","gold":4.,"prompt_ids":[10,11,13],"cot_ids":[23,24],
                   "cot_text":"Useful analysis.</think>","stop_reason":"think_end","fallback":False}]
            model.eval()
            out=model.batch(rows,Tokenizer())
            assert out["cot_target_count"]==[2,2]
            parts={"score":out["score_ce"].mean(),"cot":out["cot_ce"].mean()}
            loss,terms=objective(out,rows,.2,.5)
            parts.update(pair=terms["pair"],huber=terms["huber"])
            params=[p for p in model.parameters() if p.requires_grad]
            energies={}
            for name,part in parts.items():
                grads=torch.autograd.grad(part,params,retain_graph=True,allow_unused=True)
                energies[name]=sum(float(g.float().square().sum()) for g in grads if g is not None)
                assert energies[name]>0,(name,energies)
            loss.backward()
            assert model.lm_head.weight.grad is None
            assert all(p.grad is None for name,p in model.backbone.named_parameters() if "lora_" not in name)
            torch.optim.AdamW(params,lr=1e-4).step();model.zero_grad(set_to_none=True)
            with torch.inference_mode():ref=model.batch(rows,Tokenizer(),False)["prediction"]
            model.save(folder/"best")
            reloaded=ThinkingPolicy(folder,list(range(6)),folder/"best",trainable=False).eval()
            with torch.inference_mode():actual=reloaded.batch(rows,Tokenizer(),False)["prediction"]
            assert torch.equal(ref,actual),(ref,actual)
            # RNG state must survive sampled native decoding, successful or not.
            before=torch.get_rng_state().clone()
            first=reloaded.generate_one(rows[0],Tokenizer(),4,2027,0)
            assert torch.equal(before,torch.get_rng_state())
            assert first==dict(reloaded.generate_one(rows[0],Tokenizer(),4,2027,0),
                               generation_seconds=first["generation_seconds"])
            equal=[dict(r,gold=2.) for r in rows]
            _,ties=objective(reloaded.batch(equal,Tokenizer(),False),equal,.2,0)
            assert ties["eligible_pair"]==0 and float(ties["pair"])==0
            proofs.append({"tied":tied,"native_cpu_logits_diff":0,"cache_bf16_delta":delta,
                           "objective_gradient_energies":energies,"mask_counts":[2,2],
                           "reload_diff":0,"sampling_rng_isolated":True})
    print(json.dumps({"status":"PASS","scope":"tiny random CPU fixtures, not real-model quality",
                      "fixtures":proofs}),flush=True)


if __name__=="__main__":main()
