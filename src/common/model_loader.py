import os, yaml, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_cfg(path=None):
    path = path or os.environ.get("AE_MODELS_CONFIG", "configs/models.yaml")
    return yaml.safe_load(open(path))

def resolve(key, cfg):
    m = cfg["models"][key]
    return m.get("local_path") or m["hf_id"]   # 本地路径优先,为空回退HF hub

def load(key, cfg=None, for_attention=False):
    cfg = cfg or load_cfg(); d = cfg["defaults"]; m = cfg["models"][key]
    kw = dict(torch_dtype=getattr(torch, d["dtype"]),
              device_map=d["device_map"], trust_remote_code=True)
    if m.get("quantization"): kw.update(**m["quantization"])
    if for_attention: kw["attn_implementation"] = d["attn_implementation"]; kw["output_attentions"] = True
    tok = AutoTokenizer.from_pretrained(resolve(key, cfg), trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(resolve(key, cfg), **kw)
    return mdl.eval(), tok
