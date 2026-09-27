#!/usr/bin/env python3
# 在 src/ 目录下运行; 复用 extract_attention.py 的文本/路径,分词调用逐字一致
import math, numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from extract_attention import MODEL_CONFIGS          # ← 你的文本+本地路径,不重写

def bands(L):                                        # §3.2 的 25/50/25
    e, m = math.ceil(0.25*L), math.floor(0.75*L)
    return [("early",0,e),("mid",e,m),("late",m,L)]

def rho_row(A):    # 变体V1: 每行→该行自身均值(行内均匀度)
    bar = np.repeat(A.mean(1,keepdims=True), A.shape[1], 1)
    return 1 - np.linalg.norm(A-bar)/np.linalg.norm(A)

def rho_prof(A):   # 变体V2: 每行→所有查询的平均剖面(查询无关度)
    bar = np.repeat(A.mean(0,keepdims=True), A.shape[0], 0)
    return 1 - np.linalg.norm(A-bar)/np.linalg.norm(A)

rows = []
for cfg in MODEL_CONFIGS:
    tok = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.float16, device_map="auto",
        output_attentions=True, attn_implementation="eager",
        trust_remote_code=True)                      # 加载参数与 extract_attention 一致
    model.eval()
    bl = bands(model.config.num_hidden_layers)
    for lang, text in cfg["texts"].items():
        inputs = tok(text, return_tensors="pt").to(model.device)   # 与原管线同一调用
        with torch.no_grad():
            out = model(**inputs)
        for li, A in enumerate(out.attentions):
            band = next(b for b, lo, hi in bl if lo <= li < hi)
            Ah = A[0].float().cpu().numpy()
            for h in range(Ah.shape[0]):
                rows.append(dict(config=f'{cfg["name"]}_{lang}', band=band,
                                 layer=li, head=h,
                                 rho_row=rho_row(Ah[h]), rho_prof=rho_prof(Ah[h])))
        print(cfg["name"], lang, "T =", inputs["input_ids"].shape[1], "ok")
    del model; torch.cuda.empty_cache()

df = pd.DataFrame(rows)
import os as _os
_os.makedirs("tables", exist_ok=True)
df.to_csv("tables/posind_per_head.csv.gz", index=False, compression="gzip")
agg = df.groupby(["config","band"])[["rho_row","rho_prof"]].mean().round(4).reset_index()
agg.to_csv("tables/tab3_posind_by_band.csv", index=False)
print(agg.pivot(index="config", columns="band", values="rho_prof"))
print(agg.pivot(index="config", columns="band", values="rho_row"))
