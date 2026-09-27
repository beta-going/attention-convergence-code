#!/usr/bin/env python3
"""validate_alpha.py — 定位 Table 6 α 的真实口径
用法: python3 validate_alpha.py results_v8 results_v8_cloud"""
import glob, json, math, os, re, sys
from collections import Counter
from statistics import median

PAPER = {  # tab:asym_prop 原值
 ("Qwen3-0.6B","early"):3.71,("Qwen3-0.6B","mid"):0.18,("Qwen3-0.6B","late"):0.63,
 ("Qwen3-1.7B","early"):0.15,("Qwen3-1.7B","mid"):0.30,("Qwen3-1.7B","late"):0.64,
 ("Qwen3-4B","early"):0.64,("Qwen3-4B","mid"):1.23,("Qwen3-4B","late"):1.02,
 ("Qwen3-8B","early"):2.00,("Qwen3-8B","mid"):0.38,("Qwen3-8B","late"):0.84,
 ("Qwen3-8B-Base","early"):2.71,("Qwen3-8B-Base","mid"):0.59,("Qwen3-8B-Base","late"):1.25,
 ("Qwen3-14B","early"):2.16,("Qwen3-14B","mid"):0.58,("Qwen3-14B","late"):1.07,
 ("Llama-3.1-8B","early"):1.45,("Llama-3.1-8B","mid"):6.73,("Llama-3.1-8B","late"):2.60,
 ("Llama-3.1-8B-Base","early"):3.90,("Llama-3.1-8B-Base","mid"):78.8,("Llama-3.1-8B-Base","late"):2.47,
 ("Mistral-7B","early"):12.14,("Mistral-7B","mid"):3.94,("Mistral-7B","late"):1.62,
 ("InternLM3-8B-Instruct","early"):2.62,("InternLM3-8B-Instruct","mid"):1.38,("InternLM3-8B-Instruct","late"):0.92,
 ("Mixtral-8x7B","early"):5.25,("Mixtral-8x7B","mid"):1.78,("Mixtral-8x7B","late"):1.20,
}
ALIAS = {"Mixtral-8x7B-Instruct":"Mixtral-8x7B"}
MODE_KEYS=["mode","condition","intervention","intervention_mode"]
LANG_KEYS=["language","lang"]
BASE_VALS={"baseline","base","none","no_intervention","original"}
HOWS=["pooled","geo","arith","median"]

def nm(v): return str(v).replace("-","_").lower()
def nl(v):
    v=str(v).lower()
    return {"zh":"zh","chinese":"zh","cn":"zh","en":"en","english":"en"}.get(v,v)

def sent_ppl(r):
    for k in ("ppl","free_form_ppl","sentence_ppl","perplexity"):
        if k in r:
            try: return float(r[k]), r.get("n_tokens")
            except: pass
    t=r.get("n_tokens")
    for k in ("logprob_sum","total_logprob","sum_logprob"):
        if k in r and t: return math.exp(-float(r[k])/t), t
    return None,None

def agg(p,toks,how):
    if how=="arith": return sum(p)/len(p)
    if how=="median": return median(p)
    if how=="pooled" and toks and all(x is not None for x in toks):
        return math.exp(sum(t*math.log(x) for x,t in zip(p,toks))/sum(toks))
    return math.exp(sum(math.log(x) for x in p)/len(p))   # geo（pooled 的回退）

def main(roots):
    dirs=[]
    for root in roots:
        dirs+=sorted(glob.glob(os.path.join(root,"*_early"))
                    +glob.glob(os.path.join(root,"*_mid"))
                    +glob.glob(os.path.join(root,"*_late")))
    dirs=[d for d in dirs if os.path.exists(os.path.join(d,"stream.jsonl"))
          and not re.search(r"INT4|control|self_test|shuffvar",os.path.basename(d))]
    print("="*110); print("[1] schema 侦察 (前2个目录)")
    for d in dirs[:2]:
        recs=[json.loads(l) for l in open(os.path.join(d,"stream.jsonl"))]
        mk=next((k for k in MODE_KEYS if sum(k in r for r in recs)>len(recs)/2),None)
        lk=next((k for k in LANG_KEYS if sum(k in r for r in recs)>len(recs)/2),None)
        print(f"\n{os.path.basename(d)}: {len(recs)} 条; 模式键={mk}, 语言键={lk}")
        print("  键全集:", sorted({k for r in recs for k in r}))
        if mk: print("  模式取值:", Counter(nm(r.get(mk)) for r in recs))
        if lk: print("  语言取值:", Counter(nl(r.get(lk)) for r in recs))
    print("\n"+"="*110); print("[2] 8 种候选 α × 论文值 (✓=比率0.95–1.05)")
    cand=["A_pooled","A_geo","A_arith","A_median","B_pooled","B_geo","B_arith","B_median"]
    hits={c:0 for c in cand}; rows=[]
    for d in dirs:
        model,band=os.path.basename(d).rsplit("_",1)
        pk=(ALIAS.get(model,model),band)
        if pk not in PAPER: continue
        recs=[json.loads(l) for l in open(os.path.join(d,"stream.jsonl"))]
        mk=next((k for k in MODE_KEYS if sum(k in r for r in recs)>len(recs)/2),None)
        lk=next((k for k in LANG_KEYS if sum(k in r for r in recs)>len(recs)/2),None)
        if not mk or not lk: continue
        def collect(mode):
            o={"en":[],"zh":[]}; t={"en":[],"zh":[]}
            for r in recs:
                if nm(r.get(mk))!=mode: continue
                p,tt=sent_ppl(r)
                if p is None or not (0<p<1e12): continue
                L=nl(r.get(lk))
                if L in o: o[L].append(p); t[L].append(tt)
            return o,t
        uc,uc_t=collect("uniform_causal"); bs,bs_t=collect("baseline")
        if len(uc["en"])<5 or len(uc["zh"])<5:
            print(f"[skip] {pk[0]} {band}: EN={len(uc['en'])} ZH={len(uc['zh'])}"); continue
        row={"cell":f"{pk[0]} {band}","paper":PAPER[pk]}
        for h in HOWS:
            a=agg(uc["zh"],uc_t["zh"],h)/agg(uc["en"],uc_t["en"],h)
            row["A_"+h]=a
            row["B_"+h]=None
            if len(bs["en"])>=5 and len(bs["zh"])>=5:
                row["B_"+h]=(agg(uc["zh"],uc_t["zh"],h)/agg(bs["zh"],bs_t["zh"],h))/ \
                            (agg(uc["en"],uc_t["en"],h)/agg(bs["en"],bs_t["en"],h))
        rows.append(row)
        for c in cand:
            v=row.get(c)
            if v and 0.95<=v/row["paper"]<=1.05: hits[c]+=1
    print(f"{'cell':>24} {'paper':>8} "+" ".join(f"{c:>10}" for c in cand))
    for row in rows:
        cs=[f"{row['cell']:>24}",f"{row['paper']:>8.3g}"]
        for c in cand:
            v=row.get(c)
            cs.append("      —   " if v is None else
                     (f"{v:>9.3g}✓" if 0.95<=v/row['paper']<=1.05 else f"{v:>10.3g}"))
        print(" ".join(cs))
    print("\n命中统计 (共33格):");  [print(f"  {c}: {hits[c]}") for c in cand]
    best=max(cand,key=lambda c:hits[c])
    print(f"\n==> 最佳候选: {best} ({hits[best]}/33)。若无候选 ≥30, 把 [1] recon 输出原样发我。")

if __name__ == "__main__":
    roots = sys.argv[1:]
    if not roots:
        sys.exit("用法: python3 validate_alpha.py results_v8 [results_v8_cloud ...]"
                 "  （无参数 = 扫 0 个目录 = 空表，就是你刚看到的现象）")
    main(roots)

