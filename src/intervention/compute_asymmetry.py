#!/usr/bin/env python3
"""compute_asymmetry.py — Table 6 (tab:asym_prop) 官方 provenance。

alpha = (mean_ZH[ppl|UC] / mean_ZH[ppl|baseline])
      / (mean_EN[ppl|UC] / mean_EN[ppl|baseline])
mean = free-form 句级 ppl 的算术平均 (每语言 10 句);
baseline 与 UC 取自同一 band 目录、同一 run。
已验证: 33/33 复现 main_v29.tex Table 6。
用法: python3 src/compute_asymmetry.py tables/intervention_raw tables/intervention_raw_cloud
"""
import csv, glob, json, os, re, sys
from collections import defaultdict

ALIAS = {"Mixtral-8x7B-Instruct": "Mixtral-8x7B"}
PAPER = {
 ("Qwen3-0.6B","early"):3.71,("Qwen3-0.6B","mid"):0.176,("Qwen3-0.6B","late"):0.625,
 ("Qwen3-1.7B","early"):0.151,("Qwen3-1.7B","mid"):0.299,("Qwen3-1.7B","late"):0.636,
 ("Qwen3-4B","early"):0.639,("Qwen3-4B","mid"):1.23,("Qwen3-4B","late"):1.02,
 ("Qwen3-8B","early"):2.00,("Qwen3-8B","mid"):0.380,("Qwen3-8B","late"):0.838,
 ("Qwen3-8B-Base","early"):2.71,("Qwen3-8B-Base","mid"):0.593,("Qwen3-8B-Base","late"):1.25,
 ("Qwen3-14B","early"):2.16,("Qwen3-14B","mid"):0.578,("Qwen3-14B","late"):1.07,
 ("Llama-3.1-8B","early"):1.45,("Llama-3.1-8B","mid"):6.73,("Llama-3.1-8B","late"):2.60,
 ("Llama-3.1-8B-Base","early"):3.90,("Llama-3.1-8B-Base","mid"):78.8,("Llama-3.1-8B-Base","late"):2.46,
 ("Mistral-7B","early"):12.1,("Mistral-7B","mid"):3.94,("Mistral-7B","late"):1.62,
 ("InternLM3-8B-Instruct","early"):2.62,("InternLM3-8B-Instruct","mid"):1.38,("InternLM3-8B-Instruct","late"):0.922,
 ("Mixtral-8x7B","early"):5.25,("Mixtral-8x7B","mid"):1.78,("Mixtral-8x7B","late"):1.20,
}

def is_free(r):
    t = str(r.get("type", "")).lower()
    if "free" in t: return True
    if "cloze" in t or "gen" in t: return False
    return ("ppl" in r) and ("options" not in r)

def load_cell(d):
    per = defaultdict(lambda: defaultdict(dict))   # mode -> lang -> {idx: ppl}
    import os as _os
    _os.makedirs("tables", exist_ok=True)
    for line in open(os.path.join(d, "stream.jsonl")):
        r = json.loads(line)
        if not is_free(r): continue
        p = r.get("ppl")
        if p is None or not (0 < float(p) < 1e12): continue
        mode = str(r.get("mode", "")).replace("-", "_").lower()
        lang = {"zh":"zh","chinese":"zh","cn":"zh",
                "en":"en","english":"en"}.get(str(r.get("lang","")).lower())
        if lang not in ("zh","en") or mode not in ("baseline","uniform_causal"): continue
        per[mode][lang][r.get("sample_idx", r.get("text",""))] = float(p)
    return per

def amean(v, idxs): return sum(v[i] for i in idxs) / len(idxs)

def main(roots):
    rows = []
    for root in roots:
        for band in ("early","mid","late"):
            for d in sorted(glob.glob(os.path.join(root, f"*_{band}"))):
                name = os.path.basename(d)
                if re.search(r"INT4|control|self_test|shuffvar", name): continue
                if not os.path.exists(os.path.join(d, "stream.jsonl")): continue
                model = name[:-len(band)-1]
                per = load_cell(d)
                zc, zb = per["uniform_causal"]["zh"], per["baseline"]["zh"]
                ec, eb = per["uniform_causal"]["en"], per["baseline"]["en"]
                iz = sorted(set(zc) & set(zb)); ie = sorted(set(ec) & set(eb))
                if len(iz) < 5 or len(ie) < 5:
                    print(f"[skip] {name}: ZH={len(iz)} EN={len(ie)}", file=sys.stderr); continue
                alpha = (amean(zc,iz)/amean(zb,iz)) / (amean(ec,ie)/amean(eb,ie))
                rows.append([ALIAS.get(model,model), band, f"{alpha:.6g}", len(iz), len(ie)])
    with open("tables/tab6_asymmetry_recomputed.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["model","band","alpha","n_zh","n_en"]); w.writerows(rows)
    print(f"saved: tables/tab6_asymmetry_recomputed.csv ({len(rows)} cells)\n与 Table 6 核对:")
    ok = tot = 0
    for m,b,a,nz,ne in rows:
        k = (m,b)
        if k in PAPER:
            tot += 1; hit = 0.95 <= float(a)/PAPER[k] <= 1.05; ok += hit
            print(f"  {m:<24}{b:<6} calc={float(a):>8.3g} paper={PAPER[k]:>8.3g} {'OK' if hit else 'MISMATCH'}")
    print(f"==> {ok}/{tot} 匹配")
    print("(可选) 让 plot_asym_prop.py 改读 asymmetry_table.csv, 消灭 'data from tex' 注释")

if __name__ == "__main__":
    roots = sys.argv[1:]
    if not roots: sys.exit("用法: python3 compute_asymmetry.py results_v8 results_v8_cloud")
    main(roots)
