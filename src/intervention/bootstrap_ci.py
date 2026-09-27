#!/usr/bin/env python3
"""bootstrap_alpha_ci_v2.py — Table 6 alpha 的 CI (定义完全对齐: 算术均值 + 基线归一 + 配对)。
用法: python3 src/intervention/bootstrap_ci.py results_v8 results_v8_cloud --B 10000 --seed 0 --latex
"""
import argparse, csv, glob, json, os, re, sys
from collections import defaultdict

# is_free / load_cell 与 compute_asymmetry.py 完全相同, 直接 import 或复制
from compute_asymmetry import is_free, load_cell, ALIAS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    import random; rng = random.Random(a.seed)
    rows = []
    for root in a.roots:
        for band in ("early","mid","late"):
            for d in sorted(glob.glob(os.path.join(root, f"*_{band}"))):
                name = os.path.basename(d)
                if re.search(r"INT4|control|self_test|shuffvar", name): continue
                if not os.path.exists(os.path.join(d, "stream.jsonl")): continue
                model = name[:-len(band)-1]
                per = load_cell(d)
                zc, zb = per["uniform_causal"]["zh"], per["baseline"]["zh"]
                ec, eb = per["uniform_causal"]["en"], per["baseline"]["en"]
                iz = sorted(set(zc)&set(zb)); ie = sorted(set(ec)&set(eb))
                if len(iz) < 5 or len(ie) < 5: continue
                zc_v=[zc[i] for i in iz]; zb_v=[zb[i] for i in iz]
                ec_v=[ec[i] for i in ie]; eb_v=[eb[i] for i in ie]
                def alpha_of(zci, zbi, eci, ebi):
                    return (sum(zci)/len(zci)/(sum(zbi)/len(zbi))) / \
                           (sum(eci)/len(eci)/(sum(ebi)/len(ebi)))
                point = alpha_of(zc_v, zb_v, ec_v, eb_v)
                nz, ne, boots = len(zc_v), len(ec_v), []
                for _ in range(a.B):
                    jz = [rng.randrange(nz) for _ in range(nz)]
                    je = [rng.randrange(ne) for _ in range(ne)]
                    boots.append(alpha_of([zc_v[j] for j in jz],[zb_v[j] for j in jz],
                                          [ec_v[j] for j in je],[eb_v[j] for j in je]))
                boots.sort()
                lo, hi = boots[int(0.025*a.B)], boots[min(int(0.975*a.B), a.B-1)]
                excl = "ZH" if lo > 1 else ("EN" if hi < 1 else "-")
                rows.append([ALIAS.get(model, model), band, point, lo, hi, excl, nz, ne])
    rows.sort(key=lambda r: (r[0], ["early","mid","late"].index(r[1])))
    print(f"| Model | Band | alpha | 95% CI | CI排除1? | n |")
    print(f"|---|---|---|---|---|---|")
    for m,b,p,lo,hi,e,nz,ne in rows:
        print(f"| {m} | {b} | {p:.3g} | [{lo:.3g}, {hi:.3g}] | {e} | {nz}/{ne} |")
    import os as _os
    _os.makedirs("tables", exist_ok=True)
    with open("tables/tab9_alpha_ci.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["model","band","alpha","ci_lo","ci_hi","ci_excludes_1"])
        w.writerows([[m,b,f"{p:.6g}",f"{lo:.6g}",f"{hi:.6g}",e] for m,b,p,lo,hi,e,_,_ in rows])
    print("\nsaved: tables/tab9_alpha_ci.csv")
    if a.latex:
        print("\n% LaTeX 行 (Table 6 顺序):")
        for m,b,p,lo,hi,e,nz,ne in rows:
            mm = m.replace("Qwen3-14B", r"Qwen3-14B$^\dagger$")
            print(f"{mm} & {b} & {p:.2f} & [{lo:.2f}, {hi:.2f}] \\\\")

if __name__ == "__main__":
    main()
