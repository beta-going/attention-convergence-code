#!/usr/bin/env python3
"""Exp① 诊断: 完整性去重 → 汇总 → 排序保持率(按语言正确口径) → stats_clean.csv"""
import csv, sys
from collections import defaultdict
import statistics as st

import os as _os
_os.makedirs("tables", exist_ok=True)
rows = list(csv.DictReader(open("tables/crossinput_raw/stats.csv")))
print(f"[integrity] raw rows: {len(rows)} (expect 160)")
key = lambda r: (r["model"], r["lang"], r["band"], r["idx"])
last, conflicts = {}, []
for i, r in enumerate(rows):
    if key(r) in last and last[key(r)] != r:
        conflicts.append((key(r), i))
    last[key(r)] = r
print(f"[integrity] unique keys: {len(last)}; 重复键且数值不同: {len(conflicts)}")
if conflicts:
    print("!! 停 — 两次运行数据不一致,贴回本输出"); sys.exit(1)
clean = list(last.values())          # 保留最后写入 = 本次过 gate 的运行
with open("tables/stats_clean.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(clean)
print(f"[integrity] stats_clean.csv 写入 {len(clean)} 行\n")

per = defaultdict(list)
for r in clean: per[(r["model"], r["lang"])].append(r)

print("== 1) 每输入 conv+foc 下限(升级 '160/160' 用) ==")
for (m, lang), rs in sorted(per.items()):
    cf = [float(r["conv"]) + float(r["foc"]) for r in rs]
    print(f"{m:<14}{lang}  min={min(cf):.2f}  n>=0.90: {sum(c>=.9 for c in cf)}/{len(cf)}")

print("\n== 2) P 汇总(摘要/§4.1 限定词用) ==")
for (m, lang), rs in sorted(per.items()):
    P = [float(r["P"]) for r in rs]
    print(f"{m:<14}{lang}  P={st.mean(P):.3f}±{st.stdev(P):.3f}  [{min(P):.2f},{max(P):.2f}]")

print("\n== 3) Mistral l* 分布(Table 1 脚注用) ==")
for lang in ("en", "zh"):
    ls = sorted(int(r["lstar"]) for r in per[("Mistral-7B", lang)])
    print(f"Mistral {lang}: l*={ls}  zeros={ls.count(0)}  median={st.median(ls)} (L=32)")

print("\n== 4) 组内 dissociation: lpeak - l*(两种收敛次序) ==")
for (m, lang), rs in sorted(per.items()):
    d = [int(r["lpeak"]) - int(r["lstar"]) for r in rs]
    print(f"{m:<14}{lang}  median={st.median(d):+.0f}  range=[{min(d):+d},{max(d):+d}]")

MODELS = ["Qwen3-8B", "Qwen3-4B", "Llama-3.1-8B", "Mistral-7B"]
CANON = {  # 按语言各用各的 canonical —— 修掉旧 agg 的 en-套-zh bug
 "lstar_over_L": {"en": dict(zip(MODELS, [15/36, 19/36, 13/32, 0.0])),
                  "zh": dict(zip(MODELS, [16/36, 19/36, 14/32, 19/32]))},
 "lpeak":        {"en": dict(zip(MODELS, [2, 2, 13, 12])),
                  "zh": dict(zip(MODELS, [3, 2, 11, 12]))}}
rel = lambda a, b: (a > b) - (a < b)

print("\n== 5) 逐对排序保持率(对每语言 canonical; tie 单列) ==")
for metric, canonL in CANON.items():
    for lang in ("en", "zh"):
        canon = canonL[lang]
        print(f"--- {metric} / {lang} ---")
        for i, a in enumerate(MODELS):
            for b in MODELS[i+1:]:
                tot = ok = tie = 0
                for idx in range(20):
                    va = [r for r in per[(a, lang)] if r["idx"] == str(idx)]
                    vb = [r for r in per[(b, lang)] if r["idx"] == str(idx)]
                    if va and vb:
                        tot += 1
                        t = rel(float(va[0][metric]), float(vb[0][metric]))
                        c = rel(canon[a], canon[b])
                        ok += (t == c); tie += (t == 0 and c != 0)
                print(f"  {a:<14} vs {b:<14}: {ok:>2}/{tot} ({ok/tot:.0%}) "
                      f"ties={tie} canon_rel={c:+d}")

print("\n== 6) band(长度)趋势 ==")
for (m, lang), rs in sorted(per.items()):
    bb = defaultdict(list)
    for r in rs: bb[r["band"]].append(float(r["P"]))
    parts = [f"{b}:{st.mean(bb[b]):.3f}(n={len(bb[b])})"
             for b in ("short", "mid", "long") if bb.get(b)]
    print(f"{m:<14}{lang}  P by band: " + "  ".join(parts))
T_all = [int(r["T"]) for r in clean]
print(f"T range overall: {min(T_all)}-{max(T_all)}")
