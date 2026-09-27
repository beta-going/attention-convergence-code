#!/usr/bin/env python3
"""
P2: Head-level entropy distribution per model-language config.
Produces tables/table_entropy_dist.tex (booktabs, paste-ready) + console output.
Metrics per config: fraction of heads with H_norm<0.3 (converged),
0.3-0.7 (focused), >0.7 (diffuse); median H_norm; max head entropy.
"""
import glob, json, os
import numpy as np

PROBE_DIR = "data/attention"
FILES = ["Llama-3.1-8B_zh", "Llama-3.1-8B_en",
         "Qwen3-8B_zh", "Qwen3-8B_en",
         "Qwen3-4B_zh", "Qwen3-4B_en",
         "Mistral-7B_zh", "Mistral-7B_en"]

def find_file(pattern, probe_dir=PROBE_DIR):
    base = pattern.replace(".json", "")
    candidates = [
        os.path.join(probe_dir, f"consistency_{base}.json"),
        os.path.join(probe_dir, f"*{base}*.json"),
        os.path.join(probe_dir, pattern),
    ]
    seen, matches = set(), []
    for cand in candidates:
        for m in glob.glob(cand):
            if m not in seen:
                seen.add(m); matches.append(m)
    pref = [m for m in matches if "consistency_" in os.path.basename(m)]
    matches = pref or matches
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {probe_dir}")
    return sorted(matches)[0]

lines = []
for fn in FILES:
    try:
        f = find_file(fn)
    except FileNotFoundError:
        print(f"skip {fn}"); continue
    cfg = fn
    import os as _os
    _os.makedirs("tables", exist_ok=True)
    data = json.load(open(f))
    ents = []
    for key in data:
        if not key.startswith("layer_"): continue
        for a in np.array(data[key]):
            s = a.sum(axis=1, keepdims=True)
            p = np.clip(a / np.maximum(s, 1e-12), 1e-12, 1)
            supp = (a > 1e-9).sum(axis=1)
            hn = (-np.sum(p * np.log(p), axis=1)
                  / np.maximum(np.log(np.maximum(supp, 2)), 1e-12))
            ents.append(hn.mean())
    ents = np.array(ents)
    conv = (ents < 0.3).mean(); foc = ((ents >= 0.3) & (ents <= 0.7)).mean()
    diff = (ents > 0.7).mean()
    lines.append((cfg, conv, foc, diff, np.median(ents), ents.max()))

print(f"{'Config':<22}{'%conv':>7}{'%focus':>8}{'%diff':>7}{'medH':>7}{'maxH':>7}")
for c, cv, fo, di, me, mx in lines:
    print(f"{c:<22}{cv:>7.2f}{fo:>8.2f}{di:>7.2f}{me:>7.2f}{mx:>7.2f}")

with open("tables/table_entropy_dist.tex", "w") as fh:
    fh.write("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
             "\\textbf{Config} & \\textbf{\\% converged} & \\textbf{\\% focused} "
             "& \\textbf{\\% diffuse} & \\textbf{median} & \\textbf{max} \\\\\n"
             "\\midrule\n")
    for c, cv, fo, di, me, mx in lines:
        fh.write(f"{c} & {cv:.2f} & {fo:.2f} & {di:.2f} & {me:.2f} & {mx:.2f} \\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print("✅ Saved: tables/table_entropy_dist.tex")