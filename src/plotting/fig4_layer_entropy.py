#!/usr/bin/env python3
"""
P3: Layer x config heatmap of mean head entropy (H_norm).
Rows = layers (0..35, zero-padded where absent), cols = 8 configs.
Makes the entropy/ratio dissociation visible in one image.
"""
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt

PROBE_DIR = "data/attention"
FILES = ["Qwen3-8B_zh", "Qwen3-8B_en", "Qwen3-4B_zh", "Qwen3-4B_en",
         "Llama-3.1-8B_zh", "Llama-3.1-8B_en", "Mistral-7B_zh", "Mistral-7B_en"]

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

MAXL = 36
M = np.full((MAXL, len(FILES)), np.nan)
for j, cfg in enumerate(FILES):
    try:
        f = find_file(cfg)
    except FileNotFoundError:
        print(f"skip {cfg}"); continue
    data = json.load(open(f))
    for key in data:
        if not key.startswith("layer_"): continue
        L = int(key.split("_")[1])
        ents = []
        for a in np.array(data[key]):
            s = a.sum(axis=1, keepdims=True)
            p = np.clip(a / np.maximum(s, 1e-12), 1e-12, 1)
            supp = (a > 1e-9).sum(axis=1)
            hn = (-np.sum(p * np.log(p), axis=1)
                  / np.maximum(np.log(np.maximum(supp, 2)), 1e-12))
            ents.append(hn.mean())
        M[L, j] = np.mean(ents)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(FILES)))
ax.set_xticklabels([c.replace("-Instruct", "") for c in FILES], rotation=45,
                   ha="right", fontsize=8)
ax.set_yticks(range(0, MAXL, 4)); ax.set_ylabel("Layer")
ax.set_title("Mean per-head normalized entropy (layer × config)")
plt.colorbar(im, ax=ax, fraction=0.03, label="H_norm")
plt.tight_layout()
import os as _os
_os.makedirs("results/figures", exist_ok=True)
plt.savefig("results/figures/figure_layer_entropy_heatmap.png", dpi=250, bbox_inches="tight")
print("✅ Saved: figures/figure_layer_entropy_heatmap.png")