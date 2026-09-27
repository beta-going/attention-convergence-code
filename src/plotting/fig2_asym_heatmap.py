#!/usr/bin/env python3
"""figure_asym_heatmap.py — 11 configs × 3 bands, color = log10(alpha).
Data source: tables/tab6_asymmetry_recomputed.csv (recomputed from raw logs)
Output: figure_asym_heatmap.pdf (also .png for quick preview)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---- 数据: tables/tab6_asymmetry_recomputed.csv (从原始日志复算, 已验证==论文Table 6) ----
import csv
_order = ["Qwen3-0.6B","Qwen3-1.7B","Qwen3-4B","Qwen3-8B","Qwen3-8B-Base","Qwen3-14B",
          "Llama-3.1-8B","Llama-3.1-8B-Base","Mistral-7B","InternLM3-8B-Instruct","Mixtral-8x7B"]
_label = {"Qwen3-14B": r"Qwen3-14B$^\dagger$", "InternLM3-8B-Instruct": "InternLM3-8B"}
_alpha = {(r["model"], r["band"]): float(r["alpha"])
          for r in csv.DictReader(open("tables/tab6_asymmetry_recomputed.csv"))}
ROWS = [(_label.get(m, m), _alpha[(m,"early")], _alpha[(m,"mid")], _alpha[(m,"late")]) for m in _order]

labels = [r[0] for r in ROWS]
a = np.array([[r[1], r[2], r[3]] for r in ROWS])
lg = np.log10(a)                      # color value
VMAX = 2.0                            # clamp at +-2 decades
lg_c = np.clip(lg, -VMAX, VMAX)
norm = TwoSlopeNorm(vmin=-VMAX, vcenter=0.0, vmax=VMAX)

fig, ax = plt.subplots(figsize=(4.6, 5.2))
im = ax.imshow(lg_c, cmap="RdBu_r", norm=norm, aspect="auto")

# annotate raw alpha; bold white when strongly asymmetric
for i in range(lg.shape[0]):
    for j in range(3):
        strong = abs(lg[i, j]) > np.log10(1.2)
        ax.text(j, i, f"{a[i, j]:#.3g}", ha="center", va="center",
                fontsize=9,
                color="white" if abs(lg[i, j]) > 0.7 else "black",
                fontweight="bold" if strong else "normal")

ax.set_xticks(range(3), ["Early", "Mid", "Late"], fontsize=10)
ax.set_yticks(range(len(labels)), labels, fontsize=9)
ax.set_title(r"Cross-lingual asymmetry  $\log_{10}\alpha$"
             "\n" r"(uniform-causal, proportional split)", fontsize=10)

# family separators (Qwen3 | Llama | others)
ax.axhline(5.5, color="black", lw=1.2)
ax.axhline(6.5, color="black", lw=1.2)

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label(r"$\log_{10}\alpha$  (<0: EN-fragile,  >0: ZH-fragile)", fontsize=9)
cbar.set_ticks([-2, -1, 0, 1, 2])

fig.tight_layout()

# make dir if there is no ./results/figures/
import os
os.makedirs("./results/figures/", exist_ok=True)

fig.savefig("./results/figures/figure_asym_heatmap.pdf", bbox_inches="tight")
fig.savefig("./results/figures/figure_asym_heatmap.png", dpi=300, bbox_inches="tight")
print("saved: ./results/figures/figure_asym_heatmap.pdf / .png")
