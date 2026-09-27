# src/plot_alpha_grid.py —— 全协议 α 网格:11 配置 x 3 带 x 3 干预模式
# alpha = (ZH_int/ZH_base) / (EN_int/EN_base);  >1 => ZH 更脆, <1 => EN 更脆
import json, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = ["Qwen3-0.6B","Qwen3-1.7B","Qwen3-4B","Qwen3-8B","Qwen3-8B-Base","Qwen3-14B",
          "Llama-3.1-8B","Llama-3.1-8B-Base","Mistral-7B","InternLM3-8B-Instruct",
          "Mixtral-8x7B-Instruct"]
ROOT   = lambda m: "results_v8_cloud" if m.startswith("Mixtral") else "results_v8"
BANDS  = ["early","mid","late"]
MODES  = [("uniform_causal","UC","-","#d62728"),
          ("uniform_global","UG","--","#1f77b4"),
          ("shuffle_rows","SH",":","#7f7f7f")]

def get_ppl(d, mode, lang):
    try:
        v = d[mode][f"generation_{lang}"]["avg_ppl"]
    except (KeyError, TypeError):
        return None
    if v is None or not math.isfinite(v) or v <= 0 or v > 1e6:
        return None                      # 溢出/非法 -> 缺格
    return float(v)

def alpha(model, band, mode):
    p = f"{ROOT(model)}/{model}_{band}/intervention_results.json"
    if not os.path.exists(p):
        return None, "missing"
    d = json.load(open(p))
    eb, zb = get_ppl(d,"baseline","en"), get_ppl(d,"baseline","zh")
    ei, zi = get_ppl(d,mode,"en"),       get_ppl(d,mode,"zh")
    if None in (eb, zb, ei, zi):
        return None, "overflow/missing"
    return (zi/zb)/(ei/eb), "ok"

rows = {}
print(f"{'model':<24}{'band':<7}{'UC':>10}{'UG':>10}{'SH':>10}")
for m in MODELS:
    for b in BANDS:
        line = f"{m:<24}{b:<7}"
        for mode,_,_,_ in MODES:
            a, st = alpha(m, b, mode)
            rows[(m,b,mode)] = (a, st)
            line += f"{a:>10.3f}" if a else f"{'>':>4}{st[:5]:>6}"
        print(line)
    print()

os.makedirs("tables", exist_ok=True)
with open("tables/table_alpha_grid.tex","w") as f:
    f.write("\\begin{tabular}{llccc}\n\\toprule\nModel & Band & UC & UG & Shuffle\\\\\n\\midrule\n")
    for m in MODELS:
        for bi, b in enumerate(BANDS):
            cells = []
            for mode,_,_,_ in MODES:
                a, st = rows[(m,b,mode)]
                cells.append(f"{a:.2f}" if a else "--")
            f.write((m if bi==0 else "") + f" & {b} & " + " & ".join(cells) + "\\\\\n")
        f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("LaTeX fragment -> tables/table_alpha_grid.tex")

fig, axes = plt.subplots(4, 3, figsize=(15, 17), sharex=True, sharey=True)
for idx, m in enumerate(MODELS):
    ax = axes[idx//3][idx%3]
    for mode, short, ls, color in MODES:
        ys = [np.log10(a) if (a := rows[(m,b,mode)][0]) else np.nan for b in BANDS]
        ax.plot([0,1,2], ys, ls, color=color, marker="o", lw=2, ms=6, label=short)
    a0 = rows[(m,"early","uniform_causal")][0]
    if a0:
        ax.annotate(f"{a0:.4g}", (0, np.log10(a0)), textcoords="offset points",
                    xytext=(0,8), ha="center", fontsize=8, color="#d62728")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axhline(np.log10(1.2), color="gray", ls=":", lw=0.8)
    ax.axhline(np.log10(0.83), color="gray", ls=":", lw=0.8)
    ax.set_title(m, fontsize=10)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["early","mid","late"], fontsize=9)
    if idx % 3 == 0: ax.set_ylabel(r"$\log_{10}\alpha$", fontsize=9)
axes[3][2].axis("off")
h, l = axes[0][0].get_legend_handles_labels()
axes[3][2].legend(h, l, loc="center", fontsize=12, title="mode")
plt.tight_layout()
plt.savefig("figures/figure_alpha_grid.png", dpi=150, bbox_inches="tight")
print("figure -> figures/figure_alpha_grid.png")
