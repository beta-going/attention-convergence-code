"""
P1 — Punctuation columns attract attention.
Top: column-mass profiles (mean=1) of the four Appendix-G heads.
Bottom: per-model punct/non-punct column-mass ratio over ALL layer-heads
        (+ L>=1 subset). BOS column excluded everywhere.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.gridspec import GridSpec
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../common'))
import probe_utils as U

PANEL_HEADS = [
    ("Qwen3-8B_zh", 35, 14, "(a) Qwen3-8B-zh L35·H14"),
    ("Qwen3-8B_en", 35, 14, "(b) Qwen3-8B-en L35·H14"),
    ("Mistral-7B_zh", 0, 16, "(c) Mistral-7B-zh L0·H16"),
    ("Mistral-7B_zh", 0, 19, "(d) Mistral-7B-zh L0·H19"),
]
AGG_MODELS = ["Qwen3-8B_zh", "Qwen3-8B_en", "Mistral-7B_zh", "Llama-3.1-8B_zh"]

def punct_mask(seq, punct):
    is_p = np.zeros(seq, bool)
    for p in punct:
        if 1 <= p < seq:
            is_p[p] = True
    return is_p

def col_ratio(attn, is_p):
    m = U.column_mass(attn)
    valid = np.arange(len(m)) >= 1
    pm = m[is_p & valid]; nm = m[~is_p & valid]
    if pm.size == 0 or nm.size == 0:
        return float("nan")
    return float(pm.mean() / max(nm.mean(), 1e-12))

def main():
    fig = plt.figure(figsize=(20, 9))
    gs = GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.30,
                  left=0.05, right=0.97, top=0.90, bottom=0.09,
                  height_ratios=[1.1, 1])

    # ---------- top: profiles ----------
    for idx, (pat, L, H, title) in enumerate(PANEL_HEADS):
        data = U.load_data(U.find_file(pat))
        attn = U.attn_matrix(data, L)[H]
        is_p = punct_mask(attn.shape[1], U.punctuation_positions(data))

        m = U.column_mass(attn)
        valid = np.arange(len(m)) >= 1
        prof = m / m[valid].mean()

        ax = fig.add_subplot(gs[0, idx])
        ax.plot(prof, lw=0.8, color="#333333")
        ax.scatter(np.where(is_p & valid)[0], prof[is_p & valid],
                   s=14, color="#e8654a", zorder=3, label="punct column")
        ax.axhline(1.0, ls=":", lw=0.7, color="#888888")
        r = col_ratio(attn, is_p)
        ax.set_title(f"{title}\npunct/non-punct ratio = {r:.2f}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Key position", fontsize=8)
        if idx == 0:
            ax.set_ylabel("Column mass (mean=1)", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=6)
        del data

    # ---------- bottom: aggregate ----------
    axb = fig.add_subplot(gs[1, :])
    xs, means, cis, fracs, labels = [], [], [], [], []
    pos = 0
    for pat in AGG_MODELS:
        data = U.load_data(U.find_file(pat))
        punct = U.punctuation_positions(data)
        ratios_all, ratios_l1 = [], []
        for L in U.layer_ids(data):
            A = U.attn_matrix(data, L)
            is_p = punct_mask(A.shape[2], punct)
            for h in range(A.shape[0]):
                r = col_ratio(A[h], is_p)
                ratios_all.append(r)
                if L >= 1:
                    ratios_l1.append(r)
        for subset, tag in ((ratios_all, "all"), (ratios_l1, "L>=1")):
            x = np.asarray(subset, dtype=float)
            x_valid = x[~np.isnan(x)]
            if len(x_valid) == 0:
                xs.append(pos); pos += 1
                labels.append(f"{pat}\n{tag}")
                means.append(0.0)
                cis.append((0.0, 0.0))
                fracs.append(0.0)
                print(f"{pat:18s} {tag:5s} n={len(x):5d}  OMITTED (no standalone punct)")
                continue
            lo, hi = U.bootstrap_ci(x_valid)
            xs.append(pos); pos += 1
            labels.append(f"{pat}\n{tag}")
            means.append(x_valid.mean())
            cis.append((x_valid.mean() - lo, hi - x_valid.mean()))
            fracs.append((x_valid > 1).mean())
            try:
                _, p = stats.wilcoxon(x_valid, 1.0, alternative="greater")
            except Exception:
                p = float("nan")
            print(f"{pat:18s} {tag:5s} n={len(x_valid):5d}  mean_ratio={x_valid.mean():.3f} "
                  f"[{lo:.3f},{hi:.3f}]  frac>1={fracs[-1]:.0%}  p={p:.2e}")
        del data

    xs = np.asarray(xs)
    axb.bar(xs, means, yerr=np.array(cis).T, capsize=4, width=0.8, alpha=0.85,
            color=["#6b2fa0", "#b5367a"])
    for x, y, f in zip(xs, means, fracs):
        axb.text(x, y + 0.05, f"{f:.0%}>1", ha="center", fontsize=8)
    axb.axhline(1.0, ls="--", lw=0.8, color="#888888")
    axb.set_xticks(xs); axb.set_xticklabels(labels, fontsize=7)
    axb.set_ylabel("punct / non-punct column-mass ratio", fontsize=9)
    axb.set_title("Aggregated over all layer·heads (BOS excluded)",
                  fontsize=10, fontweight="bold")
    axb.tick_params(labelsize=7)

    fig.suptitle("P1: Punctuation columns attract attention",
                 fontsize=14, fontweight="bold", y=0.98)
    import os as _os
    _os.makedirs("results/figures", exist_ok=True)
    for ext in ("png", "pdf"):
        out = f"results/figures/figure_punct_alignment.{ext}"
        plt.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"✅ Saved: {out}")
    plt.close()

if __name__ == "__main__":
    main()
