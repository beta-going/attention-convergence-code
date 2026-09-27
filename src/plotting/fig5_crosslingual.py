"""P4 — Cross-lingual consistency of head roles (zh vs en)."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../common'))
import probe_utils as U

MODELS = ["Qwen3-8B", "Qwen3-4B", "Mistral-7B", "Llama-3.1-8B"]
STAR = (35, 14)

def stats_dict(data):
    d = {}
    for L in U.layer_ids(data):
        A = U.attn_matrix(data, L)
        for h in range(A.shape[0]):
            d[(L, h)] = (U.mean_norm_entropy(A[h]), U.off_diag_mass(A[h]))
    return d

def main():
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ci, m in enumerate(MODELS):
        zh = stats_dict(U.load_data(U.find_file(f"{m}_zh")))
        en = stats_dict(U.load_data(U.find_file(f"{m}_en")))
        keys = sorted(set(zh) & set(en))
        ez = np.array([zh[k][0] for k in keys])
        ee = np.array([en[k][0] for k in keys])
        oz = np.array([zh[k][1] for k in keys])
        oe = np.array([en[k][1] for k in keys])

        for ri, (x, y, lab) in enumerate(((ez, ee, "H_norm"), (oz, oe, "off-diag"))):
            ax = axes[ri, ci]
            ax.scatter(x, y, s=8, alpha=0.5, color="#6b2fa0")
            rho, _ = stats.spearmanr(x, y)
            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, ls="--", lw=0.8, color="#888888")
            if m == "Qwen3-8B" and STAR in keys:
                i = keys.index(STAR)
                ax.scatter([x[i]], [y[i]], s=150, marker="*",
                           edgecolor="red", facecolor="none", zorder=5)
            ax.set_title(f"{m} {lab}\nSpearman rho={rho:.2f}",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel(f"{lab} (zh)", fontsize=8)
            ax.set_ylabel(f"{lab} (en)", fontsize=8)
            ax.tick_params(labelsize=7)
            print(f"{m:14s} {lab:9s} rho={rho:.3f}")
    fig.suptitle("P4: Cross-lingual consistency of head roles (zh vs en)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    import os as _os
    _os.makedirs("results/figures", exist_ok=True)
    for ext in ("png", "pdf"):
        out = f"results/figures/figure_crosslingual.{ext}"
        plt.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"✅ Saved: {out}")
    plt.close()

if __name__ == "__main__":
    main()