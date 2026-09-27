#!/usr/bin/env python3
"""
plot_convergence_depth.py — v5 (robust plateau entry)

ℓ* 判据修订（v4 的"永不跌破"对噪声过严，全 nan）：
  P    = median( smooth(C_final)[ℓ ≥ L/2] )
  ℓ*   = 首个 ℓ，使得 #{ℓ'≥ℓ : smooth(C_final)(ℓ') < PLATEAU_FRAC·P} ≤ TOL
  即"进入 plateau 后允许最多 TOL 个 dip，但不系统性回落"。
并列佐证：ℓ*_consec = 首个 ℓ，使得此后 smooth(C_consec) 持续 ≥ C_CONSEC_THR（默认 0.7，同样允许 TOL 次违反）。
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("data/attention")
LANG = "en"
MODELS = [
    ("Qwen3-8B",     "#d43d2a"),
    ("Qwen3-4B",     "#f0a08c"),
    ("Llama-3.1-8B", "#5ba8cd"),
    ("Mistral-7B",   "#0f9488"),
]

EPS           = 1e-12
SMOOTH_W      = 3
PLATEAU_FRAC  = 0.9
TOL           = 2      # plateau 内允许的最大 dip 层数
C_CONSEC_THR  = 0.7
H_SMOOTH_W    = 5

def load_layers(path):
    import os as _os
    _os.makedirs("results/figures", exist_ok=True)
    with open(path) as f:
        d = json.load(f)
    n, S = d["n_layers"], d["seq_len"]
    return [np.asarray(d[f"layer_{i}"], dtype=np.float64) for i in range(n)], S

def norm_row_entropy_per_head(A, S):
    a = np.clip(A, EPS, 1.0)
    ent = -(a * np.log(a)).sum(-1)
    denom = np.empty(S); denom[0] = 1.0
    denom[1:] = np.log(np.arange(2, S + 1))
    return (ent / denom[None, :])[:, 1:].mean(axis=1)

def jsd_dist(P, Q):
    p = np.clip(P, EPS, None); q = np.clip(Q, EPS, None)
    m = 0.5 * (p + q)
    jsd = 0.5 * (p * np.log(p / m)).sum(-1) + 0.5 * (q * np.log(q / m)).sum(-1)
    return jsd.mean() / np.log(2)

def movavg(x, w=SMOOTH_W):
    return np.convolve(x, np.ones(w) / w, mode="same")

def entry_with_tol(s, thr, tol):
    """首个 ℓ，使得 s[ℓ:] 中低于 thr 的点数 ≤ tol。"""
    n = len(s)
    below = (s < thr).astype(int)
    # 后缀和：suffix[i] = #{j≥i : s[j] < thr}
    suffix = np.concatenate(([0], np.cumsum(below[::-1])))[::-1]
    for l in range(n):
        if suffix[l] <= tol:
            return l
    return np.nan

def main():
    fig,  axes  = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey=True)
    fig2, axes2 = plt.subplots(2, 2, figsize=(12.5, 7.0))
    rows = {}
    for ax, ax2, (name, color) in zip(axes.flat, axes2.flat, MODELS):
        layers, S = load_layers(DATA_DIR / f"consistency_{name}_{LANG}.json")
        L = len(layers)

        Hh = np.stack([norm_row_entropy_per_head(A, S) for A in layers]).T
        H  = Hh.mean(axis=0)
        c_consec = np.array([1 - jsd_dist(layers[l], layers[l + 1]) for l in range(L - 1)])
        c_final  = np.array([1 - jsd_dist(layers[l], layers[-1])    for l in range(L - 1)])

        sf, sc = movavg(c_final), movavg(c_consec)
        P   = float(np.median(sf[L // 2:]))
        ell_f = entry_with_tol(sf, PLATEAU_FRAC * P, TOL)
        ell_c = entry_with_tol(sc, C_CONSEC_THR, TOL)
        peak  = int(np.argmax(movavg(H, H_SMOOTH_W)))

        x = np.arange(L)
        ax.plot(x, H, "-",  color=color, lw=2.0, label=r"$\bar{H}$ (entropy)")
        ax.plot(x[:-1], c_consec, "--", color=color, lw=1.4, label=r"$C_{\mathrm{consec}}$")
        ax.plot(x[:-1], c_final,  ":",  color=color, lw=2.0, label=r"$C_{\mathrm{final}}$")
        ax.axhline(0.7, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(0.3, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(P, color=color, ls="-", lw=0.7, alpha=0.35)
        if not np.isnan(ell_f):
            ax.axvline(ell_f, color="0.3", ls="-.", lw=1.0, alpha=0.8)
        ax.set_title(f"{name}  (L={L}, ℓ*={ell_f:.0f})" if not np.isnan(ell_f)
                     else f"{name}  (L={L}, ℓ*=–)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)

        im = ax2.pcolormesh(x, np.arange(Hh.shape[0]), Hh, cmap="viridis", vmin=0, vmax=1)
        ax2.set_title(f"{name}: head×layer $\\bar{{H}}$")
        fig2.colorbar(im, ax=ax2, fraction=0.046)

        rows[name] = (L, P, ell_f, ell_c, peak, H[-5:].mean())
        del layers

    for a in axes[:, 0]:  a.set_ylabel("value in [0,1]")
    for a in axes[-1, :]: a.set_xlabel("layer ℓ")
    for a in axes2[:, 0]:  a.set_ylabel("head")
    for a in axes2[-1, :]: a.set_xlabel("layer ℓ")
    axes[0, 0].legend(fontsize=9, loc="upper right")
    fig.tight_layout();  fig.savefig("results/figures/figure_convergence_depth.png", dpi=200)
    fig2.tight_layout(); fig2.savefig("results/figures/figure_convergence_heatmap.png", dpi=200)
    print("✅ Saved: figures/figure_convergence_depth.png, figures/figure_convergence_heatmap.png\n")

    print(f"thresholds: SMOOTH_W={SMOOTH_W}, PLATEAU_FRAC={PLATEAU_FRAC}, TOL={TOL}, "
          f"C_CONSEC_THR={C_CONSEC_THR}")
    print(f"\n{'model':<14}{'L':>4}{'P':>7}{'l*_final':>9}{'l*/L':>7}"
          f"{'l*_consec':>10}{'Hpk@':>6}{'Hdeep':>8}")
    for name, (L, P, ef, ec, peak, hdeep) in rows.items():
        ff = ef / L if not np.isnan(ef) else float("nan")
        print(f"{name:<14}{L:>4}{P:>7.2f}{ef:>9.0f}{ff:>7.2f}{ec:>10.0f}{peak:>6}{hdeep:>8.3f}")

if __name__ == "__main__":
    main()