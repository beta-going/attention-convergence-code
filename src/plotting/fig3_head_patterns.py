#!/usr/bin/env python3
"""
Appendix G v5: log-scale raw row + data-driven subtitles + Llama panel swap.
Changes vs v4:
  1. Top row: log10(x+eps) display (eps=p1, vmax=p99.9 of causal cone).
     Linear raw is mathematically unreadable at seq>=250 (off-diag ~1e-3);
     log scale restores 10x contrasts while staying a monotone transform of raw.
  2. Subtitle numbers (col0, off-diag) computed from data, not hard-coded.
  3. Llama panel: default = L0 head with max off-diag mass (diffuse control,
     matches (c)(d) narrative); --llama-converged reverts to min-entropy L30 head.
"""
import argparse, glob, json, os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

PROBE_DIR = "data/attention"

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

def row_normalize(attn):
    return attn / np.maximum(attn.max(axis=1, keepdims=True), 1e-8)

def col0_mass(attn):
    return float(row_normalize(attn)[:, 0].mean())

def off_diag_mass(attn, band=5):
    seq = attn.shape[0]
    i = np.arange(seq)[:, None]; j = np.arange(seq)[None, :]
    mask = (j <= i) & (np.abs(i - j) > band)
    return float(row_normalize(attn)[mask].mean())

def pick_head(filepath, layer, mode):
    """mode='offdiag' -> head with max off-diag mass; 'converged' -> min entropy."""
    with open(filepath) as f:
        A = np.array(json.load(f)[f"layer_{layer}"])
    best_h, best_v = 0, -np.inf if mode == "offdiag" else np.inf
    for h in range(A.shape[0]):
        if mode == "offdiag":
            v = off_diag_mass(A[h])
            if v > best_v: best_v, best_h = v, h
        else:
            p = np.clip(A[h] / np.maximum(A[h].sum(1, keepdims=True), 1e-12), 1e-12, 1)
            v = -np.sum(p * np.log(p), axis=1).mean()
            if v < best_v: best_v, best_h = v, h
    return best_h

def log_display(attn):
    """Log-scale raw: eps=p1 of causal cone, clip at p99.9, rescale to [0,1]."""
    seq = attn.shape[0]
    cone = attn[np.tril(np.ones((seq, seq), dtype=bool))]
    eps = max(np.percentile(cone, 1), 1e-12)
    vmax = np.percentile(cone, 99.9)
    t = np.log10(np.clip(attn, 0, vmax) + eps)
    lo, hi = np.log10(eps), np.log10(vmax + eps)
    return np.clip((t - lo) / (hi - lo), 0, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-converged", action="store_true",
                    help="use L30 min-entropy head instead of L0 max-offdiag")
    args = ap.parse_args()

    llama_f = find_file("Llama-3.1-8B_zh")
    if args.llama_converged:
        lh, ltitle, lsub = pick_head(llama_f, 30, "converged"), \
            "(e) Llama-3.1-8B-zh  L30", "Late-layer global convergence"
    else:
        lh, ltitle, lsub = pick_head(llama_f, 0, "offdiag"), \
            "(e) Llama-3.1-8B-zh  L0", "Early-layer diffuse control"

    panels = [
        ("Qwen3-8B_zh", 35, 14, "(a) Qwen3-8B-zh  L35·H14",
         "Top-layer segmentation head\nlocal blocks + punctuation columns"),
        ("Qwen3-8B_en", 35, 14, "(b) Qwen3-8B-en  L35·H14",
         "Same layer/head, English\nstructural match => language-invariant"),
        ("Mistral-7B_zh", 0, 16, "(c) Mistral-7B-zh  L0·H16",
         "Diffuse, NO sink\npure off-diagonal structure"),
        ("Mistral-7B_zh", 0, 19, "(d) Mistral-7B-zh  L0·H19",
         "Diffuse WITH sink\nsentence-boundary grid"),
        ("Llama-3.1-8B_zh", None, lh, ltitle, lsub),  # layer set below
    ]
    panels[4] = (panels[4][0], 0 if not args.llama_converged else 30,
                 lh, ltitle, lsub)

    n = len(panels)
    fig = plt.figure(figsize=(5.0 * n, 10))
    gs = GridSpec(2, n, figure=fig, hspace=0.35, wspace=0.25,
                  left=0.05, right=0.91, top=0.88, bottom=0.08)
    cmap = LinearSegmentedColormap.from_list(
        "attn", ["#000000", "#0d0221", "#261447", "#6b2fa0",
                 "#b5367a", "#e8654a", "#f5b731", "#fef08a"], N=256)
    last_im = None

    for idx, (pattern, layer, head, title, subtitle) in enumerate(panels):
        fp = find_file(pattern)
        with open(fp) as f:
            meta = json.load(f)
        attn = np.array(meta[f"layer_{layer}"])[head]
        c0, od = col0_mass(attn), off_diag_mass(attn)
        #punct = meta.get("punctuation_positions", [])
                # ← CHANGED: field name corrected to match probe output (`punct_pos`).
        # Sourced from rerun_probe_long_input.py, computed by is_standalone_punct
        # (same predicate as P1 v5). For zh-Qwen / Llama-zh this is [] → no cyan.
        punct = meta.get("punct_pos", [])                # ← CHANGED (was "punctuation_positions")
        sub_full = f"{subtitle}\ncol0={c0:.2f}, off-diag={od:.3f}"

        ax_raw = fig.add_subplot(gs[0, idx])
        ax_raw.imshow(log_display(attn), aspect="auto", cmap=cmap, vmin=0,
                      vmax=1, interpolation="nearest")
        ax_raw.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax_raw.set_xlabel("Key position", fontsize=8)
        if idx == 0:
            ax_raw.set_ylabel("Query position", fontsize=9, fontweight="bold")
        ax_raw.tick_params(labelsize=6)

        ax_norm = fig.add_subplot(gs[1, idx])
        last_im = ax_norm.imshow(row_normalize(attn), aspect="auto", cmap=cmap,
                                 vmin=0, vmax=1, interpolation="nearest")
        ax_norm.set_xlabel("Key position", fontsize=8)
        if idx == 0:
            ax_norm.set_ylabel("Query position", fontsize=9, fontweight="bold")
        ax_norm.tick_params(labelsize=6)
        ax_norm.text(0.5, -0.22, sub_full, transform=ax_norm.transAxes,
                     fontsize=8, ha="center", va="top", style="italic",
                     color="#555555", linespacing=1.4)

        for ax in (ax_raw, ax_norm):
            # ← CHANGED: empty-set guard + bounds check
            if punct:
                for p in punct:
                    if p < attn.shape[0]:                 # ← ADDED: safety bound
                        ax.axvline(x=p, color="cyan", alpha=0.3, linewidth=0.5,
                                   linestyle="--")
            else:
                # ← ADDED: explicit note when punctuation is never standalone
                ax.text(0.01, 0.99, "punctuation not standalone\n(no cyan markers)",
                        transform=ax.transAxes, va="top", fontsize=7, color="#888")
            for s in ax.spines.values():
                s.set_linewidth(0.5); s.set_color("#aaaaaa")
        print(f"[{title}] seq={attn.shape[0]} col0={c0:.2f} offdiag={od:.3f}")

    fig.text(0.012, 0.68, "Raw attention\n(log scale, p99.9 clip)",
             fontsize=10, fontweight="bold", ha="center", va="center", rotation=90)
    fig.text(0.012, 0.30, "Row-normalized\n(structure revealed)",
             fontsize=10, fontweight="bold", ha="center", va="center", rotation=90)
    cbar = fig.colorbar(last_im, cax=fig.add_axes([0.925, 0.15, 0.012, 0.7]))
    cbar.set_label("Row-normalized weight\n(top row: log-scale raw)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Appendix G: Per-Head Attention Patterns (Long Input, 252–458 tokens)",
                 fontsize=15, fontweight="bold", y=0.96)
    fig.text(0.47, 0.01,
             "← Cross-lingual (a,b)  |  Sink/no-sink (c,d)  |  Llama control (e) →",
             ha="center", fontsize=10, color="#777777", style="italic")

    import os as _os
    _os.makedirs("results/figures", exist_ok=True)
    for ext in ("png", "pdf"):
        out = f"results/figures/figure_head_patterns_long_v5.{ext}"
        plt.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"✅ Saved: {out}")
    plt.close()

if __name__ == "__main__":
    main()