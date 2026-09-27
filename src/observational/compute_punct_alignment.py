#!/usr/bin/env python3
"""
P1 v5 (Plan B): punctuation-column alignment using ONLY tokens that are
standalone punctuation in the model's own tokenization.
- A token is punctuation iff, after stripping the BPE/SentencePiece leading
  marker, it consists ENTIRELY of punctuation characters (no CJK/letter bytes).
- Byte-level garbage tokens (<0x..>, raw 0x80-0xFF bytes) are explicitly excluded.
- Configs where fewer than MIN_PUNCT standalone-punctuation tokens exist are
  OMITTED (reported as "tokenizer merges punctuation; alignment undefined"),
  rather than producing meaningless ratios.
"""
import glob, json, os, re
import numpy as np
import matplotlib.pyplot as plt

PROBE_DIR = "data/attention"
CONFIGS = ["Qwen3-8B_zh", "Qwen3-8B_en", "Mistral-7B_zh", "Llama-3.1-8B_zh"]
G_HEADS = {"Qwen3-8B_zh": (35, 14), "Qwen3-8B_en": (35, 14),
           "Mistral-7B_zh": [(0, 16), (0, 19)]}
MIN_PUNCT = 15  # below this, alignment is statistically meaningless -> omit

# Standalone punctuation: the whole token (minus leading marker) is punctuation.
PUNCT_ONLY_RE = re.compile(r"^[。！？，、；：…—·\.\!\?\,\;\:\-\—\"\'\(\)\[\]\{\}<>「」『』【】《》''\/]+$")
# Byte-fallback / raw-byte tokens to reject (Mistral byte-level, any <0x..> or high bytes)
BYTE_TOK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

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

def _tok_str(t):
    if isinstance(t, str): return t
    if isinstance(t, dict): return t.get("text", str(t))
    return str(t)

def is_standalone_punct(tok):
    s = re.sub(r"^[▁\s]+", "", _tok_str(tok))
    if not s:
        return False
    if BYTE_TOK_RE.match(s):          # reject <0xE5> etc
        return False
    if any(ord(c) > 0x7F and c not in
           "。！？，、；：…—·「」『』【】《》""''" for c in s):
        # contains a non-ASCII, non-known-punct char (e.g. a CJK hanzi) -> not pure punct
        return False
    return bool(PUNCT_ONLY_RE.match(s))

def get_punct_positions(meta):
    toks = meta.get("tokens")
    if isinstance(toks, list) and toks:
        punct = {i for i, t in enumerate(toks) if is_standalone_punct(t)}
        return punct, len(toks)
    return set(), 0

def alignment_score(attn, punct_set):
    seq = attn.shape[0]
    rn = attn / np.maximum(attn.max(axis=1, keepdims=True), 1e-8)
    prof = rn.mean(axis=0)
    pm = [j for j in range(seq) if j in punct_set]
    cm = [j for j in range(seq) if j not in punct_set]
    if not pm or not cm:
        return float("nan"), prof
    return float(prof[pm].mean() / max(prof[cm].mean(), 1e-8)), prof

# ---- Diagnose + decide which configs are usable
usable = {}
for cfg in CONFIGS:
    try:
        import os as _os
        _os.makedirs("results/figures", exist_ok=True)
        meta = json.load(open(find_file(cfg)))
    except FileNotFoundError:
        print(f"skip {cfg}"); continue
    toks = meta.get("tokens", [])
    punct, nt = get_punct_positions(meta)
    seq = meta.get("seq_len", len(toks))
    punct = {p for p in punct if p < seq}
    sample = [(i, repr(_tok_str(toks[i]))) for i in sorted(punct)[:8] if i < len(toks)]
    status = "USABLE" if len(punct) >= MIN_PUNCT else "OMITTED (punctuation not standalone)"
    print(f"[{cfg}] seq={seq} n_standalone_punct={len(punct)}  -> {status}")
    print(f"     samples={sample}")
    if len(punct) >= MIN_PUNCT:
        usable[cfg] = punct

rows = []
for cfg, punct in usable.items():
    meta = json.load(open(find_file(cfg)))
    seq = meta.get("seq_len", len(meta.get("tokens", [])))
    for key in sorted(meta.keys()):
        if not key.startswith("layer_"): continue
        L = int(key.split("_")[1])
        attn_all = np.array(meta[key])
        for h in range(attn_all.shape[0]):
            a, _ = alignment_score(attn_all[h], punct)
            if not np.isnan(a):
                rows.append((cfg, L, h, a))

rows.sort(key=lambda r: -r[3])
print(f"\n{'Config':<18}{'Layer':>6}{'Head':>5}{'Align':>8}")
for cfg, L, h, a in rows[:20]:
    print(f"{cfg:<18}{L:>6}{h:>5}{a:>8.2f}")

print("\n--- Heads with A > 1.5 (usable configs only) ---")
for cfg in usable:
    sub = [a for c, _, _, a in rows if c == cfg]
    n15 = sum(1 for a in sub if a > 1.5)
    n12 = sum(1 for a in sub if a > 1.2)
    top = max(sub) if sub else float("nan")
    print(f"{cfg:<18} total={len(sub):>4}  A>1.5:{n15:>3} ({100*n15/len(sub):.1f}%)  "
          f"A>1.2:{n12:>3}  max={top:.2f}")

# Figure: only usable configs get bars/profiles; omitted ones show a note
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for i, cfg in enumerate(CONFIGS):
    if cfg in usable:
        sub = [r for r in rows if r[0] == cfg][:8]
        axes[0, i].bar([f"L{L}H{h}" for _, L, h, _ in sub],
                       [a for *_, a in sub], color="#6b2fa0")
        axes[0, i].axhline(1.0, ls="--", c="gray", lw=0.8)
        axes[0, i].set_title(f"{cfg}\ntop-8 heads (n_punct={len(usable[cfg])})",
                             fontsize=9)
        # profile
        meta = json.load(open(find_file(cfg)))
        punct = usable[cfg]
        if cfg in G_HEADS:
            spec = G_HEADS[cfg]
            specs = spec if isinstance(spec, list) else [spec]
            L, H = specs[0]
            _, prof = alignment_score(np.array(meta[f"layer_{L}"])[H], punct)
            axes[1, i].plot(prof, lw=0.7, c="#333")
            pts = [p for p in punct if p < len(prof)]
            if pts:
                axes[1, i].scatter(pts, [prof[p] for p in pts], s=6, c="cyan", zorder=3)
            axes[1, i].set_title(f"{cfg} L{L}·H{H} column profile\n(cyan=punct)",
                                 fontsize=9)
    else:
        axes[0, i].text(0.5, 0.5, f"{cfg}\n\nOMITTED\npunctuation not\nstandalone tokens",
                        ha="center", va="center", transform=axes[0, i].transAxes,
                        fontsize=10, color="#888")
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        axes[1, i].text(0.5, 0.5, "alignment undefined\nfor this tokenizer",
                        ha="center", va="center", transform=axes[1, i].transAxes,
                        fontsize=10, color="#888")
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
    axes[0, i].tick_params(labelsize=6, axis='x', rotation=45)
fig.suptitle("Punctuation-column alignment (A>1 = punctuation attracts above-baseline "
             "attention; only configs where punctuation is a standalone token)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/figure_punct_alignment.png", dpi=250, bbox_inches="tight")
print("\n✅ Saved: figures/figure_punct_alignment.png")

import csv
with open("tables/punct_alignment.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["config", "layer", "head", "alignment"])
    w.writerows(rows)
print("✅ Saved: tables/punct_alignment.csv")