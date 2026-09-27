"""Shared helpers for P1-P4 (probe_v3_long)."""
import glob, json, os
import numpy as np

PROBE_DIR = "data/attention"

def find_file(pattern, probe_dir=PROBE_DIR):
    # Support multiple matching strategies (consistent with other scripts)
    base = pattern.replace("consistency_", "").replace(".json", "")
    candidates = [
        os.path.join(probe_dir, f"consistency_{base}.json"),
        os.path.join(probe_dir, f"*{base}*.json"),
        os.path.join(probe_dir, pattern),
    ]
    for c in candidates:
        matches = sorted(glob.glob(c))
        if matches:
            return matches[0]
    matches = [m for m in matches if "consistency_" in os.path.basename(m)] or matches
    raise FileNotFoundError(f"No file matching '{pattern}' in {probe_dir}")

def load_data(filepath):
    with open(filepath) as f:
        return json.load(f)

def layer_ids(data):
    return sorted(int(k.split("_")[1]) for k in data if k.startswith("layer"))

def attn_matrix(data, layer):
    return np.array(data[f"layer_{layer}"])

def punctuation_positions(data):
    """Return set of standalone-punctuation token positions.
    Falls back to dynamic detection from tokens if not precomputed."""
    pp = data.get("punctuation_positions")
    if pp is not None:
        return set(pp)
    toks = data.get("tokens")
    if toks is None:
        return set()
    import re, unicodedata
    _BPE_MARKERS = ("▁", "Ġ", "▏", "▎", "▍", "▌", "▋", "▊", "▉")
    _BYTE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    def _is_standalone_punct(tok):
        s = tok
        for m in _BPE_MARKERS:
            if s.startswith(m):
                s = s[len(m):]
                break
        if not s:
            return False
        if _BYTE_RE.match(s):
            return False
        if any(0x80 <= ord(c) <= 0xFF and not unicodedata.category(c).startswith("P") for c in s):
            return False
        return all(unicodedata.category(c).startswith("P") for c in s)
    return {i for i, t in enumerate(toks) if _is_standalone_punct(t)}

def row_normalize(attn):
    return attn / np.maximum(attn.max(axis=1, keepdims=True), 1e-8)

def norm_entropy_rows(attn):
    """Causal-aware normalized entropy (0..1): row i has i+1 valid keys."""
    seq = attn.shape[0]
    s = attn.sum(axis=1, keepdims=True)
    p = np.clip(attn / np.maximum(s, 1e-10), 1e-10, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    denom = np.log(np.arange(1, seq + 1))   # ln(i+1)
    denom[0] = 1.0
    return h / denom

def mean_norm_entropy(attn):
    return float(norm_entropy_rows(attn).mean())

def col0_mass(attn):
    return float(row_normalize(attn)[:, 0].mean())

def off_diag_mass(attn, band=5):
    seq = attn.shape[0]
    i = np.arange(seq)[:, None]; j = np.arange(seq)[None, :]
    mask = (j <= i) & (np.abs(i - j) > band)
    return float(row_normalize(attn)[mask].mean())

def column_mass(attn):
    """Total attention received by each key position."""
    return attn.sum(axis=0)

def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, x.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)