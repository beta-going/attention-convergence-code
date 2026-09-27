#!/usr/bin/env python3
"""Bonferroni family recomputation for Appendix D (v30). v2.
Uses the repository's own loader/entropy (compute_convergence) so numbers are
pipeline-identical; falls back to a verbatim local copy if import fails.
Outputs: per-model zh-vs-en entropy-profile Spearman (n=L), per-layer
head-level Spearman with Bonferroni families (L per model; pooled 4L),
per_layer.csv, summary.csv, and the Table-1 l* ordering sanity check."""
import argparse, json, math, os, sys, itertools
import numpy as np

_EPS_LOCAL = 1e-12
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compute_convergence import norm_row_entropy_per_head as _repo_entropy
    SRC = "repo-import"
except Exception as _e:  # verbatim fallback (EPS taken from repo source if printed below)
    _repo_entropy = None; SRC = f"local-fallback ({_e})"

def head_entropy(A, S):
    if _repo_entropy is not None:
        return np.asarray(_repo_entropy(np.asarray(A, np.float64), S), float)
    a = np.clip(np.asarray(A, np.float64), _EPS_LOCAL, 1.0)
    ent = -(a * np.log(a)).sum(-1)
    denom = np.empty(S); denom[0] = 1.0
    denom[1:] = np.log(np.arange(2, S + 1))
    return (ent / denom[None, :])[:, 1:].mean(axis=1)

def rankdata(a):
    a = np.asarray(a, float); order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a)); sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j+1] == sa[i]: j += 1
        ranks[order[i:j+1]] = (i + j) / 2.0 + 1.0; i = j + 1
    return ranks

def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = math.sqrt(float((x*x).sum() * (y*y).sum()))
    return float((x*y).sum()/d) if d > 0 else float("nan")

def spearman(a, b): return pearson(rankdata(a), rankdata(b))

def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-12, 1e-300
    qab, qap, qam = a+b, a+1.0, a-1.0
    c, d = 1.0, 1.0 - qab*x/qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1.0/d; h = d
    for m in range(1, MAXIT+1):
        m2 = 2*m
        aa = m*(b-m)*x/((qam+m2)*(a+m2))
        d = 1.0+aa*d; d = FPMIN if abs(d) < FPMIN else d
        c = 1.0+aa/c; c = FPMIN if abs(c) < FPMIN else c
        d = 1.0/d; h *= d*c
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d = 1.0+aa*d; d = FPMIN if abs(d) < FPMIN else d
        c = 1.0+aa/c; c = FPMIN if abs(c) < FPMIN else c
        d = 1.0/d; de = d*c; h *= de
        if abs(de-1.0) < EPS: break
    return h

def betai(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1.0-x))
    return bt*betacf(a,b,x)/a if x < (a+1)/(a+b+2) else 1.0 - bt*betacf(b,a,1.0-x)/b

def t_p2(t, nu): return betai(nu/2.0, 0.5, nu/(nu+t*t))

def SELFTEST():
    assert abs(spearman([1,2,3,4],[1,2,3,4]) - 1) < 1e-12
    assert abs(spearman([1,2,3,4],[4,3,2,1]) + 1) < 1e-12
    assert abs(t_p2(2.0423, 30) - 0.0500) < 2e-3
    print(f"SELFTEST-OK; entropy-src={SRC}")

def rho_p(a, b):
    n = len(a); r = spearman(a, b)
    if not math.isfinite(r): return r, float("nan")
    if abs(abs(r) - 1.0) < 1e-12: return r, 1e-16
    t = abs(r)*math.sqrt((n-2)/max(1e-15, 1-r*r))
    return r, max(t_p2(t, n-2), 1e-16)

MODELS = ["Qwen3-8B", "Qwen3-4B", "Llama-3.1-8B", "Mistral-7B"]
LSTAR = {"en": {"Qwen3-8B":15,"Qwen3-4B":19,"Llama-3.1-8B":13,"Mistral-7B":0},
         "zh": {"Qwen3-8B":16,"Qwen3-4B":19,"Llama-3.1-8B":14,"Mistral-7B":19}}

def load_profile(fp):
    d = json.load(open(fp, encoding="utf-8"))
    L, H, S = d["n_layers"], d["n_heads"], d["seq_len"]
    heads = []
    for i in range(L):
        v = head_entropy(d[f"layer_{i}"], S)
        assert len(v) == H, f"layer_{i}: expected H={H}, got {len(v)}"
        heads.append(v)
    return np.array(heads), L, H, S  # (L, H)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/attention")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="tables/bonferroni_check")
    a = ap.parse_args(); SELFTEST()
    models = ["Qwen3-8B"] if a.smoke else MODELS
    os.makedirs(a.out, exist_ok=True)
    rows, pooled, pl_rows = [], [], []
    for m in models:
        D = {}
        for lang in ("en", "zh"):
            fp = os.path.join(a.dir, f"consistency_{m}_{lang}.json")
            D[lang], L, H, S = load_profile(fp)
            print(f"LOADED {m}/{lang}: L={L} H={H} S={S}")
        Lz = min(len(D["en"]), len(D["zh"]))
        prof_e = D["en"][:Lz].mean(1); prof_z = D["zh"][:Lz].mean(1)
        rp, pp = rho_p(prof_e, prof_z)
        per = [rho_p(D["en"][i], D["zh"][i]) for i in range(Lz)]
        ps = np.array([p for _, p in per]); k = Lz
        sig_k = int((ps < 0.05/k).sum()); pooled.extend(ps.tolist())
        rows.append((m, Lz, H, f"{rp:.6f}", f"{pp:.3e}", f"{ps.min():.3e}", sig_k, k))
        print(f"SUMMARY {m}: profile rho={rp:.4f} p={pp:.2e} | per-layer min_p={ps.min():.2e} sig(0.05/{k})={sig_k}/{k}")
        pl_rows += [(m, i, f"{r:.6f}", f"{p:.3e}") for i, (r, p) in enumerate(per)]
    if not a.smoke:
        pooled = np.array(pooled); K = len(pooled)
        print(f"POOLED family k={K}: sig(0.05/{K})={int((pooled < 0.05/K).sum())}/{K} min_p={pooled.min():.2e}")
        re_ = [LSTAR["en"][m] for m in MODELS]; rz = [LSTAR["zh"][m] for m in MODELS]
        r_obs = spearman(re_, rz)
        perms = list(itertools.permutations(range(4)))
        p_ex = sum(1 for pm in perms if abs(spearman(pm, range(4))) >= abs(r_obs)-1e-12)/len(perms)
        print(f"TABLE1-ORDER l* en-vs-zh (n=4, hardcoded): rho={r_obs:.3f} exact_two_sided_p={p_ex:.4f}  -> >0.95不可能; §4.1括号须指向剖面/逐层口径")
        with open(os.path.join(a.out, "summary.csv"), "w") as f:
            f.write("model,L,H,profile_rho,profile_p,perlayer_min_p,sig_kL,kL\n")
            for r in rows: f.write(",".join(map(str, r))+"\n")
        with open(os.path.join(a.out, "per_layer.csv"), "w") as f:
            f.write("model,layer,rho,p\n")
            for r in pl_rows: f.write(",".join(map(str, r))+"\n")
        print("CSV-WRITTEN", a.out, f"({len(pl_rows)} per-layer rows)")
    print("RUN-OK" if not a.smoke else "SMOKE-OK")

if __name__ == "__main__":
    main()
