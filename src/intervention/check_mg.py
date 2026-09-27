import json

MODELS = ["Qwen3-8B", "Llama-3.1-8B"]
BANDS  = ["early", "mid", "late"]
def load(p):
    return json.load(open(p))


def meta_filter(d):
    """在 meta 里递归找名字含 keep/filter/retain 的字段"""
    found = {}
    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if any(s in str(k).lower() for s in ("keep", "filter", "retain")):
                    found[path + "/" + str(k)] = v
                walk(v, path + "/" + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o): walk(v, f"{path}[{i}]")
    walk(d.get("meta", {}), "meta")
    return found

def leaves(o, path=""):
    """把一个 dict 里所有数字叶子摊平成 {路径: 值}"""
    out = {}
    if isinstance(o, dict):
        for k, v in o.items(): out.update(leaves(v, f"{path}/{k}"))
    elif isinstance(o, list):
        for i, v in enumerate(o): out.update(leaves(v, f"{path}[{i}]"))
    elif isinstance(o, (int, float)):
        out[path] = float(o)
    return out

print("=" * 72)
print("JOB A: 中文探针筛选记录 old vs new")
okA = True
for m in MODELS:
    for b in BANDS:
        fo = meta_filter(load(f"results_v8/{m}_{b}/intervention_results.json"))
        fn = meta_filter(load(f"results_v8_mg/{m}_{b}/intervention_results.json"))
        same = (fo == fn) and bool(fo)
        okA &= same
        print(f"{m:<12}{b:<6} {'一致 ✓' if same else '不一致或未找到!'}")
        if not same:
            print("  old:", json.dumps(fo)[:200])
            print("  new:", json.dumps(fn)[:200])
            if not fo:
                d = load(f"results_v8/{m}_{b}/intervention_results.json")
                print("  (meta 顶层键:", list(d.get("meta", {}).keys()), ")")
print("JOB A 结论:", "全部一致 ✓" if okA else "有不一致 → 停,贴完整输出")

print("=" * 72)
print("JOB B: baseline 逐数比对 (同一环境确定性,应完全相同)")
okB = True
for m in MODELS:
    for b in BANDS:
        lo = leaves(load(f"results_v8/{m}_{b}/intervention_results.json")["baseline"])
        ln = leaves(load(f"results_v8_mg/{m}_{b}/intervention_results.json")["baseline"])
        bad = [(k, lo.get(k), ln.get(k)) for k in sorted(set(lo) | set(ln))
               if abs(lo.get(k, 9e9) - ln.get(k, -9e9)) > 1e-4]
        okB &= not bad
        print(f"{m:<12}{b:<6} {'完全一致 ✓' if not bad else f'不一致: {bad[:5]}'}")
print("JOB B 结论:", "全部一致 ✓ (matched_global 的数字可信)" if okB else "有漂移 → 停,贴完整输出")

print("=" * 72)
print("JOB C: matched_global vs 旧 uniform_causal/uniform_global")
for m in MODELS:
    for b in BANDS:
        do = load(f"results_v8/{m}_{b}/intervention_results.json")
        dn = load(f"results_v8_mg/{m}_{b}/intervention_results.json")
        strip = lambda sub: {k.split("/", 1)[1]: v for k, v in leaves(sub).items()}
        uc, ug, mg = strip(do["uniform_causal"]), strip(do["uniform_global"]), strip(dn["matched_global"])
        print(f"--- {m} {b} ---")
        for k in sorted(set(uc) | set(ug) | set(mg)):
            f = lambda d: f"{d.get(k)}" if k in d else "—"
            print(f"  {k:<42} UC={f(uc):<10} UG={f(ug):<10} MG={f(mg)}")
