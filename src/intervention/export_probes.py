#!/usr/bin/env python3
"""从 run_intervention.py 的模块级常量中析出探针/句子/前缀文本 -> data/probes/*.json
用法: python src/intervention/export_probes.py   (首跑后人工核对计数: cloze 20/20, 句 10/10, 前缀 5/5)"""
import ast, json, pathlib, sys
HERE = pathlib.Path(__file__).parent
tree = ast.parse((HERE/"run_intervention.py").read_text(encoding="utf-8"))
out = {}
for node in tree.body:
    if not isinstance(node, ast.Assign): continue
    tgt = node.targets[0]
    if not (isinstance(tgt, ast.Name) and tgt.id.isupper()): continue
    try: val = ast.literal_eval(node.value)
    except Exception: continue
    def looks_like_text(x):
        return isinstance(x, str) and (any('\u4e00' <= c <= '\u9fff' for c in x) or " " in x) and len(x) > 8
    if isinstance(val, list) and val and all(looks_like_text(s) for s in val): out[tgt.id] = val
    elif isinstance(val, dict) and val and all(looks_like_text(str(v)) for v in val.values()): out[tgt.id] = val
dest = HERE.parents[1]/"data/probes"
dest.mkdir(parents=True, exist_ok=True)
for k, v in out.items():
    (dest/f"{k}.json").write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    n = len(v); print(f"{k}: {n} 条 -> data/probes/{k}.json")
if not out: sys.exit("未发现候选常量——请 grep -nE '^[A-Z_]{4,} *=' run_intervention.py 人工定位")
