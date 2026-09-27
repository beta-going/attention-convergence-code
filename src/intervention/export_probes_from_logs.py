#!/usr/bin/env python3
"""export_probes_from_logs.py — 从 v8 运行日志机械恢复探针文本(权威=数据,非源码)
用法: python src/intervention/export_probes_from_logs.py tables/intervention_raw"""
import json, sys, collections
from pathlib import Path
KEYS = ("probe","text","sentence","prompt","prefix","masked","probe_text","cloze")
root, out = Path(sys.argv[1]), Path("data/probes"); out.mkdir(parents=True, exist_ok=True)
def texts(rec):
    for k in KEYS:
        v = rec.get(k)
        if isinstance(v, str) and len(v) > 8: yield k, v
        elif isinstance(v, (list, tuple)) and v and all(isinstance(s, str) for s in v):
            for s in v: yield k, s
seen = collections.defaultdict(lambda: collections.defaultdict(set))
for f in root.rglob("*"):
    if f.suffix not in (".json", ".jsonl"): continue
    lang = "zh" if "_zh" in f.stem or "ZH" in f.stem else "en"
    try: lines = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception: continue
    for rec in lines:
        if not isinstance(rec, dict): continue
        if rec.get("mode") not in (None, "baseline"): continue  # 只取基线,排除干预态污染
        for k, t in texts(rec): seen[lang][k].add(t)
for lang in seen:
    for k, s in seen[lang].items():
        p = out/f"{lang}_{k}.json"
        p.write_text(json.dumps(sorted(s), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{lang}/{k}: {len(s)} 条 -> {p}")
print("\n[闸门] EN 合计应为 20(cloze)+10(free)+5(gen); ZH 合计 20+10+5;")
print("[校验] 与 AST 旧产物 diff: diff <(cat data/probes/en_FREE*.json) data/probes/EN_FREE.json")
