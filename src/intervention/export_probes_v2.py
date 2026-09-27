#!/usr/bin/env python3
"""从 stream.jsonl 机械恢复探针文本(权威=数据)。用法:
python src/intervention/export_probes_v2.py tables/intervention_raw tables/intervention_raw_cloud
期望: {en,zh}/free=10/10(Table 6 n=10/10), cloze en=20 zh<=20, gen 5+5"""
import json, sys, pathlib, collections
out = pathlib.Path("data/probes"); out.mkdir(exist_ok=True)
seen = collections.defaultdict(lambda: collections.defaultdict(set))
for root in sys.argv[1:]:
    for f in pathlib.Path(root).rglob("stream.jsonl"):
        if any(x in f.parent.name for x in ("shuffvar","INT4","control")): continue
        for line in f.open(encoding="utf-8"):
            try: r = json.loads(line)
            except: continue
            lang = {"zh":"zh","en":"en"}.get(str(r.get("lang","")).lower())
            if not lang or str(r.get("mode","")).replace("-","_").lower() != "baseline": continue
            t, txt = str(r.get("type","")).lower(), r.get("text") or r.get("sentence") or r.get("prompt")
            if not isinstance(txt, str) or len(txt) < 8: continue
            if "cloze" in t: k = "cloze"
            elif "free" in t: k = "free"
            elif "gen" in t: k = "gen"
            elif r.get("ppl") and "options" not in r: k = "free"   # 与 compute_asymmetry.is_free 对齐
            else: continue
            seen[lang][k].add(txt.strip())
for lang in sorted(seen):
    for k in ("cloze","free","gen"):
        if seen[lang].get(k):
            (out/f"{lang}_{k}.json").write_text(json.dumps(sorted(seen[lang][k]), ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{lang}/{k}: {len(seen[lang][k])}")
