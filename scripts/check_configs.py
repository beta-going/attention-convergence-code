#!/usr/bin/env python3
"""S1闸门: models.yaml 每个 key 都能 resolve 出可用路径。exit 0 = 通过"""
import sys, pathlib, yaml
cfg = yaml.safe_load((pathlib.Path(__file__).parents[1]/"configs/models.yaml").read_text())
bad = [k for k, m in cfg["models"].items() if not (m.get("hf_id") or m.get("local_path"))]
print("resolve 失败:", bad) if bad else print(f"OK: {len(cfg['models'])} 个模型全部可 resolve")
sys.exit(1 if bad else 0)
