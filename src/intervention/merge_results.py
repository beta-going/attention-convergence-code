#!/usr/bin/env python3
"""
merge_results_v8.py — 聚合 v8.0 (及 v6 旧版) 干预实验结果。

输入目录约定:
  <root>/<Model>_<band>/intervention_results.json        主跑 (4 modes, repeats=1)
  <root>/<Model>_<band>_shuffvar/intervention_results.json  shuffle 方差子跑
  <root>/<Model>_<band>/stream.jsonl                      可选, 提供 per-probe 方差

用法:
  python3 src/merge_results_v8.py \
      --local_dir results_v8 --cloud_dir results_v8_cloud --out results_v8/merged

输出:
  final_results_merged.json / summary_table.md / summary_table.csv
"""
import argparse
import csv
import json
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime

BAND_RE = re.compile(r"^(?P<model>.+)_(?P<band>early|mid|late|all)(?P<tag>_shuffvar)?$")
MODES = ["baseline", "uniform_causal", "uniform_global", "shuffle_rows"]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scan_root(root, source_label, store, var_store):
    if not os.path.isdir(root):
        print(f"[WARN] directory not found: {root}")
        return
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        jp = os.path.join(d, "intervention_results.json")
        if not (os.path.isdir(d) and os.path.isfile(jp)):
            continue
        m = BAND_RE.match(name)
        if not m:
            print(f"[SKIP] {name}: expected <Model>_<band>[_shuffvar]")
            continue
        model, band, tag = m.group("model"), m.group("band"), m.group("tag") or ""
        if model.lower().startswith("anchor"):
            print(f"[SKIP] {name}: anchor archive excluded")
            continue
        data = load_json(jp)
        meta = data.get("meta", {})

        runs = []
        if any(k.startswith("repeat_") for k in data):
            keys = sorted((k for k in data if k.startswith("repeat_")),
                          key=lambda s: int(s.split("_")[1]))
            for k in keys:
                rep = int(k.split("_")[1])
                for mode, entry in data[k].items():
                    runs.append({"mode": mode, "repeat": rep, **entry})
        else:
            for mode in MODES:
                if mode in data:
                    runs.append({"mode": mode, "repeat": 0, **data[mode]})

        zfm = meta.get("zh_filter_meta") or []
        store[(model, band, tag)] = {
            "source": source_label,
            "model": model, "band": band, "tag": tag,
            "code_version": meta.get("code_version", "?"),
            "engine": meta.get("engine", "v6.0-era"),
            "n_en_cloze": meta.get("n_en_cloze"),
            "n_zh_cloze": meta.get("n_zh_cloze"),
            "zh_filter_kept": [x["idx"] for x in zfm if x.get("kept")],
            "runs": runs,
        }

        sp = os.path.join(d, "stream.jsonl")
        if os.path.isfile(sp):
            cloze, gens = defaultdict(list), defaultdict(list)
            with open(sp, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("type") == "cloze":
                        cloze[(r["mode"], r["lang"], r["probe_idx"])].append(
                            int(bool(r["correct"])))
                    elif r.get("type") == "gen":
                        gens[(r["mode"], r["lang"])].append(r.get("gen_ppl"))
            var_store[(model, band, tag)] = {
                "cloze": {f"{mode}|{lang}|{idx}": {
                            "n": len(v), "mean": statistics.mean(v),
                            "var": statistics.pvariance(v) if len(v) > 1 else 0.0}
                          for (mode, lang, idx), v in cloze.items()},
                "gen_ppl": {f"{mode}|{lang}": v for (mode, lang), v in gens.items()},
            }
        print(f"[LOAD] {source_label}/{name}: {len(runs)} mode-runs "
              f"(code {store[(model, band, tag)]['code_version']})")


def entry_metrics(entry):
    en = entry.get("syntactic_en") or {}
    zh = entry.get("syntactic_zh") or {}
    ge = entry.get("generation_en") or {}
    gz = entry.get("generation_zh") or {}
    return {"en_acc": en.get("accuracy"), "zh_acc": zh.get("accuracy"),
            "en_freeppl": en.get("mean_ppl"), "zh_freeppl": zh.get("mean_ppl"),
            "en_genppl": ge.get("avg_ppl"), "zh_genppl": gz.get("avg_ppl")}


def fmt_ms(vals, nd=2):
    vals = [v for v in vals if v is not None and v == v and abs(v) != float("inf")]
    if not vals:
        return ""
    m = statistics.mean(vals)
    if len(vals) == 1:
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f}±{statistics.pstdev(vals):.{nd}f}"


def build_rows(store):
    rows = []
    for (model, band, tag), rec in sorted(store.items()):
        if tag == "_shuffvar":
            continue
        all_runs = list(rec["runs"])
        sv = store.get((model, band, "_shuffvar"))
        if sv:
            all_runs += [dict(r, repeat=100 + r["repeat"]) for r in sv["runs"]]
        by_mode = defaultdict(list)
        for r in all_runs:
            by_mode[r["mode"]].append(r)
        for mode in MODES:
            rs = by_mode.get(mode, [])
            if not rs:
                continue
            ms = [entry_metrics(r) for r in rs]
            rows.append({
                "model": model, "band": band, "mode": mode,
                "n_runs": len(rs), "source": rec["source"],
                "code_version": rec["code_version"],
                "n_en": rec["n_en_cloze"], "n_zh": rec["n_zh_cloze"],
                "zh_filter_kept": rec["zh_filter_kept"],
                "en_acc": [m["en_acc"] for m in ms],
                "zh_acc": [m["zh_acc"] for m in ms],
                "en_freeppl": [m["en_freeppl"] for m in ms],
                "zh_freeppl": [m["zh_freeppl"] for m in ms],
                "en_genppl": [m["en_genppl"] for m in ms],
                "zh_genppl": [m["zh_genppl"] for m in ms],
            })
    return rows


def write_markdown(rows, path):
    lines = ["# Intervention v8.0 — Merged Results", "",
             "Values are mean±sd across runs (single run → plain mean).",
             "Gen-PPL columns are empty for `_shuffvar`-augmented shuffle rows"
             " (variance pass ran without generation).", ""]
    for band in ["early", "mid", "late", "all"]:
        sub = [r for r in rows if r["band"] == band]
        if not sub:
            continue
        lines += [f"## Band: {band}", "",
                  "| Model | Mode | N runs | EN acc | ZH acc | EN freePPL | ZH freePPL | EN genPPL | ZH genPPL |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            lines.append(
                f"| {r['model']} | {r['mode']} | {r['n_runs']} "
                f"| {fmt_ms(r['en_acc'], 1)} | {fmt_ms(r['zh_acc'], 1)} "
                f"| {fmt_ms(r['en_freeppl'], 2)} | {fmt_ms(r['zh_freeppl'], 2)} "
                f"| {fmt_ms(r['en_genppl'], 2)} | {fmt_ms(r['zh_genppl'], 2)} |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_csv(rows, path):
    header = ["model", "band", "mode", "n_runs", "source", "code_version",
              "n_en", "n_zh",
              "en_acc_mean", "en_acc_sd", "zh_acc_mean", "zh_acc_sd",
              "en_freeppl_mean", "zh_freeppl_mean",
              "en_genppl_mean", "zh_genppl_mean"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            def m(vals):
                vals = [v for v in vals if v is not None and v == v]
                return statistics.mean(vals) if vals else ""
            def sd(vals):
                vals = [v for v in vals if v is not None and v == v]
                return statistics.pstdev(vals) if len(vals) > 1 else 0.0
            w.writerow([r["model"], r["band"], r["mode"], r["n_runs"],
                        r["source"], r["code_version"], r["n_en"], r["n_zh"],
                        m(r["en_acc"]), sd(r["en_acc"]),
                        m(r["zh_acc"]), sd(r["zh_acc"]),
                        m(r["en_freeppl"]), m(r["zh_freeppl"]),
                        m(r["en_genppl"]), m(r["zh_genppl"])])


def main():
    ap = argparse.ArgumentParser(description="Merge v8.0 intervention results")
    ap.add_argument("--local_dir", default="results_v8")
    ap.add_argument("--cloud_dir", default=None,
                    help="e.g. results_v8_cloud (scp'ed from server)")
    ap.add_argument("--out", default="results_v8/merged")
    args = ap.parse_args()

    store, var_store = {}, {}
    scan_root(args.local_dir, "local", store, var_store)
    if args.cloud_dir:
        scan_root(args.cloud_dir, "cloud", store, var_store)
    if not store:
        raise SystemExit("[ERROR] no intervention_results.json found")

    rows = build_rows(store)
    os.makedirs(args.out, exist_ok=True)

    merged = {
        "meta": {"generated": datetime.now().isoformat(),
                 "local_dir": args.local_dir, "cloud_dir": args.cloud_dir,
                 "n_cells": len(rows),
                 "models": sorted({r["model"] for r in rows})},
        "table": rows,
        "per_probe_and_sample": {
            f"{model}_{band}{tag}": v
            for (model, band, tag), v in var_store.items()
        },

    }
    jp = os.path.join(args.out, "final_results_merged.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    mp = os.path.join(args.out, "summary_table.md")
    write_markdown(rows, mp)
    cp = os.path.join(args.out, "summary_table.csv")
    write_csv(rows, cp)
    print(f"\n[DONE] {jp}\n       {mp}\n       {cp}")


if __name__ == "__main__":
    main()
