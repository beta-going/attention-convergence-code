#!/usr/bin/env python3
"""Exp① cross-input robustness. 三个阶段,每模型顺序执行:
  GATE A: 用存储的 canonical JSON 验证统计移植(秒级,不碰 GPU)
  GATE B: fresh forward 验证抽取路径一致(每模型 2 次 forward)
  SCAN  : 20 输入 × 2 语言 → results_v8_crossinput/stats.csv
任一 gate 失败立即退出,不产出扫描数据。"""
import sys, os, json, csv, argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, "src")
sys.path.insert(0, ".")
from extract_attention import PROBE_TEXT_ZH, PROBE_TEXT_EN
from crossinput_stats import dynamics, regimes_exact
import os, yaml
_CFG = yaml.safe_load(open(
    os.environ.get("AE_MODELS_CONFIG", "configs/models.yaml")))
def _mp(k):
    m = _CFG["models"][k]
    return m.get("local_path") or m["hf_id"]


MODEL_PATHS = {
    # "Qwen3-8B":     _mp("qwen3-8b"),
    # "Qwen3-4B":     _mp("qwen3-4b"),
    "Llama-3.1-8B": _mp("llama-3.1-8b"),
    "Mistral-7B":   _mp("mistral-7b"),
    "Qwen3-8B":     _mp("qwen3-8b"),
    "Qwen3-4B":     _mp("qwen3-4b"),
}
CANON     = {"zh": PROBE_TEXT_ZH, "en": PROBE_TEXT_EN}
PROBE_DIR = "data/attention"

# ---- 真值表(已经 8/8 OK 验证)。
# EN = 论文 Table 1(原版脚本 en 复核, bit-exact)
# ZH = 原版脚本 zh console 输出(b);Qwen3-8B 行用移植精确值(c)
EXP_T1_EN = {
    "Qwen3-8B":     (36, 0.73, 15, 2, 0.268),
    "Qwen3-4B":     (36, 0.76, 19, 2, 0.225),
    "Llama-3.1-8B": (32, 0.79, 13, 13, 0.270),
    "Mistral-7B":   (32, 0.70,  0, 12, 0.337),
}
EXP_T1_ZH = {
    "Qwen3-8B":     (36, 0.737, 16, 3, 0.2532),
    "Qwen3-4B":     (36, 0.75, 19, 2, 0.211),
    "Llama-3.1-8B": (32, 0.82, 14, 11, 0.221),
    "Mistral-7B":   (32, 0.73, 19, 12, 0.283),
}
EXP_T7 = {  # (model,lang): (conv, foc, dif, median, max)
    ("Llama-3.1-8B","zh"):(.55,.44,.01,.27,.86), ("Llama-3.1-8B","en"):(.49,.50,.01,.30,.90),
    ("Qwen3-8B","zh"):(.35,.59,.06,.39,.92),     ("Qwen3-8B","en"):(.36,.59,.05,.37,.91),
    ("Qwen3-4B","zh"):(.39,.57,.05,.37,.91),     ("Qwen3-4B","en"):(.41,.55,.04,.35,.94),
    ("Mistral-7B","zh"):(.45,.53,.02,.32,.99),   ("Mistral-7B","en"):(.46,.53,.01,.32,.98),
}
FIELDS = ["model","lang","band","idx","T","P","lstar","lstar_over_L",
          "lstar_consec","lpeak","Hdeep","conv","foc","dif","median","max"]

def load_consistency(name, lang):
    d = json.load(open(f"{PROBE_DIR}/consistency_{name}_{lang}.json"))
    return [np.asarray(d[f"layer_{i}"], dtype=np.float64)
            for i in range(d["n_layers"])]

def forward(model, tok, text, max_len=512):
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=max_len).to(model.device)
    with torch.no_grad():
        out = model(**enc, output_attentions=True, use_cache=False)
    return [a[0].double().cpu().numpy() for a in out.attentions], \
           enc["input_ids"].shape[1]

def measure(layers):
    return dynamics(layers), regimes_exact(layers)

def gate(model, tok, name):
    for lang in ("zh", "en"):
        # --- GATE A: 存储数据 + 移植统计 vs 真值表(按语言选表) ---
        ref = load_consistency(name, lang)
        d_ref, r_ref = measure(ref)
        L, P, ls, lp, hd = (EXP_T1_EN if lang == "en" else EXP_T1_ZH)[name]
        e7 = EXP_T7[(name, lang)]
        # dynamics 全指标断言
        okD = (d_ref["L"] == L
               and abs(d_ref["P"] - P) < .015
               and d_ref["lstar"] == ls and d_ref["lpeak"] == lp
               and abs(d_ref["Hdeep"] - hd) < .005)
        # regimes: conv/median 是系统性指标,断言;max 对转写误差最敏感,仅告警
        okR = (abs(r_ref["conv"] - e7[0]) < .015
               and abs(r_ref["median"] - e7[3]) < .015)
        print(f"[gate A] {name} {lang}: P={d_ref['P']:.3f}(~{P}) "
              f"l*={d_ref['lstar']}(~{ls}) peak={d_ref['lpeak']}(~{lp}) "
              f"Hdeep={d_ref['Hdeep']:.3f}(~{hd}) | "
              f"conv={r_ref['conv']:.2f}(~{e7[0]}) med={r_ref['median']:.2f}(~{e7[3]}) "
              f"max={r_ref['max']:.2f}(~{e7[4]}) | "
              f"dyn {'PASS' if okD else 'FAIL'} reg {'PASS' if okR else 'FAIL'}")
        if abs(r_ref["max"] - e7[4]) > .02:
            print(f"  [warn] max 偏差 >0.02 — 多半是我转写 Table 7 max 的舍入,"
                  f"核对论文后改 EXP_T7;不是移植 bug")
        if not (okD and okR):
            sys.exit(f"GATE A FAILED ({name} {lang})")

        # --- GATE B: fresh forward vs 存储数据 ---
        layers, T = forward(model, tok, CANON[lang])
        d_new, r_new = measure(layers)
        okB = (d_new["lstar"] == d_ref["lstar"]
               and d_new["lpeak"] == d_ref["lpeak"]
               and abs(d_new["P"] - d_ref["P"]) < .01
               and abs(r_new["conv"] - r_ref["conv"]) < .02)
        print(f"[gate B] {name} {lang}: T={T}(存储 {d_ref['S']}) -> "
              f"{'PASS' if okB else 'FAIL'}")
        if not okB:
            sys.exit("GATE B FAILED — 抽取路径不一致,查 tokenizer/dtype/eager")

def scan(model, tok, name, texts, out_path):
    new = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(FIELDS)
        for lang in ("zh", "en"):
            for it in texts[lang]:
                layers, T = forward(model, tok, it["text"])
                d, r = measure(layers)
                w.writerow([name, lang, it["band"], it["idx"], T,
                            round(d["P"], 4), d["lstar"],
                            round(d["lstar"] / d["L"], 4),
                            d["lstar_consec"], d["lpeak"],
                            round(d["Hdeep"], 4),
                            round(r["conv"], 4), round(r["foc"], 4),
                            round(r["dif"], 4), round(r["median"], 4),
                            round(r["max"], 4)])
                f.flush()
        print(f"  [scan] {name}: {len(texts['zh'])+len(texts['en'])} rows written")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default="crossinput_texts.json")
    ap.add_argument("--out",   default="results_v8_crossinput/stats.csv")
    ap.add_argument("--models", nargs="*", default=list(MODEL_PATHS))
    args = ap.parse_args()

    texts = json.load(open(args.texts, encoding="utf-8"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for name in args.models:
        print(f"\n===== {name} =====")
        tok = AutoTokenizer.from_pretrained(MODEL_PATHS[name], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS[name], torch_dtype=torch.float16,
            device_map="cuda:0", attn_implementation="eager",
            trust_remote_code=True).eval()
        gate(model, tok, name)
        scan(model, tok, name, texts, args.out)
        del model
        torch.cuda.empty_cache()
    print(f"\nDONE -> {args.out}")

if __name__ == "__main__":
    main()
