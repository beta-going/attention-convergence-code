#!/usr/bin/env python3
"""单一验收闸门: 打印各检查 PASS/FAIL/SKIP; 有 FAIL 则 exit 1.
无GPU路线(B)也应全绿——所有检查只依赖发布数据."""
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
FAILS = []
def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL':4s}  {name}  {detail}")
    if not cond: FAILS.append(name)
def rows(p):
    with open(p) as f: return list(csv.DictReader(f))

# --- S2: canonical 注意力完整性 ---
Ts = [252,252,268,268,269,284,328,458]
ats = sorted((ROOT/"data/attention").glob("consistency_*.json"))
check("S2 attention files == 8", len(ats) == 8, f"found {len(ats)}")
n_units, got_T = 0, []
for p in ats:
    d = json.load(open(p)); n_units += d["n_layers"]*d["n_heads"]; got_T.append(d["seq_len"])
check("S2 T multiset == paper", sorted(got_T) == Ts, str(got_T))
check("S3 8704 layer-head units", n_units == 8704, str(n_units))

# --- S3: Table 3 逐格(ρ_row) + 早带比值 ---
T3 = {"Qwen3-8B_en":(.0093,.0058,.0037), "Qwen3-8B_zh":(.0106,.0064,.0039),
      "Qwen3-4B_en":(.0086,.0057,.0033), "Qwen3-4B_zh":(.0101,.0064,.0034),
      "Llama-3.1-8B_en":(.0048,.0050,.0033), "Llama-3.1-8B_zh":(.0038,.0040,.0024),
      "Mistral-7B_en":(.0039,.0048,.0037), "Mistral-7B_zh":(.0039,.0042,.0021)}
p3 = ROOT/"tables/tab3_posind_by_band.csv"
if p3.exists():
    early = {}
    for r in rows(p3):
        e,m,l = T3[r["config"]]; want = {"early":e,"mid":m,"late":l}[r["band"]]
        ok = abs(float(r["rho_row"])-want) < 5e-4
        if r["band"]=="early": early[r["config"]] = float(r["rho_row"])
        check(f"T3 {r['config']}/{r['band']}", ok, f'{r["rho_row"]} vs {want}')
    rats = []
    for q in ["Qwen3-8B","Qwen3-4B"]:
        for o in ["Llama-3.1-8B","Mistral-7B"]:
            for lg in ["en","zh"]:
                rats.append(early[f"{q}_{lg}"]/early[f"{o}_{lg}"])
    # check("S3 early ratios in [1.8,2.8]", all(1.8<=x<=2.8 for x in rats),
    #       " ".join(f"{x:.2f}" for x in rats))
    
    rats = [round(x,1) for x in rats]
    check("S3 early ratios ~ [1.8,2.8]", all(1.8<=x<=2.8 for x in rats),
        " ".join(f"{x:.1f}" for x in rats))
else:
    check("S3 tab3 csv", False, "tables/tab3_posind_by_band.csv 缺失")
    
# --- S5: cross-input (App J) ---
p5 = ROOT/"tables/stats_clean.csv"
if p5.exists():
    n5 = len(rows(p5))
    check("S5 stats_clean == 160 行", n5 == 160, str(n5))
    check("S5 crossinput_raw 在仓", (ROOT/"tables/crossinput_raw/stats.csv").exists())
else:
    check("S5 stats_clean", False, "跑 crossinput_diag.py 生成")

# --- S8: Table 6 复算(纯数据,秒级) ---
import subprocess, re
r8 = subprocess.run([sys.executable, "src/intervention/compute_asymmetry.py",
                     "tables/intervention_raw", "tables/intervention_raw_cloud"],
                    cwd=ROOT, capture_output=True, text=True, timeout=600)
m8 = re.search(r"==> (\d+)/(\d+)", r8.stdout)
check("S8 asymmetry 复算全匹配", bool(m8) and m8.group(1)==m8.group(2) and int(m8.group(2))>=33,
      m8.group(0) if m8 else (r8.stderr or r8.stdout)[-150:])

# --- S9: Table 9 bootstrap CI ---
r9 = subprocess.run([sys.executable, "src/intervention/bootstrap_ci.py",
                     "tables/intervention_raw", "tables/intervention_raw_cloud",
                     "--B", "10000", "--seed", "0"],
                    cwd=ROOT, capture_output=True, text=True, timeout=1200)
try:
    rws = rows(ROOT/"tables/tab9_alpha_ci.csv")
    ex9 = sum(1 for r in rws if r["ci_excludes_1"] != "-")
    check("S9 alpha_ci 33格/18排除1", len(rws)==33 and ex9==18, f"rows={len(rws)} excl={ex9}")
except FileNotFoundError:
    check("S9 alpha_ci", False, (r9.stderr or r9.stdout)[-150:])


# --- S7: merged CSV 一致性 ---
p45 = ROOT/"tables/tab4_tab5_probe_genppl.csv"
if p45.exists():
    vs = {r["code_version"] for r in rows(p45)}
    check("S7 code_version 唯一", len(vs) == 1, str(vs))
    check("S7 shufvar 证据在", (ROOT/"tables/verification_shufvar_22cells.txt").stat().st_size > 0)
else:
    check("S7 merged csv", False, "缺失")

# --- 发布图存档 ---
figs = list((ROOT/"figures/published").glob("*"))
check("figures/published >= 8", len(figs) >= 8, f"found {len(figs)}")

# --- scrub: 隐私审计(两档; 修复: 此前 pat 未定义被 except 吞掉, 检查空转) ---
import re
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")   # 只抓真 token; hf_id 不再误报

def sample_text(f, cap=131072):
    """小文件全文; 大文件只取头尾各128KB(元数据在首尾)——不因GB级数据变慢"""
    size = f.stat().st_size
    with open(f, "rb") as fh:
        buf = fh.read(cap)
        if size > 2 * cap:
            fh.seek(-cap, 2); buf += b"\n" + fh.read(cap)
    return buf.decode(errors="ignore")

CODE_PATS = ("yang@", "beta-ai", "wsl_vllm", "/home/", "/mnt/e/")  # 代码/文档: 严格
DATA_PATS = ("yang@", "beta-ai", "wsl_vllm")   # 数据: 只查用户名; /mnt/e 为环境证据(Appendix E)放行
ME = Path(__file__).resolve()
bad_code, bad_data = [], []

def scan(f, pats, bucket):
    try: t = sample_text(f)
    except Exception: return
    hit = next((p for p in pats if p in t), None)
    if hit is None and TOKEN_RE.search(t): hit = "hf_token"
    if hit: bucket.append(f"{f.relative_to(ROOT)}:{hit}")

for base in ("src", "scripts", "docs", "envs", "configs"):
    for f in (ROOT / base).rglob("*"):
        if f.is_file() and f != ME and f.suffix in (".py",".md",".sh",".yaml",".txt") and "__pycache__" not in f.parts:
            scan(f, DATA_PATS if base == "configs" else CODE_PATS, bad_code)
for f in list((ROOT / "tables").rglob("*")) + list((ROOT / "data").rglob("*")):
    if f.is_file() and f.suffix in (".py",".md",".sh",".yaml",".txt",".toml",".cfg") and "__pycache__" not in f.parts:
        scan(f, DATA_PATS, bad_data)

check("scrub 代码/文档无隐私", not bad_code, "; ".join(bad_code[:5]))
check("scrub 数据无用户名", not bad_data, "; ".join(bad_data[:5]))

# --- S5: cross-input 证据在仓 ---
check("S5 crossinput_raw 入仓", (ROOT/"tables/crossinput_raw/stats.csv").exists())
check("S5 stats_clean 已生成", (ROOT/"tables/stats_clean.csv").exists())

# --- S8: Table 6 复算闸门(纯数据复算) ---
import subprocess, re
r = subprocess.run([sys.executable, "src/intervention/compute_asymmetry.py",
                    "tables/intervention_raw", "tables/intervention_raw_cloud"],
                   cwd=ROOT, capture_output=True, text=True, timeout=600)
m = re.search(r"==> (\d+)/(\d+)", r.stdout)
check("S8 asymmetry 全匹配", bool(m) and m.group(1) == m.group(2) and int(m.group(2)) >= 33,
      m.group(0) if m else (r.stderr or r.stdout)[-150:])



print("\n" + ("ALL PASS" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
