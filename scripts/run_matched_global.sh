#!/bin/bash
# matched-support control (Exp2), archived from v8_mg.sh
# v2: model paths resolved from configs/models.yaml (single source of truth; S1 validates them)
# usage: bash scripts/run_matched_global.sh [--check]
set -uo pipefail
PYTHON=${PYTHON:-python}   # run inside your activated env
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
CODE="$PROJECT/src/intervention/run_intervention.py"
OUTBASE="$PROJECT/results_v8_mg"; LOGDIR="$OUTBASE/logs"
MODES="baseline,matched_global"            # 带同批 baseline, 配对更干净
BANDS="early mid late"
TIMEOUT_PER_BAND=14400
MODELS=( "Qwen3-8B||no|no" "Llama-3.1-8B||no|no" )   # 名称须与 configs/models.yaml 一致

model_path() {
  "${PYTHON}" - "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("configs/models.yaml"))
name = sys.argv[1]; hits = {}
PATH_KEYS = ("local_path", "path", "model_path", "weights")
def walk(node, pkey=None):
    if isinstance(node, dict):
        nm = node.get("name") or node.get("id") or node.get("model")
        pth = next((node[k] for k in PATH_KEYS if isinstance(node.get(k), str) and node[k]), None)
        if pth:
            key = str(nm) if nm else (str(pkey) if pkey else None)
            if key: hits.setdefault(key, pth)
        for k, v in node.items():
            if isinstance(v, (dict, list)): walk(v, k)
            elif isinstance(v, str) and "/" in v and k not in ("name","id","model","hf_id")+PATH_KEYS:
                hits.setdefault(str(k), v)
    elif isinstance(node, list):
        for x in node: walk(x, pkey)
walk(cfg)
if name not in hits:
    low = {k.lower(): v for k, v in hits.items()}
    if name.lower() not in low:
        sys.exit(f"[ERR] '{name}' not in configs/models.yaml; available: {sorted(hits)}")
    name = name.lower()
print(hits[name])

PY
}

[[ "${1:-}" == "--check" ]] && { for e in "${MODELS[@]}"; do IFS='|' read -r n _q _t <<< "$e"; echo "$n -> $(model_path "$n")"; done; exit 0; }

mkdir -p "$OUTBASE" "$LOGDIR"; exec > >(tee -a "$LOGDIR/v8_mg.log") 2>&1
run_band() {
  local name="$1" path="$2" out="${OUTBASE}/${name}_${3}"
  [[ -f "${out}/intervention_results.json" ]] && { echo "[SKIP] ${name}_${3}"; return 0; }
  mkdir -p "${out}"
  local cmd=("${PYTHON}" "${CODE}" --model "${path}" --layer_split "${3}" --modes "${MODES}" --zh_auto_filter --num_repeats 1 --output_dir "${out}")
  timeout "${TIMEOUT_PER_BAND}" "${cmd[@]}" && echo "DONE ${name}_${3}" || echo "FAILED ${name}_${3} (exit=$?)"
}
for entry in "${MODELS[@]}"; do
  IFS='|' read -r name _q4 _trc <<< "${entry}"
  path="$(model_path "${name}")" || exit 1
  for band in ${BANDS}; do run_band "${name}" "${path}" "${band}"; done
done
echo "=== MG COMPLETE ==="
