# RUNBOOK — Full Reproduction Guide
Two routes, choose either:
- **A. Full re-run** (requires GPU ≥ 24 GB): execute S0→S12 in order.
- **B. Data-only recomputation** (no GPU): S0 → S3' → S8' → S9' → S12. The released data already include `data/attention/*.json` (canonical attention) and `tables/*.csv` (all tables).
## S0 Environment
```bash
conda create -n ae python=3.10 -y && conda activate ae
pip install -r envs/requirements-workstation.txt
```
Gate: `python -c "import torch, transformers; print(torch.__version__)"` runs without error.
## S1 Configuration
Edit `configs/models.yaml`: fill in each model's `local_path`, or leave it empty (automatic fallback to the HF hub id). Optional: set `AE_MODELS_CONFIG` to point at an alternative config (e.g. a local-path variant); `configs/models_local.yaml` is git-ignored.
```bash
python scripts/check_configs.py
# Gate: exit 0
```
## S2 Canonical attention extraction
[Skippable: first run `bash scripts/unpack_attention.sh` to restore the 8 JSONs from the split volumes]
```bash
python src/observational/extract_attention.py
```
Gate: 8 JSONs; seq_len ∈ {252, 268, 328, 269, 458, 284} (re-checked by verify_all).
## S3 Observational statistics → Table 1 / 2 / 6, Fig 1
```bash
python src/observational/compute_convergence.py   # → tables/tab1*.csv + fig1
python src/observational/compute_entropy_table.py # → tables/tab7*.csv
python src/observational/compute_posind.py        # → tables/tab3_posind_by_band.csv (+per_head.gz)
```
Gate: 8,704 layer–head units; Table 2 matches the paper cell-by-cell; Qwen3 early ratio 1.8–2.8×.
## S4 Observational figures → Fig 2–8
```bash
python src/plotting/fig2_asym_heatmap.py && python src/plotting/fig3_head_patterns.py
python src/plotting/fig4_layer_entropy.py && python src/plotting/fig5_crosslingual.py
python src/plotting/fig6_8_column_mass.py
```
Gate: visual inspection matches `figures/published/` (outputs go to `results/figures/`).
## S5 Cross-input (Table 7 / App J)
```bash
python src/observational/build_crossinput_texts.py --seed 42 --out data/inputs/crossinput/crossinput_texts.json
python src/observational/crossinput_run.py
```
Gate: App J dual integrity gates; T ∈ 199–512; 160 pairs; produces stats_clean.csv.
## S6 Probe preparation (Table 11)
```bash
python src/intervention/export_probes.py
# already run during migration; verify counts 20/20, 10/10, 5/5
python src/intervention/run_intervention.py --stage filter
```
Gate: ZH retained = 10/10/9/10/11/9/10/11/13/9/10 (paper Table 11).
## S7 Intervention main experiment (Table 3 / 4) — heaviest: 11 configs × 3 bands × 4 modes
```bash
python src/intervention/run_intervention.py --bands early,mid,late --seed 42
python src/intervention/run_intervention.py --bands early --seeds 42,43,44 --shuffle-only
python src/intervention/merge_results.py
# → tables/tab4_tab5_probe_genppl.csv
```
Gate: seed-42 main run == variance run, all 22 cells equal cell-by-cell (compare tables/verification_shufvar_22cells.txt).
## S8 Asymmetry index + bootstrap CI (Table 8 / 9)
```bash
python src/intervention/compute_asymmetry.py     # 33/33 recomputed
python src/intervention/bootstrap_ci.py --B 10000 --seed 0
```
Gate: 33/33 match the paper; 18/33 exclude 1.
## S9 Generation-track grid (Table 10) [pure derivation, no GPU needed]
```bash
python src/intervention/validate_alpha.py   # numeric gate: Table 10 matches cell-by-cell (figure: plot_alpha_grid.py)
```
Gate: matches paper Table 10 cell-by-cell.
## S10 Matched-global control (Table 12 / App T)
```bash
bash scripts/run_matched_global.sh   # formerly v8_mg.sh; the production invocation is archived as-is
python src/intervention/check_mg.py  # App T dual integrity gates
```
Gate: App T dual integrity gates; UC/UG cells reproduce Table 4.
## S11 INT4 control (App R)
```bash
# No standalone script: the INT4 control = the main intervention script run with the
# Qwen3-8B-INT4-control configuration
# Per-sentence logs ship with the repo: tables/intervention_raw/Qwen3-8B-INT4-control_early/
# (source of the App R numbers)
```
Gate: App R numbers (90/90/85; −10 pp; 19.43→24.57; 1.24→1.39).
## S12 Final check
```bash
python scripts/verify_all.py   # PASS/FAIL overview of all gates; exit 1 on any FAIL
```
## S12 Statistical re-computation backing (App D)
```bash
python src/observational/bonferroni_check.py
# self-test gate → 136/136 layers significant + profile ρ 0.86–0.99; outputs under tables/bonferroni_check/
```