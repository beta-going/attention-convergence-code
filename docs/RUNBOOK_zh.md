# RUNBOOK — 完整复现指南

两条路线任选:
- **A 全量重跑**(需 GPU ≥24GB): S0→S12 顺序执行。
- **B 数据复算**(无 GPU): S0 → S3' → S8' → S9' → S12。发布数据已含
  `data/attention/*.json`(canonical 注意力)与 `tables/*.csv`(各表)。

## S0 环境
```bash
conda create -n ae python=3.10 -y && conda activate ae
pip install -r envs/requirements-workstation.txt
```
闸门: `python -c "import torch, transformers; print(torch.__version__)"` 无报错。
## S1 配置
编辑 `configs/models.yaml`: 填各模型的 `local_path`,或留空(自动回退 HF hub id)。可选: 设 `AE_MODELS_CONFIG` 指向替代配置(如本地路径变体);`configs/models_local.yaml` 已被 git-ignore。
```bash
python scripts/check_configs.py        # 闸门: exit 0
```
## S2 canonical 注意力抽取 【可跳: 先 `bash scripts/unpack_attention.sh` 从分卷还原这 8 个 JSON】
```bash
python src/observational/extract_attention.py
```
闸门: 8 个 JSON; seq_len ∈ {252, 268, 328, 269, 458, 284}(verify_all 覆查)。
## S3 观测统计 → Table 1 / 2 / 6, Fig 1
```bash
python src/observational/compute_convergence.py     # → tables/tab1*.csv + fig1
python src/observational/compute_entropy_table.py   # → tables/tab7*.csv
python src/observational/compute_posind.py          # → tables/tab3_posind_by_band.csv (+per_head.gz)
```
闸门: 8,704 layer–head 单元; Table 2 逐格 == 论文; Qwen3 early 比值 1.8–2.8×。
## S4 观测图 → Fig 2–8
```bash
python src/plotting/fig2_asym_heatmap.py && python src/plotting/fig3_head_patterns.py
python src/plotting/fig4_layer_entropy.py && python src/plotting/fig5_crosslingual.py
python src/plotting/fig6_8_column_mass.py
```
闸门: 与 `figures/published/` 目检一致(输出在 `results/figures/`)。
## S5 cross-input (Table 7 / App J)
```bash
python src/observational/build_crossinput_texts.py --seed 42 --out data/inputs/crossinput/crossinput_texts.json
python src/observational/crossinput_run.py
```
闸门: App J 双完整性闸门; T∈199–512; 160 对; 产出 stats_clean.csv。
## S6 探针准备 (Table 11)
```bash
python src/intervention/export_probes.py    # 已随迁移跑过; 核对计数 20/20, 10/10, 5/5
python src/intervention/run_intervention.py --stage filter
```
闸门: ZH retained = 10/10/9/10/11/9/10/11/13/9/10(论文 Table 11)。
## S7 干预主实验 (Table 3 / 4) — 最重: 11 配置 × 3 带 × 4 模
```bash
python src/intervention/run_intervention.py --bands early,mid,late --seed 42
python src/intervention/run_intervention.py --bands early --seeds 42,43,44 --shuffle-only
python src/intervention/merge_results.py    # → tables/tab4_tab5_probe_genppl.csv
```
闸门: seed42 主跑 == variance 跑 22 格逐格相等(对照 tables/verification_shufvar_22cells.txt)。
## S8 不对称指数 + bootstrap CI (Table 8 / 9)
```bash
python src/intervention/compute_asymmetry.py          # 33/33 复算
python src/intervention/bootstrap_ci.py --B 10000 --seed 0
```
闸门: 33/33 与论文一致; 18/33 排除 1。
## S9 generation-track 网格 (Table 10) 【纯推导, 无需 GPU】
```bash
python src/intervention/validate_alpha.py    # 数值闸门: Table 10 逐格一致 (图: plot_alpha_grid.py)
```
闸门: 与论文 Table 10 逐格一致。
## S10 matched-global 对照 (Table 12 / App T)
```bash
bash scripts/run_matched_global.sh   # 原 v8_mg.sh, 生产调用原样存档
python src/intervention/check_mg.py  # App T 完整性双闸门
```
闸门: 完整性双闸门(App T); UC/UG 格复现 Table 4。
## S11 INT4 对照 (App R)
```bash
无独立脚本: INT4 对照 = 主干预脚本以 Qwen3-8B-INT4-control 配置运行
# 逐句日志已随仓: tables/intervention_raw/Qwen3-8B-INT4-control_early/ (App R 数字来源)
```
闸门: App R 数字(90/90/85; −10pp; 19.43→24.57; 1.24→1.39)。
## S12 终检
```bash
python scripts/verify_all.py    # 全部闸门 PASS/FAIL 一览, 有 FAIL 则 exit 1
```

## S12 统计重算背书 (App D)

python src/observational/bonferroni_check.py  # 自测闸 → 136/136 逐层显著 + 剖面ρ 0.86–0.99; 输出 tables/bonferroni_check/
