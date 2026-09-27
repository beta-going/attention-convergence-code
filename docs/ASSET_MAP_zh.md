# ASSET_MAP — 论文物件 ↔ 脚本 ↔ 数据 四方映射 (标签列 = 论文 LaTeX \label, 可直接检索)

> **命名说明：** 脚本输出前缀（`tab1*`、`tab3_*`、`tab4_tab5*`、`tab6_*`、`tab7*`）沿用早期草稿的编号，**有意不与**论文最终表号对应（重命名会破坏 `python scripts/verify_all.py` 与存档日志路径）。本文件的映射为唯一权威。

## 主表

| 论文物件 | label | 生成脚本 | 输入 | RUNBOOK 阶段 |
|---|---|---|---|---|
| Table 1 收敛统计 | tab:convergence | src/observational/compute_convergence.py | data/attention/*.json | S3 |
| Table 13 压缩分类 | tab:compression | (由 Table 1 机械推导: phase 边界公式见其 caption) | tables/tab1* | S3 后推导 |
| Table 2 ρ_pos 分带 | tab:posind | src/observational/compute_posind.py | data/attention/*.json | S3 |
| Table 3 探针准确率 | tab:scaling | run_intervention.py + merge_results.py | data/probes/*.json | S6–S7 |
| Table 4 gen-PPL | tab:scaling_genppl | 同上(同一次运行的 genppl 列) | 同上 | S7 |
| Table 8 不对称 α | tab:asym_prop | compute_asymmetry.py | per-sentence 日志 | S8 |
| Table 6 熵分区 | tab:entropy | compute_entropy_table.py | data/attention/*.json | S3 |
| Table 7 cross-input | tab:crossinput | crossinput_run.py | data/inputs/crossinput/ | S5 |
| Table 9 bootstrap CI | tab:alpha_ci | bootstrap_ci.py (--B 10000 --seed 0) | per-sentence 日志 | S8 |
| Table 10 α_gen 网格 | tab:gen_alpha | validate_alpha.py / plot_alpha_grid.py | tables/tab4_tab5* | S9 |
| Table 11 探针计数 | tab:probe_counts | run_intervention.py --stage filter | data/probes/*.json | S6 |
| Table 12 matched-global | tab:matchedglobal | matched_global.py + check_mg.py | 同 Table 3/4 设置 | S10 |

## 图

| 论文图 | label | 脚本 | 已发布原图 |
|---|---|---|---|
| Fig 1 收敛动力学 | fig:convergence | compute_convergence.py | figures/published/figure_convergence_depth.png |
| Fig 2 不对称热图 | fig:asym_heatmap | plotting/fig2_asym_heatmap.py | .../figure_asym_heatmap.png |
| Fig 3 头级模式 | fig:heads | plotting/fig3_head_patterns.py | .../figure_head_patterns_long_v5.png |
| Fig 4 层×模型熵 | fig:layerentropy | plotting/fig4_layer_entropy.py | .../figure_layer_entropy_heatmap.png |
| Fig 5 跨语言一致 | fig:figure_crosslingual | plotting/fig5_crosslingual.py | .../figure_crosslingual_consistency.png |
| Fig 6–8 列质量 | fig:colmass_* | plotting/fig6_8_column_mass.py | .../fig_column_mass_*.pdf, fig8_deepdive_llama_en.pdf |

## 附录

| 附录 | 内容 | 出处 |
|---|---|---|
| App A | 输入/探针协议 | data/inputs/canonical/*.txt + data/probes/*.json |
| App D | 统计检验重算 | src/observational/bonferroni_check.py + tables/bonferroni_check/ |
| App E | 双环境声明 | envs/README_zh.md(版本逐字对应) |
| App J | cross-input 协议 | build_crossinput_texts.py(seed 42) + stats_clean.csv |
| App L | 标点列对齐 | compute_punct_alignment.py |
| App O | bootstrap CI | bootstrap_ci.py |
| App P | 生成轨网格 | validate_alpha.py / plot_alpha_grid.py |
| App R | INT4 对照 | 主干预脚本 INT4 配置变体 → tables/intervention_raw/Qwen3-8B-INT4-control_early/ |
| App T | matched-global | matched_global.py + check_mg.py + tables/matched_global_raw/ |

## 正文关键数字的直查路径

- 引言轴 B / §4.2 "1.8–2.8×": tables/tab3_posind_by_band.csv → early 列交叉比值
- "Welch t≥7.2": tables/posind_per_head.csv.gz → scripts/posind_per_head_check.py
- "22 格复现": tables/verification_shufvar_22cells.txt
- "33/33 复算": python src/intervention/compute_asymmetry.py tables/intervention_raw tables/intervention_raw_cloud → 生成 tables/tab6_asymmetry_recomputed.csv，33 格全部等于论文 Table 8

## 原始数据资产（tables/ 下的发布证据）

| 资产 | 内容 |
|---|---|
| tables/intervention_raw/ | Table 3/4/8/9 的原始逐句日志。第二环境（工作站：PyTorch 2.10.0+cu128 / Transformers 4.52.4，附录E）的 v8.0 统一重跑；7 个先前报告过数值的配置与旧值逐格相等（由 33/33 复算体现，旧值记录于 compute_asymmetry.py 的 PAPER 字典） |
| tables/intervention_raw_cloud/ | 第三环境（服务器：PyTorch 2.8.0+cu128 / Transformers 5.16.1，附录E）：Mixtral-8x7B 三个 band + early_shuffvar；Table 8 三格 3/3 复现 |
| tables/selftest/ | 干预引擎内置自检（--self-test）的完整运行输出存档（Qwen3-8B，工作站环境）：hooks 生效、关断后逐位恢复、value-space≈weight-space 等价性。附录E 所述 max deviation 2.4e-4 为 Mixtral 服务器环境的自检，其运行输出未单独保留，以论文附录E与随仓引擎代码为准 |
| tables/bonferroni_check/ | App D 重算输出: summary.csv + per_layer.csv(136行逐层ρ) + l*排序sanity |
| tables/matched_global_raw/ | Table 12 原始日志 + 完整性检查记录 |
| tables/crossinput_raw/ 与 tables/stats_clean.csv | Table 7 原始数据与清理后数据（160 个键唯一、无冲突） |
| tables/tab6_asymmetry_raw.csv 与 tab6_asymmetry_recomputed.csv | Table 8 的历史快照 / 本次从日志复算的结果 |
| tables/tab9_alpha_ci.csv | Table 9 复算结果（B=10000, seed=0） |
| tables/verification_shufvar_22cells.txt | seed42 主跑与方差跑 22 格逐格相等的对照记录 |
| tables/crossinput_diag_report.txt | cross-input 完整性检查报告（240 行→160 键、收敛/聚焦≥0.90、T∈199–512） |

## 探针数据集（data/probes/，6 个 JSON）

en/zh × cloze(20/20) / free(10/10) / gen(5/5)。zh_cloze.json 为附录 S 的 20 题原始池；各模型保留子集（9–13 题）由自动 Δlog-prob>0.5 过滤产生，逐 probe 元数据在各运行日志中。

路径保留政策：原始日志里的模型本地绝对路径和软件版本号是环境证据（附录 E），有意保留；用户名/主机名/环境名已替换为 user@host/env。

## 数据还原

> 原始注意力矩阵单文件>100MB，按 7MB gzip 分卷存于 data/attention_parts/（兼顾 GitHub 100MB 单文件限制与 anonymous.4open.science 的 8MB 上限）；克隆后先运行 bash scripts/unpack_attention.sh 还原 data/attention/*.json，再跑 `python scripts/verify_all.py`。本地已还原的不受影响。
