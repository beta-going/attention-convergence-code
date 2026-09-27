# ASSET_MAP — Four-way mapping: paper artifact ↔ script ↔ data (label column = paper LaTeX \label, directly searchable)
> **Naming note:** script-output prefixes (`tab1*`, `tab3_*`, `tab4_tab5*`, `tab6_*`, `tab7*`) preserve an earlier draft's numbering and deliberately do NOT match final paper table numbers (renaming would break `python scripts/verify_all.py` and archived-log paths). The mapping in this file is authoritative.

## Main tables
| Paper artifact | label | Generating script | Input | RUNBOOK stage |
|---|---|---|---|---|
| Table 1 convergence statistics | tab:convergence | src/observational/compute_convergence.py | data/attention/*.json | S3 |
| Table 13 compression classification | tab:compression | (mechanically derived from Table 1: phase-boundary formulas in its caption) | tables/tab1* | derived after S3 |
| Table 2 ρ_pos by band | tab:posind | src/observational/compute_posind.py | data/attention/*.json | S3 |
| Table 3 probe accuracy | tab:scaling | run_intervention.py + merge_results.py | data/probes/*.json | S6–S7 |
| Table 4 gen-PPL | tab:scaling_genppl | same scripts (genppl columns of the same run) | same as above | S7 |
| Table 8 asymmetry α | tab:asym_prop | compute_asymmetry.py | per-sentence logs | S8 |
| Table 6 entropy regimes | tab:entropy | compute_entropy_table.py | data/attention/*.json | S3 |
| Table 7 cross-input | tab:crossinput | crossinput_run.py | data/inputs/crossinput/ | S5 |
| Table 9 bootstrap CI | tab:alpha_ci | bootstrap_ci.py (--B 10000 --seed 0) | per-sentence logs | S8 |
| Table 10 α_gen grid | tab:gen_alpha | validate_alpha.py / plot_alpha_grid.py | tables/tab4_tab5* | S9 |
| Table 11 probe counts | tab:probe_counts | run_intervention.py --stage filter | data/probes/*.json | S6 |
| Table 12 matched-global | tab:matchedglobal | matched_global.py + check_mg.py | same setup as Tables 3/4 | S10 |
## Figures
| Paper figure | label | Script | Published original |
|---|---|---|---|
| Fig 1 convergence dynamics | fig:convergence | compute_convergence.py | figures/published/figure_convergence_depth.png |
| Fig 2 asymmetry heatmap | fig:asym_heatmap | plotting/fig2_asym_heatmap.py | .../figure_asym_heatmap.png |
| Fig 3 head patterns | fig:heads | plotting/fig3_head_patterns.py | .../figure_head_patterns_long_v5.png |
| Fig 4 layer×model entropy | fig:layerentropy | plotting/fig4_layer_entropy.py | .../figure_layer_entropy_heatmap.png |
| Fig 5 cross-lingual consistency | fig:figure_crosslingual | plotting/fig5_crosslingual.py | .../figure_crosslingual_consistency.png |
| Fig 6–8 column mass | fig:colmass_* | plotting/fig6_8_column_mass.py | .../fig_column_mass_*.pdf, fig8_deepdive_llama_en.pdf |
## Appendices
| Appendix | Content | Source |
|---|---|---|
| App A | Input/probe protocol | data/inputs/canonical/*.txt + data/probes/*.json |
| App D | Statistical-test recomputation | src/observational/bonferroni_check.py + tables/bonferroni_check/ |
| App E | Dual-environment statement | envs/README_en.md (versions correspond verbatim) |
| App J | Cross-input protocol | build_crossinput_texts.py (seed 42) + stats_clean.csv |
| App L | Punctuation column alignment | compute_punct_alignment.py |
| App O | Bootstrap CI | bootstrap_ci.py |
| App P | Generation-track grid | validate_alpha.py / plot_alpha_grid.py |
| App R | INT4 control | main intervention script with the INT4 config variant → tables/intervention_raw/Qwen3-8B-INT4-control_early/ |
| App T | Matched-global | matched_global.py + check_mg.py + tables/matched_global_raw/ |
## Direct lookup paths for key numbers in the main text
- Intro axis B / §4.2 "1.8–2.8×": tables/tab3_posind_by_band.csv → cross ratios in the early column
- "Welch t≥7.2": tables/posind_per_head.csv.gz → scripts/posind_per_head_check.py
- "22-cell reproduction": tables/verification_shufvar_22cells.txt
- "33/33 recomputation": `python src/intervention/compute_asymmetry.py tables/intervention_raw tables/intervention_raw_cloud` → generates tables/tab6_asymmetry_recomputed.csv, all 33 cells equal paper Table 8
## Raw data assets (published evidence under tables/)
| Asset | Content |
|---|---|
| tables/intervention_raw/ | Raw per-sentence logs behind Tables 3/4/8/9. v8.0 unified re-run in the second environment (workstation: PyTorch 2.10.0+cu128 / Transformers 4.52.4, Appendix E); the 7 configurations with previously reported values match the old values cell-by-cell (witnessed by the 33/33 recomputation; old values recorded in the PAPER dict of compute_asymmetry.py) |
| tables/intervention_raw_cloud/ | Third environment (server: PyTorch 2.8.0+cu128 / Transformers 5.16.1, Appendix E): Mixtral-8x7B three bands + early_shuffvar; Table 8's three cells reproduced 3/3 |
| tables/selftest/ | Full archived output of the intervention engine's built-in self-test (--self-test) (Qwen3-8B, workstation environment): hooks fire, bit-exact restoration after disabling, value-space ≈ weight-space equivalence. The max deviation 2.4e-4 stated in Appendix E is the self-test in the Mixtral server environment; its run output was not archived separately — see paper Appendix E and the engine code shipped with the repo |
| tables/bonferroni_check/ | App D recomputation output: summary.csv + per_layer.csv (136 rows, per-layer ρ) + ℓ* ordering sanity |
| tables/matched_global_raw/ | Table 12 raw logs + integrity-check records |
| tables/crossinput_raw/ and tables/stats_clean.csv | Table 7 raw data and cleaned data (160 unique keys, no conflicts) |
| tables/tab6_asymmetry_raw.csv and tab6_asymmetry_recomputed.csv | Historical snapshot of Table 8 / result recomputed from the logs |
| tables/tab9_alpha_ci.csv | Table 9 recomputation (B=10000, seed=0) |
| tables/verification_shufvar_22cells.txt | Record that the seed-42 main run and the variance run match cell-by-cell in all 22 cells |
| tables/crossinput_diag_report.txt | Cross-input integrity-check report (240 rows → 160 keys, converged/focused ≥ 0.90, T ∈ 199–512) |
## Probe datasets (data/probes/, 6 JSONs)
en/zh × cloze (20/20) / free (10/10) / gen (5/5). zh_cloze.json is the raw 20-item pool backing Appendix S; each model's retained subset (9–13 items) is produced by the automatic Δlog-prob > 0.5 filter; per-probe metadata lives in each run log.
Path-retention policy: the model-local absolute paths and software version strings in the raw logs are environment evidence (Appendix E) and are intentionally retained; usernames / hostnames / environment names have been replaced with user@host/env.
## Data restoration
> Individual raw attention-matrix files exceed 100 MB; they are stored as 7 MB gzip split volumes under data/attention_parts/ (reconciling the GitHub 100 MB single-file limit with the 8 MB upload cap of anonymous.4open.science). After cloning, first run `bash scripts/unpack_attention.sh` to restore data/attention/*.json, then run `python scripts/verify_all.py`. Local copies that are already restored are unaffected.