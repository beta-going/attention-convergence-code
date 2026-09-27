# entropy-crosslingual
Code, data, and one-shot verification for the preprint **"Attention Convergence Across Architectures: Cross-Lingual Structure and Surrogate-Dependent Causal Effects"** (Wan-Jiu Yang, 2026; DOI: 10.5281/zenodo.22987817).
## What this repo contains
- **Observational study** — canonical attention for 8 model×language configs (Llama-3.1-8B, Mistral-7B, Qwen3-4B/8B × en/zh): full attention matrices, convergence/compression statistics, positional-induction ρ_pos by depth band (paper Tables 1, 2, 6, 7; Figs 1, 3–8).
- **Intervention study** — attention-band interventions (early/mid/late; shuffle & mean modes) across 11 configurations (Qwen3-0.6B…14B incl. Base/Instruct/INT4 variants, Llama-3.1-8B Base/Instruct, InternLM3-8B, Mistral-7B) plus Mixtral-8x7B on a second environment and matched-global controls, over {en,zh}×{cloze 20, free 10, gen 5} probes (paper Tables 3, 4, 8–12; Fig 2).
- **Raw evidence** — per-probe/per-sentence streaming logs behind every published number (`tables/*_raw/`), and a one-shot verifier that re-derives the numbers from them.
Headline results reproduced in this artifact: Qwen3 early-band ρ_pos ratios **1.8–2.8×** over Llama/Mistral (per-head Welch t ≥ 7.2) and 1.5–1.7× vs. its own mid-band; Table 8 asymmetry **recomputed 33/33** from raw logs; bootstrap CIs (B=10 000, seed 0) exclude 1 in **18/33** cells; seed-42 main run equals the variance run in **22/22** cells.
## Repository layout
```
configs/models.yaml         # model ids / local paths (S1)
data/attention/                # rebuilt by scripts/unpack_attention.sh
data/inputs/canonical/         # en/zh prompt sets (App A)
data/probes/                   # 6 probe sets: {en,zh}×{cloze,free,gen}
docs/ASSET_MAP_en.md           # paper table/figure ↔ script ↔ data ↔ log map
docs/RUNBOOK_en.md             # full reproduction guide, S0–S12
envs/                          # pinned requirements (workstation / server)
figures/published/             # final paper figures (Figs 1–8)
scripts/unpack_attention.sh    # rebuild data/attention/*.json from split parts
scripts/run_matched_global.sh  # matched-global rerun (archived production call)
scripts/verify_all.py          # one-shot verification of all published numbers
scripts/check_configs.py       # S1 gate
scripts/posind_per_head_check.py  # per-head Welch tests (t ≥ 7.2)
src/common/                    # model loader, probe utils
src/observational/             # S2–S5: extraction + paper Tables 1/2/6/7
src/intervention/              # S6–S11: probes, interventions, asymmetry, CI (paper Tables 3/4, 8–12)
src/plotting/                  # Figs 2–8
tables/                        # per-table CSVs + raw logs (intervention_raw*, matched_global_raw, selftest)
```
## Quick start (no GPU, ~10 min)
```bash
bash scripts/unpack_attention.sh
pip install -r envs/requirements-workstation.txt
python scripts/verify_all.py        # expect: ALL PASS
```
## Full reproduction
`docs/RUNBOOK_en.md`: **Route A** — full rerun with GPU (S0→S12); **Route B** — data-only recomputation. To locate the file/script/log behind any paper number, see `docs/ASSET_MAP_en.md`.
## Data notes
- Canonical attention JSONs are ~100–500 MB each (each exceeds GitHub's 100 MB file limit); the full files are additionally archived at https://doi.org/10.5281/zenodo.22990102 (Zenodo, CC BY 4.0) — byte-exact 7 MB gzip parts are kept in this repo, `scripts/unpack_attention.sh` restores them, `cmp`-verified.
- Raw logs intentionally keep local model paths and library versions (environment evidence, Appendix E); usernames/hostnames are scrubbed to `user@host/env`.
- `data/probes/zh_cloze.json` is the full 20-item pool (Appendix S); per-model retained subsets (9–13 probes, Δlog-prob > 0.5) are recorded in per-run logs (Table 11).
## License
- Code: MIT (see LICENSE).
- Released data (probe datasets, statistics, logs): CC BY 4.0.
- Model weights are not redistributed; each model is used under its own license (Appendix E of the paper).
## Citation
```bibtex
@misc{yang2026atlas,
  title = {Attention Convergence Across Architectures: Cross-Lingual Structure and Surrogate-Dependent Causal Effects},
  author = {Yang, Wan-Jiu},
  year = {2026},
  doi = {10.5281/zenodo.22987817},
  url = {https://doi.org/10.5281/zenodo.22987817}
}
