# entropy-crosslingual
预印本 **"Attention Convergence Across Architectures: Cross-Lingual Structure and Surrogate-Dependent Causal Effects"**（Wan-Jiu Yang, 2026；DOI: 10.5281/zenodo.22987817）的代码、数据与单命令验证。
## 仓库内容
- **观测研究** — 8 个模型×语言配置（Llama-3.1-8B、Mistral-7B、Qwen3-4B/8B × en/zh）的 canonical 注意力：完整注意力矩阵、收敛/压缩统计、分深度带的 positional-induction ρ_pos（论文 Tables 1、2、6、7；Figs 1、3–8）。
- **干预研究** — 注意力带干预（early/mid/late；shuffle 与 mean 两种模式），覆盖 11 个配置（Qwen3-0.6B…14B 含 Base/Instruct/INT4 变体、Llama-3.1-8B Base/Instruct、InternLM3-8B、Mistral-7B），另有 Mixtral-8x7B 在另一套环境上运行以及 matched-global 对照；探针为 {en,zh}×{cloze 20, free 10, gen 5}（论文 Tables 3、4、8–12；Fig 2）。
- **原始证据** — 每个已发表数字背后的逐 probe/逐句流式日志（`tables/*_raw/`），以及从中重新推导这些数字的单命令验证器。本 artifact 复现的头条结果：Qwen3 early-band ρ_pos 相对 Llama/Mistral 的比值 **1.8–2.8×**（逐头 Welch t ≥ 7.2）、相对其自身 mid-band 为 1.5–1.7×；Table 8 不对称指数从原始日志**复算 33/33**；bootstrap CI（B=10 000，seed 0）在 **18/33** 格排除 1；seed-42 主跑与方差跑 **22/22** 格相等。
## 仓库结构
```
configs/models.yaml            # 模型 id / 本地路径（S1）
data/attention/                # 由 scripts/unpack_attention.sh 重建
data/inputs/canonical/         # en/zh 提示集（App A）
data/probes/                   # 6 个探针集：{en,zh}×{cloze,free,gen}
docs/ASSET_MAP_zh.md           # 论文表/图 ↔ 脚本 ↔ 数据 ↔ 日志 映射
docs/RUNBOOK_zh.md             # 完整复现指南，S0–S12
envs/                          # 固定版本的依赖（workstation / server）
figures/published/             # 论文最终图（Figs 1–8）
scripts/unpack_attention.sh    # 从分卷重建 data/attention/*.json
scripts/run_matched_global.sh  # matched-global 重跑（生产调用原样存档）
scripts/verify_all.py          # 所有已发表数字的单命令验证
scripts/check_configs.py       # S1 闸门
scripts/posind_per_head_check.py # 逐头 Welch 检验（t ≥ 7.2）
src/common/                    # 模型加载器、探针工具
src/observational/             # S2–S5：抽取 + 论文 Tables 1/2/6/7
src/intervention/              # S6–S11：探针、干预、不对称指数、CI（论文 Tables 3/4、8–12）
src/plotting/                  # Figs 2–8
tables/                        # 各表 CSV + 原始日志（intervention_raw*, matched_global_raw, selftest）
```
## 快速开始（无 GPU，约 10 分钟）
```bash
bash scripts/unpack_attention.sh
pip install -r envs/requirements-workstation.txt
python scripts/verify_all.py    # 预期输出: ALL PASS
```
## 完整复现
`docs/RUNBOOK_zh.md`：**路线 A** — 有 GPU 全量重跑（S0→S12）；**路线 B** — 纯数据复算。要定位任一论文数字背后的文件/脚本/日志，见 `docs/ASSET_MAP_zh.md`。
## 数据说明
- 完整文件已另存档于 https://doi.org/10.5281/zenodo.22990102（Zenodo，CC BY 4.0）；仓库内保留按字节精确切的 7 MB gzip 分卷。
- 原始日志有意保留模型本地路径与库版本（环境证据，Appendix E）；用户名/主机名已清洗为 `user@host/env`。
- `data/probes/zh_cloze.json` 是完整的 20 题原始池（Appendix S）；各模型保留子集（9–13 题，Δlog-prob > 0.5）记录于逐次运行日志（Table 11）。
## 许可证
- 代码：MIT（见 LICENSE）。
- 发布数据（探针数据集、统计结果、日志）：CC BY 4.0。
- 模型权重不予再分发；各模型按其自身许可使用（论文 Appendix E）。
## 引用
```bibtex
@misc{yang2026atlas,
  title = {Attention Convergence Across Architectures: Cross-Lingual Structure and Surrogate-Dependent Causal Effects},
  author = {Yang, Wan-Jiu},
  year = {2026},
  doi = {10.5281/zenodo.22987817},
  url = {https://doi.org/10.5281/zenodo.22987817}
}
```