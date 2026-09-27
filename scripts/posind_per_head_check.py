import pandas as pd
from scipy import stats
df = pd.read_csv("posind_per_head.csv.gz")
e = df[df.band=="early"]
for qc in ["Qwen3-8B_en","Qwen3-8B_zh","Qwen3-4B_en","Qwen3-4B_zh"]:
    for oc in ["Llama-3.1-8B_en","Llama-3.1-8B_zh","Mistral-7B_en","Mistral-7B_zh"]:
        if qc[-2:] != oc[-2:]: continue
        t = stats.ttest_ind(e[e.config==qc].rho_row, e[e.config==oc].rho_row, equal_var=False)
        u = stats.mannwhitneyu(e[e.config==qc].rho_row, e[e.config==oc].rho_row)
        print(f"{qc} vs {oc}: Welch t={t.statistic:.1f}, p={t.pvalue:.1e}; MWU p={u.pvalue:.1e}")
