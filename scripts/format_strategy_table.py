#!/usr/bin/env python3
from pathlib import Path

import pandas as pd

p = Path('.results/strategy_comparison.tsv')
full = Path('.results/strategy_comparison_full.tsv')
md = Path('.results/strategy_comparison.md')

df = pd.read_csv(p, sep='\t')
df.to_csv(full, sep='\t', index=False)

cols = [
    'model',
    'accuracy', 'accuracy_%',
    'f1_macro', 'f1_macro_%',
    'f1_bacteria', 'f1_bacteria_%',
    'f1_eukaryota', 'f1_eukaryota_%',
    'f1_plasmid', 'f1_plasmid_%',
    'f1_virus', 'f1_virus_%',
    'recall_bacteria', 'recall_bacteria_%',
    'recall_eukaryota', 'recall_eukaryota_%',
    'recall_plasmid', 'recall_plasmid_%',
    'recall_virus', 'recall_virus_%',
    'p_thr', 'v_thr', 'circ_len'
]
out = df[[c for c in cols if c in df.columns]].copy()

# Normalize names if already formatted
rename_back = {
    'accuracy_%': 'accuracy',
    'f1_macro_%': 'f1_macro',
    'f1_bacteria_%': 'f1_bacteria',
    'f1_eukaryota_%': 'f1_eukaryota',
    'f1_plasmid_%': 'f1_plasmid',
    'f1_virus_%': 'f1_virus',
    'recall_bacteria_%': 'recall_bacteria',
    'recall_eukaryota_%': 'recall_eukaryota',
    'recall_plasmid_%': 'recall_plasmid',
    'recall_virus_%': 'recall_virus',
}
for old, new in rename_back.items():
    if old in out.columns and new not in out.columns:
        out = out.rename(columns={old: new})

for c in out.columns:
    if c != 'model':
        out[c] = pd.to_numeric(out[c], errors='coerce')

pct_cols = [
    'accuracy', 'f1_macro',
    'f1_bacteria', 'f1_eukaryota', 'f1_plasmid', 'f1_virus',
    'recall_bacteria', 'recall_eukaryota', 'recall_plasmid', 'recall_virus'
]
already_percent = any(c.endswith('_%') for c in df.columns)

for c in pct_cols:
    if c in out.columns:
        if already_percent:
            out[c] = out[c].round(2)
        else:
            out[c] = (out[c] * 100).round(2)

for c in ['p_thr', 'v_thr']:
    if c in out.columns:
        out[c] = out[c].round(2)

if 'circ_len' in out.columns:
    out['circ_len'] = out['circ_len'].round(0).astype('Int64')

sort_cols = [c for c in ['accuracy', 'f1_macro'] if c in out.columns]
out = out.sort_values(sort_cols, ascending=False)

out = out.rename(columns={
    'accuracy': 'accuracy_%',
    'f1_macro': 'f1_macro_%',
    'f1_bacteria': 'f1_bacteria_%',
    'f1_eukaryota': 'f1_eukaryota_%',
    'f1_plasmid': 'f1_plasmid_%',
    'f1_virus': 'f1_virus_%',
    'recall_bacteria': 'recall_bacteria_%',
    'recall_eukaryota': 'recall_eukaryota_%',
    'recall_plasmid': 'recall_plasmid_%',
    'recall_virus': 'recall_virus_%',
})

out.to_csv(p, sep='\t', index=False)

headers = list(out.columns)
md_lines = []
md_lines.append("| " + " | ".join(headers) + " |")
md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
for _, r in out.iterrows():
    vals = []
    for h in headers:
        v = r[h]
        if pd.isna(v):
            vals.append("")
        else:
            vals.append(str(v))
    md_lines.append("| " + " | ".join(vals) + " |")
md.write_text("\n".join(md_lines) + "\n")

print(out.to_string(index=False))
