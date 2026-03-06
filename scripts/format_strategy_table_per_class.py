#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

base = Path('.results')
src_tsv = base / 'strategy_comparison.tsv'
full = base / 'strategy_comparison_full.tsv'
out_tsv = base / 'strategy_comparison.tsv'
out_md = base / 'strategy_comparison.md'
out_extra = base / 'strategy_comparison_per_class.tsv'

df = pd.read_csv(src_tsv, sep='\t')
df.to_csv(full, sep='\t', index=False)

# Keep only per-class precision/recall/f1 + model
cols = [
    'model',
    'precision_bacteria', 'recall_bacteria', 'f1_bacteria',
    'precision_eukaryota', 'recall_eukaryota', 'f1_eukaryota',
    'precision_plasmid', 'recall_plasmid', 'f1_plasmid',
    'precision_virus', 'recall_virus', 'f1_virus',
]
sub = df[[c for c in cols if c in df.columns]].copy()

for c in sub.columns:
    if c != 'model':
        sub[c] = pd.to_numeric(sub[c], errors='coerce')
        sub[c] = (sub[c] * 100).round(2)

sub = sub.rename(columns={
    'precision_bacteria': 'precision_bacteria_%',
    'recall_bacteria': 'recall_bacteria_%',
    'f1_bacteria': 'f1_bacteria_%',
    'precision_eukaryota': 'precision_eukaryota_%',
    'recall_eukaryota': 'recall_eukaryota_%',
    'f1_eukaryota': 'f1_eukaryota_%',
    'precision_plasmid': 'precision_plasmid_%',
    'recall_plasmid': 'recall_plasmid_%',
    'f1_plasmid': 'f1_plasmid_%',
    'precision_virus': 'precision_virus_%',
    'recall_virus': 'recall_virus_%',
    'f1_virus': 'f1_virus_%',
})

# Order with final model first, then strongest alternatives
preferred_order = [
    'final_model_rescue',
    'best_tuned_accuracy',
    'pred_hybrid_rescue',
    'pred_circular_rescue',
    'pred_hyb',
    'pred_hierarchical',
    'pred_4cac',
    'pred_dmc',
]
order_map = {m: i for i, m in enumerate(preferred_order)}
sub['_ord'] = sub['model'].map(lambda x: order_map.get(x, 999))
if 'f1_plasmid_%' in sub.columns:
    sub = sub.sort_values(['_ord', 'f1_plasmid_%'], ascending=[True, False])
else:
    sub = sub.sort_values(['_ord'])
sub = sub.drop(columns=['_ord'])

sub.to_csv(out_tsv, sep='\t', index=False)
sub.to_csv(out_extra, sep='\t', index=False)

headers = list(sub.columns)
md_lines = []
md_lines.append('| ' + ' | '.join(headers) + ' |')
md_lines.append('|' + '|'.join(['---'] * len(headers)) + '|')
for _, r in sub.iterrows():
    vals = []
    for h in headers:
        v = r[h]
        vals.append('' if pd.isna(v) else str(v))
    md_lines.append('| ' + ' | '.join(vals) + ' |')
out_md.write_text('\n'.join(md_lines) + '\n')

print(sub.to_string(index=False))
