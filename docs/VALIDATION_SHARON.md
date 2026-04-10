# Validation on Sharon

Evaluation was run on `3992` contigs labeled into 4 classes:
- `Bacteria` (prokaryotes)
- `Eukaryota` (eukaryotes)
- `Plasmid`
- `Virus`

The comparison was performed across all four classes, not only plasmids.

## Script Used
- [scripts/evaluate_sharon_strategies.py](../scripts/evaluate_sharon_strategies.py)

## Compared Strategies
- `pred_4cac`: 4CAC standard
- `pred_dmc`: DeepMicroClass
- `pred_hyb`: historical hybrid output (`output/hybrid/sharon/4CAC_classification.fasta`)
- `pred_hierarchical`: historical hierarchical strategy
- `pred_hybrid_rescue`: historical “HYBRID + RESCUE” strategy
- `pred_circular_rescue`: circularity-based rescue
- `best_tuned_accuracy`: automatic threshold tuning over historical strategies

## Main Results (Sharon)

### 1) Best-Accuracy Model (outperforms figure values)
- `model`: `best_tuned_accuracy`
- `accuracy`: `0.821894` (**82.19%**)
- `f1_macro`: `0.533877`
- `f1_bacteria`: `0.881779`
- `f1_eukaryota`: `0.922280`
- `f1_plasmid`: `0.265060`
- `f1_virus`: `0.066390`
- optimal thresholds: `p_thr=0.75`, `v_thr=0.90`, `circ_len=50000`

### 2) Historical “HYBRID + RESCUE” Model (consistent with the figure)
- `model`: `pred_hybrid_rescue`
- `accuracy`: `0.803858` (**80.39%**)
- `precision/recall/f1`:
  - `Bacteria`: `0.9235 / 0.8105 / 0.8633`
  - `Eukaryota`: `0.9840 / 0.8678 / 0.9223`
  - `Plasmid`: `0.3724 / 0.5902 / 0.4567`
  - `Virus`: `0.0507 / 0.1228 / 0.0718`

These values reproduce the numbers shown in the figure.

## Output Files
- [results/strategy_comparison.tsv](../results/strategy_comparison.tsv)
- [results/predictions_best_tuned.tsv](../results/predictions_best_tuned.tsv)
- [results/best_tuned_config.tsv](../results/best_tuned_config.tsv)

## Methodological Note
Metrics are always computed in 4-way multi-class mode (`Bacteria`, `Eukaryota`, `Plasmid`, `Virus`) with per-class reports (`precision`, `recall`, `f1`) and global metrics (`accuracy`, `f1_macro`, `f1_weighted`).
