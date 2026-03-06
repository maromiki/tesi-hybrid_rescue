# Validazione su Sharon

Valutazione eseguita su `3992` contig etichettati in 4 classi:
- `Bacteria` (procarioti)
- `Eukaryota` (eucarioti)
- `Plasmid`
- `Virus`

Il confronto è stato fatto su tutte e quattro le classi, non solo sui plasmidi.

## Script usato
- [scripts/evaluate_sharon_strategies.py](../scripts/evaluate_sharon_strategies.py)

## Strategie confrontate
- `pred_4cac`: 4CAC standard
- `pred_dmc`: DeepMicroClass
- `pred_hyb`: output ibrido storico (`output/hybrid/sharon/4CAC_classification.fasta`)
- `pred_hierarchical`: strategia gerarchica storica
- `pred_hybrid_rescue`: strategia storica “HYBRID + RESCUE”
- `pred_circular_rescue`: rescue con circolarità
- `best_tuned_accuracy`: tuning automatico soglie su strategie storiche

## Risultati principali (Sharon)

### 1) Modello con accuracy migliore (supera i risultati in figura)
- `model`: `best_tuned_accuracy`
- `accuracy`: `0.821894` (**82.19%**)
- `f1_macro`: `0.533877`
- `f1_bacteria`: `0.881779`
- `f1_eukaryota`: `0.922280`
- `f1_plasmid`: `0.265060`
- `f1_virus`: `0.066390`
- soglie ottime: `p_thr=0.75`, `v_thr=0.90`, `circ_len=50000`

### 2) Modello storico “HYBRID + RESCUE” (coerente con la figura)
- `model`: `pred_hybrid_rescue`
- `accuracy`: `0.803858` (**80.39%**)
- `precision/recall/f1`:
  - `Bacteria`: `0.9235 / 0.8105 / 0.8633`
  - `Eukaryota`: `0.9840 / 0.8678 / 0.9223`
  - `Plasmid`: `0.3724 / 0.5902 / 0.4567`
  - `Virus`: `0.0507 / 0.1228 / 0.0718`

Questi valori riproducono i numeri mostrati nella figura.

## File output
- [results/strategy_comparison.tsv](../results/strategy_comparison.tsv)
- [results/predictions_best_tuned.tsv](../results/predictions_best_tuned.tsv)
- [results/best_tuned_config.tsv](../results/best_tuned_config.tsv)

## Nota metodologica
Le metriche sono sempre calcolate in modalità multi-classe 4-way (`Bacteria`, `Eukaryota`, `Plasmid`, `Virus`) con report per classe (`precision`, `recall`, `f1`) e metriche globali (`accuracy`, `f1_macro`, `f1_weighted`).
