# Modello Hybrid + Rescue

Architettura ibrida che combina le predizioni basate sul Deep Learning di DeepMicroClass con la propagazione delle etichette dell'algoritmo di 4CAC. Sono state applicate delle modifiche al meccanismo di propagazione, usando le Soft Labels di DMC direttamente nel grafo, ed è stata implementata l'euristica di recupero plasmidico.

## Struttura
- `scripts/hybrid_dmc_4cac.py`: pipeline completa + grid search + confronto baseline.
- `config/sharon_paths.json`: percorsi input Sharon.
- `docs/TOOL_ANALYSIS.md`: sintesi tecnica DMC/4CAC e integrazione.
- `results/`: output di validazione.

## Requisiti
Ambiente Python con: `numpy`, `pandas`, `scikit-learn`.

## Esecuzione
### 1) Baseline
```bash
python scripts/hybrid_dmc_4cac.py compare-baseline \
  --c4-file data/output/4cac/sharon/4CAC_classification.fasta \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/baseline
```

### 2) Ricerca soglia anchor
```bash
python scripts/hybrid_dmc_4cac.py grid-search \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gfa-file data/output/metaspades/sharon/assembly_graph_with_scaffolds.gfa \
  --paths-file data/output/metaspades/sharon/scaffolds.paths \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/grid_search \
  --anchor-thresholds 0.55:0.95:0.05 \
  --plasmid-rescue-threshold 0.6 \
  --temperature 1.0 \
  --alpha 0.65 \
  --n-iter 20
```

Con il comportamento corrente, `--anchor-thresholds` serve soprattutto per analisi/reportistica (conteggio/rate degli anchor), mentre l'input probabilistico verso 4CAC resta quello originale di DMC anche per i contig incerti.

In pratica, cambiando `--anchor-thresholds` si aggiornano i campi diagnostici (`anchors_n`, `anchors_rate`) ma non viene più applicata alcuna riscrittura dei punteggi per i contig incerti.

### 3) Run finale con soglia ottima
```bash
python scripts/hybrid_dmc_4cac.py run \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gfa-file data/output/metaspades/sharon/assembly_graph_with_scaffolds.gfa \
  --paths-file data/output/metaspades/sharon/scaffolds.paths \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/final_model \
  --anchor-threshold <BEST> \
  --plasmid-rescue-threshold 0.6 \
  --temperature 1.0 \
  --alpha 0.65 \
  --n-iter 20
```

### 4) Confronto strategie storiche + tuning (4 classi)
```bash
python scripts/evaluate_sharon_strategies.py \
  --gt data/output/sharon/sharon_ground_truth.csv \
  --dmc data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --hyb data/output/hybrid/sharon/4CAC_classification.fasta \
  --c4 data/output/4cac/sharon/4CAC_classification.fasta \
  --circular data/output/metaspades/sharon/circular_contigs_filtered.txt \
  --out results/strategy_comparison.tsv
```

## Output principali
- `predictions_hybrid.tsv`: classe finale per contig.
- `node_state.tsv`: stato nodi (label + probabilità propagate).
- `metrics_hybrid.tsv`: metriche complete su Sharon + campi diagnostici `anchors_n`, `anchors_rate` e `rescued_plasmids`.
- `grid_search_summary.tsv`: confronto configurazioni; con la logica corrente è utile soprattutto per tracciare diagnostica anchor, non per forzare uniformazione dei contig incerti.
- `strategy_comparison.tsv`: confronto completo su 4 classi tra baseline e strategie ibride.
- `predictions_best_tuned.tsv`: predizioni del modello ottimizzato per accuracy.
- `best_tuned_config.tsv`: soglie ottime del tuning.

## Workflow CAMISIM contig-level (short vs long)
Per benchmark controllati su **contig** (non su reads), con simulazione e assemblaggio in SLURM:

- Config scenari: `config/camisim_contig_scenarios_1000.json`
- Script creazione subset contig: `scripts/create_camisim_contig_scenarios.py`
- Script valutazione scenari: `scripts/evaluate_camisim_contig_scenarios.py`
- Plot scenari: `scripts/plot_camisim_contig_scenarios.py`
- Hybrid da output DMC: `scripts/run_hybrid_from_dmc.py`
- SLURM pipeline completa: `slurm/contig_scenarios/submit_workflow.sh`

La pipeline genera due rami separati:
- **short**: CAMISIM metagenomico (`art`) + assembly con `metaSPAdes --meta`
- **long**: CAMISIM metagenomico (`nanosim3`) + assembly con `Flye --meta`

Output principali in:
- `validazione/camisim_contig_scenarios/runs/short/`
- `validazione/camisim_contig_scenarios/runs/long/`
- `validazione/camisim_contig_scenarios/results_short/`
- `validazione/camisim_contig_scenarios/results_long/`
- `validazione/camisim_contig_scenarios/plots_short/`
- `validazione/camisim_contig_scenarios/plots_long/`

Avvio:
```bash
bash slurm/contig_scenarios/submit_workflow.sh
```
