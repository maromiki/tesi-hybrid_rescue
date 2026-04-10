# finalissima - Hybrid DMC + 4CAC (Sharon)

Reproducible pipeline to combine DeepMicroClass probabilistic predictions with 4CAC-style topological propagation.

Current behavior note: contigs below the anchor threshold ("uncertain") are not overwritten with `0.25/0.25/0.25/0.25`; their original DMC probability vectors are preserved.

## Structure
- `scripts/hybrid_dmc_4cac.py`: full pipeline + grid search + baseline comparison.
- `config/sharon_paths.json`: Sharon input paths.
- `docs/TOOL_ANALYSIS.md`: technical DMC/4CAC integration summary.
- `results/`: validation outputs.

## Requirements
Python environment with: `numpy`, `pandas`, `scikit-learn`.

## Execution
### 1) Baseline
```bash
python scripts/hybrid_dmc_4cac.py compare-baseline \
  --c4-file data/output/4cac/sharon/4CAC_classification.fasta \
  --dmc-file data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --gt-file data/output/sharon/sharon_ground_truth.csv \
  --output-dir results/baseline
```

### 2) Anchor Threshold Search
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

With the current behavior, `--anchor-thresholds` is primarily useful for diagnostics/reporting (anchor counts/rates), while the probabilistic input to 4CAC remains the original DMC vector even for uncertain contigs.

In practice, changing `--anchor-thresholds` updates diagnostic fields (`anchors_n`, `anchors_rate`) but does not rewrite uncertain-contig class scores.

### 3) Final Run With Best Threshold
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

### 4) Historical Strategy Comparison + Tuning (4 classes)
```bash
python scripts/evaluate_sharon_strategies.py \
  --gt data/output/sharon/sharon_ground_truth.csv \
  --dmc data/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv \
  --hyb data/output/hybrid/sharon/4CAC_classification.fasta \
  --c4 data/output/4cac/sharon/4CAC_classification.fasta \
  --circular data/output/metaspades/sharon/circular_contigs_filtered.txt \
  --out results/strategy_comparison.tsv
```

## Main Outputs
- `predictions_hybrid.tsv`: final class per contig.
- `node_state.tsv`: node states (labels + propagated probabilities).
- `metrics_hybrid.tsv`: full Sharon metrics + diagnostics `anchors_n`, `anchors_rate`, and `rescued_plasmids`.
- `grid_search_summary.tsv`: configuration comparison; with current logic it is mainly diagnostic and not used to enforce uncertain-contig score uniformization.
- `strategy_comparison.tsv`: complete 4-class comparison across baseline and hybrid strategies.
- `predictions_best_tuned.tsv`: predictions from the best accuracy-tuned model.
- `best_tuned_config.tsv`: optimal tuned thresholds.

## Workflow CAMISIM contig-level (short vs long)
For controlled **contig-level** benchmarking (not read-level), using simulation and assembly on SLURM:

- Scenario config: `config/camisim_contig_scenarios_1000.json`
- Contig subset creation script: `scripts/create_camisim_contig_scenarios.py`
- Scenario evaluation script: `scripts/evaluate_camisim_contig_scenarios.py`
- Scenario plotting script: `scripts/plot_camisim_contig_scenarios.py`
- Hybrid from DMC output: `scripts/run_hybrid_from_dmc.py`
- Full SLURM pipeline: `slurm/contig_scenarios/submit_workflow.sh`

The pipeline generates two separate branches:
- **short**: metagenomic CAMISIM (`art`) + assembly with `metaSPAdes --meta`
- **long**: metagenomic CAMISIM (`nanosim3`) + assembly with `Flye --meta`

Main outputs are written to:
- `validation/camisim_contig_scenarios/runs/short/`
- `validation/camisim_contig_scenarios/runs/long/`
- `validation/camisim_contig_scenarios/results_short/`
- `validation/camisim_contig_scenarios/results_long/`
- `validation/camisim_contig_scenarios/plots_short/`
- `validation/camisim_contig_scenarios/plots_long/`

Run:
```bash
bash slurm/contig_scenarios/submit_workflow.sh
```
