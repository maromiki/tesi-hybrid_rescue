#!/bin/bash
#SBATCH --job-name=hybrid_sharon
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/nfsd/bcb/bcbg/marongiumi/thesis/finalissima/results/hybrid_sharon_%j.out
#SBATCH --error=/nfsd/bcb/bcbg/marongiumi/thesis/finalissima/results/hybrid_sharon_%j.err

set -euo pipefail

BASE="/nfsd/bcb/bcbg/marongiumi/thesis/finalissima"
PY="/nfsd/bcb/bcbg/marongiumi/miniconda3/bin/python"
SCRIPT="$BASE/scripts/hybrid_dmc_4cac.py"

DMC="/nfsd/bcb/bcbg/marongiumi/output/dmc/sharon/scaffolds/scaffolds.fasta_pred_one-hot_hybrid.tsv"
GFA="/nfsd/bcb/bcbg/marongiumi/output/metaspades/sharon/assembly_graph_with_scaffolds.gfa"
PATHS="/nfsd/bcb/bcbg/marongiumi/output/metaspades/sharon/scaffolds.paths"
GT="/nfsd/bcb/bcbg/marongiumi/output/sharon/sharon_ground_truth.csv"
C4="/nfsd/bcb/bcbg/marongiumi/output/4cac/sharon/4CAC_classification.fasta"

mkdir -p "$BASE/results"

$PY "$SCRIPT" compare-baseline \
  --c4-file "$C4" \
  --dmc-file "$DMC" \
  --gt-file "$GT" \
  --output-dir "$BASE/results/baseline" \
  --temperature 1.0

$PY "$SCRIPT" grid-search \
  --dmc-file "$DMC" \
  --gfa-file "$GFA" \
  --paths-file "$PATHS" \
  --gt-file "$GT" \
  --output-dir "$BASE/results/grid_search" \
  --anchor-thresholds 0.55:0.95:0.05 \
  --plasmid-rescue-threshold 0.6 \
  --temperature 1.0 \
  --alpha 0.65 \
  --n-iter 20 \
  --min-anchor-len 1000

BEST=$(python - << 'PYEOF'
import pandas as pd
p='/nfsd/bcb/bcbg/marongiumi/thesis/finalissima/results/grid_search/grid_search_summary.tsv'
df=pd.read_csv(p, sep='\t')
print(float(df.sort_values(['f1_macro','f1_plasmid','accuracy'], ascending=False).iloc[0]['anchor_threshold']))
PYEOF
)

$PY "$SCRIPT" run \
  --dmc-file "$DMC" \
  --gfa-file "$GFA" \
  --paths-file "$PATHS" \
  --gt-file "$GT" \
  --output-dir "$BASE/results/final_model" \
  --anchor-threshold "$BEST" \
  --plasmid-rescue-threshold 0.6 \
  --temperature 1.0 \
  --alpha 0.65 \
  --n-iter 20 \
  --min-anchor-len 1000
