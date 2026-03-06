#!/bin/bash
#SBATCH --job-name=scenarios_long
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=.slurm/contig_scenarios/logs/%x_%j.out
#SBATCH --error=.slurm/contig_scenarios/logs/%x_%j.err

set -euo pipefail

PY="data/miniconda3/envs/thesis_env/bin/python"
BASE="data/thesis/finalissima"
RUN="${BASE}/validazione/camisim_contig_scenarios/runs/long"

"${PY}" "${BASE}/scripts/create_camisim_contig_scenarios.py" \
  --gt "${RUN}/ground_truth/assembly_gt.tsv" \
  --assembly "${RUN}/assembly_flye/assembly.fasta" \
  --config "${BASE}/config/camisim_contig_scenarios_long_feasible.json" \
  --out-dir "${RUN}/scenarios" \
  --seed 7 \
  --min-length 500

echo "DONE long scenarios -> ${RUN}/scenarios"
