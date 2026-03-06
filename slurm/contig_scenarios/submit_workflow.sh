#!/bin/bash
set -euo pipefail

BASE=".slurm/contig_scenarios"
mkdir -p "${BASE}/logs"

# SHORT branch
J1=$(sbatch --parsable "${BASE}/run_camisim_short_meta.slurm")
J2=$(sbatch --parsable --dependency=afterok:${J1} "${BASE}/assemble_short_metaspades.slurm")
J3=$(sbatch --parsable --dependency=afterok:${J2} "${BASE}/map_gt_short.slurm")
J4=$(sbatch --parsable --dependency=afterok:${J3} "${BASE}/create_scenarios_short.slurm")
J5=$(sbatch --parsable --dependency=afterok:${J4} "${BASE}/classify_short_all.slurm")

# LONG branch
L1=$(sbatch --parsable "${BASE}/run_camisim_long_meta.slurm")
L2=$(sbatch --parsable --dependency=afterok:${L1} "${BASE}/assemble_long_flye.slurm")
L3=$(sbatch --parsable --dependency=afterok:${L2} "${BASE}/map_gt_long.slurm")
L4=$(sbatch --parsable --dependency=afterok:${L3} "${BASE}/create_scenarios_long.slurm")
L5=$(sbatch --parsable --dependency=afterok:${L4} "${BASE}/classify_long_all.slurm")

echo "Submitted SHORT chain: ${J1} -> ${J2} -> ${J3} -> ${J4} -> ${J5}"
echo "Submitted LONG  chain: ${L1} -> ${L2} -> ${L3} -> ${L4} -> ${L5}"

echo "Track with: squeue -j ${J1},${J2},${J3},${J4},${J5},${L1},${L2},${L3},${L4},${L5}"
