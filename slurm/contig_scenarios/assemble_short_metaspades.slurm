#!/bin/bash
#SBATCH --job-name=assemble_short_meta
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/nfsd/bcb/bcbg/marongiumi/thesis/finalissima/slurm/contig_scenarios/logs/%x_%j.out
#SBATCH --error=/nfsd/bcb/bcbg/marongiumi/thesis/finalissima/slurm/contig_scenarios/logs/%x_%j.err

set -euo pipefail

READS="/nfsd/bcb/bcbg/marongiumi/thesis/finalissima/validazione/camisim_contig_scenarios/runs/short/camisim/sample_0/reads/anonymous_reads.fq.gz"
OUT_DIR="/nfsd/bcb/bcbg/marongiumi/thesis/finalissima/validazione/camisim_contig_scenarios/runs/short/assembly_metaspades"
METASPADES="/nfsd/bcb/bcbg/tools/metaSPAdes/SPAdes-4.2.0-Linux/bin/metaspades.py"
SCRATCH="/ext/${USER}/metaspades_${SLURM_JOB_ID}"

mkdir -p "${OUT_DIR}"
mkdir -p "${SCRATCH}"
cp "${READS}" "${SCRATCH}/reads.fq.gz"

python3 "${METASPADES}" \
  --12 "${SCRATCH}/reads.fq.gz" \
  --meta \
  -t "${SLURM_CPUS_PER_TASK}" \
  -m 120 \
  -o "${SCRATCH}/spades_out"

cp "${SCRATCH}/spades_out/scaffolds.fasta" "${OUT_DIR}/"
cp "${SCRATCH}/spades_out/scaffolds.paths" "${OUT_DIR}/"
cp "${SCRATCH}/spades_out/assembly_graph_with_scaffolds.gfa" "${OUT_DIR}/"
cp "${SCRATCH}/spades_out/spades.log" "${OUT_DIR}/"
cp "${SCRATCH}/spades_out/params.txt" "${OUT_DIR}/"

rm -rf "${SCRATCH}"
echo "DONE short assembly -> ${OUT_DIR}"
