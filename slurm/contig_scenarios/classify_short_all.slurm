#!/bin/bash
#SBATCH --job-name=classify_short_all
#SBATCH --partition=allgroups
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=10:00:00
#SBATCH --output=.slurm/contig_scenarios/logs/%x_%j.out
#SBATCH --error=.slurm/contig_scenarios/logs/%x_%j.err

set -euo pipefail

BASE="data/thesis/finalissima"
RUN="${BASE}/validazione/camisim_contig_scenarios/runs/short"
ASM_DIR="${RUN}/assembly_metaspades"
ASM_FASTA="${ASM_DIR}/scaffolds.fasta"
GT="${RUN}/ground_truth/scaffolds_gt.tsv"

DMC_OUT="${RUN}/predictions/dmc"
C4_OUT="${RUN}/predictions/4cac"
HYB_OUT="${RUN}/predictions/hybrid"
EVAL_OUT="${BASE}/validazione/camisim_contig_scenarios/results_short"

mkdir -p "${DMC_OUT}" "${C4_OUT}" "${HYB_OUT}" "${EVAL_OUT}"

# 1) DeepMicroClass (GPU)
singularity exec --nv \
  --bind /nfsd/bcb/bcbg/marongiumi:/nfsd/bcb/bcbg/marongiumi \
  tools/DeepMicroClass/deepmicroclass.sif \
  DeepMicroClass predict \
    --input "${ASM_FASTA}" \
    --output_dir "${DMC_OUT}" \
    --device cuda

# 2) 4CAC
source data/miniconda3/etc/profile.d/conda.sh
conda activate 4cac

python tools/4CAC/classify_xgb.py \
  -f "${ASM_FASTA}" \
  -o "${C4_OUT}/scaffolds.fasta.probs_xgb_4class.out" \
  -p "${SLURM_CPUS_PER_TASK}"

python tools/4CAC/classify_4CAC.py \
  --assembler metaSPAdes \
  --asmdir "${ASM_DIR}/" \
  --classdir "${C4_OUT}/" \
  --outdir "${C4_OUT}/"

# 3) Hybrid (DMC probs + graph correction)
data/miniconda3/envs/thesis_env/bin/python "${BASE}/scripts/run_hybrid_from_dmc.py" \
  --dmc "${DMC_OUT}/scaffolds.fasta_pred_one-hot_hybrid.tsv" \
  --asmdir "${ASM_DIR}" \
  --out-dir "${HYB_OUT}" \
  --assembler metaSPAdes \
  --contig-base scaffolds.fasta \
  --fourcac-script tools/4CAC/classify_4CAC.py \
  --fourcac-env 4cac \
  --temperature 1.0 \
  --anchor-threshold 0.30 \
  --fallback-probs "${C4_OUT}/scaffolds.fasta.probs_xgb_4class.out" \
  --phage-threshold 0.7 \
  --plasmid-threshold 0.7

# 4) Evaluate on scenario subsets
data/miniconda3/envs/thesis_env/bin/python "${BASE}/scripts/evaluate_camisim_contig_scenarios.py" \
  --mode short \
  --scenario-root "${RUN}/scenarios" \
  --dmc "${DMC_OUT}/scaffolds.fasta_pred_one-hot_hybrid.tsv" \
  --c4 "${C4_OUT}/4CAC_classification.fasta" \
  --hybrid "${HYB_OUT}/4CAC_classification.fasta" \
  --out-dir "${EVAL_OUT}"

# 5) Plot
data/miniconda3/envs/thesis_env/bin/python "${BASE}/scripts/plot_camisim_contig_scenarios.py" \
  --metrics "${EVAL_OUT}/metrics_short_per_scenario.tsv" \
  --out-dir "${BASE}/validazione/camisim_contig_scenarios/plots_short" \
  --mode short

echo "DONE short classification/evaluation"
