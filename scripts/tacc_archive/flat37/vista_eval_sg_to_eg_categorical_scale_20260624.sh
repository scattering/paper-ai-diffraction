#!/bin/bash
#SBATCH -J sgeg-scale-eval
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/09870/williamratcliff/sgeg_scale_eval_%j.out
#SBATCH -e /scratch/09870/williamratcliff/sgeg_scale_eval_%j.err

set -euo pipefail

: "${SCALE_FRACTION:?Set SCALE_FRACTION to one of 1_2, 1_4, 1_8.}"

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA

RESULT_DIR="/scratch/09870/williamratcliff/sg_eg_categorical_referee_20260624"
SG_TAG="pubfix_sgeg_scale_20260624_sg_${SCALE_FRACTION}_s1337"
EG_TAG="pubfix_sgeg_scale_20260624_eg_${SCALE_FRACTION}_s1337"

python evaluate_sg_to_eg_control.py \
  --sg-checkpoint "/scratch/09870/williamratcliff/ai_diffraction_models/xrd_model_${SG_TAG}_best.pth" \
  --sg-config "${RESULT_DIR}/configs/${SG_TAG}.json" \
  --sg-eval-data-path /scratch/09870/williamratcliff/ai_diffraction_generated/uniform_2m_blackbird_20260401_sg_trainready.hdf5 \
  --eg-checkpoint "/scratch/09870/williamratcliff/ai_diffraction_models/xrd_model_${EG_TAG}_best.pth" \
  --eg-config "${RESULT_DIR}/configs/${EG_TAG}.json" \
  --eg-truth-data-path /scratch/09870/williamratcliff/ai_diffraction_generated/uniform_2m_blackbird_20260401_trainready.hdf5 \
  --canonical-table-path /scratch/09870/williamratcliff/ai-diffraction/Code/Post_Processing/canonical_extinction_to_space_group.csv \
  --output-json "${RESULT_DIR}/sg230_collapsed_vs_eg99_categorical_scale_${SCALE_FRACTION}_${SLURM_JOB_ID}.json" \
  --output-csv "${RESULT_DIR}/sg230_collapsed_vs_eg99_categorical_scale_${SCALE_FRACTION}_${SLURM_JOB_ID}_topk.csv"
