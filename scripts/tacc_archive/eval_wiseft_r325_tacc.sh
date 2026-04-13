#!/bin/bash
#SBATCH -J vista-wiseft-r325
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/09870/williamratcliff/vista_wiseft_r325_%j.out
#SBATCH -e /scratch/09870/williamratcliff/vista_wiseft_r325_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA

BASE=/scratch/09870/williamratcliff/ai_diffraction_models/xrd_model_9rwv1qly_best.pth
PO=/scratch/09870/williamratcliff/ai_diffraction_models/xrd_model_cscjfdwk_best.pth
CFG=/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_from_ic6gfmvm_physpe_coord.json
EVAL=/work2/09870/williamratcliff/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5
PRIOR_STAGE2=/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5
PRIOR_PO=/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_200k_po_v1_trainready.hdf5

CKPT_A03=/scratch/09870/williamratcliff/ai_diffraction_models/xrd_model_wiseft_9rwv1qly_cscjfdwk_a03_best.pth
CKPT_A05=/scratch/09870/williamratcliff/ai_diffraction_models/xrd_model_wiseft_9rwv1qly_cscjfdwk_a05_best.pth

python make_wise_ft_checkpoint.py \
  --base-checkpoint "$BASE" \
  --finetuned-checkpoint "$PO" \
  --alpha 0.3 \
  --output-checkpoint "$CKPT_A03"

python make_wise_ft_checkpoint.py \
  --base-checkpoint "$BASE" \
  --finetuned-checkpoint "$PO" \
  --alpha 0.5 \
  --output-checkpoint "$CKPT_A05"

python evaluate_calibration_metrics.py \
  --checkpoint "$CKPT_A03" \
  --config "$CFG" \
  --eval-data-path "$EVAL" \
  --prior-data-path "$PRIOR_STAGE2" \
  --aux-temperature 5.0 \
  --bootstrap 1000 \
  --output-json /scratch/09870/williamratcliff/wiseft_a03_stage2prior_r325.json

python evaluate_calibration_metrics.py \
  --checkpoint "$CKPT_A03" \
  --config "$CFG" \
  --eval-data-path "$EVAL" \
  --prior-data-path "$PRIOR_PO" \
  --aux-temperature 5.0 \
  --bootstrap 1000 \
  --output-json /scratch/09870/williamratcliff/wiseft_a03_poprior_r325.json

python evaluate_calibration_metrics.py \
  --checkpoint "$CKPT_A05" \
  --config "$CFG" \
  --eval-data-path "$EVAL" \
  --prior-data-path "$PRIOR_STAGE2" \
  --aux-temperature 5.0 \
  --bootstrap 1000 \
  --output-json /scratch/09870/williamratcliff/wiseft_a05_stage2prior_r325.json

python evaluate_calibration_metrics.py \
  --checkpoint "$CKPT_A05" \
  --config "$CFG" \
  --eval-data-path "$EVAL" \
  --prior-data-path "$PRIOR_PO" \
  --aux-temperature 5.0 \
  --bootstrap 1000 \
  --output-json /scratch/09870/williamratcliff/wiseft_a05_poprior_r325.json
