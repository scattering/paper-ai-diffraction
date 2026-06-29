#!/bin/bash
#SBATCH -J rt-f37-eval
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/09870/williamratcliff/rt_flat37_eval_%j.out
#SBATCH -e /scratch/09870/williamratcliff/rt_flat37_eval_%j.err

set -euo pipefail

MODEL_DIR="/scratch/09870/williamratcliff/ai_diffraction_models"
TAG="pubfix_rt_flat37_20260624_stage2c_r2346k_s1337"
CHECKPOINT="${MODEL_DIR}/xrd_model_${TAG}_best.pth"
CONFIG="config_rruff_conditioned_2346k_rt_flat37_stage2c_20260624.json"
PRIOR="/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5"
R325="/work2/09870/williamratcliff/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5"
R473="/work2/09870/williamratcliff/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5"
RESULT_DIR="/scratch/09870/williamratcliff/rt_flat37_referee_20260624/${SLURM_JOB_ID}"

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

mkdir -p "$RESULT_DIR"

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA
python -m unittest test_extinction_multilabel_mapping.py -v

cd /scratch/09870/williamratcliff/ai-diffraction/Code/Reg_Transformer_FlashAttn

python evaluate_decoder_variants_multilabel.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --eval-data-path "$R325" \
  --prior-data-path "$PRIOR" \
  --aux-temperature 5.0 \
  --group-keys minerals \
  --output-json "${RESULT_DIR}/${TAG}_r325_t5p0.json" \
  --failure-json "${RESULT_DIR}/${TAG}_r325_t5p0_failure_modes.json" \
  --failure-model-name "rt_flat37_aux_bayes_t5"

python evaluate_decoder_variants_multilabel.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --eval-data-path "$R473" \
  --prior-data-path "$PRIOR" \
  --aux-temperature 5.0 \
  --output-json "${RESULT_DIR}/${TAG}_r473_t5p0.json"

echo "[INFO] wrote RT flat37 evaluation outputs to ${RESULT_DIR}"
