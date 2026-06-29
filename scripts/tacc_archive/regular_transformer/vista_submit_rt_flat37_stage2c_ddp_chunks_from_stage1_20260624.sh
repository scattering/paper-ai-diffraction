#!/bin/bash
set -euo pipefail

: "${STAGE1_JOB_ID:?Set STAGE1_JOB_ID to the running/completed stage-1 Slurm job id}"

ACCOUNT="${ACCOUNT:-CDA24014}"
PARTITION="${PARTITION:-gh}"
EVAL_PARTITION="${EVAL_PARTITION:-gh-dev}"
DDP_NODES="${DDP_NODES:-4}"
WALLTIME="${WALLTIME:-08:00:00}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MODEL_DIR="/scratch/09870/williamratcliff/ai_diffraction_models"
STAGE1_TAG="pubfix_rt_flat37_20260624_stage1_u2m_s1337"
STAGE2_TAG="pubfix_rt_flat37_20260624_stage2c_r2346k_s1337"
STAGE1_BEST="${MODEL_DIR}/xrd_model_${STAGE1_TAG}_best.pth"
STAGE2_LATEST="${MODEL_DIR}/xrd_model_${STAGE2_TAG}_latest.pth"

cd /scratch/09870/williamratcliff/ai-diffraction/Code/Reg_Transformer_FlashAttn

submit_job() {
  local raw job_id
  raw="$(sbatch --parsable "$@")"
  printf '%s\n' "$raw" >&2
  job_id="$(printf '%s\n' "$raw" | awk '/^[0-9]+(;|$)/ { sub(/;.*/, "", $1); print $1 }' | tail -n 1)"
  if [ -z "$job_id" ]; then
    echo "[ERROR] could not parse Slurm job id from sbatch output" >&2
    exit 1
  fi
  printf '%s\n' "$job_id"
}

common_sbatch_args=(
  -A "$ACCOUNT"
  -p "$PARTITION"
  -N "$DDP_NODES"
  --ntasks-per-node=1
  --cpus-per-task=12
  -t "$WALLTIME"
)

epoch1_job="$(submit_job "${common_sbatch_args[@]}" --dependency="afterok:${STAGE1_JOB_ID}" \
  --export="ALL,TARGET_NUM_EPOCHS=1,RESUME_CHECKPOINT=${STAGE1_BEST},RESUME_WEIGHTS_ONLY=1,GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}" \
  vista_train_rt_flat37_stage2c_epoch_chunk_ddp_20260624.sh)"

epoch2_job="$(submit_job "${common_sbatch_args[@]}" --dependency="afterok:${epoch1_job}" \
  --export="ALL,TARGET_NUM_EPOCHS=2,RESUME_CHECKPOINT=${STAGE2_LATEST},RESUME_WEIGHTS_ONLY=0,GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}" \
  vista_train_rt_flat37_stage2c_epoch_chunk_ddp_20260624.sh)"

epoch3_job="$(submit_job "${common_sbatch_args[@]}" --dependency="afterok:${epoch2_job}" \
  --export="ALL,TARGET_NUM_EPOCHS=3,RESUME_CHECKPOINT=${STAGE2_LATEST},RESUME_WEIGHTS_ONLY=0,GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}" \
  vista_train_rt_flat37_stage2c_epoch_chunk_ddp_20260624.sh)"

eval_job="$(submit_job -A "$ACCOUNT" -p "$EVAL_PARTITION" --dependency="afterok:${epoch3_job}" \
  vista_eval_rt_flat37_stage2c_20260624.sh)"

echo "RT stage2 DDP epoch1 job: ${epoch1_job}"
echo "RT stage2 DDP epoch2 job: ${epoch2_job}"
echo "RT stage2 DDP epoch3 job: ${epoch3_job}"
echo "RT eval job: ${eval_job}"
