#!/bin/bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-CDA24014}"
PARTITION="${PARTITION:-gh}"
EVAL_PARTITION="${EVAL_PARTITION:-gh-dev}"

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

stage1_job="$(submit_job -A "$ACCOUNT" -p "$PARTITION" vista_train_rt_flat37_stage1_20260624.sh)"
stage2_job="$(submit_job -A "$ACCOUNT" -p "$PARTITION" --dependency="afterok:${stage1_job}" vista_train_rt_flat37_stage2c_20260624.sh)"
eval_job="$(submit_job -A "$ACCOUNT" -p "$EVAL_PARTITION" --dependency="afterok:${stage2_job}" vista_eval_rt_flat37_stage2c_20260624.sh)"

echo "RT stage1 job: ${stage1_job}"
echo "RT stage2 job: ${stage2_job}"
echo "RT eval job: ${eval_job}"
