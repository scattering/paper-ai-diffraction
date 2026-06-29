#!/bin/bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-CDA24014}"
PARTITION="${PARTITION:-gh}"
EVAL_PARTITION="${EVAL_PARTITION:-gh-dev}"
fractions=(1_2 1_4 1_8)

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA

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

for frac in "${fractions[@]}"; do
  echo "Submitting categorical EG scaling run for fraction ${frac}"
  eg_jobid="$(submit_job -A "$ACCOUNT" -p "$PARTITION" \
    --export="ALL,SCALE_MODE=eg,SCALE_FRACTION=${frac}" \
    vista_uniform_2m_sgeg_categorical_scale_train_20260624.sh)"

  echo "Submitting categorical SG scaling run for fraction ${frac}"
  sg_jobid="$(submit_job -A "$ACCOUNT" -p "$PARTITION" \
    --export="ALL,SCALE_MODE=sg,SCALE_FRACTION=${frac}" \
    vista_uniform_2m_sgeg_categorical_scale_train_20260624.sh)"

  echo "Submitting categorical SG->EG eval for fraction ${frac}"
  eval_jobid="$(submit_job -A "$ACCOUNT" -p "$EVAL_PARTITION" \
    --dependency="afterok:${sg_jobid}:${eg_jobid}" \
    --export="ALL,SCALE_FRACTION=${frac}" \
    vista_eval_sg_to_eg_categorical_scale_20260624.sh)"

  echo "fraction=${frac} sg=${sg_jobid} eg=${eg_jobid} eval=${eval_jobid}"
done
