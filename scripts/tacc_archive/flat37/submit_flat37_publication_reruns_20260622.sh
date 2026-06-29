#!/bin/bash
set -euo pipefail

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA

PREFIX="${PREFIX:-pubfix_20260622}"
SEED="${SEED:-1337}"
TRAIN_PARTITION="${TRAIN_PARTITION:-gh}"
EVAL_PARTITION="${EVAL_PARTITION:-gh}"
STAGE1_TIME="${STAGE1_TIME:-16:00:00}"
STAGE2_TIME="${STAGE2_TIME:-16:00:00}"
PO_TIME="${PO_TIME:-04:00:00}"
MIXED_TIME="${MIXED_TIME:-16:00:00}"
EVAL_TIME="${EVAL_TIME:-06:00:00}"
ABLATION_STAGE1_TIME="${ABLATION_STAGE1_TIME:-08:00:00}"
ABLATION_STAGE2_TIME="${ABLATION_STAGE2_TIME:-08:00:00}"
SUBMIT_MAIN="${SUBMIT_MAIN:-1}"
SUBMIT_PO="${SUBMIT_PO:-1}"
SUBMIT_MIXED="${SUBMIT_MIXED:-1}"
SUBMIT_ABLATION="${SUBMIT_ABLATION:-0}"
SUBMIT_AUX025="${SUBMIT_AUX025:-0}"
ABLATION_MODES="${ABLATION_MODES:-on off}"
ABLATION_SEEDS="${ABLATION_SEEDS:-1337 2027 31415}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"
REMOTE_DATA_HOST="${REMOTE_DATA_HOST:-stampede3}"
EXISTING_STAGE1_JOB="${EXISTING_STAGE1_JOB:-}"
EXISTING_STAGE1_META="${EXISTING_STAGE1_META:-}"

ROOT="${ROOT:-/scratch/09870/williamratcliff/flat37_publication_20260622}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/vista_train_flat37_publication_20260622.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/vista_eval_flat37_publication_20260622.sh}"
mkdir -p "$ROOT"

STAGE1_CFG="/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/config_uniform_2m_blackbird_flat37_stage1_paper_20260622.json"
STAGE2_CFG="/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_flat37_stage2c_20260622.json"
STAGE2_AUX025_CFG="/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_fixedmap_stage2c_aux025_20260622.json"
PO_CFG="/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/config_rruff_conditioned_200k_po_1e_flat37_20260622.json"
MIXED_CFG="/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/config_rruff_conditioned_dualsource_2346k_500kpo_flat37_20260622.json"
RRUFF_PRIOR="/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5"
PO200_PRIOR="/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_200k_po_v1_trainready.hdf5"
PO500_PRIOR="/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5"

submit_train() {
  local job_name="$1"
  local base_config="$2"
  local output_config="$3"
  local meta_out="$4"
  local job_tag="$5"
  local time_limit="$6"
  local prior_path="$7"
  local dependency="${8:-}"
  local resume_meta="${9:-}"
  local seed_value="${10:-$SEED}"

  local sbatch_args=(--parsable -p "$TRAIN_PARTITION" -t "$time_limit" --job-name "$job_name")
  if [ -n "$dependency" ]; then
    sbatch_args+=(--dependency="afterok:${dependency}")
  fi

  local export_vars="ALL,BASE_CONFIG=${base_config},OUTPUT_CONFIG=${output_config},META_OUT=${meta_out},JOB_TAG=${job_tag},SEED=${seed_value},PRIOR_DATA_PATH=${prior_path},DISABLE_WANDB=${DISABLE_WANDB},WAIT_FOR_DATA=1,REMOTE_DATA_HOST=${REMOTE_DATA_HOST}"
  if [ -n "$resume_meta" ]; then
    export_vars+=",RESUME_META=${resume_meta}"
  fi

  sbatch "${sbatch_args[@]}" --export="$export_vars" "$TRAIN_SCRIPT" | tail -n 1
}

submit_eval() {
  local job_name="$1"
  local meta_in="$2"
  local dependency="$3"

  sbatch --parsable \
    -p "$EVAL_PARTITION" \
    -t "$EVAL_TIME" \
    --job-name "$job_name" \
    --dependency="afterok:${dependency}" \
    --export=ALL,META_IN="$meta_in" \
    "$EVAL_SCRIPT" | tail -n 1
}

stage1_job=""
stage1_meta="$ROOT/${PREFIX}_stage1_u2m_s${SEED}_meta.json"
if [ "$SUBMIT_MAIN" = "1" ] || [ "$SUBMIT_PO" = "1" ] || [ "$SUBMIT_MIXED" = "1" ] || [ "$SUBMIT_AUX025" = "1" ]; then
  if [ -n "$EXISTING_STAGE1_JOB" ] && [ -n "$EXISTING_STAGE1_META" ]; then
    stage1_job="$EXISTING_STAGE1_JOB"
    stage1_meta="$EXISTING_STAGE1_META"
    echo "reusing stage1: train=${stage1_job} meta=${stage1_meta}"
  else
    stage1_tag="${PREFIX}_stage1_u2m_s${SEED}"
    stage1_job="$(submit_train \
      "${stage1_tag}" \
      "$STAGE1_CFG" \
      "$ROOT/${stage1_tag}_config.json" \
      "$stage1_meta" \
      "$stage1_tag" \
      "$STAGE1_TIME" \
      "$RRUFF_PRIOR")"
    echo "${stage1_tag}: train=${stage1_job} meta=${stage1_meta}"
  fi
fi

if [ "$SUBMIT_MAIN" = "1" ]; then
  stage2_tag="${PREFIX}_stage2c_r2346k_s${SEED}"
  stage2_meta="$ROOT/${stage2_tag}_meta.json"
  stage2_job="$(submit_train \
    "${stage2_tag}" \
    "$STAGE2_CFG" \
    "$ROOT/${stage2_tag}_config.json" \
    "$stage2_meta" \
    "$stage2_tag" \
    "$STAGE2_TIME" \
    "$RRUFF_PRIOR" \
    "$stage1_job" \
    "$stage1_meta")"
  eval_job="$(submit_eval "${stage2_tag}_eval" "$stage2_meta" "$stage2_job")"
  echo "${stage2_tag}: train=${stage2_job} eval=${eval_job} meta=${stage2_meta}"
fi

if [ "$SUBMIT_AUX025" = "1" ]; then
  aux025_tag="${PREFIX}_stage2c_r2346k_aux025_s${SEED}"
  aux025_meta="$ROOT/${aux025_tag}_meta.json"
  aux025_job="$(submit_train \
    "${aux025_tag}" \
    "$STAGE2_AUX025_CFG" \
    "$ROOT/${aux025_tag}_config.json" \
    "$aux025_meta" \
    "$aux025_tag" \
    "$STAGE2_TIME" \
    "$RRUFF_PRIOR" \
    "$stage1_job" \
    "$stage1_meta")"
  aux025_eval="$(submit_eval "${aux025_tag}_eval" "$aux025_meta" "$aux025_job")"
  echo "${aux025_tag}: train=${aux025_job} eval=${aux025_eval} meta=${aux025_meta}"
fi

if [ "$SUBMIT_PO" = "1" ]; then
  po_tag="${PREFIX}_po200k_s${SEED}"
  po_meta="$ROOT/${po_tag}_meta.json"
  po_job="$(submit_train \
    "${po_tag}" \
    "$PO_CFG" \
    "$ROOT/${po_tag}_config.json" \
    "$po_meta" \
    "$po_tag" \
    "$PO_TIME" \
    "$PO200_PRIOR" \
    "$stage1_job" \
    "$stage1_meta")"
  po_eval="$(submit_eval "${po_tag}_eval" "$po_meta" "$po_job")"
  echo "${po_tag}: train=${po_job} eval=${po_eval} meta=${po_meta}"
fi

if [ "$SUBMIT_MIXED" = "1" ]; then
  mixed_tag="${PREFIX}_dualsource2500k_s${SEED}"
  mixed_meta="$ROOT/${mixed_tag}_meta.json"
  mixed_job="$(submit_train \
    "${mixed_tag}" \
    "$MIXED_CFG" \
    "$ROOT/${mixed_tag}_config.json" \
    "$mixed_meta" \
    "$mixed_tag" \
    "$MIXED_TIME" \
    "$PO500_PRIOR" \
    "$stage1_job" \
    "$stage1_meta")"
  mixed_eval="$(submit_eval "${mixed_tag}_eval" "$mixed_meta" "$mixed_job")"
  echo "${mixed_tag}: train=${mixed_job} eval=${mixed_eval} meta=${mixed_meta}"
fi

if [ "$SUBMIT_ABLATION" = "1" ]; then
  for mode in $ABLATION_MODES; do
    for ablation_seed in $ABLATION_SEEDS; do
      ablation_stage1_cfg="/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/config_uniform_premaint_1381k_splitablation_${mode}.json"
      ablation_stage2_cfg="/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/config_rruff_conditioned_2346k_splitablation_${mode}.json"
      ablation_prefix="${PREFIX}_splitablate_${mode}_s${ablation_seed}"
      ablation_stage1_meta="$ROOT/${ablation_prefix}_stage1_meta.json"
      ablation_stage2_meta="$ROOT/${ablation_prefix}_stage2_meta.json"

      ablation_stage1_job="$(submit_train \
        "${ablation_prefix}_u1381k" \
        "$ablation_stage1_cfg" \
        "$ROOT/${ablation_prefix}_stage1_config.json" \
        "$ablation_stage1_meta" \
        "${ablation_prefix}_u1381k" \
        "$ABLATION_STAGE1_TIME" \
        "$RRUFF_PRIOR" \
        "" \
        "" \
        "$ablation_seed")"

      ablation_stage2_job="$(submit_train \
        "${ablation_prefix}_r2346k" \
        "$ablation_stage2_cfg" \
        "$ROOT/${ablation_prefix}_stage2_config.json" \
        "$ablation_stage2_meta" \
        "${ablation_prefix}_r2346k" \
        "$ABLATION_STAGE2_TIME" \
        "$RRUFF_PRIOR" \
        "$ablation_stage1_job" \
        "$ablation_stage1_meta" \
        "$ablation_seed")"

      ablation_eval="$(submit_eval "${ablation_prefix}_eval" "$ablation_stage2_meta" "$ablation_stage2_job")"
      echo "${ablation_prefix}: stage1=${ablation_stage1_job} stage2=${ablation_stage2_job} eval=${ablation_eval}"
    done
  done
fi
