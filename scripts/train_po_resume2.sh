#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src/core:$ROOT/src/eval:$ROOT/src/topology:$ROOT/src/utils${PYTHONPATH:+:$PYTHONPATH}"

BASE_CONFIG="${BASE_CONFIG:-$ROOT/configs/po_resume2.json}"
PO_TRAINREADY_H5="${PO_TRAINREADY_H5:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT/external/checkpoints/xrd_model_cscjfdwk_best.pth}"
MODEL_OUTDIR="${MODEL_OUTDIR:-$ROOT/results/checkpoints}"
CANONICAL_CSV="${CANONICAL_CSV:-}"
FINAL_TABLE_CSV="${FINAL_TABLE_CSV:-}"
SG_LOOKUP_CSV="${SG_LOOKUP_CSV:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"

if [[ -z "$PO_TRAINREADY_H5" ]]; then
  echo "Set PO_TRAINREADY_H5 before running." >&2
  exit 2
fi

mkdir -p "$MODEL_OUTDIR" "$ROOT/results/tmp"
TMP_CONFIG="$(mktemp "$ROOT/results/tmp/poresume2_XXXX.json")"
trap 'rm -f "$TMP_CONFIG"' EXIT

ARGS=(
  --base-config "$BASE_CONFIG"
  --output-config "$TMP_CONFIG"
  --set "data_path=$PO_TRAINREADY_H5"
  --set "resume_checkpoint=$CHECKPOINT_PATH"
  --set "model_path=$MODEL_OUTDIR"
)
[[ -n "$CANONICAL_CSV" ]] && ARGS+=(--set "canonical_table_path=$CANONICAL_CSV")
[[ -n "$FINAL_TABLE_CSV" ]] && ARGS+=(--set "final_table_path=$FINAL_TABLE_CSV")
[[ -n "$SG_LOOKUP_CSV" ]] && ARGS+=(--set "sg_lookup_path=$SG_LOOKUP_CSV")

python3 "$ROOT/scripts/_write_overridden_config.py" "${ARGS[@]}"
python3 "$ROOT/src/core/train.py" --config "$TMP_CONFIG" ${TRAIN_EXTRA_ARGS}
