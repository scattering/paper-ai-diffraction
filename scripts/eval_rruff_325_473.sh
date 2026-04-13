#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -c "import paper_ai_diffraction" >/dev/null 2>&1 || {
  echo "Install the repo first with: pip install -e ." >&2
  exit 2
}

CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT/external/checkpoints/xrd_model_82ept35h_best.pth}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/final_mixed_2500k_dualsource.json}"
RRUFF_325_H5="${RRUFF_325_H5:-}"
RRUFF_473_H5="${RRUFF_473_H5:-}"
PRIOR_H5="${PRIOR_H5:-}"
OUTDIR="${OUTDIR:-$ROOT/results/eval_rerun}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AUX_TEMP="${AUX_TEMP:-5.0}"
BOOTSTRAP="${BOOTSTRAP:-1000}"

mkdir -p "$OUTDIR"

if [[ -z "$RRUFF_325_H5" || -z "$RRUFF_473_H5" || -z "$PRIOR_H5" ]]; then
  echo "Set RRUFF_325_H5, RRUFF_473_H5, and PRIOR_H5 before running." >&2
  exit 2
fi

python3 -m paper_ai_diffraction.eval.evaluate_calibration_metrics \
  --checkpoint "$CHECKPOINT_PATH" \
  --config "$CONFIG_PATH" \
  --eval-data-path "$RRUFF_325_H5" \
  --prior-data-path "$PRIOR_H5" \
  --aux-temperature "$AUX_TEMP" \
  --num-workers "$NUM_WORKERS" \
  --bootstrap "$BOOTSTRAP" \
  --output-json "$OUTDIR/rruff325_calibration.json"

python3 -m paper_ai_diffraction.eval.evaluate_calibration_metrics \
  --checkpoint "$CHECKPOINT_PATH" \
  --config "$CONFIG_PATH" \
  --eval-data-path "$RRUFF_473_H5" \
  --prior-data-path "$PRIOR_H5" \
  --aux-temperature "$AUX_TEMP" \
  --num-workers "$NUM_WORKERS" \
  --bootstrap "$BOOTSTRAP" \
  --output-json "$OUTDIR/rruff473_calibration.json"

python3 -m paper_ai_diffraction.eval.evaluate_split_head_validity \
  --checkpoint "$CHECKPOINT_PATH" \
  --config "$CONFIG_PATH" \
  --eval-data-path "$RRUFF_325_H5" \
  --output-json "$OUTDIR/rruff325_split_validity.json"

python3 -m paper_ai_diffraction.eval.evaluate_split_head_validity \
  --checkpoint "$CHECKPOINT_PATH" \
  --config "$CONFIG_PATH" \
  --eval-data-path "$RRUFF_473_H5" \
  --output-json "$OUTDIR/rruff473_split_validity.json"

echo "Wrote evaluation outputs to $OUTDIR"
