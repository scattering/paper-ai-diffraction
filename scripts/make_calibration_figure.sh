#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src/core:$ROOT/src/eval:$ROOT/src/topology:$ROOT/src/utils${PYTHONPATH:+:$PYTHONPATH}"

CAL_SWEEP_JSON="${CAL_SWEEP_JSON:-}"
OUTDIR="${OUTDIR:-$ROOT/results/figures}"
OUTPUT_SVG="${OUTPUT_SVG:-$OUTDIR/calibration_sweep.svg}"
TITLE="${TITLE:-Calibration Sweep on Real RRUFF}"

mkdir -p "$OUTDIR"

if [[ -z "$CAL_SWEEP_JSON" ]]; then
  echo "Set CAL_SWEEP_JSON to a compatible decoder temperature sweep JSON before running." >&2
  exit 2
fi

python3 "$ROOT/src/eval/plot_calibration_sweep.py" \
  --input-json "$CAL_SWEEP_JSON" \
  --output-svg "$OUTPUT_SVG" \
  --title "$TITLE"

echo "Wrote calibration sweep figure to $OUTPUT_SVG"
