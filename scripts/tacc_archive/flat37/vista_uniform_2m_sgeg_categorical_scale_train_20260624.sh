#!/bin/bash
#SBATCH -J sgeg-scale
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 02:00:00
#SBATCH -o /scratch/09870/williamratcliff/sgeg_scale_train_%j.out
#SBATCH -e /scratch/09870/williamratcliff/sgeg_scale_train_%j.err

set -euo pipefail

: "${SCALE_MODE:?Set SCALE_MODE to sg or eg.}"
: "${SCALE_FRACTION:?Set SCALE_FRACTION to one of 1_2, 1_4, 1_8.}"

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA

case "$SCALE_MODE" in
  sg) TEMPLATE="config_uniform_2m_blackbird_sg230_categorical_referee_20260624.json" ;;
  eg) TEMPLATE="config_uniform_2m_blackbird_eg99_categorical_referee_20260624.json" ;;
  *) echo "Unsupported SCALE_MODE=$SCALE_MODE" >&2; exit 1 ;;
esac

case "$SCALE_FRACTION" in
  1_2) TRAIN_SAMPLES=800000 ;;
  1_4) TRAIN_SAMPLES=400000 ;;
  1_8) TRAIN_SAMPLES=200000 ;;
  *) echo "Unsupported SCALE_FRACTION=$SCALE_FRACTION" >&2; exit 1 ;;
esac

RESULT_DIR="/scratch/09870/williamratcliff/sg_eg_categorical_referee_20260624"
CONFIG_DIR="${RESULT_DIR}/configs"
RUN_TAG="pubfix_sgeg_scale_20260624_${SCALE_MODE}_${SCALE_FRACTION}_s1337"
CONFIG_OUT="${CONFIG_DIR}/${RUN_TAG}.json"

export TEMPLATE TRAIN_SAMPLES SCALE_MODE SCALE_FRACTION RUN_TAG CONFIG_OUT
mkdir -p "$CONFIG_DIR"
python - <<'PY'
import json
import os
from pathlib import Path

cfg = json.loads(Path(os.environ["TEMPLATE"]).read_text())
cfg["run_tag"] = os.environ["RUN_TAG"]
cfg["max_samples_train"] = int(os.environ["TRAIN_SAMPLES"])
cfg["scaling_mode"] = os.environ["SCALE_MODE"]
cfg["scaling_fraction"] = os.environ["SCALE_FRACTION"]
cfg.pop("resume_checkpoint", None)
cfg["resume_weights_only"] = False

out = Path(os.environ["CONFIG_OUT"])
out.write_text(json.dumps(cfg, indent=2))
print(f"[INFO] wrote {out}")
print(f"[INFO] run_tag={cfg['run_tag']}")
print(f"[INFO] max_samples_train={cfg['max_samples_train']}")
PY

python train.py --config "$CONFIG_OUT" --disable_wandb
