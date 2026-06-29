#!/bin/bash
#SBATCH -J rt-f37-s2c-e
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 20:00:00
#SBATCH -o /scratch/09870/williamratcliff/rt_flat37_stage2c_epoch_%j.out
#SBATCH -e /scratch/09870/williamratcliff/rt_flat37_stage2c_epoch_%j.err

set -euo pipefail

: "${TARGET_NUM_EPOCHS:?TARGET_NUM_EPOCHS must be set to the cumulative epoch target}"
: "${RESUME_CHECKPOINT:?RESUME_CHECKPOINT must be set}"

RESUME_WEIGHTS_ONLY="${RESUME_WEIGHTS_ONLY:-0}"
BASE_CONFIG="${BASE_CONFIG:-config_rruff_conditioned_2346k_rt_flat37_stage2c_20260624.json}"
CONFIG_DIR="/scratch/09870/williamratcliff/rt_flat37_referee_20260624/configs"
CONFIG_OUT="${CONFIG_DIR}/rt_stage2c_epoch${TARGET_NUM_EPOCHS}_${SLURM_JOB_ID}.json"

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA
python -m unittest test_extinction_multilabel_mapping.py -v

cd /scratch/09870/williamratcliff/ai-diffraction/Code/Reg_Transformer_FlashAttn
mkdir -p "$CONFIG_DIR"

export BASE_CONFIG CONFIG_OUT TARGET_NUM_EPOCHS RESUME_CHECKPOINT RESUME_WEIGHTS_ONLY
python - <<'PY'
import json
import os
from pathlib import Path

with open(os.environ["BASE_CONFIG"], "r") as handle:
    cfg = json.load(handle)

cfg["num_epochs"] = int(os.environ["TARGET_NUM_EPOCHS"])
cfg["resume_checkpoint"] = os.environ["RESUME_CHECKPOINT"]
cfg["resume_weights_only"] = os.environ["RESUME_WEIGHTS_ONLY"] == "1"

out = Path(os.environ["CONFIG_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as handle:
    json.dump(cfg, handle, indent=2)

print(f"[INFO] wrote chunk config {out}")
print(f"[INFO] target cumulative epochs: {cfg['num_epochs']}")
print(f"[INFO] resume checkpoint: {cfg['resume_checkpoint']}")
print(f"[INFO] resume weights only: {cfg['resume_weights_only']}")
PY

python train_multilabel.py \
  --config "$CONFIG_OUT" \
  --disable_wandb
