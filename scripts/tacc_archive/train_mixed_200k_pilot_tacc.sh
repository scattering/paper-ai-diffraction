#!/bin/bash
#SBATCH -J vista-rruff-mixed-200k
#SBATCH -A <PROJECT_CODE>
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/$USER/vista_rruff_mixed_200k_%j.out
#SBATCH -e /scratch/$USER/vista_rruff_mixed_200k_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/$USER/ai-diffraction-venv/bin/activate

export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/$USER/ai-diffraction/.wandb_api_key)"
DATA_PATH="/scratch/$USER/ai_diffraction_generated/rruff_conditioned_mixed_200k_v1_trainready.hdf5"

cd /scratch/$USER/ai-diffraction/Code/ViT_NVIDIA
while true; do
  if [[ -f "$DATA_PATH" ]]; then
    echo "[INFO] mixed dataset ready at $DATA_PATH"
    break
  fi
  echo "[INFO] waiting for mixed dataset at $DATA_PATH"
  sleep 15
done

python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY
python train.py --config config_rruff_conditioned_mixed_200k_from_ic6gfmvm_physpe_coord.json
