#!/bin/bash
#SBATCH -J vista-rruff-200k-po-r2
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 02:00:00
#SBATCH -o /scratch/09870/williamratcliff/vista_rruff_conditioned_200k_po_r2_%j.out
#SBATCH -e /scratch/09870/williamratcliff/vista_rruff_conditioned_200k_po_r2_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

export WANDB_API_KEY
WANDB_API_KEY="$(cat /scratch/09870/williamratcliff/ai-diffraction/.wandb_api_key)"
DATA_PATH="/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_200k_po_v1_trainready.hdf5"
EXPECTED_BYTES=5889557080

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA
while true; do
  if [[ -f "$DATA_PATH" ]]; then
    CURRENT_BYTES="$(stat -c%s "$DATA_PATH")"
    echo "[INFO] data_path=$DATA_PATH current_bytes=$CURRENT_BYTES expected_bytes=$EXPECTED_BYTES"
    if [[ "$CURRENT_BYTES" -eq "$EXPECTED_BYTES" ]]; then
      break
    fi
  else
    echo "[INFO] waiting for dataset at $DATA_PATH"
  fi
  sleep 30
done

python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY
python train.py --config config_rruff_conditioned_200k_po_resume2_from_cscjfdwk_physpe_coord.json
