#!/bin/bash
#SBATCH -J vista-dualsrc-2500k
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 16:00:00
#SBATCH -o /scratch/09870/williamratcliff/vista_dualsource_2500k_%j.out
#SBATCH -e /scratch/09870/williamratcliff/vista_dualsource_2500k_%j.err

set -euo pipefail

module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

export WANDB_API_KEY
export WANDB_NAME="rruff-dualsource-2500k-gh-${SLURM_JOB_ID}"
WANDB_API_KEY="$(cat /scratch/09870/williamratcliff/ai-diffraction/.wandb_api_key)"
STD_DATA_PATH="/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_2346k_v1_trainready.hdf5"
PO_DATA_PATH="/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5"
REMOTE_PO_PATH="/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5"
REMOTE_HOST="stampede3"
META_PATH="/scratch/09870/williamratcliff/dualsource2500k_train_${SLURM_JOB_ID}.json"
START_TS="$(date +%s)"
export META_PATH START_TS

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA

while true; do
  std_ready=0
  po_ready=0
  if [[ -f "$STD_DATA_PATH" ]]; then
    std_ready=1
  fi
  if [[ -f "$PO_DATA_PATH" ]]; then
    po_ready=1
  fi
  if [[ $std_ready -eq 1 && $po_ready -eq 1 ]]; then
    echo "[INFO] standard_data_path ready at $STD_DATA_PATH"
    echo "[INFO] po_data_path ready at $PO_DATA_PATH"
    break
  fi
  if [[ $po_ready -eq 0 ]] && ssh "$REMOTE_HOST" "test -f '$REMOTE_PO_PATH'"; then
    echo "[INFO] staging PO dataset from ${REMOTE_HOST}:${REMOTE_PO_PATH}"
    rsync -a --partial "$REMOTE_HOST:$REMOTE_PO_PATH" "$PO_DATA_PATH"
  fi
  if [[ $std_ready -eq 0 ]]; then
    echo "[INFO] waiting for standard dataset at $STD_DATA_PATH"
  fi
  if [[ $po_ready -eq 0 ]]; then
    echo "[INFO] waiting for PO dataset at ${REMOTE_HOST}:${REMOTE_PO_PATH}"
  fi
  sleep 120
done

python - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
PY

python train.py --config config_rruff_conditioned_dualsource_2346k_500kpo_from_ic6gfmvm_physpe_coord.json

python - <<'PY'
import glob
import json
import os

model_dir = "/scratch/09870/williamratcliff/ai_diffraction_models"
cfg_path = "/scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA/config_rruff_conditioned_dualsource_2346k_500kpo_from_ic6gfmvm_physpe_coord.json"
prior_path = "/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5"
meta_path = os.environ["META_PATH"]
start_ts = int(os.environ["START_TS"])

candidates = []
for path in glob.glob(os.path.join(model_dir, "xrd_model_*_best.pth")):
    try:
        mtime = int(os.path.getmtime(path))
    except OSError:
        continue
    if mtime >= start_ts - 60:
        candidates.append((mtime, path))

if not candidates:
    raise SystemExit("No best checkpoint produced by dualsource2500k training job")

candidates.sort()
best_path = candidates[-1][1]
payload = {
    "checkpoint": best_path,
    "config": cfg_path,
    "prior_data_path": prior_path,
    "train_job_id": os.environ["SLURM_JOB_ID"]
}
with open(meta_path, "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"[INFO] wrote metadata to {meta_path}")
print(f"[INFO] best checkpoint {best_path}")
PY
