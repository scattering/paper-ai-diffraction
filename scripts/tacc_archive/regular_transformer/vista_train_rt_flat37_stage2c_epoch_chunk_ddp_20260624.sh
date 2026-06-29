#!/bin/bash
#SBATCH -J rt-f37-s2c-ddp
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -t 08:00:00
#SBATCH -o /scratch/09870/williamratcliff/rt_flat37_stage2c_ddp_%j.out
#SBATCH -e /scratch/09870/williamratcliff/rt_flat37_stage2c_ddp_%j.err

set -euo pipefail

: "${TARGET_NUM_EPOCHS:?TARGET_NUM_EPOCHS must be set to the cumulative epoch target}"

BASE_CONFIG="${BASE_CONFIG:-config_rruff_conditioned_2346k_rt_flat37_stage2c_20260624.json}"
CONFIG_DIR="/scratch/09870/williamratcliff/rt_flat37_20260624/configs"
CONFIG_OUT="${CONFIG_DIR}/rt_stage2c_ddp_epoch${TARGET_NUM_EPOCHS}_${SLURM_JOB_ID}.json"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-$(( GLOBAL_BATCH_SIZE / SLURM_NNODES ))}"

if [ "$BATCH_SIZE_PER_RANK" -lt 1 ]; then
  echo "[ERROR] computed BATCH_SIZE_PER_RANK=${BATCH_SIZE_PER_RANK}" >&2
  exit 1
fi

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA
python -m unittest test_extinction_multilabel_mapping.py -v

cd /scratch/09870/williamratcliff/ai-diffraction/Code/Reg_Transformer_FlashAttn
mkdir -p "$CONFIG_DIR"

export BASE_CONFIG CONFIG_OUT TARGET_NUM_EPOCHS RESUME_CHECKPOINT RESUME_WEIGHTS_ONLY
export BATCH_SIZE_PER_RANK RUN_TAG_OVERRIDE MAX_SAMPLES_TRAIN MAX_SAMPLES_VAL MAX_SAMPLES_TEST
python - <<'PY'
import json
import os
from pathlib import Path

with open(os.environ["BASE_CONFIG"], "r") as handle:
    cfg = json.load(handle)

cfg["num_epochs"] = int(os.environ["TARGET_NUM_EPOCHS"])
cfg["batch_size"] = int(os.environ["BATCH_SIZE_PER_RANK"])

resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", "")
if resume_checkpoint and resume_checkpoint.upper() != "NONE":
    cfg["resume_checkpoint"] = resume_checkpoint
    cfg["resume_weights_only"] = os.environ.get("RESUME_WEIGHTS_ONLY", "0") == "1"
else:
    cfg.pop("resume_checkpoint", None)
    cfg["resume_weights_only"] = False

run_tag = os.environ.get("RUN_TAG_OVERRIDE", "")
if run_tag:
    cfg["run_tag"] = run_tag

for env_key, cfg_key in (
    ("MAX_SAMPLES_TRAIN", "max_samples_train"),
    ("MAX_SAMPLES_VAL", "max_samples_val"),
    ("MAX_SAMPLES_TEST", "max_samples_test"),
):
    value = os.environ.get(env_key, "")
    if value:
        cfg[cfg_key] = int(value)

out = Path(os.environ["CONFIG_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as handle:
    json.dump(cfg, handle, indent=2)

print(f"[INFO] wrote DDP chunk config {out}")
print(f"[INFO] target cumulative epochs: {cfg['num_epochs']}")
print(f"[INFO] per-rank batch size: {cfg['batch_size']}")
print(f"[INFO] run tag: {cfg.get('run_tag')}")
print(f"[INFO] resume checkpoint: {cfg.get('resume_checkpoint', 'NONE')}")
print(f"[INFO] resume weights only: {cfg.get('resume_weights_only')}")
for key in ("max_samples_train", "max_samples_val", "max_samples_test"):
    if key in cfg:
        print(f"[INFO] {key}: {cfg[key]}")
PY

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$(( 20000 + SLURM_JOB_ID % 40000 ))}"
export MASTER_ADDR MASTER_PORT OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

echo "[INFO] nodes=${SLURM_NNODES} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[INFO] global_batch_size=${GLOBAL_BATCH_SIZE} batch_size_per_rank=${BATCH_SIZE_PER_RANK}"

srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -lc '
  echo "[INFO] host=$(hostname) slurm_procid=${SLURM_PROCID} localid=${SLURM_LOCALID}"
  python -m torch.distributed.run \
    --nnodes="${SLURM_NNODES}" \
    --nproc_per_node=1 \
    --node_rank="${SLURM_PROCID}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train_multilabel.py \
      --config "'"${CONFIG_OUT}"'" \
      --distributed \
      --disable_wandb
'
