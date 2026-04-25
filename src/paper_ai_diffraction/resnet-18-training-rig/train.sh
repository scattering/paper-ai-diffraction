#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE
CONFIG_PATH="$1"
NPROC_PER_NODE="$2"
PROJECT="final-cnn-results"
NNODES=${SLURM_JOB_NUM_NODES:-1}
NODE_RANK=${SLURM_PROCID:-0}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -n 1)
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=$(( ( RANDOM % 16383 ) + 49152 ))
unset WANDB_RUN_ID
CUDA_VISIBLE_DEVICES=$(python3 <<EOF
import subprocess, re

nproc = $NPROC_PER_NODE   # bash expands this before python sees it

out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
gpus = []
current = None
for line in out.strip().splitlines():
    gm = re.match(r'GPU \d+:.*UUID: (GPU-[^\)]+)\)', line)
    mm = re.match(r'\s+MIG.*UUID: (MIG-[^\)]+)\)', line)
    if gm:
        current = {"uuid": gm.group(1).strip(), "mig": []}
        gpus.append(current)
    elif mm and current:
        current["mig"].append(mm.group(1).strip())

# Take only the first nproc physical GPUs
gpus = gpus[:nproc]
devices = [g["mig"][0] if g["mig"] else g["uuid"] for g in gpus]
print(",".join(devices))
EOF
)

export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Processes per node:   $NPROC_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# ── Environment ───────────────────────────────────────────────────────────────
module load miniforge
hash -r
export PATH="$CONDA_PREFIX/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate compute

# ── W&B pre-create run ────────────────────────────────────────────────────────
export WANDB_PROJECT="$PROJECT"
if [ "$NODE_RANK" -eq 0 ]; then
    RUN_ID=$(python -c "
import wandb, json
raw = json.load(open('$CONFIG_PATH'))
if 'parameters' in raw:
    config = {k: v.get('value', v.get('values',[None])[0]) for k,v in raw['parameters'].items()}
else:
    config = raw
run = wandb.init(project='$PROJECT', config=config, reinit=True)
print(run.id)
run.finish()
")
    [ -z "$RUN_ID" ] && echo "ERROR: W&B run creation failed" && exit 1
    echo "W&B run: $RUN_ID"
fi
export WANDB_RUN_ID="${RUN_ID:-}"

# ── Debug flags ───────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF

# ── Launch ────────────────────────────────────────────────────────────────────
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --rdzv_id=123 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py \
  --config "$CONFIG_PATH" \
  --sweep_id "none" \
  --num_sweeps 1 \
  --distributed
