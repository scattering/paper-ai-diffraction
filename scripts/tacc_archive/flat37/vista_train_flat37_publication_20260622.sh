#!/bin/bash
#SBATCH -J pubfix-train
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 12:00:00
#SBATCH -o /scratch/09870/williamratcliff/pubfix_train_%x_%j.out
#SBATCH -e /scratch/09870/williamratcliff/pubfix_train_%x_%j.err

set -euo pipefail

: "${BASE_CONFIG:?BASE_CONFIG must be set}"
: "${OUTPUT_CONFIG:?OUTPUT_CONFIG must be set}"
: "${META_OUT:?META_OUT must be set}"
: "${JOB_TAG:?JOB_TAG must be set}"

SEED="${SEED:-1337}"
RESUME_META="${RESUME_META:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
PRIOR_DATA_PATH="${PRIOR_DATA_PATH:-}"
DISABLE_WANDB="${DISABLE_WANDB:-1}"
RUN_MAPPING_TEST="${RUN_MAPPING_TEST:-1}"
WAIT_FOR_DATA="${WAIT_FOR_DATA:-1}"
MAX_DATA_WAIT_SECONDS="${MAX_DATA_WAIT_SECONDS:-21600}"
REMOTE_DATA_HOST="${REMOTE_DATA_HOST:-stampede3}"
NUM_WORKERS_OVERRIDE="${NUM_WORKERS_OVERRIDE:-}"
PREFETCH_FACTOR_OVERRIDE="${PREFETCH_FACTOR_OVERRIDE:-}"
MODEL_DIR="/scratch/09870/williamratcliff/ai_diffraction_models"
RESULT_ROOT="/scratch/09870/williamratcliff/flat37_publication"

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

mkdir -p "$MODEL_DIR" "$RESULT_ROOT" "$(dirname "$OUTPUT_CONFIG")" "$(dirname "$META_OUT")"
cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA

if [ "$RUN_MAPPING_TEST" = "1" ]; then
  python -m unittest test_extinction_multilabel_mapping.py -v
fi

MAPPING_JSON="${RESULT_ROOT}/${JOB_TAG}_eg37_mapping.json"
MAPPING_CSV="${RESULT_ROOT}/${JOB_TAG}_eg37_mapping.csv"
python export_extinction_mapping_artifact.py --json-out "$MAPPING_JSON" --csv-out "$MAPPING_CSV"
export MAPPING_JSON MAPPING_CSV
MAPPING_SHA="$(python - <<'PY'
import json
import os
with open(os.environ["MAPPING_JSON"], "r") as handle:
    print(json.load(handle)["sha256_without_checksum"])
PY
)"
export BASE_CONFIG OUTPUT_CONFIG META_OUT JOB_TAG SEED RESUME_META RESUME_CHECKPOINT PRIOR_DATA_PATH MAPPING_SHA NUM_WORKERS_OVERRIDE PREFETCH_FACTOR_OVERRIDE

python - <<'PY'
import json
import os
from pathlib import Path

with open(os.environ["BASE_CONFIG"], "r") as handle:
    cfg = json.load(handle)

cfg["seed"] = int(os.environ["SEED"])
cfg["run_tag"] = os.environ["JOB_TAG"]

resume_meta = os.environ.get("RESUME_META", "")
resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", "")
if resume_meta:
    with open(resume_meta, "r") as handle:
        meta = json.load(handle)
    resume_checkpoint = meta["checkpoint"]

if resume_checkpoint:
    cfg["resume_checkpoint"] = resume_checkpoint
    cfg["resume_weights_only"] = True
else:
    cfg.pop("resume_checkpoint", None)
    cfg.pop("resume_weights_only", None)

if os.environ.get("NUM_WORKERS_OVERRIDE"):
    cfg["num_workers"] = int(os.environ["NUM_WORKERS_OVERRIDE"])
if os.environ.get("PREFETCH_FACTOR_OVERRIDE"):
    cfg["prefetch_factor"] = int(os.environ["PREFETCH_FACTOR_OVERRIDE"])

out = Path(os.environ["OUTPUT_CONFIG"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as handle:
    json.dump(cfg, handle, indent=2)
print(f"[INFO] wrote config {out}")
print(f"[INFO] mapping_sha256_without_checksum={os.environ['MAPPING_SHA']}")
PY

if [ "$WAIT_FOR_DATA" = "1" ]; then
  START_WAIT="$(date +%s)"
  while true; do
    missing="$(python - <<'PY'
import json
import os
from pathlib import Path

with open(os.environ["OUTPUT_CONFIG"], "r") as handle:
    cfg = json.load(handle)

paths = []
if cfg.get("data_path"):
    paths.append(cfg["data_path"])
if cfg.get("standard_data_path"):
    paths.append(cfg["standard_data_path"])
if cfg.get("po_data_path"):
    paths.append(cfg["po_data_path"])

for path in paths:
    if not Path(path).is_file():
        print(path)
PY
)"
    if [ -z "$missing" ]; then
      echo "[INFO] all training data paths are present"
      break
    fi

    while IFS= read -r path; do
      [ -z "$path" ] && continue
      echo "[INFO] missing data path: $path"
      if ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_DATA_HOST" "test -f '$path'" 2>/dev/null; then
        echo "[INFO] staging ${REMOTE_DATA_HOST}:${path}"
        mkdir -p "$(dirname "$path")"
        rsync -a --partial "$REMOTE_DATA_HOST:$path" "$path"
      fi
    done <<< "$missing"

    now="$(date +%s)"
    if [ "$((now - START_WAIT))" -ge "$MAX_DATA_WAIT_SECONDS" ]; then
      echo "[ERROR] timed out waiting for training data paths" >&2
      exit 1
    fi
    sleep 120
  done
fi

train_args=(--config "$OUTPUT_CONFIG")
if [ "$DISABLE_WANDB" = "1" ]; then
  train_args+=(--disable_wandb)
fi
python train.py "${train_args[@]}"

LATEST_CHECKPOINT="$MODEL_DIR/xrd_model_${JOB_TAG}_best.pth"
if [ ! -s "$LATEST_CHECKPOINT" ]; then
  echo "[ERROR] expected checkpoint was not written: $LATEST_CHECKPOINT" >&2
  exit 1
fi
export LATEST_CHECKPOINT

python - <<'PY'
import json
import os
from pathlib import Path

with open(os.environ["OUTPUT_CONFIG"], "r") as handle:
    cfg = json.load(handle)

prior_path = os.environ.get("PRIOR_DATA_PATH") or cfg.get("data_path") or cfg.get("standard_data_path")
payload = {
    "train_job_id": os.environ["SLURM_JOB_ID"],
    "job_tag": os.environ["JOB_TAG"],
    "checkpoint": os.environ["LATEST_CHECKPOINT"],
    "config": os.environ["OUTPUT_CONFIG"],
    "base_config": os.environ["BASE_CONFIG"],
    "seed": int(os.environ["SEED"]),
    "use_split_head": bool(cfg.get("use_split_head", False)),
    "prior_data_path": prior_path,
    "mapping_json": os.environ["MAPPING_JSON"],
    "mapping_csv": os.environ["MAPPING_CSV"],
    "mapping_sha256_without_checksum": os.environ["MAPPING_SHA"],
}
for key in ("data_path", "standard_data_path", "po_data_path"):
    if key in cfg:
        payload[key] = cfg[key]

out = Path(os.environ["META_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as handle:
    json.dump(payload, handle, indent=2)
print(json.dumps(payload, indent=2))
PY
