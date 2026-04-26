#!/bin/bash
#SBATCH -J eval-mixed2500k-r325473
#SBATCH -A <PROJECT_CODE>
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /scratch/$USER/eval-mixed2500k-r325473.%j.out
#SBATCH -e /scratch/$USER/eval-mixed2500k-r325473.%j.err

set -euo pipefail
module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/$USER/ai-diffraction-venv/bin/activate
cd /scratch/$USER/ai-diffraction/Code/ViT_NVIDIA

: "${TRAIN_JOB_ID:?TRAIN_JOB_ID must be set}"
META="/scratch/$USER/mixed2500k_train_${TRAIN_JOB_ID}.json"
SPEC="/scratch/$USER/ai-diffraction/docs/review_notes/mixed2500k_r325_specs_${TRAIN_JOB_ID}.json"
export META SPEC

python - <<'PY'
import json
import os
meta = os.environ["META"]
spec = os.environ["SPEC"]
with open(meta, "r") as fh:
    payload = json.load(fh)
spec_payload = [
    {
        "name": f"mixed2500k_{payload['train_job_id']}_aux_bayes_t5",
        "checkpoint": payload["checkpoint"],
        "config": payload["config"],
        "decoder": "aux_bayes",
        "temperature": 5.0,
    }
]
with open(spec, "w") as fh:
    json.dump(spec_payload, fh, indent=2)
print(f"[INFO] wrote specs to {spec}")
print(f"[INFO] using checkpoint {payload['checkpoint']}")
PY

ckpt="$(python - <<'PY'
import json, os
with open(os.environ["META"], "r") as fh:
    print(json.load(fh)["checkpoint"])
PY
)"
cfg="$(python - <<'PY'
import json, os
with open(os.environ["META"], "r") as fh:
    print(json.load(fh)["config"])
PY
)"
prior="$(python - <<'PY'
import json, os
with open(os.environ["META"], "r") as fh:
    print(json.load(fh)["prior_data_path"])
PY
)"

python evaluate_calibration_metrics.py --checkpoint "$ckpt" --config "$cfg" --eval-data-path /work2/<PROJECT_CODE>/$USER/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 --prior-data-path "$prior" --aux-temperature 5.0 --bootstrap 1000 --output-json "/scratch/$USER/mixed2500k_calibration_metrics_325_${TRAIN_JOB_ID}.json"

python evaluate_calibration_metrics.py --checkpoint "$ckpt" --config "$cfg" --eval-data-path /work2/<PROJECT_CODE>/$USER/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 --prior-data-path "$prior" --aux-temperature 5.0 --bootstrap 1000 --output-json "/scratch/$USER/mixed2500k_calibration_metrics_473_${TRAIN_JOB_ID}.json"

python evaluate_split_head_validity.py --checkpoint "$ckpt" --config "$cfg" --eval-data-path /work2/<PROJECT_CODE>/$USER/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 --output-json "/scratch/$USER/mixed2500k_split_validity_325_${TRAIN_JOB_ID}.json"

python evaluate_split_head_validity.py --checkpoint "$ckpt" --config "$cfg" --eval-data-path /work2/<PROJECT_CODE>/$USER/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5 --output-json "/scratch/$USER/mixed2500k_split_validity_473_${TRAIN_JOB_ID}.json"

python compare_325_failure_modes.py --specs-json "$SPEC" --eval-data-path /work2/<PROJECT_CODE>/$USER/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 --prior-data-path "$prior" --output-json "/scratch/$USER/mixed2500k_compare_325_failure_modes_${TRAIN_JOB_ID}.json"
