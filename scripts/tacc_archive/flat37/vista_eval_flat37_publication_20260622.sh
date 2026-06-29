#!/bin/bash
#SBATCH -J pubfix-eval
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00
#SBATCH -o /scratch/09870/williamratcliff/pubfix_eval_%x_%j.out
#SBATCH -e /scratch/09870/williamratcliff/pubfix_eval_%x_%j.err

set -euo pipefail

: "${META_IN:?META_IN must be set}"

TEMPS="${TEMPS:-5.0 7.5}"
TEMP_GRID="${TEMP_GRID:-1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10.0 11.0 12.0 13.0 14.0 15.0 16.0 18.0 20.0 25.0 30.0 40.0 50.0 75.0 100.0 150.0 200.0}"
R325="/work2/09870/williamratcliff/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5"
R473="/work2/09870/williamratcliff/rruff-benchmark/RRUFF_option1_473_with_buckets_maxnorm.hdf5"
RESULT_ROOT="/scratch/09870/williamratcliff/corrected_split_publication"

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate
cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA

python -m unittest test_extinction_multilabel_mapping.py -v

export META_IN
ckpt="$(python - <<'PY'
import json, os
with open(os.environ["META_IN"], "r") as handle:
    print(json.load(handle)["checkpoint"])
PY
)"
cfg="$(python - <<'PY'
import json, os
with open(os.environ["META_IN"], "r") as handle:
    print(json.load(handle)["config"])
PY
)"
prior="$(python - <<'PY'
import json, os
with open(os.environ["META_IN"], "r") as handle:
    print(json.load(handle)["prior_data_path"])
PY
)"
tag="$(python - <<'PY'
import json, os
with open(os.environ["META_IN"], "r") as handle:
    print(json.load(handle)["job_tag"])
PY
)"
split_enabled="$(python - <<'PY'
import json, os
with open(os.environ["META_IN"], "r") as handle:
    print("1" if json.load(handle)["use_split_head"] else "0")
PY
)"

OUT_DIR="${RESULT_ROOT}/${tag}_eval_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
cp "$META_IN" "$OUT_DIR/train_meta.json"

for temp in $TEMPS; do
  temp_tag="${temp//./p}"
  python evaluate_calibration_metrics.py \
    --checkpoint "$ckpt" \
    --config "$cfg" \
    --eval-data-path "$R325" \
    --prior-data-path "$prior" \
    --aux-temperature "$temp" \
    --bootstrap 1000 \
    --output-json "$OUT_DIR/${tag}_r325_t${temp_tag}.json"

  python evaluate_calibration_metrics.py \
    --checkpoint "$ckpt" \
    --config "$cfg" \
    --eval-data-path "$R473" \
    --prior-data-path "$prior" \
    --aux-temperature "$temp" \
    --bootstrap 1000 \
    --output-json "$OUT_DIR/${tag}_r473_t${temp_tag}.json"
done

python evaluate_grouped_temperature_cv.py \
  --checkpoint "$ckpt" \
  --config "$cfg" \
  --eval-data-path "$R325" \
  --prior-data-path "$prior" \
  --group-key minerals \
  --n-folds 5 \
  --temperature-grid $TEMP_GRID \
  --selection-metric nll \
  --output-json "$OUT_DIR/${tag}_r325_grouped_temp_cv_nll.json"

python evaluate_grouped_temperature_cv.py \
  --checkpoint "$ckpt" \
  --config "$cfg" \
  --eval-data-path "$R325" \
  --prior-data-path "$prior" \
  --group-key minerals \
  --n-folds 5 \
  --temperature-grid $TEMP_GRID \
  --selection-metric top5 \
  --output-json "$OUT_DIR/${tag}_r325_grouped_temp_cv_top5.json"

if [ "$split_enabled" = "1" ]; then
  python evaluate_split_head_validity.py \
    --checkpoint "$ckpt" \
    --config "$cfg" \
    --eval-data-path "$R325" \
    --output-json "$OUT_DIR/${tag}_split_validity_325.json"

  python evaluate_split_head_validity.py \
    --checkpoint "$ckpt" \
    --config "$cfg" \
    --eval-data-path "$R473" \
    --output-json "$OUT_DIR/${tag}_split_validity_473.json"

  python evaluate_split_head_components_h5.py \
    --checkpoint "$ckpt" \
    --config "$cfg" \
    --data-path "$R325" \
    --split test \
    --output-json "$OUT_DIR/${tag}_split_components_325.json"

  python evaluate_split_head_components_h5.py \
    --checkpoint "$ckpt" \
    --config "$cfg" \
    --data-path "$R473" \
    --split test \
    --output-json "$OUT_DIR/${tag}_split_components_473.json"
fi

SPECS_JSON="$OUT_DIR/${tag}_failure_mode_specs.json"
export SPECS_JSON ckpt cfg tag split_enabled
python - <<'PY'
import json
import os

specs = [
    {
        "name": f"{os.environ['tag']}_aux_bayes_t5",
        "checkpoint": os.environ["ckpt"],
        "config": os.environ["cfg"],
        "decoder": "aux_bayes",
        "temperature": 5.0,
    }
]
if os.environ["split_enabled"] == "1":
    specs.append(
        {
            "name": f"{os.environ['tag']}_split_bayes",
            "checkpoint": os.environ["ckpt"],
            "config": os.environ["cfg"],
            "decoder": "split_bayes",
            "temperature": 1.0,
        }
    )
with open(os.environ["SPECS_JSON"], "w") as handle:
    json.dump(specs, handle, indent=2)
print(json.dumps(specs, indent=2))
PY

python compare_325_failure_modes.py \
  --specs-json "$SPECS_JSON" \
  --eval-data-path "$R325" \
  --prior-data-path "$prior" \
  --output-json "$OUT_DIR/${tag}_compare_325_failure_modes.json"

echo "[INFO] wrote evaluation outputs to $OUT_DIR"
