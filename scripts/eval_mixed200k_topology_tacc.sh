#!/bin/bash
#SBATCH -J cmp-mixed200k-r325
#SBATCH -A CDA24014
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/09870/williamratcliff/cmp-mixed200k-r325.%j.out
#SBATCH -e /scratch/09870/williamratcliff/cmp-mixed200k-r325.%j.err

set -euo pipefail
module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate
cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA

python compare_325_failure_modes.py --specs-json /scratch/09870/williamratcliff/ai-diffraction/docs/review_notes/mixed200k_r325_specs_eeru8svx.json --eval-data-path /work2/09870/williamratcliff/rruff-benchmark/RRUFF_usable_plus_recoverable_325_with_labels_maxnorm.hdf5 --prior-data-path /scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_mixed_200k_v1_trainready.hdf5 --output-json /scratch/09870/williamratcliff/mixed200k_compare_325_failure_modes_eeru8svx.json
