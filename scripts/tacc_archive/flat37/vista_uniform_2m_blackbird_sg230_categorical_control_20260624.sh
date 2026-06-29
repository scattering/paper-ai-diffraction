#!/bin/bash
#SBATCH -J sgeg-sg230
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 12:00:00
#SBATCH -o /scratch/09870/williamratcliff/sgeg_sg230_%j.out
#SBATCH -e /scratch/09870/williamratcliff/sgeg_sg230_%j.err

set -euo pipefail

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA

python train.py \
  --config sg230_categorical_control_20260624.json \
  --disable_wandb
