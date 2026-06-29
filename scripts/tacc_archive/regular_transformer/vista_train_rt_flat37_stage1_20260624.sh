#!/bin/bash
#SBATCH -J rt-f37-s1
#SBATCH -A CDA24014
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -t 48:00:00
#SBATCH -o /scratch/09870/williamratcliff/rt_flat37_stage1_%j.out
#SBATCH -e /scratch/09870/williamratcliff/rt_flat37_stage1_%j.err

set -euo pipefail

module purge
module load gcc/13.2.0 cuda/12.5 python3/3.11.8
source /scratch/09870/williamratcliff/ai-diffraction-venv/bin/activate

cd /scratch/09870/williamratcliff/ai-diffraction/Code/ViT_NVIDIA
python -m unittest test_extinction_multilabel_mapping.py -v

cd /scratch/09870/williamratcliff/ai-diffraction/Code/Reg_Transformer_FlashAttn
python train_multilabel.py \
  --config config_uniform_2m_blackbird_rt_flat37_stage1_20260624.json \
  --disable_wandb
