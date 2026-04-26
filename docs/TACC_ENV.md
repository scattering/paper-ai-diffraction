# TACC Environment Notes

This repo has two environment tiers:

1. [environment.yml](../environment.yml)
   - light paper/figure environment
   - intended for editable install, table regeneration, and topology figures
2. [environment-train-eval.yml](../environment-train-eval.yml)
   - full training/evaluation environment
   - intended for checkpoint evaluation and training reruns

## What Was Validated on Stampede

The following path was validated on Stampede `skx-dev` for the paper-facing figure layer:

```bash
module purge
module load gcc/13.2.0
module load python/3.12.11

python3 -m venv /scratch/$USER/venvs/paper-ai-diffraction-py312
source /scratch/$USER/venvs/paper-ai-diffraction-py312/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy scipy pandas matplotlib networkx h5py tqdm

cd /scratch/$USER/paper-ai-diffraction-smoke
python -m pip install -e .
```

Using that environment, the following succeeded on Stampede:

- editable install
- top-level package import
- table regeneration
- topology-distance figure generation
- topology-flow figure generation

## Full Training/Eval Environment

For full evaluation or training reruns on TACC, use the heavier environment:

```bash
module purge
module load gcc/13.2.0
module load python/3.12.11

python3 -m venv /scratch/$USER/venvs/paper-ai-diffraction-train-eval-py312
source /scratch/$USER/venvs/paper-ai-diffraction-train-eval-py312/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

Then install the packages from [environment-train-eval.yml](../environment-train-eval.yml) using your preferred Conda or pip workflow.

The important practical constraint is:

- do not rely on older shared Python 3.9 environments for this repo
- use a fresh Python 3.12 environment for reproducible results

## Reviewer Notebook On TAP

For the interactive reviewer notebook, the intended TACC path is TAP on Stampede3.

Minimal sequence:

```bash
module purge
module load gcc/13.2.0
module load python/3.12.11

python3 -m venv /scratch/$USER/venvs/paper-ai-diffraction-train-eval-py312
source /scratch/$USER/venvs/paper-ai-diffraction-train-eval-py312/bin/activate

python -m pip install --upgrade pip setuptools wheel
cd /path/to/paper-ai-diffraction
python -m pip install -e .
mkdir -p external/checkpoints
```

Then place the released checkpoint at:

```text
external/checkpoints/xrd_model_82ept35h_best.pth
```

After that:

1. open [tap.tacc.utexas.edu](https://tap.tacc.utexas.edu)
2. launch Jupyter on Stampede3
3. open [reviewer_walkthrough.ipynb](../notebooks/reviewer_walkthrough.ipynb)

The shipped notebook walkthrough on TAP works with the compact repo assets plus checkpoint alone. It does not require the full benchmark HDF5s.

Only set `PRIOR_H5` if you want paper-faithful calibrated inference rather than checkpoint-only inference.
