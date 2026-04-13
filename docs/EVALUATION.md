# Evaluation

This prototype contains both the raw evaluation code and the compact JSON outputs used in the paper.

Before using the wrappers below:

```bash
conda env create -f environment-train-eval.yml
conda activate paper-ai-diffraction-train-eval
pip install -e .
```

For TACC-specific notes, see:

- [TACC_ENV.md](/tmp/paper-ai-diffraction/docs/TACC_ENV.md)

## Included Evaluation Code

- [evaluate_calibration_metrics.py](/tmp/paper-ai-diffraction/src/paper_ai_diffraction/eval/evaluate_calibration_metrics.py)
- [evaluate_split_head_validity.py](/tmp/paper-ai-diffraction/src/paper_ai_diffraction/eval/evaluate_split_head_validity.py)
- [compare_325_failure_modes.py](/tmp/paper-ai-diffraction/src/paper_ai_diffraction/topology/compare_325_failure_modes.py)
- [analyze_topological_error_distance.py](/tmp/paper-ai-diffraction/src/paper_ai_diffraction/topology/analyze_topological_error_distance.py)

## Included Evaluation Results

The `results/` directory intentionally contains only compact JSON outputs for:

- final mixed champion (`82ept35h`)
- mixed-200k pilot (`eeru8svx`)

These JSONs are enough to regenerate the main benchmark summaries without rerunning the models. Generated figures and derived summaries should be written under `results/figures/` and are not tracked.

## Paper-Relevant Benchmark Summary

The public repo keeps the essential benchmark outcomes needed to interpret the bundled JSONs.

Final large mixed champion (`82ept35h`):

- `RRUFF-325` calibrated Bayesian auxiliary `Top-1 / Top-5 = 10.46 / 43.69`
- `RRUFF-473` calibrated Bayesian auxiliary `Top-1 / Top-5 = 9.94 / 50.74`
- split-head validity:
  - `RRUFF-325 = 47.38%`
  - `RRUFF-473 = 49.26%`

Mixed-200k pilot (`eeru8svx`):

- `RRUFF-325` calibrated Bayesian auxiliary `Top-1 / Top-5 = 13.23 / 40.92`
- `RRUFF-473` calibrated Bayesian auxiliary `Top-1 / Top-5 = 13.53 / 49.05`
- split-head validity:
  - `RRUFF-325 = 1.54%`
  - `RRUFF-473 = 1.27%`

These numbers are included here so the public repo still carries the minimal scientific lineage after removing the dated internal notes.

## Canonical Evaluation Wrapper

Use:

- [eval_rruff_325_473.sh](/tmp/paper-ai-diffraction/scripts/eval_rruff_325_473.sh)

This script expects:

- `CHECKPOINT_PATH`
- `CONFIG_PATH`
- `RRUFF_325_H5`
- `RRUFF_473_H5`
- `PRIOR_H5`

Those variables and their intended roles are documented in:

- [dataset_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/dataset_manifest.csv)
- [checkpoint_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/checkpoint_manifest.csv)

and then runs:

1. calibration metrics on `RRUFF-325`
2. calibration metrics on `RRUFF-473`
3. split-head validity on both

## Topology and Table Regeneration

Use:

- [make_topology_flow_figure.sh](/tmp/paper-ai-diffraction/scripts/make_topology_flow_figure.sh)
- [make_calibration_figure.sh](/tmp/paper-ai-diffraction/scripts/make_calibration_figure.sh)
- [make_main_tables.py](/tmp/paper-ai-diffraction/scripts/make_main_tables.py)

The topology-flow wrapper renders the staged DAG figure set. The calibration wrapper renders a Top-1/Top-5 versus temperature SVG from a compatible sweep JSON. The table script prints compact CSV rows for the mixed-200k and final mixed champion paper numbers.

## Historical TACC Evaluation Launchers

The original cluster-bound evaluation launchers are preserved under:

- [/tmp/paper-ai-diffraction/scripts/tacc_archive]( /tmp/paper-ai-diffraction/scripts/tacc_archive )

Those scripts are retained for provenance, not as the preferred public interface.
