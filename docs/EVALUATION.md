# Evaluation

This prototype contains both the raw evaluation code and the compact JSON outputs used in the paper.

## Included Evaluation Code

- [evaluate_calibration_metrics.py](/tmp/paper-ai-diffraction/src/eval/evaluate_calibration_metrics.py)
- [evaluate_split_head_validity.py](/tmp/paper-ai-diffraction/src/eval/evaluate_split_head_validity.py)
- [compare_325_failure_modes.py](/tmp/paper-ai-diffraction/src/topology/compare_325_failure_modes.py)
- [analyze_topological_error_distance.py](/tmp/paper-ai-diffraction/src/topology/analyze_topological_error_distance.py)

## Included Evaluation Results

The `results/` directory already contains compact JSON outputs for:

- final mixed champion (`82ept35h`)
- mixed-200k pilot (`eeru8svx`)

These JSONs are enough to regenerate the main benchmark summaries without rerunning the models.

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
