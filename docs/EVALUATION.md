# Evaluation

This repo contains both the raw evaluation code and the compact JSON outputs used in the paper.

Before using the wrappers below:

```bash
conda env create -f environment-train-eval.yml
conda activate paper-ai-diffraction-train-eval
pip install -e .
```

For TACC-specific notes, see:

- [TACC_ENV.md](TACC_ENV.md)

## Included Evaluation Code

- [evaluate_calibration_metrics.py](../src/paper_ai_diffraction/eval/evaluate_calibration_metrics.py)
- [compare_325_failure_modes.py](../src/paper_ai_diffraction/topology/compare_325_failure_modes.py)
- [analyze_topological_error_distance.py](../src/paper_ai_diffraction/topology/analyze_topological_error_distance.py)

## Included Evaluation Results

The `results/` directory intentionally contains compact paper-backed outputs:

- revised summary rows in [revised_paper_summary.json](../results/revised_paper_summary.json)
- grouped mineral-family calibration assets under [flat37/calibration](../results/flat37/calibration)
- information-channel and topology summaries under [flat37/information_theory](../results/flat37/information_theory)
- older mixed-curriculum and positional-ablation JSONs retained for provenance

These JSONs are enough to regenerate the main benchmark summaries without rerunning the models. Generated figures and derived summaries should be written under `results/figures/` and are not tracked.

## Paper-Relevant Benchmark Summary

The public repo keeps the essential benchmark outcomes needed to interpret the bundled JSONs.

Stage-2c flat-37 ViT, calibrated Bayesian auxiliary output at `T=5`:

- `RRUFF-325` Top-1 / Top-3 / Top-5 = `10.77 / 26.15 / 42.46`
- `RRUFF-473` Top-1 / Top-5 = `11.84 / 49.68`
- RRUFF-325 calibration metrics: `ECE = 0.059`, `NLL = 3.69`, `Brier = 0.960`

Preferred-orientation and mixed-curriculum flat-37 variants:

- pure PO: `RRUFF-325` Top-1 / Top-5 = `11.69 / 45.23`; `RRUFF-473` = `12.26 / 53.28`
- large mixed: `RRUFF-325` Top-1 / Top-5 = `11.69 / 43.69`; `RRUFF-473` = `12.47 / 51.37`

Matched regular transformer:

- `RRUFF-325` Top-1 / Top-3 / Top-5 = `10.77 / 30.15 / 43.69`
- `RRUFF-473` Top-1 / Top-3 / Top-5 = `12.68 / 34.46 / 51.16`

Matched SG-to-EG control on the same 2.0M uniform corpus:

- raw SG space: `8.54 / 17.89 / 24.15`
- SG probabilities collapsed to EG space: `10.22 / 21.67 / 29.30`
- direct EG: `12.05 / 24.17 / 31.96`

The compact source for these rows is [revised_paper_summary.json](../results/revised_paper_summary.json).

## Canonical Evaluation Wrapper

Use:

- [eval_rruff_325_473.sh](../scripts/eval_rruff_325_473.sh)

This script expects:

- `CHECKPOINT_PATH`
- `CONFIG_PATH`
- `RRUFF_325_H5`
- `RRUFF_473_H5`
- `PRIOR_H5`

Those variables and their intended roles are documented in:

- [dataset_manifest.csv](../reproducibility/dataset_manifest.csv)
- [checkpoint_manifest.csv](../reproducibility/checkpoint_manifest.csv)

and then runs calibration metrics on:

1. `RRUFF-325`
2. `RRUFF-473`

Set `RUN_SPLIT_VALIDITY=1` only for historical checkpoints that explicitly use that evaluation path.

## Topology and Table Regeneration

Use:

- [make_topology_flow_figure.sh](../scripts/make_topology_flow_figure.sh)
- [make_calibration_figure.sh](../scripts/make_calibration_figure.sh)
- [make_main_tables.py](../scripts/make_main_tables.py)

The topology-flow wrapper renders the staged DAG figure set. The calibration wrapper renders a Top-1/Top-5 versus temperature SVG from a compatible sweep JSON. The table script prints compact CSV rows from the revised summary JSON.

## Historical TACC Evaluation Launchers

The original cluster-bound evaluation launchers are preserved under:

- [../scripts/tacc_archive]( ../scripts/tacc_archive )

Those scripts are retained for provenance, not as the preferred public interface.
