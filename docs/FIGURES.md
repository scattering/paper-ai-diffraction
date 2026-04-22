# Figures

The strongest figure-generation path currently migrated into this prototype is the topology/DAG path.

Before using the wrappers below:

```bash
conda env create -f environment.yml
conda activate paper-ai-diffraction
pip install -e .
```

For the validated Stampede figure path, see:

- [TACC_ENV.md](/tmp/paper-ai-diffraction/docs/TACC_ENV.md)

## Included Figure Scripts

- [plot_extinction_topology_flow.py](/tmp/paper-ai-diffraction/src/paper_ai_diffraction/topology/plot_extinction_topology_flow.py)
- [plot_topological_error_distance.py](/tmp/paper-ai-diffraction/src/paper_ai_diffraction/topology/plot_topological_error_distance.py)
- [plot_calibration_sweep.py](/tmp/paper-ai-diffraction/src/paper_ai_diffraction/eval/plot_calibration_sweep.py)
- [make_curriculum_real_holdout.py](/tmp/paper-ai-diffraction/scripts/make_curriculum_real_holdout.py)
- [make_stage_decoder_tradeoffs_rruff473.py](/tmp/paper-ai-diffraction/scripts/make_stage_decoder_tradeoffs_rruff473.py)
- [make_physics_pe_q2_ruler.py](/tmp/paper-ai-diffraction/scripts/make_physics_pe_q2_ruler.py)

## Included Figure Asset

- [extinction_group_adjacency.json](/tmp/paper-ai-diffraction/assets/topology/extinction_group_adjacency.json)

## Included Compact Figure Inputs

- [stage2c_r325_temp_sweep.json](/tmp/paper-ai-diffraction/assets/figure_data/stage2c_r325_temp_sweep.json)
- [physics_pe_curve_82ept35h.json](/tmp/paper-ai-diffraction/assets/figure_data/physics_pe_curve_82ept35h.json)

## Figure Wrappers

- [make_topology_flow_figure.sh](/tmp/paper-ai-diffraction/scripts/make_topology_flow_figure.sh)
- [make_topology_distance_figure.sh](/tmp/paper-ai-diffraction/scripts/make_topology_distance_figure.sh)
- [make_calibration_figure.sh](/tmp/paper-ai-diffraction/scripts/make_calibration_figure.sh)

Generated outputs are written under `results/figures/` and are intentionally not tracked in git.

## Current Gaps

Current external inputs still needed for figure regeneration are:

- `CANONICAL_CSV` for topology-flow rendering

Validated Stampede example for `CANONICAL_CSV`:

```bash
export CANONICAL_CSV=/scratch/$USER/ai-diffraction/Code/Post_Processing/canonical_extinction_to_space_group.csv
```

Those inputs are documented in:

- [dataset_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/dataset_manifest.csv)

What still remains incomplete:

- attention or curriculum plots retained in the manuscript
- any figure paths that still depend on unreduced source-project scripts

Current paper plots now source-backed directly in this repo are:

- curriculum holdout
- RRUFF-473 decoder tradeoffs
- calibration sweep from the bundled Stage-2c JSON
- physics-PE ruler supplementary figure
