# Figures

The strongest figure-generation path currently migrated into this prototype is the topology/DAG path.

## Included Figure Scripts

- [plot_extinction_topology_flow.py](/tmp/paper-ai-diffraction/src/topology/plot_extinction_topology_flow.py)
- [plot_topological_error_distance.py](/tmp/paper-ai-diffraction/src/topology/plot_topological_error_distance.py)
- [plot_calibration_sweep.py](/tmp/paper-ai-diffraction/src/eval/plot_calibration_sweep.py)

## Included Figure Asset

- [extinction_group_adjacency.json](/tmp/paper-ai-diffraction/assets/topology/extinction_group_adjacency.json)

## Figure Wrappers

- [make_topology_flow_figure.sh](/tmp/paper-ai-diffraction/scripts/make_topology_flow_figure.sh)
- [make_topology_distance_figure.sh](/tmp/paper-ai-diffraction/scripts/make_topology_distance_figure.sh)
- [make_calibration_figure.sh](/tmp/paper-ai-diffraction/scripts/make_calibration_figure.sh)

## Current Gaps

Current external inputs still needed for figure regeneration are:

- `CANONICAL_CSV` for topology-flow rendering
- `CAL_SWEEP_JSON` for calibration-sweep rendering

Those inputs are documented in:

- [dataset_manifest.csv](/tmp/paper-ai-diffraction/reproducibility/dataset_manifest.csv)

What still remains incomplete:

- attention or curriculum plots retained in the manuscript
- any figure paths that still depend on unreduced source-project scripts
