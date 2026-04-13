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

## Current Gaps

This prototype still needs a final paper-facing wrapper layer for:

- calibration sweep figures
- topological distance summary figures
- any attention or curriculum plots retained in the manuscript

The underlying analysis code exists in the source project, but only the topology-flow path is wired cleanly here so far.
