#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -c "import paper_ai_diffraction" >/dev/null 2>&1 || {
  echo "Install the repo first with: pip install -e ." >&2
  exit 2
}

GRAPH_JSON="${GRAPH_JSON:-$ROOT/assets/topology/extinction_group_adjacency.json}"
MIXED200K_FAILURE_JSON="${MIXED200K_FAILURE_JSON:-$ROOT/results/mixed200k_compare_325_failure_modes_eeru8svx.json}"
MIXED2500K_FAILURE_JSON="${MIXED2500K_FAILURE_JSON:-$ROOT/results/mixed2500k_compare_325_failure_modes_655279.json}"
OUTDIR="${OUTDIR:-$ROOT/results/figures}"

mkdir -p "$OUTDIR"

python3 -m paper_ai_diffraction.topology.analyze_topological_error_distance \
  --graph-json "$GRAPH_JSON" \
  --failure-json "$MIXED200K_FAILURE_JSON" \
  --output-json "$OUTDIR/mixed200k_topology_summary.json" >/dev/null

python3 -m paper_ai_diffraction.topology.analyze_topological_error_distance \
  --graph-json "$GRAPH_JSON" \
  --failure-json "$MIXED2500K_FAILURE_JSON" \
  --output-json "$OUTDIR/mixed2500k_topology_summary.json" >/dev/null

python3 - <<'PY' "$OUTDIR/mixed200k_topology_summary.json" "$OUTDIR/mixed2500k_topology_summary.json" "$OUTDIR/combined_topology_summary.json"
import json, pathlib, sys
paths = [pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])]
combined = {"models": {}}
for p in paths:
    blob = json.loads(p.read_text())
    combined["models"].update(blob["models"])
pathlib.Path(sys.argv[3]).write_text(json.dumps(combined, indent=2))
PY

python3 -m paper_ai_diffraction.topology.plot_topological_error_distance \
  --summary-json "$OUTDIR/combined_topology_summary.json" \
  --output-svg "$OUTDIR/topological_error_distance.svg"

echo "Wrote topology-distance summary to $OUTDIR/topological_error_distance.svg"
