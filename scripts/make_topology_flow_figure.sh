#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -c "import paper_ai_diffraction" >/dev/null 2>&1 || {
  echo "Install the repo first with: pip install -e ." >&2
  exit 2
}

GRAPH_JSON="${GRAPH_JSON:-$ROOT/assets/topology/extinction_group_adjacency.json}"
CANONICAL_CSV="${CANONICAL_CSV:-}"
FAILURE_JSON="${FAILURE_JSON:-$ROOT/results/mixed2500k_compare_325_failure_modes_655279.json}"
OUTDIR="${OUTDIR:-$ROOT/results/figures}"
LABEL="${LABEL:-champion_mixed_2500k}"

mkdir -p "$OUTDIR"

if [[ -z "$CANONICAL_CSV" ]]; then
  echo "Set CANONICAL_CSV to the canonical extinction-group CSV before running." >&2
  exit 2
fi

python3 -m paper_ai_diffraction.topology.plot_extinction_topology_flow \
  --graph-json "$GRAPH_JSON" \
  --canonical-csv "$CANONICAL_CSV" \
  --model "${LABEL}:${FAILURE_JSON}" \
  --output-dir "$OUTDIR" \
  --top-k-flows 24 \
  --min-count 3 \
  --staged-build

echo "Wrote topology-flow figures to $OUTDIR"
