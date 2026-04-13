#!/usr/bin/env python3
"""
Render slide-friendly extinction-group topology graphs with error-flow overlays.

Design goals:
- Use the condensed extinction-group DAG rather than space-group space.
- Color nodes by crystal system so nearby branches are visually interpretable.
- Overlay model-specific true->pred error traffic with colors keyed to
  descendant / ancestor / branch-jump directionality.
- Emit both SVG and PNG for manuscript / PowerPoint use.
- Optionally emit sequential frames for simple PowerPoint animation.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.patches import FancyArrowPatch


SYSTEM_COLORS = {
    "Triclinic": "#6C757D",
    "Monoclinic": "#4C78A8",
    "Orthorhombic": "#72B7B2",
    "Tetragonal": "#54A24B",
    "Trigonal": "#ECA82C",
    "Hexagonal": "#B279A2",
    "Cubic": "#E45756",
    "Unknown": "#9D755D",
}

FLOW_COLORS = {
    "descendant": "#2F6FB0",
    "ancestor": "#D64550",
    "branch_jump": "#A06B2B",
}

FLOW_ZORDER = {
    "branch_jump": 4,
    "ancestor": 5,
    "descendant": 6,
}


@dataclass
class ModelSpec:
    name: str
    failure_json: str
    summary_json: str | None = None


def parse_args():
    parser = argparse.ArgumentParser(description="Plot extinction-group topology flow graphs.")
    parser.add_argument("--graph-json", required=True)
    parser.add_argument("--canonical-csv", required=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as label:path/to/failure.json[:path/to/summary.json]",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k-flows", type=int, default=24)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--show-labels", type=int, default=18, help="How many busiest nodes to label.")
    parser.add_argument("--animation-frames", action="store_true", help="Emit sequential build frames for PPT.")
    parser.add_argument("--staged-build", action="store_true", help="Emit a 6-frame staged build for each model.")
    parser.add_argument("--hide-flow-counts", action="store_true", help="Suppress numeric labels on flow arrows.")
    return parser.parse_args()


def parse_model_spec(raw: str) -> ModelSpec:
    parts = raw.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid --model spec: {raw}")
    summary_json = parts[2] if len(parts) == 3 else None
    return ModelSpec(name=parts[0], failure_json=parts[1], summary_json=summary_json)


def parse_space_group_list(raw):
    if isinstance(raw, list):
        return [int(v) for v in raw]
    if pd.isna(raw):
        return []
    return [int(v) for v in ast.literal_eval(str(raw))]


def sg_to_system(sg_num: int) -> str:
    if 1 <= sg_num <= 2:
        return "Triclinic"
    if 3 <= sg_num <= 15:
        return "Monoclinic"
    if 16 <= sg_num <= 74:
        return "Orthorhombic"
    if 75 <= sg_num <= 142:
        return "Tetragonal"
    if 143 <= sg_num <= 167:
        return "Trigonal"
    if 168 <= sg_num <= 194:
        return "Hexagonal"
    if 195 <= sg_num <= 230:
        return "Cubic"
    return "Unknown"


def load_graph(graph_json: str):
    with open(graph_json, "r") as handle:
        blob = json.load(handle)

    graph = nx.DiGraph()
    label_to_node = {}
    for node, attrs in blob["nodes"].items():
        graph.add_node(str(node), **attrs)
        for merged in attrs.get("merged_from", []):
            label_to_node[int(merged) + 1] = str(node)
    for parent, children in blob["adjacency"].items():
        for child in children:
            graph.add_edge(str(parent), str(child))
    return graph, label_to_node


def undirected_shortest_distance(undirected: nx.Graph, src: str, dst: str) -> int | None:
    try:
        return nx.shortest_path_length(undirected, src, dst)
    except nx.NetworkXNoPath:
        return None


def classify_direction(graph: nx.DiGraph, true_node: str, pred_node: str) -> str:
    if nx.has_path(graph, true_node, pred_node):
        return "descendant"
    if nx.has_path(graph, pred_node, true_node):
        return "ancestor"
    return "branch_jump"


def compute_levels(graph: nx.DiGraph) -> dict[str, int]:
    # The condensed graph has a single leaf in practice, but compute generally.
    sinks = [n for n in graph.nodes if graph.out_degree(n) == 0]
    rev = graph.reverse(copy=False)
    levels = {}
    for node in graph.nodes:
        dists = []
        for sink in sinks:
            try:
                dists.append(nx.shortest_path_length(rev, node, sink))
            except nx.NetworkXNoPath:
                continue
        levels[node] = min(dists) if dists else 0
    return levels


def build_positions(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    levels = compute_levels(graph)
    grouped = defaultdict(list)
    for node, level in levels.items():
        grouped[level].append(node)

    pos = {}
    for level, pairs in grouped.items():
        nodes = sorted(
            pairs,
            key=lambda node: (
                len(graph.nodes[node].get("space_groups", [])),
                str(graph.nodes[node].get("name", "")),
            ),
        )
        count = len(nodes)
        for idx, node in enumerate(nodes):
            x = 0.0 if count == 1 else idx - (count - 1) / 2.0
            y = -float(level)
            pos[node] = (x, y)
    return pos


def load_systems(canonical_csv: str) -> dict[int, str]:
    df = pd.read_csv(canonical_csv)
    systems = {}
    for _, row in df.iterrows():
        eg = int(row["Index"]) + 1
        sgs = parse_space_group_list(row["Space Group Numbers"])
        systems[eg] = sg_to_system(sgs[0]) if sgs else "Unknown"
    return systems


def short_name(name: str) -> str:
    text = str(name)
    text = text.replace(" (equiv:", "\n(eqv:")
    return text


def load_summary(summary_json: str | None, model_key: str):
    if not summary_json or not os.path.exists(summary_json):
        return None
    with open(summary_json, "r") as handle:
        blob = json.load(handle)
    return blob["models"].get(model_key)


def aggregate_flows(graph, label_to_node, failure_json: str):
    with open(failure_json, "r") as handle:
        blob = json.load(handle)
    model_key = list(blob["models"])[0]
    examples = blob["models"][model_key]["examples"]
    flows = Counter()
    node_load = Counter()
    direction_counts = Counter()
    hop_counts = Counter()
    for ex in examples:
        if ex["correct"]:
            continue
        true_node = label_to_node.get(int(ex["true_eg"]))
        pred_node = label_to_node.get(int(ex["pred_eg"]))
        if true_node is None or pred_node is None:
            continue
        direction = classify_direction(graph, true_node, pred_node)
        hop = undirected_shortest_distance(graph.to_undirected(), true_node, pred_node)
        flows[(true_node, pred_node, direction)] += 1
        node_load[true_node] += 1
        node_load[pred_node] += 1
        direction_counts[direction] += 1
        if hop is not None:
            hop_counts[hop] += 1
    return model_key, flows, node_load, direction_counts, hop_counts


def draw_curved_arrow(ax, p1, p2, color, width, alpha, rad):
    patch = FancyArrowPatch(
        p1,
        p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=8 + width * 1.5,
        lw=width,
        color=color,
        alpha=alpha,
        zorder=FLOW_ZORDER["branch_jump"],
    )
    ax.add_patch(patch)


def render_model(
    graph,
    pos,
    systems_by_eg,
    label_to_node,
    spec: ModelSpec,
    args,
    zoom: bool = False,
    flow_directions: set[str] | None = None,
    show_summary: bool = True,
    show_legends: bool = True,
    show_labels: bool = True,
    title_suffix: str | None = None,
):
    model_key, flows, node_load, direction_counts, hop_counts = aggregate_flows(
        graph, label_to_node, spec.failure_json
    )
    summary = load_summary(spec.summary_json, model_key)

    node_system = {}
    for node in graph.nodes:
        merged = graph.nodes[node].get("merged_from", [])
        eg = int(merged[0]) + 1 if merged else None
        node_system[node] = systems_by_eg.get(eg, "Unknown")

    top_flows = [(k, v) for k, v in flows.items() if v >= args.min_count]
    top_flows.sort(key=lambda item: item[1], reverse=True)
    top_flows = top_flows[: args.top_k_flows]

    highlighted = set()
    for (src, dst, _), _count in top_flows:
        highlighted.add(src)
        highlighted.add(dst)

    busiest_nodes = [node for node, _ in node_load.most_common(args.show_labels)]
    label_nodes = set(busiest_nodes) | highlighted

    plot_graph = graph
    plot_pos = pos
    if zoom:
        active_nodes = set()
        undirected = graph.to_undirected()
        for (src, dst, _direction), _count in top_flows:
            active_nodes.add(src)
            active_nodes.add(dst)
            try:
                active_nodes.update(nx.shortest_path(undirected, src, dst))
            except nx.NetworkXNoPath:
                pass
        plot_graph = graph.subgraph(active_nodes).copy()
        plot_pos = nx.spring_layout(
            plot_graph.to_undirected(),
            seed=7,
            k=max(0.45, 2.4 / math.sqrt(max(len(plot_graph.nodes), 1))),
            scale=10.0,
            center=(0.0, 0.0),
            iterations=300,
        )
        label_nodes = set(plot_graph.nodes)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_facecolor("white")

    # Base DAG edges.
    nx.draw_networkx_edges(
        plot_graph,
        plot_pos,
        ax=ax,
        edge_color="#D8D8D8",
        width=1.2 if zoom else 1.0,
        alpha=0.8 if zoom else 0.7,
        arrows=False,
    )

    # Base nodes.
    node_sizes = []
    node_colors = []
    edge_colors = []
    edge_widths = []
    for node in plot_graph.nodes:
        traffic = node_load.get(node, 0)
        node_sizes.append((260 if zoom else 120) + (40 if zoom else 26) * math.sqrt(max(traffic, 0)))
        node_colors.append(SYSTEM_COLORS[node_system[node]])
        edge_colors.append("#202020" if node in highlighted else "#FFFFFF")
        edge_widths.append(1.8 if node in highlighted else 0.6)

    nx.draw_networkx_nodes(
        plot_graph,
        plot_pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=edge_colors,
        linewidths=edge_widths,
        alpha=0.95,
    )

    filtered_flows = top_flows
    if flow_directions is not None:
        filtered_flows = [item for item in top_flows if item[0][2] in flow_directions]

    # Overlay main error flows.
    for (src, dst, direction), count in filtered_flows:
        if src not in plot_graph.nodes or dst not in plot_graph.nodes:
            continue
        p1 = plot_pos[src]
        p2 = plot_pos[dst]
        width = 1.4 + 0.45 * math.sqrt(count)
        alpha = min(0.92, 0.25 + 0.05 * count)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        rad_mag = 0.12 if abs(dx) >= 0.5 else 0.22
        rad = rad_mag if dy >= 0 else -rad_mag
        if direction == "descendant":
            rad *= 0.7
        elif direction == "ancestor":
            rad *= -0.7
        draw_curved_arrow(ax, p1, p2, FLOW_COLORS[direction], width, alpha, rad)
        mx = (p1[0] + p2[0]) / 2.0
        my = (p1[1] + p2[1]) / 2.0 + rad * 0.55
        if not args.hide_flow_counts:
            ax.text(
                mx,
                my,
                str(count),
                fontsize=9,
                color=FLOW_COLORS[direction],
                ha="center",
                va="center",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                zorder=10,
            )

    # Labels only on highlighted / busy nodes.
    if show_labels:
        for node in label_nodes:
            if node not in plot_graph.nodes:
                continue
            x, y = plot_pos[node]
            merged = graph.nodes[node].get("merged_from", [])
            eg = int(merged[0]) + 1 if merged else None
            label = f"EG {eg}"
            ax.text(
                x,
                y - (0.22 if zoom else 0.38),
                label,
                fontsize=11 if zoom else 9,
                ha="center",
                va="top",
                weight="bold",
                color="#111111",
                zorder=11,
            )

    suffix = title_suffix or ("focused hop subgraph" if zoom else "extinction-group topology error flows")
    title = f"{spec.name}: {suffix}"
    ax.set_title(title, fontsize=18, weight="bold", pad=20)

    summary_lines = [
        f"Top flows shown: {len(top_flows)}",
        f"descendant={direction_counts.get('descendant', 0)}",
        f"ancestor={direction_counts.get('ancestor', 0)}",
        f"branch={direction_counts.get('branch_jump', 0)}",
    ]
    if summary is not None:
        summary_lines.extend(
            [
                f"top-1={summary['top1_pct']:.2f}%",
                f"<=2 hops={summary['pct_wrong_le2']:.2f}%",
                f"mean DAG dist={summary['mean_wrong_dist']:.2f}",
            ]
        )
    else:
        if hop_counts:
            le2 = hop_counts.get(1, 0) + hop_counts.get(2, 0)
            wrong = sum(hop_counts.values())
            summary_lines.append(f"<=2 hops={100.0 * le2 / wrong:.2f}%")
    if show_summary:
        ax.text(
            0.015,
            0.985,
            "\n".join(summary_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#CFCFCF", alpha=0.95),
        )

    # Legends.
    system_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=system, markerfacecolor=color, markersize=10)
        for system, color in SYSTEM_COLORS.items()
    ]
    flow_handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=label.replace("_", " "))
        for label, color in FLOW_COLORS.items()
    ]
    if show_legends:
        leg1 = ax.legend(handles=system_handles, loc="lower left", title="Node color: crystal system", frameon=True)
        ax.add_artist(leg1)
        ax.legend(handles=flow_handles, loc="lower right", title="Flow color: error direction", frameon=True)

    xs = [xy[0] for xy in plot_pos.values()]
    ys = [xy[1] for xy in plot_pos.values()]
    ax.set_xlim(min(xs) - 2.5, max(xs) + 2.5)
    ax.set_ylim(min(ys) - 2.0, max(ys) + 2.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    return fig


def save_figure(fig, output_base: Path, dpi: int):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    specs = [parse_model_spec(raw) for raw in args.model]
    graph, label_to_node = load_graph(args.graph_json)
    pos = build_positions(graph)
    systems_by_eg = load_systems(args.canonical_csv)
    output_dir = Path(args.output_dir)

    for idx, spec in enumerate(specs):
        fig = render_model(graph, pos, systems_by_eg, label_to_node, spec, args, zoom=False)
        stem = spec.name.lower().replace(" ", "_").replace("/", "_")
        save_figure(fig, output_dir / stem, args.dpi)

        fig = render_model(graph, pos, systems_by_eg, label_to_node, spec, args, zoom=True)
        save_figure(fig, output_dir / f"{stem}_zoom", args.dpi)

        if args.animation_frames:
            fig = render_model(graph, pos, systems_by_eg, label_to_node, spec, args, zoom=True)
            save_figure(fig, output_dir / f"frame_{idx+1:02d}_{stem}", args.dpi)

        if args.staged_build:
            stages = [
                ("01_base_dag", None, False, False, False, "base DAG"),
                ("02_system_nodes", set(), False, True, False, "node colors by crystal system"),
                ("03_descendant", {"descendant"}, False, True, True, "descendant hops"),
                ("04_ancestor", {"descendant", "ancestor"}, False, True, True, "descendant + ancestor hops"),
                ("05_branch", {"descendant", "ancestor", "branch_jump"}, False, True, True, "all hop types"),
                ("06_final", {"descendant", "ancestor", "branch_jump"}, True, True, True, "final annotated view"),
            ]
            for stage_name, dirs, show_summary, show_legends, show_labels_flag, suffix in stages:
                fig = render_model(
                    graph,
                    pos,
                    systems_by_eg,
                    label_to_node,
                    spec,
                    args,
                    zoom=True,
                    flow_directions=dirs,
                    show_summary=show_summary,
                    show_legends=show_legends,
                    show_labels=show_labels_flag,
                    title_suffix=suffix,
                )
                save_figure(fig, output_dir / f"{stage_name}_{stem}", args.dpi)


if __name__ == "__main__":
    main()
