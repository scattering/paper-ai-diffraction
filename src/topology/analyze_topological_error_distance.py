#!/usr/bin/env python3
import argparse
import json
from collections import Counter, deque


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze model prediction errors on the extinction-group subgroup graph.")
    parser.add_argument("--graph-json", required=True, help="Path to extinction_group_adjacency.json")
    parser.add_argument("--failure-json", required=True, help="Path to compare_325_failure_modes.json")
    parser.add_argument("--output-json", required=True, help="Path to write summary JSON")
    return parser.parse_args()


def build_graph(graph_json_path):
    with open(graph_json_path, "r") as handle:
        data = json.load(handle)

    nodes = {str(key): value for key, value in data["nodes"].items()}
    directed = {str(key): [str(child) for child in value] for key, value in data["adjacency"].items()}
    undirected = {node: set() for node in nodes}
    for parent, children in directed.items():
        for child in children:
            undirected.setdefault(parent, set()).add(child)
            undirected.setdefault(child, set()).add(parent)

    # The graph assets use 0..98 for extinction-group IDs, while model outputs use 1..99.
    label_to_node = {}
    for node, attrs in nodes.items():
        for raw_eg in attrs.get("merged_from", []):
            label_to_node[int(raw_eg) + 1] = node

    return nodes, directed, undirected, label_to_node


def shortest_distance(undirected, src, dst):
    if src == dst:
        return 0
    queue = deque([(src, 0)])
    seen = {src}
    while queue:
        node, dist = queue.popleft()
        for nxt in undirected.get(node, ()):
            if nxt == dst:
                return dist + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, dist + 1))
    return None


def has_directed_path(directed, src, dst):
    if src == dst:
        return True
    queue = deque([src])
    seen = {src}
    while queue:
        node = queue.popleft()
        for nxt in directed.get(node, ()):
            if nxt == dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def summarize_model(model_blob, directed, undirected, label_to_node):
    directionality = Counter()
    wrong_distances = []

    for example in model_blob["examples"]:
        true_node = label_to_node.get(int(example["true_eg"]))
        pred_node = label_to_node.get(int(example["pred_eg"]))
        if true_node is None or pred_node is None:
            directionality["unmapped"] += 1
            continue
        if true_node == pred_node:
            directionality["correct"] += 1
            continue

        dist = shortest_distance(undirected, true_node, pred_node)
        wrong_distances.append("inf" if dist is None else dist)

        if has_directed_path(directed, true_node, pred_node):
            directionality["descendant_pred_lower_sym"] += 1
        elif has_directed_path(directed, pred_node, true_node):
            directionality["ancestor_pred_higher_sym"] += 1
        else:
            directionality["branch_jump"] += 1

    counter = Counter(wrong_distances)
    finite = [value for value in wrong_distances if value != "inf"]
    return {
        "top1_pct": model_blob["accuracy_top1_pct"],
        "correct": int(directionality["correct"]),
        "wrong_n": int(len(wrong_distances)),
        "distance_buckets": {
            "1": int(counter.get(1, 0)),
            "2": int(counter.get(2, 0)),
            "3": int(counter.get(3, 0)),
            "4plus": int(sum(v for k, v in counter.items() if k not in (1, 2, 3, "inf"))),
            "inf": int(counter.get("inf", 0)),
        },
        "pct_wrong_le2": (100.0 * (counter.get(1, 0) + counter.get(2, 0)) / len(wrong_distances)) if wrong_distances else None,
        "mean_wrong_dist": (sum(finite) / len(finite)) if finite else None,
        "directionality": {key: int(value) for key, value in directionality.items()},
    }


def main():
    args = parse_args()
    _, directed, undirected, label_to_node = build_graph(args.graph_json)

    with open(args.failure_json, "r") as handle:
        failure_data = json.load(handle)

    summary = {
        "graph_json": args.graph_json,
        "failure_json": args.failure_json,
        "models": {},
    }
    for name, blob in failure_data["models"].items():
        summary["models"][name] = summarize_model(blob, directed, undirected, label_to_node)

    with open(args.output_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary["models"], indent=2))


if __name__ == "__main__":
    main()
