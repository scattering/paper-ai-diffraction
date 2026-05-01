#!/usr/bin/env python3
"""
Prior distribution baseline: computes label distribution and top-1/3/5 accuracy
obtained by always predicting the k most frequent labels.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

DEFAULT_DATASETS = {
    "325": r"",
    "473": r"",
}


def load_labels(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f["y_test"][:]


def compute_distribution(labels: np.ndarray) -> list[tuple[int, int, float]]:
    counts = Counter(labels.tolist())
    total = len(labels)
    return sorted(
        [(label, count, count / total) for label, count in counts.items()],
        key=lambda x: x[1],
        reverse=True,
    )


def topk_accuracy(labels: np.ndarray, dist: list[tuple[int, int, float]], k: int) -> float:
    top_k_set = {label for label, _, _ in dist[:k]}
    return sum(1 for label in labels if label in top_k_set) / len(labels)


def evaluate_dataset(h5_path: str, name: str) -> dict:
    labels = load_labels(h5_path)
    dist = compute_distribution(labels)
    n = len(labels)
    n_classes = len(dist)

    results = {
        "name": name,
        "h5_path": str(h5_path),
        "n_samples": n,
        "n_classes": n_classes,
        "distribution": [{"label": label, "count": count, "fraction": round(frac, 6)} for label, count, frac in dist],
        "prior_baseline": {
            f"top{k}": {
                "accuracy": round(topk_accuracy(labels, dist, k), 6),
                "predicted_labels": [label for label, _, _ in dist[:k]],
            }
            for k in (1, 3, 5)
        },
    }

    print(f"\n{'=' * 62}")
    print(f"Dataset : {name}  ({Path(h5_path).name})")
    print(f"  Samples : {n}")
    print(f"  Classes : {n_classes} unique labels")
    print(f"\n  Label distribution (top 15 of {n_classes}):")
    print(f"  {'Label':>8}  {'Count':>8}  {'Fraction':>10}")
    for label, count, frac in dist[:15]:
        print(f"  {label:>8}  {count:>8}  {frac:>10.4f}")
    if n_classes > 15:
        print(f"  ... ({n_classes - 15} more classes)")
    print(f"\n  Prior baseline (always predict top-k most frequent labels):")
    for k in (1, 3, 5):
        acc = results["prior_baseline"][f"top{k}"]["accuracy"]
        top_labels = results["prior_baseline"][f"top{k}"]["predicted_labels"]
        print(f"  Top-{k}: {acc:.4f}  (labels: {top_labels})")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prior distribution baseline for extinction group prediction datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS.keys()),
        help="Dataset names to evaluate (default: all). Must match keys in DEFAULT_DATASETS or be overridden via --paths.",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        metavar="NAME=PATH",
        help="Override or add dataset paths as NAME=PATH pairs, e.g. --paths 325=/data/325.hdf5",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to save results as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets = dict(DEFAULT_DATASETS)
    if args.paths:
        for entry in args.paths:
            name, _, path = entry.partition("=")
            datasets[name.strip()] = path.strip()

    names = args.datasets if args.datasets else list(datasets.keys())
    all_results = []
    for name in names:
        if name not in datasets:
            print(f"[WARNING] Unknown dataset '{name}', skipping.")
            continue
        all_results.append(evaluate_dataset(datasets[name], name))

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
