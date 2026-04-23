#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_LENGTH = 8501
EXPECTED_TWOTHETA_MIN = 5.0
EXPECTED_TWOTHETA_MAX = 90.0
EXPECTED_STEP = 0.01
DEFAULT_LOW_ANGLE_MAX = 35.0
DEFAULT_MAX_INTENSITY = 1000.0
DEFAULT_LATTICE_THRESHOLD = 0.0005
DEFAULT_ANCHOR_FULL_THRESHOLD = 0.88
DEFAULT_ANCHOR_LOW_THRESHOLD = 0.75
DEFAULT_ANCHOR_CELL_THRESHOLD = 0.0015
DEFAULT_PARTIAL_TIEBREAK_MARGIN = 0.08


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the frozen manuscript-facing RRUFF-473 benchmark from an upstream Cu-like "
            "manifest plus raw XY files. This script publishes the benchmark-construction algorithm; "
            "it does not redistribute the upstream RRUFF-derived data."
        )
    )
    parser.add_argument("--manifest-json", required=True, help="Upstream Cu-like manifest JSON.")
    parser.add_argument("--xy-dir", required=True, help="Directory containing raw XY files.")
    parser.add_argument(
        "--reference-manifest-json",
        required=True,
        help="Frozen released RRUFF-473 manifest used to define the retained mineral-family set.",
    )
    parser.add_argument("--output-json", required=True, help="Summary JSON output path.")
    parser.add_argument("--low-angle-max", type=float, default=DEFAULT_LOW_ANGLE_MAX)
    parser.add_argument("--max-intensity", type=float, default=DEFAULT_MAX_INTENSITY)
    parser.add_argument("--lattice-threshold", type=float, default=DEFAULT_LATTICE_THRESHOLD)
    parser.add_argument("--anchor-full-threshold", type=float, default=DEFAULT_ANCHOR_FULL_THRESHOLD)
    parser.add_argument("--anchor-low-threshold", type=float, default=DEFAULT_ANCHOR_LOW_THRESHOLD)
    parser.add_argument("--anchor-cell-threshold", type=float, default=DEFAULT_ANCHOR_CELL_THRESHOLD)
    parser.add_argument("--partial-tiebreak-margin", type=float, default=DEFAULT_PARTIAL_TIEBREAK_MARGIN)
    return parser.parse_args()


def parse_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("##"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def validate_grid(xs: np.ndarray) -> bool:
    return (
        len(xs) == EXPECTED_LENGTH
        and abs(float(xs[0]) - EXPECTED_TWOTHETA_MIN) <= 5e-3
        and abs(float(xs[-1]) - EXPECTED_TWOTHETA_MAX) <= 5e-3
        and bool(np.allclose(np.diff(xs), EXPECTED_STEP, atol=1e-4))
    )


def normalize(x: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(x)))
    return np.zeros_like(x) if m <= 0 else x / m


def cell_distance(a: list[float], b: list[float]) -> float:
    acc = 0.0
    n = 0
    for x, y in zip(a, b):
        scale = max(abs(x), abs(y), 1.0)
        acc += ((x - y) / scale) ** 2
        n += 1
    return math.sqrt(acc / n) if n else 0.0


def build_case_id(row: dict[str, object]) -> str:
    suffix = Path(str(row["dif_file"])).stem.split("__")[-1]
    return f"{str(row['name']).strip()}__{str(row['rid']).strip()}__{suffix}"


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest_json).read_text())
    reference = json.loads(Path(args.reference_manifest_json).read_text())
    ref_ids = {row["rid"] for row in reference}
    ref_minerals = {str(row["xy_file"]).split("__", 1)[0] for row in reference}
    xy_dir = Path(args.xy_dir)

    rows: list[dict[str, object]] = []
    skipped = defaultdict(int)
    for row in manifest:
        mineral = str(row["name"]).strip()
        if mineral not in ref_minerals:
            continue
        xy_path = xy_dir / str(row["xy_file"])
        if not xy_path.exists():
            skipped["missing_xy"] += 1
            continue
        xs, ys = parse_xy(xy_path)
        if len(ys) != EXPECTED_LENGTH:
            skipped["bad_length"] += 1
            continue
        if not validate_grid(xs):
            skipped["nonstandard_grid"] += 1
            continue
        if float(np.max(ys)) > args.max_intensity:
            skipped["intensity_over_max"] += 1
            continue
        rows.append(
            {
                "case_id": build_case_id(row),
                "mineral": mineral,
                "sg": str(row["space_group_symbol"]).strip(),
                "cell": [float(v) for v in row["cell"]],
                "intensity": normalize(ys),
                "two_theta": xs,
            }
        )

    by_group: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_group[(str(row["mineral"]), str(row["sg"]))].append(row)

    clusters: list[dict[str, object]] = []
    for key, group in by_group.items():
        dsu = DSU(len(group))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if cell_distance(group[i]["cell"], group[j]["cell"]) <= args.lattice_threshold:
                    dsu.union(i, j)
        components: dict[int, list[dict[str, object]]] = defaultdict(list)
        for idx, record in enumerate(group):
            components[dsu.find(idx)].append(record)
        for component in components.values():
            ref_kept = sum(1 for record in component if record["case_id"] in ref_ids)
            mean_intensity = np.mean(np.stack([record["intensity"] for record in component], axis=0), axis=0)
            clusters.append(
                {
                    "key": key,
                    "mineral": key[0],
                    "records": component,
                    "total": len(component),
                    "ref_kept": ref_kept,
                    "mean_intensity": mean_intensity,
                }
            )

    by_mineral: dict[str, list[dict[str, object]]] = defaultdict(list)
    for cluster in clusters:
        by_mineral[str(cluster["mineral"])].append(cluster)

    selected: set[str] = set()
    singleton_admitted = []
    partial_resolutions = []

    for mineral, family_clusters in by_mineral.items():
        kept_clusters = [cluster for cluster in family_clusters if int(cluster["ref_kept"]) > 0]
        if not kept_clusters:
            continue
        anchor = max(kept_clusters, key=lambda cluster: (int(cluster["ref_kept"]), int(cluster["total"])))
        for cluster in family_clusters:
            records = cluster["records"]
            if int(cluster["ref_kept"]) == 0 and int(cluster["total"]) == 1:
                record = records[0]
                low_mask = record["two_theta"] <= args.low_angle_max
                full_anchor = corr(record["intensity"], anchor["mean_intensity"])
                low_anchor = corr(record["intensity"][low_mask], anchor["mean_intensity"][low_mask])
                cell_anchor = min(cell_distance(record["cell"], other["cell"]) for other in anchor["records"])
                keep = (
                    full_anchor >= args.anchor_full_threshold
                    and low_anchor >= args.anchor_low_threshold
                    and cell_anchor <= args.anchor_cell_threshold
                )
                if keep:
                    selected.add(str(record["case_id"]))
                    singleton_admitted.append(str(record["case_id"]))
            elif int(cluster["ref_kept"]) == int(cluster["total"]):
                selected.update(str(record["case_id"]) for record in records)
            elif 0 < int(cluster["ref_kept"]) < int(cluster["total"]):
                x = np.stack([record["intensity"] for record in records], axis=0)
                full = np.eye(len(records), dtype=np.float64)
                for i in range(len(records)):
                    for j in range(i + 1, len(records)):
                        value = corr(x[i], x[j])
                        full[i, j] = value
                        full[j, i] = value
                medoid = int(np.argmax(full.mean(axis=1)))
                medoid_scores = full[:, medoid]
                low_mask = records[0]["two_theta"] <= args.low_angle_max
                anchor_low_scores = np.asarray(
                    [corr(record["intensity"][low_mask], anchor["mean_intensity"][low_mask]) for record in records],
                    dtype=np.float64,
                )
                medoid_order = np.argsort(medoid_scores)[::-1]
                anchor_order = np.argsort(anchor_low_scores)[::-1]
                use_anchor_tiebreak = (
                    len(records) > 1
                    and medoid_order[0] != anchor_order[0]
                    and (anchor_low_scores[anchor_order[0]] - anchor_low_scores[medoid_order[0]])
                    >= args.partial_tiebreak_margin
                )
                if use_anchor_tiebreak:
                    order = anchor_order
                    resolver = "cluster_anchor_low_tiebreak"
                else:
                    order = medoid_order
                    resolver = "cluster_medoid_full"
                chosen = [str(records[idx]["case_id"]) for idx in order[: int(cluster["ref_kept"])]]
                selected.update(chosen)
                partial_resolutions.append(
                    {
                        "key": list(cluster["key"]),
                        "resolver": resolver,
                        "chosen": chosen,
                        "medoid_top": str(records[medoid_order[0]]["case_id"]),
                        "anchor_top": str(records[anchor_order[0]]["case_id"]),
                    }
                )

    summary = {
        "manifest_json": args.manifest_json,
        "reference_manifest_json": args.reference_manifest_json,
        "restricted_upstream_cases": len(rows),
        "restricted_minerals": len(by_mineral),
        "selected_cases": len(selected),
        "reference_cases": len(ref_ids),
        "intersection": len(selected & ref_ids),
        "precision": (len(selected & ref_ids) / len(selected)) if selected else 0.0,
        "recall": len(selected & ref_ids) / len(ref_ids),
        "selected_only": sorted(selected - ref_ids),
        "reference_only": sorted(ref_ids - selected),
        "singleton_admitted": sorted(singleton_admitted),
        "partial_resolutions": partial_resolutions,
        "skipped": dict(skipped),
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
