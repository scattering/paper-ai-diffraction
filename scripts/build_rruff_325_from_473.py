#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically rebuild the manuscript-facing RRUFF-325 subset from the frozen "
            "RRUFF-473 benchmark by recomputing nuisance-fit severity from fixed Rwp thresholds."
        )
    )
    parser.add_argument("--input-h5", required=True, help="Frozen RRUFF-473 HDF5.")
    parser.add_argument("--output-h5", required=True, help="Output path for the reconstructed RRUFF-325 HDF5.")
    parser.add_argument(
        "--output-json",
        help="Optional summary JSON path. Defaults to <output-h5 stem>_summary.json.",
    )
    parser.add_argument("--usable-max-rwp", type=float, default=0.12)
    parser.add_argument("--recoverable-max-rwp", type=float, default=0.22)
    parser.add_argument("--poor-max-rwp", type=float, default=0.50)
    return parser.parse_args()


def classify_severity(rwp: float, usable_max: float, recoverable_max: float, poor_max: float) -> str:
    if rwp > poor_max:
        return "catastrophic"
    if rwp > recoverable_max:
        return "poor"
    if rwp > usable_max:
        return "recoverable"
    return "usable_or_better"


def decode_strings(values: object) -> list[str]:
    out = []
    for value in values.tolist():
        out.append(value.decode() if isinstance(value, bytes) else str(value))
    return out


def main() -> None:
    args = parse_args()

    import h5py
    import numpy as np

    retain = {"usable_or_better", "recoverable"}

    with h5py.File(args.input_h5, "r") as src:
        datasets = {key: src[key][:] for key in src.keys() if key != "two_theta"}
        two_theta = src["two_theta"][:]

    rwp = np.asarray(datasets["rwp"], dtype=np.float64)
    severity = np.asarray(
        [
            classify_severity(
                float(value),
                usable_max=args.usable_max_rwp,
                recoverable_max=args.recoverable_max_rwp,
                poor_max=args.poor_max_rwp,
            )
            for value in rwp
        ],
        dtype=object,
    )
    mask = np.asarray([value in retain for value in severity], dtype=bool)

    output_h5 = Path(args.output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5, "w") as dst:
        for key, value in datasets.items():
            if key == "severity":
                dst.create_dataset(
                    key,
                    data=np.asarray([s.encode("utf-8") for s in severity[mask].tolist()]),
                )
            else:
                dst.create_dataset(key, data=value[mask])
        dst.create_dataset("two_theta", data=two_theta)

    output_json = Path(args.output_json) if args.output_json else output_h5.with_name(f"{output_h5.stem}_summary.json")
    minerals = decode_strings(datasets["minerals"][mask])
    summary = {
        "input_h5": args.input_h5,
        "output_h5": str(output_h5),
        "retain_severity": ["recoverable", "usable_or_better"],
        "thresholds": {
            "usable_max_rwp": args.usable_max_rwp,
            "recoverable_max_rwp": args.recoverable_max_rwp,
            "poor_max_rwp": args.poor_max_rwp,
        },
        "n_patterns": int(mask.sum()),
        "unique_minerals": int(len(set(minerals))),
        "unique_extinction_groups": int(len(set(np.asarray(datasets["y_test"][mask], dtype=np.int64).tolist()))),
        "counts_by_severity": {
            name: int(np.sum(severity[mask] == name))
            for name in ["usable_or_better", "recoverable", "poor", "catastrophic"]
        },
        "source_counts_by_recomputed_severity": {
            name: int(np.sum(severity == name))
            for name in ["usable_or_better", "recoverable", "poor", "catastrophic"]
        },
    }
    output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
