#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Export reviewer-friendly RRUFF example CSVs from a benchmark HDF5.")
    parser.add_argument("--benchmark-h5", required=True)
    parser.add_argument("--failure-json", default=None, help="Optional per-example inference summary with correct/wrong flags.")
    parser.add_argument("--model-key", default=None, help="Model key inside failure JSON. Defaults to the only entry if unambiguous.")
    parser.add_argument("--correct-index", type=int, default=None)
    parser.add_argument("--wrong-index", type=int, default=None)
    parser.add_argument("--metadata-csv", default=None, help="Optional authoritative metadata CSV keyed by case_id.")
    parser.add_argument("--metadata-case-id-col", default="case_id")
    parser.add_argument("--metadata-sg-col", default="space_group_number")
    parser.add_argument("--metadata-sg-symbol-col", default="space_group_symbol")
    parser.add_argument("--metadata-sg-source-col", default="space_group_source")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _load_metadata_lookup(args):
    if not args.metadata_csv:
        return {}
    df = pd.read_csv(args.metadata_csv)
    required = args.metadata_case_id_col
    if required not in df.columns:
        raise KeyError(f"{args.metadata_csv} does not contain required case-id column {required!r}")
    lookup = {}
    for _, row in df.iterrows():
        case_id = row.get(args.metadata_case_id_col)
        if pd.isna(case_id):
            continue
        key = str(case_id)
        entry = {}
        sg_col = args.metadata_sg_col
        sg_symbol_col = args.metadata_sg_symbol_col
        sg_source_col = args.metadata_sg_source_col
        if sg_col in df.columns and not pd.isna(row.get(sg_col)):
            entry["true_space_group"] = int(row[sg_col])
        if sg_symbol_col in df.columns and not pd.isna(row.get(sg_symbol_col)):
            entry["true_space_group_symbol"] = str(row[sg_symbol_col])
        if sg_source_col in df.columns and not pd.isna(row.get(sg_source_col)):
            entry["true_space_group_source"] = str(row[sg_source_col])
        lookup[key] = entry
    return lookup


def _choose_examples(failure_json_path: str | None, model_key: str | None, correct_index: int | None, wrong_index: int | None):
    if correct_index is not None and wrong_index is not None:
        return correct_index, wrong_index, {}
    if not failure_json_path:
        raise ValueError("Either explicit indices or --failure-json must be provided")
    data = json.loads(Path(failure_json_path).read_text())
    models = data["models"]
    if model_key is None:
        if len(models) != 1:
            raise ValueError("Failure JSON contains multiple models; pass --model-key")
        model_key = next(iter(models))
    examples = models[model_key]["examples"]
    correct = next(item for item in examples if item["correct"])
    wrong = next(item for item in examples if not item["correct"])
    return int(correct["index"]), int(wrong["index"]), {correct["index"]: correct, wrong["index"]: wrong}


def _write_example(out_dir: Path, tag: str, index: int, h5: h5py.File, failure_meta: dict, metadata_lookup: dict):
    tt = h5["two_theta"][index]
    intensity = h5["X_test"][index]
    df = pd.DataFrame({"2theta": tt, "intensity": intensity})

    case_id = _decode(h5["case_ids"][index]) if "case_ids" in h5 else f"example_{index}"
    mineral = _decode(h5["minerals"][index]) if "minerals" in h5 else None
    severity = _decode(h5["severity"][index]) if "severity" in h5 else None
    fit_bucket = _decode(h5["fit_bucket"][index]) if "fit_bucket" in h5 else None
    rwp = float(h5["rwp"][index]) if "rwp" in h5 else None
    true_eg = int(h5["y_test"][index]) if "y_test" in h5 else None
    authority = metadata_lookup.get(case_id, {})

    stem = f"{tag}_{index:03d}_{case_id}".replace("/", "_")
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "benchmark_index": int(index),
                "case_id": case_id,
                "mineral": mineral,
                "severity": severity,
                "fit_bucket": fit_bucket,
                "rwp": rwp,
                "true_eg": true_eg,
                **authority,
                "paper_model_summary": failure_meta or None,
                "csv_path": str(csv_path),
            },
            indent=2,
        )
    )
    return {"csv_path": str(csv_path), "json_path": str(json_path), "case_id": case_id, "true_eg": true_eg}


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    correct_idx, wrong_idx, failure_lookup = _choose_examples(
        args.failure_json,
        args.model_key,
        args.correct_index,
        args.wrong_index,
    )
    metadata_lookup = _load_metadata_lookup(args)

    with h5py.File(args.benchmark_h5, "r") as h5:
        correct = _write_example(out_dir, "correct_case", correct_idx, h5, failure_lookup.get(correct_idx, {}), metadata_lookup)
        wrong = _write_example(out_dir, "failure_case", wrong_idx, h5, failure_lookup.get(wrong_idx, {}), metadata_lookup)

    manifest = {
        "benchmark_h5_basename": Path(args.benchmark_h5).name,
        "failure_json": args.failure_json,
        "model_key": args.model_key,
        "metadata_csv": Path(args.metadata_csv).name if args.metadata_csv else None,
        "examples": {"correct_case": correct, "failure_case": wrong},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
