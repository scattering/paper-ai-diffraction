#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import pandas as pd
import torch

from paper_ai_diffraction.reviewer.notebook_support import build_model_bundle, infer_single_pattern


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute per-example benchmark inference summaries for reviewer use.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--benchmark-h5", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--prior-h5", default=None)
    parser.add_argument("--decoder", default="aux_bayes")
    parser.add_argument("--aux-temperature", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def main():
    args = parse_args()
    bundle = build_model_bundle(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        prior_h5_path=args.prior_h5,
    )

    rows = []
    with h5py.File(args.benchmark_h5, "r") as h5:
        total = h5["X_test"].shape[0]
        limit = min(args.limit, total) if args.limit is not None else total
        for idx in range(limit):
            intensity = h5["X_test"][idx]
            result = infer_single_pattern(
                bundle,
                intensity,
                aux_temperature=args.aux_temperature,
                decoder=args.decoder,
                prior_h5_path=args.prior_h5,
            )
            rows.append(
                {
                    "index": idx,
                    "case_id": _decode(h5["case_ids"][idx]) if "case_ids" in h5 else None,
                    "mineral": _decode(h5["minerals"][idx]) if "minerals" in h5 else None,
                    "true_eg": int(h5["y_test"][idx]) if "y_test" in h5 else None,
                    "rwp": float(h5["rwp"][idx]) if "rwp" in h5 else None,
                    "severity": _decode(h5["severity"][idx]) if "severity" in h5 else None,
                    "fit_bucket": _decode(h5["fit_bucket"][idx]) if "fit_bucket" in h5 else None,
                    "selected_decoder": result["selected_decoder"],
                    "top1_eg": result["top1_eg"],
                    "top1_symbol": result["top1_symbol"],
                    "top1_candidate_space_groups": result["top1_candidate_space_groups"],
                    "top1_probability": result["decoders"][result["selected_decoder"]]["top5_prob"][0],
                    "top5_eg": result["decoders"][result["selected_decoder"]]["top5_eg"],
                    "top5_prob": result["decoders"][result["selected_decoder"]]["top5_prob"],
                }
            )

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "benchmark_h5": args.benchmark_h5,
        "prior_h5": args.prior_h5,
        "decoder": args.decoder,
        "aux_temperature": args.aux_temperature,
        "n_examples": len(rows),
        "examples": rows,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output_json} with {len(rows)} examples")


if __name__ == "__main__":
    main()
