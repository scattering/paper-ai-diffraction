#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: Path):
    return json.loads(path.read_text())


def calibration_point(path: Path):
    blob = load(path)
    point = blob["calibrated_bayesian_aux"]["point_estimate"]
    return point["top1"] * 100.0, point["top5"] * 100.0


def split_valid(path: Path):
    blob = load(path)
    if "rates_pct" in blob and "valid_single_match" in blob["rates_pct"]:
        return float(blob["rates_pct"]["valid_single_match"])
    value = blob["valid_single_match"]
    total = blob["n_samples"]
    return value * 100.0 if value <= 1.0 else 100.0 * value / total


def main():
    rows = [
        (
            "mixed200k",
            RESULTS / "mixed200k_calibration_metrics_325_eeru8svx.json",
            RESULTS / "mixed200k_calibration_metrics_473_eeru8svx.json",
            RESULTS / "mixed200k_split_validity_325_eeru8svx.json",
            RESULTS / "mixed200k_split_validity_473_eeru8svx.json",
        ),
        (
            "mixed2500k",
            RESULTS / "mixed2500k_calibration_metrics_325_655279.json",
            RESULTS / "mixed2500k_calibration_metrics_473_655279.json",
            RESULTS / "mixed2500k_split_validity_325_655279.json",
            RESULTS / "mixed2500k_split_validity_473_655279.json",
        ),
    ]

    print("model,rruff325_top1,rruff325_top5,rruff473_top1,rruff473_top5,rruff325_split_valid,rruff473_split_valid")
    for name, c325, c473, s325, s473 in rows:
        top1_325, top5_325 = calibration_point(c325)
        top1_473, top5_473 = calibration_point(c473)
        valid_325 = split_valid(s325)
        valid_473 = split_valid(s473)
        print(f"{name},{top1_325:.2f},{top5_325:.2f},{top1_473:.2f},{top5_473:.2f},{valid_325:.2f},{valid_473:.2f}")


if __name__ == "__main__":
    main()
