#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "revised_paper_summary.json"


def fmt(value):
    return "" if value is None else f"{float(value):.2f}"


def main():
    summary = json.loads(SUMMARY.read_text())
    benchmarks = summary["benchmarks"]

    print("model,rruff325_top1,rruff325_top3,rruff325_top5,rruff473_top1,rruff473_top3,rruff473_top5")
    for name, payload in benchmarks.items():
        r325 = payload.get("rruff325", {})
        r473 = payload.get("rruff473", {})
        print(
            ",".join(
                [
                    name,
                    fmt(r325.get("top1")),
                    fmt(r325.get("top3")),
                    fmt(r325.get("top5")),
                    fmt(r473.get("top1")),
                    fmt(r473.get("top3")),
                    fmt(r473.get("top5")),
                ]
            )
        )

    control = summary["matched_sg_to_eg_control"]["full_scale"]
    print()
    print("sg_eg_control,top1,top3,top5")
    for name, payload in control.items():
        print(",".join([name, fmt(payload.get("top1")), fmt(payload.get("top3")), fmt(payload.get("top5"))]))

    margins = summary["matched_sg_to_eg_control"]["scale_margins_direct_eg_minus_collapsed_sg"]
    print()
    print("sg_eg_scale_margin,delta_top1,delta_top3,delta_top5")
    for name, payload in margins.items():
        print(",".join([name, fmt(payload.get("top1")), fmt(payload.get("top3")), fmt(payload.get("top5"))]))


if __name__ == "__main__":
    main()
