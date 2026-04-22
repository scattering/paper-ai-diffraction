#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outdir = root / "results" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    output_png = outdir / "stage_decoder_tradeoffs_rruff473.png"

    labels = [
        "Split masked",
        "Bayesian aux",
        "Fused a=0.25\n(Stage-2a)",
        "Fused a=0.50\n(Stage-2b)",
    ]
    top1 = np.array([14.80, 14.59, 15.64, 16.70], dtype=float)
    top5 = np.array([34.04, 52.22, 45.24, 41.01], dtype=float)

    x = np.arange(len(labels), dtype=float)
    width = 0.38

    plt.figure(figsize=(10, 5.5))
    ax = plt.gca()
    ax.bar(x - width / 2, top1, width, label="Top-1", color="#2C7FB8")
    ax.bar(x + width / 2, top5, width, label="Top-5", color="#FF7F0E")

    ax.set_title("RRUFF-473 Benchmark: Decoder Tradeoffs")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 55)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
