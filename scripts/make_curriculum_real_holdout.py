#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outdir = root / "results" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    output_png = outdir / "curriculum_real_holdout.png"

    stages = ["Stage-1", "Stage-2a", "Stage-2b"]
    split = np.array([7.04, 11.21, 11.46], dtype=float)
    aux = np.array([3.19, 11.13, 10.23], dtype=float)

    x = np.arange(len(stages), dtype=float)
    width = 0.34

    plt.figure(figsize=(8, 4.8))
    ax = plt.gca()
    ax.bar(x - width / 2, split, width, label="Split-head decode", color="#4C78A8")
    ax.bar(x + width / 2, aux, width, label="Bayesian aux", color="#F58518")

    for xpos, value in zip(x - width / 2, split):
        ax.text(xpos, value + 0.18, f"{value:.1f}", ha="center", va="bottom", fontsize=10)
    for xpos, value in zip(x + width / 2, aux):
        ax.text(xpos, value + 0.18, f"{value:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x, stages)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0, 13)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
