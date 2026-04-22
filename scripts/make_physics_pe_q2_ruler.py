#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render the physics PE supplementary figure from a bundled curve JSON.")
    parser.add_argument(
        "--input-json",
        default=str(root / "assets" / "figure_data" / "physics_pe_curve_82ept35h.json"),
    )
    parser.add_argument(
        "--output-svg",
        default=str(root / "results" / "figures" / "physics_pe_q2_ruler.svg"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input_json).read_text())

    two_theta = np.asarray(payload["two_theta"], dtype=float)
    q2_like = np.asarray(payload["sin2theta_norm"], dtype=float)
    pc1 = np.asarray(payload["pc1_norm"], dtype=float)
    emb_norm = np.asarray(payload["emb_norm_norm"], dtype=float)
    pc1_var = 100.0 * float(payload["pc1_explained_var"])
    corr_q2 = float(payload["corr_pc1_sin2theta"])

    output_svg = Path(args.output_svg)
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)

    axes[0].plot(two_theta, pc1, color="#1f77b4", linewidth=2.0, label="PC1")
    axes[0].plot(two_theta, q2_like, color="#d62728", linewidth=1.8, linestyle="--", label=r"$Q^2$-like coordinate")
    axes[0].set_ylabel("Normalized value")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(color="#e6e6e6", linewidth=0.8)

    axes[1].plot(two_theta, emb_norm, color="#2ca02c", linewidth=2.0)
    axes[1].set_xlabel(r"$2\\theta$ (degrees)")
    axes[1].set_ylabel("Embedding norm\n(normalized)")
    axes[1].grid(color="#e6e6e6", linewidth=0.8)

    fig.suptitle("Learned physics PE behaves as a reciprocal-space ruler", y=0.98)
    fig.text(
        0.5,
        0.02,
        f"PC1 explains {pc1_var:.2f}% of embedding variance; corr(PC1, $Q^2$-like coordinate) = {corr_q2:.5f}",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output_svg)
    plt.close(fig)
    print(f"Wrote {output_svg}")


if __name__ == "__main__":
    main()
