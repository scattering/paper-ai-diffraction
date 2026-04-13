#!/usr/bin/env python3
import argparse
import json
import math
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Top-1/Top-5 accuracy vs temperature from a sweep JSON.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-svg", required=True)
    parser.add_argument("--series-prefix", default="aux_temp_")
    parser.add_argument("--title", default="Calibration Sweep on Real 325")
    return parser.parse_args()


def scale_x(x, xmin, xmax, left, width):
    if xmax == xmin:
        return left + width / 2
    return left + (x - xmin) / (xmax - xmin) * width


def scale_y(y, ymin, ymax, top, height):
    if ymax == ymin:
        return top + height / 2
    return top + height - (y - ymin) / (ymax - ymin) * height


def polyline(points, color):
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="2" '
        f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" />'
    )


def circles(points, color):
    return "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" />' for x, y in points
    )


def main():
    args = parse_args()
    with open(args.input_json, "r") as handle:
        payload = json.load(handle)

    rows = []
    for key, metrics in payload["results"].items():
        if not key.startswith(args.series_prefix):
            continue
        m = re.match(r"aux_temp_(.+)", key)
        if not m:
            continue
        temp = float(m.group(1))
        rows.append((temp, float(metrics["top1"]), float(metrics["top5"])))

    rows.sort(key=lambda x: x[0])
    temps = [r[0] for r in rows]
    top1 = [r[1] for r in rows]
    top5 = [r[2] for r in rows]

    width = 800
    height = 500
    left = 80
    right = 30
    top = 60
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xmin, xmax = min(temps), max(temps)
    ymin, ymax = 0.0, math.ceil(max(top5) / 5.0) * 5.0

    top1_pts = [(scale_x(x, xmin, xmax, left, plot_w), scale_y(y, ymin, ymax, top, plot_h)) for x, y in zip(temps, top1)]
    top5_pts = [(scale_x(x, xmin, xmax, left, plot_w), scale_y(y, ymin, ymax, top, plot_h)) for x, y in zip(temps, top5)]

    y_ticks = list(range(0, int(ymax) + 1, 5))
    x_ticks = temps

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="20" font-family="Helvetica,Arial,sans-serif">{args.title}</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222" />',
    ]

    for y in y_ticks:
        py = scale_y(y, ymin, ymax, top, plot_h)
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{left+plot_w}" y2="{py:.1f}" stroke="#ddd" />')
        parts.append(f'<text x="{left-10}" y="{py+4:.1f}" text-anchor="end" font-size="12" font-family="Helvetica,Arial,sans-serif">{y}</text>')

    for x in x_ticks:
        px = scale_x(x, xmin, xmax, left, plot_w)
        label = f"{x:g}"
        parts.append(f'<line x1="{px:.1f}" y1="{top+plot_h}" x2="{px:.1f}" y2="{top+plot_h+6}" stroke="#222" />')
        parts.append(f'<text x="{px:.1f}" y="{top+plot_h+22}" text-anchor="middle" font-size="12" font-family="Helvetica,Arial,sans-serif">{label}</text>')

    parts.extend(
        [
            polyline(top1_pts, "#1f77b4"),
            circles(top1_pts, "#1f77b4"),
            polyline(top5_pts, "#d62728"),
            circles(top5_pts, "#d62728"),
            f'<text x="{width/2:.1f}" y="{height-20}" text-anchor="middle" font-size="14" font-family="Helvetica,Arial,sans-serif">Temperature</text>',
            f'<text x="20" y="{height/2:.1f}" transform="rotate(-90 20 {height/2:.1f})" text-anchor="middle" font-size="14" font-family="Helvetica,Arial,sans-serif">Accuracy (%)</text>',
            f'<line x1="{width-180}" y1="70" x2="{width-150}" y2="70" stroke="#1f77b4" stroke-width="2" />',
            f'<circle cx="{width-165}" cy="70" r="4" fill="#1f77b4" />',
            f'<text x="{width-140}" y="75" font-size="12" font-family="Helvetica,Arial,sans-serif">Top-1</text>',
            f'<line x1="{width-180}" y1="95" x2="{width-150}" y2="95" stroke="#d62728" stroke-width="2" />',
            f'<circle cx="{width-165}" cy="95" r="4" fill="#d62728" />',
            f'<text x="{width-140}" y="100" font-size="12" font-family="Helvetica,Arial,sans-serif">Top-5</text>',
        ]
    )
    parts.append("</svg>")

    with open(args.output_svg, "w") as handle:
        handle.write("\n".join(parts))


if __name__ == "__main__":
    main()
