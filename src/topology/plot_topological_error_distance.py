#!/usr/bin/env python3
import argparse
import json


BAR_COLORS = {
    "1": "#4C78A8",
    "2": "#72B7B2",
    "3": "#F2CF5B",
    "4plus": "#E45756",
    "inf": "#B279A2",
}

DIRECTION_COLORS = {
    "descendant_pred_lower_sym": "#4C78A8",
    "ancestor_pred_higher_sym": "#E45756",
    "branch_jump": "#9D755D",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot stacked topological error-distance bars from summary JSON.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-svg", required=True)
    return parser.parse_args()


def svg_rect(x, y, w, h, fill):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" />'


def svg_text(x, y, text, size=14, anchor="middle", weight="normal"):
    safe = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" font-family="Arial, sans-serif">{safe}</text>'


def main():
    args = parse_args()
    with open(args.summary_json, "r") as handle:
        summary = json.load(handle)["models"]

    names = list(summary.keys())
    buckets = ["1", "2", "3", "4plus", "inf"]
    width = 1180
    height = 620
    top = 70
    bottom = 135
    left1 = 85
    right1 = 40
    panel_gap = 70
    panel_w = 470
    panel_h = height - top - bottom
    left2 = left1 + panel_w + panel_gap
    right2 = 55
    bar_gap = 22
    bar_w = (panel_w - bar_gap * (len(names) - 1)) / len(names)

    display_names = {
        "ic6gfmvm_split_bayes": "Stage-1",
        "wwuvp1kj_aux_bayes_t1": "Legacy",
        "7pv3pv3y_aux_bayes_t1": "350k baseline",
        "9rwv1qly_aux_bayes_t5": "Stage-2c",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_text(left1 + panel_w / 2, 34, "A. Error distance", size=18, weight="bold"),
        svg_text(left2 + panel_w / 2, 34, "B. Error direction", size=18, weight="bold"),
    ]

    for frac in range(0, 101, 20):
        y = top + panel_h - panel_h * (frac / 100.0)
        parts.append(f'<line x1="{left1}" y1="{y:.1f}" x2="{left1 + panel_w}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1" />')
        parts.append(svg_text(left1 - 12, y + 4, f"{frac}%", size=12, anchor="end"))

    for idx, name in enumerate(names):
        model = summary[name]
        wrong_n = max(int(model["wrong_n"]), 1)
        x = left1 + idx * (bar_w + bar_gap)
        cumulative = 0.0
        for bucket in buckets:
            frac = model["distance_buckets"][bucket] / wrong_n
            h = frac * panel_h
            y = top + panel_h - cumulative - h
            parts.append(svg_rect(x, y, bar_w, h, BAR_COLORS[bucket]))
            cumulative += h

        label = display_names.get(name, name.replace("_", " "))
        parts.append(svg_text(x + bar_w / 2, top + panel_h + 22, label, size=13))
        parts.append(svg_text(x + bar_w / 2, top + panel_h + 40, f"{model['top1_pct']:.1f}% top-1", size=11))

    # Panel B: directionality
    dir_keys = ["descendant_pred_lower_sym", "ancestor_pred_higher_sym", "branch_jump"]
    row_gap = 28
    row_h = (panel_h - row_gap * (len(names) - 1)) / len(names)
    for idx, name in enumerate(names):
        model = summary[name]
        wrong_n = max(int(model["wrong_n"]), 1)
        y = top + idx * (row_h + row_gap)
        parts.append(svg_text(left2 - 10, y + row_h / 2 + 4, display_names.get(name, name.replace("_", " ")), size=13, anchor="end"))
        x = left2
        for key in dir_keys:
            frac = model["directionality"].get(key, 0) / wrong_n
            w = frac * panel_w
            if w > 0:
                parts.append(svg_rect(x, y, w, row_h, DIRECTION_COLORS[key]))
            x += w
        parts.append(svg_text(left2 + panel_w + 8, y + row_h / 2 + 4, f"{model['top1_pct']:.1f}%", size=11, anchor="start"))

    legend_x = left1
    legend_y = height - 42
    step = 100
    bucket_labels = {"1": "1 hop", "2": "2 hops", "3": "3 hops", "4plus": "4+", "inf": "disc."}
    for idx, bucket in enumerate(buckets):
        x = legend_x + idx * step
        parts.append(svg_rect(x, legend_y - 12, 18, 18, BAR_COLORS[bucket]))
        parts.append(svg_text(x + 26, legend_y + 2, bucket_labels[bucket], size=12, anchor="start"))

    dir_legend_x = left2
    dir_legend_y = height - 42
    dir_labels = {
        "descendant_pred_lower_sym": "descendant",
        "ancestor_pred_higher_sym": "ancestor",
        "branch_jump": "branch jump",
    }
    dir_step = 150
    for idx, key in enumerate(dir_keys):
        x = dir_legend_x + idx * dir_step
        parts.append(svg_rect(x, dir_legend_y - 12, 18, 18, DIRECTION_COLORS[key]))
        parts.append(svg_text(x + 26, dir_legend_y + 2, dir_labels[key], size=12, anchor="start"))

    parts.append("</svg>")
    with open(args.output_svg, "w") as handle:
        handle.write("\n".join(parts))


if __name__ == "__main__":
    main()
