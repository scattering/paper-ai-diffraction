#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Write a temporary config JSON with simple overrides.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--set", action="append", default=[], help="Override in key=value form.")
    return parser.parse_args()


def parse_value(raw):
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def main():
    args = parse_args()
    with open(args.base_config, "r") as fh:
        config = json.load(fh)

    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"Invalid --set override: {item}")
        key, value = item.split("=", 1)
        config[key] = parse_value(value)

    output = Path(args.output_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        json.dump(config, fh, indent=2)


if __name__ == "__main__":
    main()
