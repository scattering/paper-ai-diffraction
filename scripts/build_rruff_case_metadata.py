#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import gemmi
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small case_id -> space-group metadata CSV from matched RRUFF DIF files."
    )
    parser.add_argument("--case-id", action="append", required=True, help="Case ID such as Mineral__R050071-1__6130")
    parser.add_argument("--dif-root", required=True, help="Directory containing RRUFF Powder DIF files")
    parser.add_argument("--final-table-path", required=True, help="SG -> EG table used by the paper")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def infer_dif_path(case_id: str, dif_root: Path) -> Path:
    parts = case_id.split("__")
    if len(parts) < 3:
        raise ValueError(f"Unexpected case_id format: {case_id}")
    dif_name = f"{parts[0]}__{parts[1]}__Powder__DIF_File__{parts[2]}.txt"
    dif_path = dif_root / dif_name
    if not dif_path.exists():
        raise FileNotFoundError(f"Could not find DIF file for {case_id}: {dif_path}")
    return dif_path


def parse_dif_space_group(path: Path) -> str:
    text = path.read_text(errors="ignore")
    match = re.search(r"SPACE GROUP\s*[:=]\s*([^\n\r]+)", text)
    if not match:
        raise ValueError(f"Could not parse SPACE GROUP from {path}")
    return match.group(1).strip()


def normalize_hm_symbol(symbol: str) -> list[str]:
    raw = symbol.strip().replace("\\par", "").strip()
    candidates = [raw]

    s = raw.replace("_", "")
    s = re.sub(r"^([A-ZR])(?=[0-9-])", r"\1 ", s)
    candidates.append(s)

    s2 = raw.replace("_", "")
    s2 = re.sub(r"^([A-ZR])(?=[0-9-])", r"\1 ", s2)
    s2 = s2.replace("/", " /")
    candidates.append(s2)

    out = []
    for cand in candidates:
        cand = " ".join(cand.split())
        if cand and cand not in out:
            out.append(cand)
    return out


def space_group_symbol_to_number(symbol: str) -> int:
    for candidate in normalize_hm_symbol(symbol):
        sg = gemmi.find_spacegroup_by_name(candidate)
        if sg is not None:
            return int(sg.number)
    raise ValueError(f"Could not map space-group symbol to number: {symbol!r}")


def load_sg_to_eg_map(final_table_path: Path) -> dict[int, int]:
    with final_table_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = {int(row["Space Group"]): int(row["Extinction Group"]) for row in reader}
    if len(mapping) != 230:
        raise ValueError(f"Expected 230 SG -> EG mappings, found {len(mapping)}")
    return mapping


def main() -> None:
    args = parse_args()
    dif_root = Path(args.dif_root)
    final_table_path = Path(args.final_table_path)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    sg_to_eg = load_sg_to_eg_map(final_table_path)
    rows = []
    for case_id in args.case_id:
        dif_path = infer_dif_path(case_id, dif_root)
        sg_symbol = parse_dif_space_group(dif_path)
        sg_number = space_group_symbol_to_number(sg_symbol)
        rows.append(
            {
                "case_id": case_id,
                "dif_filename": dif_path.name,
                "space_group_symbol": sg_symbol,
                "space_group_number": sg_number,
                "mapped_extinction_group": sg_to_eg.get(sg_number),
                "space_group_source": "rruff_fit_dif",
            }
        )

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(output_csv)


if __name__ == "__main__":
    main()
