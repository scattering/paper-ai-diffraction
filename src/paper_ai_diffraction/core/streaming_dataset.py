import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from paper_ai_diffraction.utils.extinction_multilabel import (
    DEFAULT_FINAL_TABLE,
    build_extinction_templates,
    ext_group_to_multilabel_target,
)

try:
    import spglib
    from pycrysfml import cfml_py_utilities
    from pyxtal import pyxtal
    from pyxtal.symmetry import get_wyckoffs
except ImportError:  # pragma: no cover
    spglib = None
    cfml_py_utilities = None
    pyxtal = None
    get_wyckoffs = None


ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",
]


@dataclass
class StreamingConfig:
    samples_per_epoch: int
    batch_size: int
    num_workers: int
    twotheta_min: float = 10.0
    twotheta_max: float = 110.0
    step_size: float = 0.1
    wavelength: float = 1.54030
    u: float = 0.025
    v: float = -0.025
    w: float = 0.025
    max_structure_attempts: int = 50
    seed: int = 1234
    weighted_sampling: bool = False
    label_mode: str = "multilabel"
    final_table_path: str | Path = DEFAULT_FINAL_TABLE
    canonical_table_path: Optional[str | Path] = None
    sg_lookup_path: Optional[str | Path] = None

    @property
    def num_points(self) -> int:
        return int((self.twotheta_max - self.twotheta_min) / self.step_size) + 1


def _load_ext_group_to_sgs(final_table_path: str | Path) -> Dict[int, List[int]]:
    mapping: Dict[int, List[int]] = {}
    with Path(final_table_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ext_group = int(row["Extinction Group"])
            sg = int(row["Space Group"])
            mapping.setdefault(ext_group, []).append(sg)
    return mapping


def _sample_valid_composition(space_group: int, rng: random.Random):
    wyckoff_data = get_wyckoffs(space_group, organized=True)
    valid_counts = []
    for positions in wyckoff_data:
        total_atoms = sum(len(pos) for pos in positions)
        if total_atoms > 1:
            valid_counts.append(total_atoms)

    if not valid_counts:
        return ["C"], [2]

    num_atoms = rng.choice(valid_counts)
    num_elements = rng.randint(2, 3)
    species = rng.sample(ELEMENTS, num_elements)
    valid_ratios = [r for r in range(2, num_atoms + 1) if num_atoms % r == 0]
    ratios = [rng.choice(valid_ratios) for _ in range(num_elements)] if valid_ratios else [1] * num_elements
    return species, ratios


def _validate_structure(structure, expected_space_group: int) -> bool:
    atoms = structure.to_ase()
    lattice = atoms.get_cell()
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    detected = spglib.get_spacegroup((lattice, positions, numbers), symprec=1e-2)
    if detected is None:
        return False
    return f"({expected_space_group})" in detected


def _extract_structure_info(structure):
    atoms = structure.to_ase()
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    positions = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()
    atom_list = [{"symbol": sym, "position": pos} for sym, pos in zip(symbols, positions)]
    return structure.group.number, a, b, c, alpha, beta, gamma, atom_list


def _create_pattern_dict(space_group, a, b, c, alpha, beta, gamma, atom_list, cfg: StreamingConfig):
    sg_symbol = spglib.get_spacegroup_type(space_group)["international_short"]
    return {
        "phases": [
            {
                "placeholder": {
                    "_space_group_name_H-M_alt": sg_symbol,
                    "_cell_length_a": float(a),
                    "_cell_length_b": float(b),
                    "_cell_length_c": float(c),
                    "_cell_angle_alpha": float(alpha),
                    "_cell_angle_beta": float(beta),
                    "_cell_angle_gamma": float(gamma),
                    "_atom_site": [
                        {
                            "_label": atom["symbol"],
                            "_type_symbol": atom["symbol"],
                            "_fract_x": float(atom["position"][0]),
                            "_fract_y": float(atom["position"][1]),
                            "_fract_z": float(atom["position"][2]),
                            "_occupancy": 1.0,
                            "_adp_type": "iso",
                            "_B_iso_or_equiv": 0,
                        }
                        for atom in atom_list
                    ],
                }
            }
        ],
        "experiments": [
            {
                "NPD": {
                    "_diffrn_radiation_probe": "xray",
                    "_diffrn_radiation_wavelength": cfg.wavelength,
                    "_pd_instr_resolution_u": cfg.u,
                    "_pd_instr_resolution_v": cfg.v,
                    "_pd_instr_resolution_w": cfg.w,
                    "_pd_instr_resolution_x": 0.0015,
                    "_pd_instr_resolution_y": 0,
                    "_pd_instr_reflex_asymmetry_p1": 0,
                    "_pd_instr_reflex_asymmetry_p2": 0,
                    "_pd_instr_reflex_asymmetry_p3": 0,
                    "_pd_instr_reflex_asymmetry_p4": 0,
                    "_pd_meas_2theta_offset": 0,
                    "_pd_meas_2theta_range_min": cfg.twotheta_min,
                    "_pd_meas_2theta_range_max": cfg.twotheta_max,
                    "_pd_meas_2theta_range_inc": cfg.step_size,
                    "_phase": [{"_label": "placeholder", "_scale": 1}],
                    "_pd_background": [
                        {"_2theta": cfg.twotheta_min, "_intensity": 0},
                        {"_2theta": cfg.twotheta_max, "_intensity": 0},
                    ],
                }
            }
        ],
    }


class OnTheFlyPatternDataset(IterableDataset):
    def __init__(self, cfg: StreamingConfig):
        super().__init__()
        if any(mod is None for mod in (spglib, cfml_py_utilities, pyxtal, get_wyckoffs)):
            raise ImportError("Streaming generation requires pyxtal, pycrysfml, and spglib")

        self.cfg = cfg
        self.ext_group_to_sgs = _load_ext_group_to_sgs(cfg.final_table_path)
        self.ext_groups = sorted(self.ext_group_to_sgs.keys())
        self.templates = build_extinction_templates(
            canonical_table_path=cfg.canonical_table_path,
            final_table_path=cfg.final_table_path,
            sg_lookup_path=cfg.sg_lookup_path,
        )

    def _sample_ext_group(self, rng: random.Random) -> int:
        if not self.cfg.weighted_sampling:
            return rng.choice(self.ext_groups)
        weights = [len(self.ext_group_to_sgs[ext]) for ext in self.ext_groups]
        return rng.choices(self.ext_groups, weights=weights, k=1)[0]

    def _generate_structure(self, space_group: int, rng: random.Random):
        for _ in range(self.cfg.max_structure_attempts):
            try:
                structure = pyxtal()
                if space_group == 1:
                    species = rng.sample(ELEMENTS, rng.randint(1, 3))
                    counts = [1] * len(species)
                    structure.from_random(dim=3, group=1, species=species, numIons=counts)
                else:
                    species, counts = _sample_valid_composition(space_group, rng)
                    structure.from_random(dim=3, group=space_group, species=species, numIons=counts)
                if _validate_structure(structure, space_group):
                    return structure
            except Exception:
                continue
        raise RuntimeError(f"Failed to generate valid structure for SG {space_group}")

    def _generate_pattern(self, structure):
        space_group, a, b, c, alpha, beta, gamma, atom_list = _extract_structure_info(structure)
        pattern_dict = _create_pattern_dict(space_group, a, b, c, alpha, beta, gamma, atom_list, self.cfg)
        _, intensity = cfml_py_utilities.cw_powder_pattern_from_dict(pattern_dict)
        intensity = np.asarray(intensity, dtype=np.float32)
        intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
        if intensity.max() > 0:
            intensity = 1000.0 * intensity / intensity.max()
        return intensity

    def __iter__(self) -> Iterator:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rng = random.Random(self.cfg.seed + worker_id)

        local_count = 0
        while local_count < self.cfg.samples_per_epoch:
            global_idx = worker_id + local_count * num_workers
            if global_idx >= self.cfg.samples_per_epoch:
                break

            ext_group = self._sample_ext_group(rng)
            sg = rng.choice(self.ext_group_to_sgs[ext_group])
            try:
                structure = self._generate_structure(sg, rng)
                intensity = self._generate_pattern(structure)
            except Exception:
                continue

            x = torch.tensor(intensity, dtype=torch.float32)
            if self.cfg.label_mode == "multilabel":
                y = ext_group_to_multilabel_target(ext_group, self.templates)
                yield x, y, torch.tensor(ext_group - 1, dtype=torch.long)
            else:
                yield x, torch.tensor(ext_group - 1, dtype=torch.long)
            local_count += 1


def _streaming_collate(batch):
    xs = torch.stack([item[0] for item in batch], dim=0)
    ys = torch.stack([item[1] for item in batch], dim=0)
    ext = torch.stack([item[2] for item in batch], dim=0)
    return xs, ys, ext


def get_streaming_dataloader(cfg: StreamingConfig) -> DataLoader:
    dataset = OnTheFlyPatternDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_streaming_collate,
    )
