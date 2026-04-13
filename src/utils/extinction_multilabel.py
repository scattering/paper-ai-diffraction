import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


FEATURES = [
    "Sys_Triclinic",
    "Sys_Monoclinic",
    "Sys_Orthorhombic",
    "Sys_Tetragonal",
    "Sys_Trigonal",
    "Sys_Hexagonal",
    "Sys_Cubic",
    "Lattice_P",
    "Lattice_I",
    "Lattice_F",
    "Lattice_C",
    "Lattice_R",
    "Pos1_21",
    "Pos1_31",
    "Pos1_41",
    "Pos1_42",
    "Pos1_61",
    "Pos1_62",
    "Pos1_63",
    "Pos1_a",
    "Pos1_b",
    "Pos1_c",
    "Pos1_d",
    "Pos1_n",
    "Pos2_21",
    "Pos2_a",
    "Pos2_b",
    "Pos2_c",
    "Pos2_d",
    "Pos2_n",
    "Pos3_21",
    "Pos3_a",
    "Pos3_b",
    "Pos3_c",
    "Pos3_d",
    "Pos3_n",
    "Pos3_ab",
]

FEATURE_TO_INDEX = {name: idx for idx, name in enumerate(FEATURES)}
SYSTEM_SLICE = slice(0, 7)
LATTICE_SLICE = slice(7, 12)
OPERATOR_SLICE = slice(12, len(FEATURES))

DEFAULT_CANONICAL_TABLE = Path(__file__).resolve().parents[1] / "Post_Processing" / "canonical_extinction_to_space_group.csv"
DEFAULT_FINAL_TABLE = Path(__file__).resolve().parents[1] / "Post_Processing" / "FINAL_SPG_ExtG_CrysS_Table.csv"
DEFAULT_SG_LOOKUP = Path(__file__).resolve().parents[1] / "Post_Processing" / "spacegroup_lookup.csv"


@dataclass(frozen=True)
class ExtinctionTemplate:
    ext_group: int
    canonical_index: int
    canonical_symbol: str
    crystal_system: str
    vector: torch.Tensor


def _normalize_symbol(symbol: str) -> str:
    canon = symbol.split("(equiv:")[0].strip()
    canon = canon.replace("R (obv)-", "R_obv ")
    canon = canon.replace("R (obv) -", "R_obv ")
    canon = canon.replace("R (obv)", "R_obv")
    canon = canon.replace("R (rev)-", "R_rev ")
    canon = canon.replace("R (rev) -", "R_rev ")
    canon = canon.replace("R (rev)", "R_rev")
    canon = canon.replace("(ab)", "ab")
    return " ".join(canon.split())


def _get_crystal_system_from_sg(space_group: int) -> str:
    if 1 <= space_group <= 2:
        return "Triclinic"
    if 3 <= space_group <= 15:
        return "Monoclinic"
    if 16 <= space_group <= 74:
        return "Orthorhombic"
    if 75 <= space_group <= 142:
        return "Tetragonal"
    if 143 <= space_group <= 167:
        return "Trigonal"
    if 168 <= space_group <= 194:
        return "Hexagonal"
    if 195 <= space_group <= 230:
        return "Cubic"
    raise ValueError(f"Invalid space group {space_group}")


def _symbol_to_feature_vector(symbol: str, crystal_system: str) -> torch.Tensor:
    parts = _normalize_symbol(symbol).split()
    if not parts:
        raise ValueError(f"Could not parse canonical symbol: {symbol}")

    vector = torch.zeros(len(FEATURES), dtype=torch.float32)
    vector[FEATURE_TO_INDEX[f"Sys_{crystal_system}"]] = 1.0

    lattice = parts[0]
    lattice_map = {
        "P": "Lattice_P",
        "I": "Lattice_I",
        "F": "Lattice_F",
        "C": "Lattice_C",
        "A": "Lattice_C",
        "B": "Lattice_C",
        "R_obv": "Lattice_R",
        "R_rev": "Lattice_R",
        "R": "Lattice_R",
    }
    lattice_feature = lattice_map.get(lattice)
    if lattice_feature is None:
        raise ValueError(f"Unsupported lattice token '{lattice}' in symbol '{symbol}'")
    vector[FEATURE_TO_INDEX[lattice_feature]] = 1.0

    for pos_idx, token in enumerate(parts[1:], start=1):
        if pos_idx > 3:
            break
        for op in token.split("/"):
            if op in {"-", "1"}:
                continue
            feat = f"Pos{pos_idx}_{op}"
            if feat in FEATURE_TO_INDEX:
                vector[FEATURE_TO_INDEX[feat]] = 1.0

    return vector


def build_extinction_templates(
    canonical_table_path: Path | str = DEFAULT_CANONICAL_TABLE,
    final_table_path: Path | str = DEFAULT_FINAL_TABLE,
    sg_lookup_path: Path | str = DEFAULT_SG_LOOKUP,
) -> Dict[int, ExtinctionTemplate]:
    canonical_table_path = Path(canonical_table_path or DEFAULT_CANONICAL_TABLE)
    final_table_path = Path(final_table_path or DEFAULT_FINAL_TABLE)
    sg_lookup_path = Path(sg_lookup_path or DEFAULT_SG_LOOKUP)

    canonical_rows: Dict[int, Dict[str, str]] = {}
    with canonical_table_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            canonical_rows[int(row["Index"])] = row

    sg_to_lookup: Dict[int, Dict[str, str]] = {}
    with sg_lookup_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sg_to_lookup[int(row["Space Group Number"])] = row

    ext_templates: Dict[int, ExtinctionTemplate] = {}
    with final_table_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ext_group = int(row["Extinction Group"])
            space_group = int(row["Space Group"])
            lookup = sg_to_lookup[space_group]
            canonical_index = int(lookup["Index"])
            canonical_row = canonical_rows[canonical_index]
            sg_numbers = ast.literal_eval(canonical_row["Space Group Numbers"])
            crystal_system = _get_crystal_system_from_sg(sg_numbers[0])
            vector = _symbol_to_feature_vector(canonical_row["Canonical Extinction Group"], crystal_system)

            if ext_group in ext_templates:
                existing = ext_templates[ext_group]
                if existing.canonical_index != canonical_index:
                    raise ValueError(f"Extinction group {ext_group} maps to multiple canonical indices")
            else:
                ext_templates[ext_group] = ExtinctionTemplate(
                    ext_group=ext_group,
                    canonical_index=canonical_index,
                    canonical_symbol=canonical_row["Canonical Extinction Group"],
                    crystal_system=crystal_system,
                    vector=vector,
                )

    if len(ext_templates) != 99:
        raise ValueError(f"Expected 99 extinction templates, found {len(ext_templates)}")

    return ext_templates


def build_template_bank(
    canonical_table_path: Path | str = DEFAULT_CANONICAL_TABLE,
    final_table_path: Path | str = DEFAULT_FINAL_TABLE,
    sg_lookup_path: Path | str = DEFAULT_SG_LOOKUP,
) -> Tuple[torch.Tensor, List[int], Dict[int, ExtinctionTemplate]]:
    templates = build_extinction_templates(canonical_table_path, final_table_path, sg_lookup_path)
    ext_groups = sorted(templates.keys())
    template_bank = torch.stack([templates[ext].vector for ext in ext_groups], dim=0)
    return template_bank, ext_groups, templates


def template_mask_from_ext_groups(ext_group_order: List[int], allowed_ext_groups: List[int] | set[int]) -> torch.Tensor:
    allowed = set(int(ext) for ext in allowed_ext_groups)
    return torch.tensor([ext in allowed for ext in ext_group_order], dtype=torch.bool)


def ext_group_to_multilabel_target(
    ext_group: int,
    templates: Dict[int, ExtinctionTemplate],
) -> torch.Tensor:
    return templates[ext_group].vector.clone()


def split_multilabel_logits(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return logits[:, SYSTEM_SLICE], logits[:, LATTICE_SLICE], logits[:, OPERATOR_SLICE]


def multilabel_targets_to_split_targets(targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    system_targets = torch.argmax(targets[:, SYSTEM_SLICE], dim=1)
    lattice_targets = torch.argmax(targets[:, LATTICE_SLICE], dim=1)
    operator_targets = targets[:, OPERATOR_SLICE]
    return system_targets, lattice_targets, operator_targets


def build_system_operator_allowed_mask(template_bank: torch.Tensor) -> torch.Tensor:
    bank = template_bank.to(dtype=torch.bool)
    system_indices = torch.argmax(bank[:, SYSTEM_SLICE].to(dtype=torch.float32), dim=1)
    operator_bank = bank[:, OPERATOR_SLICE]
    allowed = torch.zeros((SYSTEM_SLICE.stop - SYSTEM_SLICE.start, OPERATOR_SLICE.stop - OPERATOR_SLICE.start), dtype=torch.bool)
    for system_idx in range(allowed.shape[0]):
        system_rows = operator_bank[system_indices == system_idx]
        if system_rows.numel() > 0:
            allowed[system_idx] = system_rows.any(dim=0)
    return allowed


def score_multilabel_templates(
    logits: torch.Tensor,
    template_bank: torch.Tensor,
    metric: str = "euclidean",
    allowed_mask: torch.Tensor | None = None,
    hierarchical: bool = False,
    log_priors: torch.Tensor | None = None,
    prior_weight: float = 0.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    bank = template_bank.to(device=probs.device, dtype=probs.dtype)

    if metric == "cosine":
        probs_norm = F.normalize(probs, dim=1)
        bank_norm = F.normalize(bank, dim=1)
        scores = probs_norm @ bank_norm.T
    elif metric == "bernoulli":
        eps = torch.finfo(probs.dtype).eps
        probs = probs.clamp(min=eps, max=1 - eps)
        log_p = torch.log(probs).unsqueeze(1)
        log_one_minus_p = torch.log1p(-probs).unsqueeze(1)
        bank_expanded = bank.unsqueeze(0)
        scores = (bank_expanded * log_p + (1.0 - bank_expanded) * log_one_minus_p).sum(dim=2)
    else:
        distances = torch.cdist(probs, bank)
        scores = -distances

    if hierarchical:
        pred_system = torch.argmax(probs[:, SYSTEM_SLICE], dim=1)
        pred_lattice = torch.argmax(probs[:, LATTICE_SLICE], dim=1)
        bank_system = torch.argmax(bank[:, SYSTEM_SLICE], dim=1)
        bank_lattice = torch.argmax(bank[:, LATTICE_SLICE], dim=1)
        hierarchy_mask = (bank_system.unsqueeze(0) == pred_system.unsqueeze(1)) & (
            bank_lattice.unsqueeze(0) == pred_lattice.unsqueeze(1)
        )
        scores = scores.masked_fill(~hierarchy_mask, float("-inf"))

    if allowed_mask is not None:
        allowed_mask = allowed_mask.to(device=probs.device)
        scores = scores.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))

    if log_priors is not None and prior_weight != 0.0:
        log_priors = log_priors.to(device=probs.device, dtype=probs.dtype)
        scores = scores + prior_weight * log_priors.unsqueeze(0)

    invalid_rows = torch.isinf(scores).all(dim=1)
    if invalid_rows.any():
        fallback_scores = score_multilabel_templates(
            logits[invalid_rows],
            template_bank,
            metric=metric,
            allowed_mask=allowed_mask,
            hierarchical=False,
        )
        scores = scores.clone()
        scores[invalid_rows] = fallback_scores

    return scores


def score_split_head_templates(
    logits: torch.Tensor,
    template_bank: torch.Tensor,
    allowed_mask: torch.Tensor | None = None,
    log_priors: torch.Tensor | None = None,
    prior_weight: float = 0.0,
    impossible_operator_masking: bool = False,
    impossible_operator_prob: float = 1e-3,
) -> torch.Tensor:
    sys_logits, lat_logits, op_logits = split_multilabel_logits(logits)
    sys_log_probs = F.log_softmax(sys_logits, dim=1)
    lat_log_probs = F.log_softmax(lat_logits, dim=1)

    bank = template_bank.to(device=logits.device, dtype=logits.dtype)
    bank_sys = torch.argmax(bank[:, SYSTEM_SLICE], dim=1)
    bank_lat = torch.argmax(bank[:, LATTICE_SLICE], dim=1)
    bank_ops = bank[:, OPERATOR_SLICE]

    sys_scores = sys_log_probs[:, bank_sys]
    lat_scores = lat_log_probs[:, bank_lat]

    eps = torch.finfo(logits.dtype).eps
    op_probs = torch.sigmoid(op_logits).clamp(min=eps, max=1 - eps)
    if impossible_operator_masking:
        system_operator_allowed = build_system_operator_allowed_mask(bank).to(device=logits.device)
        pred_system = torch.argmax(sys_logits, dim=1)
        allowed_ops = system_operator_allowed[pred_system]
        impossible_prob = max(float(impossible_operator_prob), float(eps))
        op_probs = torch.where(allowed_ops, op_probs, torch.full_like(op_probs, impossible_prob))
    op_log_p = torch.log(op_probs).unsqueeze(1)
    op_log_not_p = torch.log1p(-op_probs).unsqueeze(1)
    op_scores = (bank_ops.unsqueeze(0) * op_log_p + (1.0 - bank_ops.unsqueeze(0)) * op_log_not_p).sum(dim=2)

    scores = sys_scores + lat_scores + op_scores

    if allowed_mask is not None:
        allowed_mask = allowed_mask.to(device=logits.device)
        scores = scores.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))

    if log_priors is not None and prior_weight != 0.0:
        log_priors = log_priors.to(device=logits.device, dtype=logits.dtype)
        scores = scores + prior_weight * log_priors.unsqueeze(0)

    return scores


def decode_multilabel_logits(
    logits: torch.Tensor,
    template_bank: torch.Tensor,
    ext_group_order: List[int],
    metric: str = "euclidean",
    allowed_mask: torch.Tensor | None = None,
    hierarchical: bool = False,
    log_priors: torch.Tensor | None = None,
    prior_weight: float = 0.0,
) -> torch.Tensor:
    scores = score_multilabel_templates(
        logits,
        template_bank,
        metric=metric,
        allowed_mask=allowed_mask,
        hierarchical=hierarchical,
        log_priors=log_priors,
        prior_weight=prior_weight,
    )
    indices = torch.argmax(scores, dim=1)

    ext_group_tensor = torch.tensor(ext_group_order, device=logits.device, dtype=torch.long)
    return ext_group_tensor[indices]


def topk_decoded_ext_groups(
    logits: torch.Tensor,
    template_bank: torch.Tensor,
    ext_group_order: List[int],
    k: int,
    metric: str = "euclidean",
    allowed_mask: torch.Tensor | None = None,
    hierarchical: bool = False,
    log_priors: torch.Tensor | None = None,
    prior_weight: float = 0.0,
) -> torch.Tensor:
    scores = score_multilabel_templates(
        logits,
        template_bank,
        metric=metric,
        allowed_mask=allowed_mask,
        hierarchical=hierarchical,
        log_priors=log_priors,
        prior_weight=prior_weight,
    )
    topk_indices = torch.topk(scores, k=k, dim=1).indices
    ext_group_tensor = torch.tensor(ext_group_order, device=logits.device, dtype=torch.long)
    return ext_group_tensor[topk_indices]


def decode_split_head_logits(
    logits: torch.Tensor,
    template_bank: torch.Tensor,
    ext_group_order: List[int],
    allowed_mask: torch.Tensor | None = None,
    log_priors: torch.Tensor | None = None,
    prior_weight: float = 0.0,
    impossible_operator_masking: bool = False,
    impossible_operator_prob: float = 1e-3,
) -> torch.Tensor:
    scores = score_split_head_templates(
        logits,
        template_bank,
        allowed_mask=allowed_mask,
        log_priors=log_priors,
        prior_weight=prior_weight,
        impossible_operator_masking=impossible_operator_masking,
        impossible_operator_prob=impossible_operator_prob,
    )
    indices = torch.argmax(scores, dim=1)
    ext_group_tensor = torch.tensor(ext_group_order, device=logits.device, dtype=torch.long)
    return ext_group_tensor[indices]


def topk_decoded_split_head_ext_groups(
    logits: torch.Tensor,
    template_bank: torch.Tensor,
    ext_group_order: List[int],
    k: int,
    allowed_mask: torch.Tensor | None = None,
    log_priors: torch.Tensor | None = None,
    prior_weight: float = 0.0,
    impossible_operator_masking: bool = False,
    impossible_operator_prob: float = 1e-3,
) -> torch.Tensor:
    scores = score_split_head_templates(
        logits,
        template_bank,
        allowed_mask=allowed_mask,
        log_priors=log_priors,
        prior_weight=prior_weight,
        impossible_operator_masking=impossible_operator_masking,
        impossible_operator_prob=impossible_operator_prob,
    )
    topk_indices = torch.topk(scores, k=k, dim=1).indices
    ext_group_tensor = torch.tensor(ext_group_order, device=logits.device, dtype=torch.long)
    return ext_group_tensor[topk_indices]
