from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from paper_ai_diffraction.core.model import VIT_model, adapt_patch_embed_input_channels
from paper_ai_diffraction.utils.extinction_multilabel import (
    build_template_bank,
    score_split_head_templates,
    template_mask_from_ext_groups,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PACKAGE_ROOT.parents[1] / "assets"
REVIEWER_LOOKUP_DIR = ASSET_ROOT / "lookups"
REVIEWER_TOPOLOGY_JSON = ASSET_ROOT / "topology" / "extinction_group_adjacency.json"
REVIEWER_EXAMPLE_DIR = ASSET_ROOT / "reviewer_examples"


def decode_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str)
    return value


def read_pattern_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {col.lower(): col for col in df.columns}
    if "2theta" in lower:
        theta_col = lower["2theta"]
    elif "two_theta" in lower:
        theta_col = lower["two_theta"]
    else:
        raise ValueError("CSV must contain a '2theta' or 'two_theta' column")
    if "intensity" not in lower:
        raise ValueError("CSV must contain an 'intensity' column")

    out = df[[theta_col, lower["intensity"]]].copy()
    out.columns = ["two_theta", "intensity"]
    out = out.dropna().sort_values("two_theta").drop_duplicates("two_theta")
    if out.empty:
        raise ValueError("CSV does not contain any valid rows after cleaning")
    return out.reset_index(drop=True)


def normalize_and_interpolate_pattern(
    df: pd.DataFrame,
    two_theta_min: float,
    two_theta_max: float,
    spec_length: int,
) -> dict[str, Any]:
    src_theta = df["two_theta"].to_numpy(dtype=np.float32)
    src_intensity = df["intensity"].to_numpy(dtype=np.float32)

    src_intensity = np.nan_to_num(src_intensity, nan=0.0, posinf=0.0, neginf=0.0)
    src_intensity = src_intensity - float(src_intensity.min())
    peak = float(src_intensity.max())
    if peak > 0:
        src_intensity = src_intensity / peak

    target_theta = np.linspace(two_theta_min, two_theta_max, spec_length, dtype=np.float32)
    interp_intensity = np.interp(target_theta, src_theta, src_intensity, left=0.0, right=0.0).astype(np.float32)

    warnings: list[str] = []
    if src_theta[0] > two_theta_min or src_theta[-1] < two_theta_max:
        warnings.append("Input range does not fully span the model grid; uncovered regions were filled with zeros.")
    if len(src_theta) > 1:
        median_step = float(np.median(np.diff(src_theta)))
        target_step = float(target_theta[1] - target_theta[0])
        if median_step > 2.5 * target_step:
            warnings.append("Input grid is substantially coarser than the model grid; interpolation may smooth peak structure.")

    return {
        "source_two_theta": src_theta,
        "source_intensity": src_intensity,
        "target_two_theta": target_theta,
        "target_intensity": interp_intensity,
        "warnings": warnings,
    }


def load_example_manifest(manifest_path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(manifest_path or (REVIEWER_EXAMPLE_DIR / "manifest.json"))
    if manifest_path.name.startswith("._"):
        manifest_path = manifest_path.with_name(manifest_path.name[2:])
    with manifest_path.open("r") as handle:
        return json.load(handle)


def load_example_metadata(example_csv_path: str | Path) -> dict[str, Any]:
    example_csv_path = Path(example_csv_path)
    if example_csv_path.name.startswith("._"):
        example_csv_path = example_csv_path.with_name(example_csv_path.name[2:])
    metadata_path = example_csv_path.with_suffix(".json")
    if metadata_path.name.startswith("._"):
        metadata_path = metadata_path.with_name(metadata_path.name[2:])
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing paired metadata JSON for {example_csv_path}")
    with metadata_path.open("r") as handle:
        return json.load(handle)


def normalize_prediction_record(
    record: dict[str, Any],
    resources_or_bundle: dict[str, Any] | ModelBundle | None = None,
) -> dict[str, Any]:
    normalized = dict(record)

    top5_eg = normalized.get("top5_eg") or []
    top5_prob = normalized.get("top5_prob") or []
    if normalized.get("top1_eg") is None:
        if normalized.get("predicted_ext_group") is not None:
            normalized["top1_eg"] = int(normalized["predicted_ext_group"])
        elif top5_eg:
            normalized["top1_eg"] = int(top5_eg[0])
    if normalized.get("top1_probability") is None and top5_prob:
        normalized["top1_probability"] = float(top5_prob[0])

    if normalized.get("top1_candidate_space_groups") is None and normalized.get("top1_eg") is not None and resources_or_bundle is not None:
        normalized["top1_candidate_space_groups"] = ext_group_to_space_groups(int(normalized["top1_eg"]), resources_or_bundle)

    return normalized


def load_precomputed_benchmark_summary(
    summary_path: str | Path,
    resources_or_bundle: dict[str, Any] | ModelBundle | None = None,
) -> dict[str, Any]:
    summary_path = Path(summary_path)
    with summary_path.open("r") as handle:
        summary = json.load(handle)
    summary["examples"] = [
        normalize_prediction_record(example, resources_or_bundle=resources_or_bundle)
        for example in summary.get("examples", [])
    ]
    return summary


def _get_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor] | Any:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _build_model(config: dict[str, Any], checkpoint: Any, device: torch.device) -> nn.Module:
    model = VIT_model(
        spec_length=config["spec_length"],
        num_output=config.get("num_labels", 37),
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        drop_ratio=config["dropout"],
        use_rope=config["use_rope"],
        use_mlp_head=config["use_mlp_head"],
        mlp_head_hidden_dim=config["mlp_head_hidden_dim"],
        use_physics_pe=config.get("use_physics_pe", False),
        physics_pe_mode=config.get("physics_pe_mode", "sin2theta"),
        two_theta_min=float(config.get("two_theta_min", 5.0)),
        two_theta_max=float(config.get("two_theta_max", 90.0)),
        physics_pe_scale=float(config.get("physics_pe_scale", 1.0)),
        use_coordinate_channel=config.get("use_coordinate_channel", False),
        coordinate_mode=config.get("coordinate_mode", "sin2theta"),
    )
    if config.get("use_split_head", False):
        model.head_sys = nn.Linear(model.embed_dim, 7)
        model.head_lat = nn.Linear(model.embed_dim, 5)
        model.head_ops = nn.Linear(model.embed_dim, 25)
    if config.get("use_aux_ext_head", False):
        model.aux_ext_head = nn.Linear(model.embed_dim, config.get("num_classes", 99))
        aux_state = checkpoint.get("aux_ext_head_state_dict") if isinstance(checkpoint, dict) else None
        if aux_state is not None:
            model.aux_ext_head.load_state_dict(aux_state)
    adapted_state_dict = adapt_patch_embed_input_channels(_get_model_state_dict(checkpoint), model)
    model.load_state_dict(adapted_state_dict, strict=False)
    return model.to(device).eval()


def load_extinction_resources(
    canonical_table_path: str | Path | None = None,
    final_table_path: str | Path | None = None,
    sg_lookup_path: str | Path | None = None,
) -> dict[str, Any]:
    canonical_table_path = Path(canonical_table_path or (REVIEWER_LOOKUP_DIR / "canonical_extinction_to_space_group.csv"))
    final_table_path = Path(final_table_path or (REVIEWER_LOOKUP_DIR / "FINAL_SPG_ExtG_CrysS_Table.csv"))
    sg_lookup_path = Path(sg_lookup_path or (REVIEWER_LOOKUP_DIR / "spacegroup_lookup.csv"))

    template_bank, ext_group_order, templates = build_template_bank(
        canonical_table_path=canonical_table_path,
        final_table_path=final_table_path,
        sg_lookup_path=sg_lookup_path,
    )

    final_df = pd.read_csv(final_table_path)
    ext_to_sgs = (
        final_df.groupby("Extinction Group")["Space Group"]
        .apply(lambda col: sorted(int(v) for v in col.tolist()))
        .to_dict()
    )

    sg_lookup_df = pd.read_csv(sg_lookup_path)
    sg_to_eg = {int(row["Space Group Number"]): int(row["Index"]) for _, row in sg_lookup_df.iterrows()}
    final_map = {int(row["Space Group"]): int(row["Extinction Group"]) for _, row in final_df.iterrows()}
    sg_to_eg = {sg: final_map.get(sg) for sg in sg_to_eg}

    return {
        "template_bank": template_bank,
        "ext_group_order": ext_group_order,
        "templates": templates,
        "ext_to_sgs": ext_to_sgs,
        "sg_to_eg": sg_to_eg,
        "canonical_table_path": canonical_table_path,
        "final_table_path": final_table_path,
        "sg_lookup_path": sg_lookup_path,
    }


def load_topology_assets(graph_json: str | Path | None = None) -> dict[str, Any]:
    graph_json = Path(graph_json or REVIEWER_TOPOLOGY_JSON)
    with graph_json.open("r") as handle:
        data = json.load(handle)

    nodes = {str(key): value for key, value in data["nodes"].items()}
    directed = {str(key): [str(child) for child in value] for key, value in data["adjacency"].items()}
    undirected = {node: set() for node in nodes}
    for parent, children in directed.items():
        for child in children:
            undirected.setdefault(parent, set()).add(child)
            undirected.setdefault(child, set()).add(parent)

    label_to_node: dict[int, str] = {}
    for node, attrs in nodes.items():
        for raw_eg in attrs.get("merged_from", []):
            label_to_node[int(raw_eg) + 1] = node

    return {
        "nodes": nodes,
        "directed": directed,
        "undirected": undirected,
        "label_to_node": label_to_node,
    }


def _shortest_path(undirected: dict[str, set[str]], src: str, dst: str) -> list[str] | None:
    if src == dst:
        return [src]
    queue = deque([src])
    parent = {src: None}
    while queue:
        node = queue.popleft()
        for nxt in undirected.get(node, set()):
            if nxt in parent:
                continue
            parent[nxt] = node
            if nxt == dst:
                path = [dst]
                cur = dst
                while parent[cur] is not None:
                    cur = parent[cur]
                    path.append(cur)
                return list(reversed(path))
            queue.append(nxt)
    return None


def _has_directed_path(directed: dict[str, list[str]], src: str, dst: str) -> bool:
    if src == dst:
        return True
    queue = deque([src])
    seen = {src}
    while queue:
        node = queue.popleft()
        for nxt in directed.get(node, []):
            if nxt == dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def describe_topology_relation(pred_eg: int, true_eg: int, topology_assets: dict[str, Any]) -> dict[str, Any]:
    label_to_node = topology_assets["label_to_node"]
    directed = topology_assets["directed"]
    undirected = topology_assets["undirected"]

    true_node = label_to_node.get(int(true_eg))
    pred_node = label_to_node.get(int(pred_eg))
    if true_node is None or pred_node is None:
        return {"relation": "unmapped", "distance": None, "path": None}
    if true_node == pred_node:
        return {"relation": "exact", "distance": 0, "path": [true_node]}

    path = _shortest_path(undirected, true_node, pred_node)
    distance = None if path is None else len(path) - 1
    if _has_directed_path(directed, true_node, pred_node):
        relation = "descendant"
    elif _has_directed_path(directed, pred_node, true_node):
        relation = "ancestor"
    else:
        relation = "branch_jump"
    return {"relation": relation, "distance": distance, "path": path}


@dataclass
class ModelBundle:
    model: nn.Module
    config: dict[str, Any]
    device: torch.device
    template_bank: torch.Tensor
    ext_group_order: list[int]
    templates: dict[int, Any]
    ext_to_sgs: dict[int, list[int]]
    sg_to_eg: dict[int, int | None]
    prior_h5_path: str | None = None


def build_model_bundle(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: str | None = None,
    canonical_table_path: str | Path | None = None,
    final_table_path: str | Path | None = None,
    sg_lookup_path: str | Path | None = None,
    prior_h5_path: str | None = None,
) -> ModelBundle:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    with open(config_path, "r") as handle:
        config.update(json.load(handle))

    resources = load_extinction_resources(
        canonical_table_path=canonical_table_path,
        final_table_path=final_table_path,
        sg_lookup_path=sg_lookup_path,
    )

    if canonical_table_path is None:
        config["canonical_table_path"] = str(resources["canonical_table_path"])
    if final_table_path is None:
        config["final_table_path"] = str(resources["final_table_path"])
    if sg_lookup_path is None:
        config["sg_lookup_path"] = str(resources["sg_lookup_path"])

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(config, checkpoint, torch_device)
    return ModelBundle(
        model=model,
        config=config,
        device=torch_device,
        template_bank=resources["template_bank"].to(torch_device),
        ext_group_order=resources["ext_group_order"],
        templates=resources["templates"],
        ext_to_sgs=resources["ext_to_sgs"],
        sg_to_eg=resources["sg_to_eg"],
        prior_h5_path=prior_h5_path,
    )


def _load_log_priors(h5_path: str | Path, ext_group_order: list[int], smoothing: float = 1.0) -> torch.Tensor:
    with h5py.File(h5_path, "r") as handle:
        if "y_train" not in handle:
            raise KeyError(f"{h5_path} does not contain y_train for prior estimation")
        y_train = handle["y_train"][:]
    counts = {int(ext): float(smoothing) for ext in ext_group_order}
    for value in y_train.tolist():
        counts[int(value)] = counts.get(int(value), float(smoothing)) + 1.0
    probs = torch.tensor([counts[int(ext)] for ext in ext_group_order], dtype=torch.float32)
    probs = probs / probs.sum()
    return torch.log(probs)


def _load_seen_mask(h5_path: str | Path, ext_group_order: list[int]) -> torch.Tensor:
    with h5py.File(h5_path, "r") as handle:
        if "y_train" not in handle:
            raise KeyError(f"{h5_path} does not contain y_train for seen-class masking")
        allowed = sorted(set(int(v) for v in handle["y_train"][:].tolist()))
    return template_mask_from_ext_groups(ext_group_order, allowed)


def infer_single_pattern(
    bundle: ModelBundle,
    target_intensity: np.ndarray,
    *,
    aux_temperature: float = 5.0,
    decoder: str = "aux_bayes",
    prior_h5_path: str | None = None,
) -> dict[str, Any]:
    x = torch.tensor(target_intensity, dtype=torch.float32, device=bundle.device).unsqueeze(0)
    with torch.no_grad():
        cls = bundle.model(x, return_cls_embedding=True)

        results: dict[str, Any] = {}
        prior_source = prior_h5_path or bundle.prior_h5_path

        if hasattr(bundle.model, "aux_ext_head"):
            aux_logits = bundle.model.aux_ext_head(cls)
            aux_probs = torch.softmax(aux_logits, dim=1)
            top5 = torch.topk(aux_probs, k=5, dim=1)
            results["aux_raw"] = {
                "top5_eg": [bundle.ext_group_order[int(idx)] for idx in top5.indices[0].tolist()],
                "top5_prob": [float(v) for v in top5.values[0].tolist()],
            }
            if prior_source:
                log_priors = _load_log_priors(prior_source, bundle.ext_group_order).to(bundle.device)
                bayes_scores = aux_logits / float(aux_temperature) + log_priors.unsqueeze(0).to(dtype=aux_logits.dtype)
                bayes_probs = torch.softmax(bayes_scores, dim=1)
                bayes_top5 = torch.topk(bayes_probs, k=5, dim=1)
                results["aux_bayes"] = {
                    "top5_eg": [bundle.ext_group_order[int(idx)] for idx in bayes_top5.indices[0].tolist()],
                    "top5_prob": [float(v) for v in bayes_top5.values[0].tolist()],
                    "temperature": float(aux_temperature),
                    "prior_h5_path": str(prior_source),
                }

        if bundle.config.get("use_split_head", False):
            logits = torch.cat(
                [bundle.model.head_sys(cls), bundle.model.head_lat(cls), bundle.model.head_ops(cls)],
                dim=1,
            )
            split_kwargs: dict[str, Any] = {
                "allowed_mask": None,
                "log_priors": None,
                "prior_weight": 0.0,
                "impossible_operator_masking": True,
                "impossible_operator_prob": 1e-3,
            }
            if prior_source:
                split_kwargs["allowed_mask"] = _load_seen_mask(prior_source, bundle.ext_group_order).to(bundle.device)
                split_kwargs["log_priors"] = _load_log_priors(prior_source, bundle.ext_group_order).to(bundle.device)
                split_kwargs["prior_weight"] = 1.0
            split_scores = score_split_head_templates(
                logits,
                bundle.template_bank,
                **split_kwargs,
            )
            split_probs = torch.softmax(split_scores, dim=1)
            split_top5 = torch.topk(split_probs, k=5, dim=1)
            results["split_bayes" if prior_source else "split_raw"] = {
                "top5_eg": [bundle.ext_group_order[int(idx)] for idx in split_top5.indices[0].tolist()],
                "top5_prob": [float(v) for v in split_top5.values[0].tolist()],
            }

        selected_key = decoder if decoder in results else ("aux_bayes" if "aux_bayes" in results else next(iter(results)))
        selected = results[selected_key]
        top1_eg = int(selected["top5_eg"][0])
        return {
            "selected_decoder": selected_key,
            "top1_eg": top1_eg,
            "top1_symbol": bundle.templates[top1_eg].canonical_symbol,
            "top1_crystal_system": bundle.templates[top1_eg].crystal_system,
            "top1_candidate_space_groups": bundle.ext_to_sgs.get(top1_eg, []),
            "decoders": results,
        }


def ext_group_to_space_groups(ext_group: int, resources_or_bundle: dict[str, Any] | ModelBundle) -> list[int]:
    mapping = resources_or_bundle.ext_to_sgs if isinstance(resources_or_bundle, ModelBundle) else resources_or_bundle["ext_to_sgs"]
    return mapping.get(int(ext_group), [])


def space_group_to_ext_group(space_group: int, resources_or_bundle: dict[str, Any] | ModelBundle) -> int | None:
    mapping = resources_or_bundle.sg_to_eg if isinstance(resources_or_bundle, ModelBundle) else resources_or_bundle["sg_to_eg"]
    return mapping.get(int(space_group))


def plot_pattern_overlay(prep: dict[str, Any]) -> tuple[Any, Any]:
    """Plot the raw input pattern against the model-grid resampling.

    Color convention:
    - blue: cleaned and normalized source CSV on its native 2theta grid
    - orange: interpolated pattern on the model's fixed input grid
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prep["source_two_theta"], prep["source_intensity"], label="source CSV", alpha=0.7)
    ax.plot(
        prep["target_two_theta"],
        prep["target_intensity"],
        label="model-grid interpolation",
        linewidth=1.2,
    )
    ax.set_xlabel("2theta")
    ax.set_ylabel("normalized intensity")
    ax.set_title("Pattern preprocessing: original vs. resampled")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_topk_probabilities(result: dict[str, Any]) -> tuple[Any, Any]:
    """Plot the selected decoder's top-k extinction-group probabilities.

    Color convention:
    - dark blue: top-1 extinction-group prediction
    - light blue: lower-ranked candidates in the same top-k list
    """
    selected = result["decoders"][result["selected_decoder"]]
    labels = [f"EG {int(eg)}" for eg in selected["top5_eg"]]
    probs = [float(v) for v in selected["top5_prob"]]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(labels, probs, color=["#1f77b4"] + ["#9ecae1"] * max(len(labels) - 1, 0))
    ax.set_ylabel("probability")
    ax.set_title(f"Top-k extinction-group probabilities ({result['selected_decoder']})")
    ax.set_ylim(0.0, max(probs) * 1.2 if probs else 1.0)
    ax.grid(axis="y", alpha=0.2)
    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{prob:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    return fig, ax


def plot_topology_path(
    relation_info: dict[str, Any],
    topology_assets: dict[str, Any],
    *,
    true_eg: int | None = None,
    pred_eg: int | None = None,
) -> tuple[Any, Any]:
    """Plot the condensed-DAG path between the true and predicted extinction groups.

    Color convention:
    - green: true extinction-group node
    - red: predicted extinction-group node
    - gray: intermediate condensed-DAG nodes on the shortest path
    """
    path = relation_info.get("path") or []
    nodes = topology_assets["nodes"]

    labels: list[str] = []
    colors: list[str] = []
    for idx, node in enumerate(path):
        merged = [int(v) + 1 for v in nodes.get(node, {}).get("merged_from", [])]
        label = f"node {node}\nEG {', '.join(str(v) for v in merged)}"
        if idx == 0:
            label += "\ntrue"
            colors.append("#2ca02c")
        elif idx == len(path) - 1:
            label += "\npred"
            colors.append("#d62728")
        else:
            colors.append("#bdbdbd")
        labels.append(label)

    if not labels:
        fig, ax = plt.subplots(figsize=(6, 1.8))
        ax.text(0.5, 0.5, "No topology path available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig, ax

    x = np.arange(len(labels))
    y = np.zeros_like(x, dtype=float)
    fig_w = max(6.0, 1.8 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 2.6))
    ax.plot(x, y, color="#6b6b6b", linewidth=2, zorder=1)
    ax.scatter(x, y, s=400, c=colors, edgecolors="black", zorder=2)
    for xi, label in zip(x, labels):
        ax.text(xi, 0.15, label, ha="center", va="bottom", fontsize=9)

    title = f"Condensed DAG path: {relation_info.get('relation', 'unknown')}"
    if true_eg is not None and pred_eg is not None:
        title += f" (true EG {true_eg} -> predicted EG {pred_eg})"
    ax.set_title(title)
    ax.set_ylim(-0.35, 0.9)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def plot_precomputed_summary(precomputed: dict[str, Any], topology_assets: dict[str, Any]) -> tuple[Any, Any]:
    """Summarize the frozen RRUFF-325 benchmark results with two compact plots.

    Left panel color convention:
    - green: exact predictions
    - blue: descendant-local errors
    - yellow: ancestor-local errors
    - red: branch jumps
    - gray: unmapped cases

    Right panel color convention:
    - green histogram: exact predictions
    - red histogram: wrong predictions
    """
    examples = precomputed.get("examples", [])
    if not examples:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.text(0.5, 0.5, "No precomputed examples available", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig, ax

    rows = []
    for ex in examples:
        relation = describe_topology_relation(int(ex["top1_eg"]), int(ex["true_eg"]), topology_assets)
        rows.append(
            {
                "correct": int(ex["top1_eg"]) == int(ex["true_eg"]),
                "top1_probability": float(ex.get("top1_probability", 0.0)),
                "relation": relation["relation"],
            }
        )
    df = pd.DataFrame(rows)

    relation_order = ["exact", "descendant", "ancestor", "branch_jump", "unmapped"]
    counts = df["relation"].value_counts().reindex(relation_order, fill_value=0)
    exact_conf = df.loc[df["correct"], "top1_probability"].to_numpy(dtype=float)
    wrong_conf = df.loc[~df["correct"], "top1_probability"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    axes[0].bar(counts.index, counts.values, color=["#2ca02c", "#1f77b4", "#ffbf00", "#d62728", "#7f7f7f"])
    axes[0].set_title("RRUFF-325 topology relation counts")
    axes[0].set_ylabel("examples")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.2)

    bins = np.linspace(0.0, 1.0, 16)
    if len(exact_conf):
        axes[1].hist(exact_conf, bins=bins, alpha=0.6, label="exact", color="#2ca02c")
    if len(wrong_conf):
        axes[1].hist(wrong_conf, bins=bins, alpha=0.6, label="wrong", color="#d62728")
    axes[1].set_title("Top-1 confidence on precomputed RRUFF-325 set")
    axes[1].set_xlabel("top-1 probability")
    axes[1].set_ylabel("examples")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    return fig, axes
