#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a unified paper qualitative comparison for FAPR-Depth v6.

The script performs one test-split inference pass, automatically selects
representative cases, and generates:

1. unified_qualitative_depth_paper.{png,pdf}
   Columns:
   RGB | Raw Depth | Backbone Baseline | Posterior w/o Safety |
   Safe Posterior | Full Candidate | Ground Truth

2. unified_qualitative_error_paper.{png,pdf}
   Columns:
   Raw Error | Backbone Error | Posterior Error | Safe Error | Full Error

3. unified_qualitative_selection.csv
   Reproducible sample identifiers and selection scores.

Automatic case types:
- Boundary Gain: largest Full-vs-Backbone gain on boundary pixels.
- Safe Rescue: largest Safe-vs-unconstrained-Posterior gain.
- Hard-Region Gain: largest Full-vs-Backbone gain on the hardest 20% pixels.
- Proposal Gain: largest Full-vs-Safe gain.

This is an internal unified pipeline comparison. It uses paper-facing names and
does not expose the implementation name of the external backbone.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from fapr_analysis_common import (
    add_common_args,
    bootstrap,
    chw_rgb,
    iter_forward,
    map2d,
    select_top_records,
    write_csv,
    write_json,
    write_run_manifest,
)


CASE_ORDER = [
    "boundary_gain",
    "safe_rescue",
    "hard_gain",
    "proposal_gain",
]

CASE_LABELS = {
    "boundary_gain": "Boundary Gain",
    "safe_rescue": "Safe Rescue",
    "hard_gain": "Hard-Region Gain",
    "proposal_gain": "Proposal Gain",
}

DEPTH_COLUMNS = [
    "RGB",
    "Raw Depth",
    "Backbone Baseline",
    "Posterior w/o Safety",
    "Safe Posterior",
    "Full Candidate",
    "Ground Truth",
]

ERROR_COLUMNS = [
    "Raw Error",
    "Backbone Error",
    "Posterior Error",
    "Safe Error",
    "Full Error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate unified FAPR v6 qualitative comparison figures."
    )
    add_common_args(parser, "07_unified_qualitative", default_phase="joint")
    parser.add_argument(
        "--num-cases",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Number of automatically selected case types.",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=12,
        help="Top records retained per selection criterion before de-duplication.",
    )
    parser.add_argument(
        "--manual-global-indices",
        type=str,
        default="",
        help=(
            "Optional comma-separated global sample indices. When supplied, "
            "automatic selection is disabled."
        ),
    )
    parser.add_argument("--paper-dpi", type=int, default=600)
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Remove overall figure titles.",
    )
    return parser.parse_args()


def choose_serif_font() -> str:
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"):
        if candidate in installed:
            return candidate
    return "serif"


def apply_paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": choose_serif_font(),
            "font.size": 9.5,
            "axes.titlesize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask > 0.5
    count = int(mask.sum().item())
    if count <= 0:
        return float("-inf")
    return float(value[mask].float().mean().item())


def hard_top20_mask(
    anchor_error: torch.Tensor,
    region: torch.Tensor,
) -> torch.Tensor:
    hard = torch.zeros_like(region, dtype=torch.bool)
    for bi in range(anchor_error.shape[0]):
        valid_values = anchor_error[bi][region[bi] > 0.5]
        if valid_values.numel() == 0:
            continue
        threshold = torch.quantile(valid_values.float(), 0.80)
        hard[bi] = (region[bi] > 0.5) & (anchor_error[bi] >= threshold)
    return hard


def create_record(
    inp: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    bi: int,
    global_index: int,
    meta: Dict[str, Any],
    case_scores: Dict[str, float],
) -> Dict[str, Any]:
    gt = inp["gt"][bi : bi + 1]
    raw = inp["raw"][bi : bi + 1]
    anchor = out["anchor_depth"][bi : bi + 1]
    posterior = out["legacy_fused"][bi : bi + 1]
    safe = out["safe_posterior"][bi : bi + 1]
    full = out["candidate"][bi : bi + 1]

    return {
        "global_index": int(global_index),
        "sample_in_shard": int(meta["sample_offset"] + bi),
        "shard_index": int(meta["shard_index"]),
        "source_shard": str(meta["source_shard"]),
        "sample_key": (
            f"shard_{int(meta['shard_index']):04d}_"
            f"sample_{int(meta['sample_offset'] + bi):04d}"
        ),
        "case_scores": dict(case_scores),
        "rgb": chw_rgb(inp["rgb"], bi),
        "mask": map2d(inp["mask"], bi),
        "valid": map2d(inp["valid"], bi),
        "boundary": map2d(inp["boundary"], bi),
        "raw": map2d(raw, 0),
        "anchor": map2d(anchor, 0),
        "posterior": map2d(posterior, 0),
        "safe": map2d(safe, 0),
        "full": map2d(full, 0),
        "gt": map2d(gt, 0),
        "raw_error": map2d(torch.abs(raw - gt), 0),
        "anchor_error": map2d(torch.abs(anchor - gt), 0),
        "posterior_error": map2d(torch.abs(posterior - gt), 0),
        "safe_error": map2d(torch.abs(safe - gt), 0),
        "full_error": map2d(torch.abs(full - gt), 0),
    }


def select_unique_cases(
    pools: Dict[str, List[Dict[str, Any]]],
    num_cases: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    selected: List[Tuple[str, Dict[str, Any]]] = []
    used_keys = set()

    for case_name in CASE_ORDER[:num_cases]:
        chosen = None
        for record in pools[case_name]:
            if record["sample_key"] not in used_keys:
                chosen = record
                break
        if chosen is None and pools[case_name]:
            chosen = pools[case_name][0]
        if chosen is not None:
            selected.append((case_name, chosen))
            used_keys.add(chosen["sample_key"])

    return selected


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manual_indices = {
        int(value.strip())
        for value in args.manual_global_indices.split(",")
        if value.strip()
    }

    pools: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in CASE_ORDER
    }
    manual_records: Dict[int, Dict[str, Any]] = {}

    global_index = 0
    progress = tqdm(
        iter_forward(ctx),
        total=len(ctx.shards),
        desc="Unified qualitative selection",
        dynamic_ncols=True,
    )

    for inp, out, meta in progress:
        batch_size = int(inp["rgb"].shape[0])

        valid_region = (
            (inp["valid"] > 0.5)
            & (inp["mask"] > 0.5)
        )
        boundary_region = valid_region & (inp["boundary"] > 0.15)

        anchor_error = torch.abs(out["anchor_depth"] - inp["gt"])
        posterior_error = torch.abs(out["legacy_fused"] - inp["gt"])
        safe_error = torch.abs(out["safe_posterior"] - inp["gt"])
        full_error = torch.abs(out["candidate"] - inp["gt"])
        hard_region = hard_top20_mask(anchor_error, valid_region.float())

        for bi in range(batch_size):
            sample_valid = valid_region[bi : bi + 1]
            sample_boundary = boundary_region[bi : bi + 1]
            sample_hard = hard_region[bi : bi + 1]

            case_scores = {
                "boundary_gain": masked_mean(
                    anchor_error[bi : bi + 1] - full_error[bi : bi + 1],
                    sample_boundary,
                ),
                "safe_rescue": masked_mean(
                    posterior_error[bi : bi + 1] - safe_error[bi : bi + 1],
                    sample_valid,
                ),
                "hard_gain": masked_mean(
                    anchor_error[bi : bi + 1] - full_error[bi : bi + 1],
                    sample_hard,
                ),
                "proposal_gain": masked_mean(
                    safe_error[bi : bi + 1] - full_error[bi : bi + 1],
                    sample_valid,
                ),
            }

            current_index = global_index + bi
            needs_record = (
                not manual_indices
                or current_index in manual_indices
                or any(np.isfinite(value) for value in case_scores.values())
            )
            if not needs_record:
                continue

            record = create_record(
                inp,
                out,
                bi,
                current_index,
                meta,
                case_scores,
            )

            if manual_indices:
                if current_index in manual_indices:
                    manual_records[current_index] = record
            else:
                for case_name, score in case_scores.items():
                    if not np.isfinite(score):
                        continue
                    candidate = dict(record)
                    candidate["selection_score"] = float(score)
                    select_top_records(
                        pools[case_name],
                        candidate,
                        key="selection_score",
                        k=int(args.candidate_pool),
                        largest=True,
                    )

        global_index += batch_size

    if manual_indices:
        missing = sorted(manual_indices - set(manual_records))
        if missing:
            raise RuntimeError(
                "Manual sample indices were not found: "
                + ", ".join(str(value) for value in missing)
            )
        selected = [
            (f"manual_{index}", manual_records[index])
            for index in sorted(manual_indices)
        ]
    else:
        selected = select_unique_cases(pools, int(args.num_cases))

    if not selected:
        raise RuntimeError("No qualitative cases were selected.")

    manifest_rows = []
    for row_index, (case_name, record) in enumerate(selected, start=1):
        label = CASE_LABELS.get(case_name, case_name)
        score = record["case_scores"].get(case_name, float("nan"))
        manifest_rows.append(
            {
                "row": row_index,
                "case": case_name,
                "case_label": label,
                "selection_score_m": score,
                "global_index": record["global_index"],
                "sample_key": record["sample_key"],
                "shard_index": record["shard_index"],
                "sample_in_shard": record["sample_in_shard"],
                "source_shard": record["source_shard"],
            }
        )

    write_csv(out_dir / "unified_qualitative_selection.csv", manifest_rows)
    write_json(
        out_dir / "unified_qualitative_selection.json",
        {
            "selected": manifest_rows,
            "selection_policy": (
                "manual_global_indices"
                if manual_indices
                else CASE_ORDER[: int(args.num_cases)]
            ),
        },
    )
    write_run_manifest(
        ctx,
        {
            "analysis": "unified_qualitative",
            "selected_samples": manifest_rows,
        },
    )

    save_depth_figure(
        selected,
        out_dir / "unified_qualitative_depth_paper.png",
        out_dir / "unified_qualitative_depth_paper.pdf",
        dpi=args.paper_dpi,
        show_title=not args.no_title,
    )
    save_error_figure(
        selected,
        out_dir / "unified_qualitative_error_paper.png",
        out_dir / "unified_qualitative_error_paper.pdf",
        dpi=args.paper_dpi,
        show_title=not args.no_title,
    )

    print("\nSelected qualitative cases")
    for row in manifest_rows:
        print(row)
    print("Saved to:", out_dir)


def sample_depth_range(record: Dict[str, Any]) -> Tuple[float, float]:
    gt = record["gt"]
    valid = record["valid"] > 0.5
    values = gt[valid & np.isfinite(gt) & (gt > 0)]
    if values.size == 0:
        values = gt[np.isfinite(gt) & (gt > 0)]
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def save_depth_figure(
    selected: Sequence[Tuple[str, Dict[str, Any]]],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
    show_title: bool,
) -> None:
    import matplotlib.pyplot as plt

    apply_paper_style()
    rows = len(selected)
    fig, axes = plt.subplots(
        rows,
        len(DEPTH_COLUMNS),
        figsize=(14.7, 2.25 * rows + 0.7),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, (case_name, record) in enumerate(selected):
        vmin, vmax = sample_depth_range(record)
        images = [
            record["rgb"],
            record["raw"],
            record["anchor"],
            record["posterior"],
            record["safe"],
            record["full"],
            record["gt"],
        ]
        case_label = CASE_LABELS.get(case_name, case_name)
        score = record["case_scores"].get(case_name, float("nan"))

        for column_index, (image, column_name) in enumerate(
            zip(images, DEPTH_COLUMNS)
        ):
            ax = axes[row_index, column_index]
            if column_name == "RGB":
                ax.imshow(image)
            else:
                ax.imshow(
                    image,
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
            if row_index == 0:
                ax.set_title(column_name)
            ax.axis("off")

        axes[row_index, 0].text(
            -0.08,
            0.5,
            (
                f"({chr(97 + row_index)}) {case_label}\n"
                f"gain={score:+.4f} m"
            ),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9.0,
        )

    if show_title:
        fig.suptitle(
            "Unified Qualitative Comparison of FAPR-Depth Output Stages",
            fontsize=13.0,
        )

    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_error_figure(
    selected: Sequence[Tuple[str, Dict[str, Any]]],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
    show_title: bool,
) -> None:
    import matplotlib.pyplot as plt

    apply_paper_style()
    rows = len(selected)
    fig, axes = plt.subplots(
        rows,
        len(ERROR_COLUMNS),
        figsize=(11.2, 2.25 * rows + 0.7),
        squeeze=False,
        constrained_layout=True,
    )

    for row_index, (case_name, record) in enumerate(selected):
        errors = [
            record["raw_error"],
            record["anchor_error"],
            record["posterior_error"],
            record["safe_error"],
            record["full_error"],
        ]
        mask = (record["valid"] > 0.5) & (record["mask"] > 0.5)
        values = np.concatenate(
            [error[mask].reshape(-1) for error in errors if np.any(mask)]
        ) if np.any(mask) else np.concatenate([error.reshape(-1) for error in errors])
        values = values[np.isfinite(values)]
        vmax = float(np.percentile(values, 98)) if values.size else 0.05
        vmax = max(vmax, 1.0e-5)

        for column_index, (error, column_name) in enumerate(
            zip(errors, ERROR_COLUMNS)
        ):
            ax = axes[row_index, column_index]
            ax.imshow(
                error * record["mask"],
                cmap="magma",
                vmin=0.0,
                vmax=vmax,
            )
            if row_index == 0:
                ax.set_title(column_name)
            ax.axis("off")

        case_label = CASE_LABELS.get(case_name, case_name)
        axes[row_index, 0].text(
            -0.08,
            0.5,
            f"({chr(97 + row_index)}) {case_label}",
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9.0,
        )

    if show_title:
        fig.suptitle(
            "Transparent-Region Absolute Error Maps",
            fontsize=13.0,
        )

    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
