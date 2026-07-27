#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper-ready expert-routing and source-allocation analysis for FAPR-Depth v6.

Compared with the original visualize_fapr_expert_routing.py, this version:

1. Removes unsupported failure states from paper figures automatically.
   In the natural test split, Missing has zero support and is therefore excluded.
2. Creates a two-panel paper figure:
      (a) failure-conditioned expert routing
      (b) failure-conditioned source allocation
3. Writes one merged paper table containing routing, source weights and entropy.
4. Uses an adaptive y-axis for the supplementary routing-entropy figure.
5. Produces compact 8-panel qualitative figures:
      RGB
      raw absolute error
      target failure state
      predicted failure state
      biased-expert routing
      boundary-expert routing
      raw-source weight
      expert-source weight
6. Keeps the original detailed diagnostic figures and CSV files.

The script performs model inference once. It uses the same cache/checkpoint
protocol as the other FAPR v6 paper-analysis scripts.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from fapr_analysis_common import (
    add_common_args,
    bootstrap,
    chw_rgb,
    iter_forward,
    map2d,
    save_heatmap,
    select_top_records,
    tensor_numpy,
    write_csv,
    write_json,
    write_run_manifest,
)


FAILURE_NAMES = ["Valid", "Missing", "Biased", "Boundary"]
EXPERT_NAMES = ["Missing expert", "Biased expert", "Boundary expert"]
SOURCE_NAMES = ["Raw source", "Relative source", "Expert source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper-ready expert-routing and source-allocation figures."
    )
    add_common_args(parser, "03_expert_routing", default_phase="joint")
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument(
        "--min-failure-pixels",
        type=int,
        default=64,
        help="Minimum non-valid pixels for a sample to enter qualitative ranking.",
    )
    parser.add_argument(
        "--paper-sample-rank",
        type=int,
        default=7,
        help=(
            "Ranked qualitative sample copied to routing_sample_selected_paper.*. "
            "Use 0 to disable. The previous run's clearest example was rank 7."
        ),
    )
    parser.add_argument("--paper-dpi", type=int, default=600)
    parser.add_argument(
        "--no-paper-title",
        action="store_true",
        help="Omit the combined figure's overall title.",
    )
    parser.add_argument(
        "--skip-detailed-qualitative",
        action="store_true",
        help="Generate only compact paper qualitative figures.",
    )
    return parser.parse_args()


def choose_serif_font() -> str:
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"):
        if candidate in installed:
            return candidate
    return "serif"


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)
    qualitative_dir = out_dir / "qualitative"
    qualitative_paper_dir = out_dir / "qualitative_paper"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    qualitative_paper_dir.mkdir(parents=True, exist_ok=True)

    route_sum = np.zeros((4, 3), dtype=np.float64)
    alpha_sum = np.zeros((4, 3), dtype=np.float64)
    entropy_sum = np.zeros(4, dtype=np.float64)
    support = np.zeros(4, dtype=np.float64)

    top_samples: List[Dict[str, Any]] = []

    progress = tqdm(
        iter_forward(ctx),
        total=len(ctx.shards),
        desc="Expert routing",
        dynamic_ncols=True,
    )
    for inp, out, meta in progress:
        labels, _ = ctx.train_mod.failure_targets(
            inp["raw"], inp["gt"], inp["valid"], inp["boundary"]
        )
        region = (inp["valid"] > 0.5) & (inp["mask"] > 0.5)
        pi = out["pi"].float()
        alpha = out["alpha"].float()
        entropy = out["route_entropy"].float()

        for cls in range(4):
            cls_region = region & (labels == cls)
            count = int(cls_region.sum().item())
            if count <= 0:
                continue
            support[cls] += count
            for channel in range(3):
                route_sum[cls, channel] += float(
                    pi[:, channel : channel + 1][cls_region].sum().item()
                )
                alpha_sum[cls, channel] += float(
                    alpha[:, channel : channel + 1][cls_region].sum().item()
                )
            entropy_sum[cls] += float(entropy[cls_region].sum().item())

        # Rank examples by the amount and severity of non-valid failure evidence.
        for bi in range(inp["rgb"].shape[0]):
            sample_region = region[bi : bi + 1]
            failure_pixels = int(
                ((labels[bi : bi + 1] > 0) & sample_region).sum().item()
            )
            if failure_pixels < int(args.min_failure_pixels):
                continue

            raw_error = torch.abs(
                inp["raw"][bi : bi + 1] - inp["gt"][bi : bi + 1]
            )
            severity = float(
                (
                    raw_error
                    * (labels[bi : bi + 1] > 0).float()
                    * sample_region.float()
                ).sum().item()
                / max(failure_pixels, 1)
            )

            record = {
                "score": failure_pixels * max(severity, 1.0e-6),
                "failure_pixels": failure_pixels,
                "severity": severity,
                "source_shard": meta["source_shard"],
                "rgb": chw_rgb(inp["rgb"], bi),
                "raw": map2d(inp["raw"], bi),
                "gt": map2d(inp["gt"], bi),
                "raw_error": map2d(raw_error, 0),
                "target": map2d(labels.float(), bi),
                "predicted": map2d(
                    torch.argmax(
                        out["fail_prob"],
                        dim=1,
                        keepdim=True,
                    ).float(),
                    bi,
                ),
                "pi": tensor_numpy(pi[bi]),
                "alpha": tensor_numpy(alpha[bi]),
                "entropy": map2d(entropy, bi),
                "candidate_error": map2d(
                    torch.abs(out["candidate"] - inp["gt"]),
                    bi,
                ),
            }
            select_top_records(
                top_samples,
                record,
                key="score",
                k=int(args.num_visualizations),
                largest=True,
            )

    route_mean = np.divide(
        route_sum,
        support[:, None],
        out=np.full_like(route_sum, np.nan),
        where=support[:, None] > 0,
    )
    alpha_mean = np.divide(
        alpha_sum,
        support[:, None],
        out=np.full_like(alpha_sum, np.nan),
        where=support[:, None] > 0,
    )
    entropy_mean = np.divide(
        entropy_sum,
        support,
        out=np.full_like(entropy_sum, np.nan),
        where=support > 0,
    )

    present_ids = [index for index, value in enumerate(support) if value > 0]
    if not present_ids:
        raise RuntimeError("No supported failure states were found.")

    routing_rows = make_routing_rows(route_mean, support)
    source_rows = make_source_rows(alpha_mean, support)
    entropy_rows = make_entropy_rows(entropy_mean, support)
    paper_rows = make_paper_summary_rows(
        present_ids,
        route_mean,
        alpha_mean,
        entropy_mean,
        support,
    )

    # Full diagnostic CSVs keep all four configured states.
    write_csv(out_dir / "routing_by_failure.csv", routing_rows)
    write_csv(out_dir / "source_fusion_by_failure.csv", source_rows)
    write_csv(out_dir / "routing_entropy_by_failure.csv", entropy_rows)

    # Paper table contains only states supported by the evaluated split.
    write_csv(out_dir / "routing_source_summary_table.csv", paper_rows)

    write_json(
        out_dir / "routing_analysis.json",
        {
            "all_failure_states": FAILURE_NAMES,
            "paper_failure_states": [FAILURE_NAMES[i] for i in present_ids],
            "experts": EXPERT_NAMES,
            "sources": SOURCE_NAMES,
            "support_pixels": support,
            "routing_matrix": route_mean,
            "source_matrix": alpha_mean,
            "route_entropy": entropy_mean,
            "paper_summary_table": paper_rows,
        },
    )
    write_run_manifest(
        ctx,
        {
            "analysis": "expert_routing_paper",
            "paper_failure_states": [FAILURE_NAMES[i] for i in present_ids],
            "unsupported_states_excluded_from_paper_figures": [
                FAILURE_NAMES[i] for i in range(4) if i not in present_ids
            ],
        },
    )

    # Original full diagnostic heatmaps.
    save_heatmap(
        route_mean,
        FAILURE_NAMES,
        EXPERT_NAMES,
        "Failure-conditioned expert routing",
        out_dir / "expert_routing_heatmap_diagnostic.png",
        value_format=".3f",
        vmin=0.0,
        vmax=1.0,
    )
    save_heatmap(
        alpha_mean,
        FAILURE_NAMES,
        SOURCE_NAMES,
        "Source-fusion weights conditioned on failure state",
        out_dir / "source_fusion_heatmap_diagnostic.png",
        value_format=".3f",
        vmin=0.0,
        vmax=1.0,
    )

    # Paper-ready figures exclude unsupported rows such as Missing on the natural test split.
    route_paper = route_mean[present_ids]
    alpha_paper = alpha_mean[present_ids]
    row_labels = [FAILURE_NAMES[i] for i in present_ids]

    save_single_paper_heatmap(
        route_paper,
        row_labels,
        EXPERT_NAMES,
        "Failure-conditioned expert routing",
        out_dir / "expert_routing_heatmap_paper.png",
        out_dir / "expert_routing_heatmap_paper.pdf",
        dpi=args.paper_dpi,
    )
    save_single_paper_heatmap(
        alpha_paper,
        row_labels,
        SOURCE_NAMES,
        "Failure-conditioned source allocation",
        out_dir / "source_fusion_heatmap_paper.png",
        out_dir / "source_fusion_heatmap_paper.pdf",
        dpi=args.paper_dpi,
    )
    save_combined_paper_heatmaps(
        route_paper,
        alpha_paper,
        row_labels,
        out_dir / "routing_and_source_heatmaps_paper.png",
        out_dir / "routing_and_source_heatmaps_paper.pdf",
        dpi=args.paper_dpi,
        show_title=not args.no_paper_title,
    )
    save_entropy_supplement(
        entropy_mean[present_ids],
        row_labels,
        out_dir / "routing_entropy_by_failure_paper.png",
        out_dir / "routing_entropy_by_failure_paper.pdf",
        dpi=args.paper_dpi,
    )

    for index, sample in enumerate(top_samples, start=1):
        if not args.skip_detailed_qualitative:
            save_detailed_qualitative(
                sample,
                qualitative_dir / f"routing_sample_{index:02d}.png",
            )

        compact_png = (
            qualitative_paper_dir / f"routing_sample_{index:02d}_paper.png"
        )
        compact_pdf = (
            qualitative_paper_dir / f"routing_sample_{index:02d}_paper.pdf"
        )
        save_compact_qualitative(
            sample,
            compact_png,
            compact_pdf,
            dpi=args.paper_dpi,
        )

    selected_rank = int(args.paper_sample_rank)
    if selected_rank > 0:
        if selected_rank > len(top_samples):
            print(
                f"[WARNING] --paper-sample-rank={selected_rank} exceeds "
                f"available samples={len(top_samples)}."
            )
        else:
            src_png = (
                qualitative_paper_dir
                / f"routing_sample_{selected_rank:02d}_paper.png"
            )
            src_pdf = (
                qualitative_paper_dir
                / f"routing_sample_{selected_rank:02d}_paper.pdf"
            )
            shutil.copy2(
                src_png,
                out_dir / "routing_sample_selected_paper.png",
            )
            shutil.copy2(
                src_pdf,
                out_dir / "routing_sample_selected_paper.pdf",
            )

    print("\nPaper failure states:", [FAILURE_NAMES[i] for i in present_ids])
    print("\nRouting matrix used by paper figure")
    print(np.array2string(route_paper, precision=5, suppress_small=True))
    print("\nSource-allocation matrix used by paper figure")
    print(np.array2string(alpha_paper, precision=5, suppress_small=True))
    print("\nMerged paper table")
    for row in paper_rows:
        print(row)
    print("\nSaved to:", out_dir)


def make_routing_rows(
    route_mean: np.ndarray,
    support: np.ndarray,
) -> List[Dict[str, float]]:
    return [
        {
            "failure_state": FAILURE_NAMES[i],
            "support_pixels": int(support[i]),
            **{
                EXPERT_NAMES[j].lower().replace(" ", "_"): float(route_mean[i, j])
                for j in range(3)
            },
        }
        for i in range(4)
    ]


def make_source_rows(
    alpha_mean: np.ndarray,
    support: np.ndarray,
) -> List[Dict[str, float]]:
    return [
        {
            "failure_state": FAILURE_NAMES[i],
            "support_pixels": int(support[i]),
            **{
                SOURCE_NAMES[j].lower().replace(" ", "_"): float(alpha_mean[i, j])
                for j in range(3)
            },
        }
        for i in range(4)
    ]


def make_entropy_rows(
    entropy_mean: np.ndarray,
    support: np.ndarray,
) -> List[Dict[str, float]]:
    return [
        {
            "failure_state": FAILURE_NAMES[i],
            "support_pixels": int(support[i]),
            "mean_normalized_route_entropy": float(entropy_mean[i]),
        }
        for i in range(4)
    ]


def make_paper_summary_rows(
    present_ids: Sequence[int],
    route_mean: np.ndarray,
    alpha_mean: np.ndarray,
    entropy_mean: np.ndarray,
    support: np.ndarray,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for i in present_ids:
        rows.append(
            {
                "failure_state": FAILURE_NAMES[i],
                "support_pixels": int(support[i]),
                "support_million": float(support[i] / 1_000_000.0),
                "route_missing_expert": float(route_mean[i, 0]),
                "route_biased_expert": float(route_mean[i, 1]),
                "route_boundary_expert": float(route_mean[i, 2]),
                "source_raw": float(alpha_mean[i, 0]),
                "source_relative": float(alpha_mean[i, 1]),
                "source_expert": float(alpha_mean[i, 2]),
                "normalized_route_entropy": float(entropy_mean[i]),
            }
        )
    return rows


def apply_paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": choose_serif_font(),
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def annotate_heatmap(
    ax,
    matrix: np.ndarray,
    digits: int = 3,
) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            text = "—" if not np.isfinite(value) else f"{value:.{digits}f}"
            # Black/white text switching improves contrast without changing the map.
            color = "white" if np.isfinite(value) and value >= 0.60 else "black"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=9.5,
            )


def save_single_paper_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    apply_paper_style()
    fig, ax = plt.subplots(
        figsize=(1.75 * len(col_labels) + 2.2, 0.85 * len(row_labels) + 2.2)
    )
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, pad=9)
    annotate_heatmap(ax, matrix)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean weight")
    fig.tight_layout()
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_combined_paper_heatmaps(
    route_matrix: np.ndarray,
    source_matrix: np.ndarray,
    row_labels: Sequence[str],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
    show_title: bool,
) -> None:
    import matplotlib.pyplot as plt

    apply_paper_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.1),
        constrained_layout=True,
    )

    matrices = [route_matrix, source_matrix]
    column_sets = [EXPERT_NAMES, SOURCE_NAMES]
    titles = [
        "(a) Expert routing within the correction branch",
        "(b) Source allocation in posterior fusion",
    ]

    image = None
    for ax, matrix, columns, title in zip(
        axes,
        matrices,
        column_sets,
        titles,
    ):
        image = ax.imshow(
            matrix,
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
            cmap="viridis",
        )
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=18, ha="right")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(title, pad=9)
        annotate_heatmap(ax, matrix)

    axes[0].set_ylabel("Ground-truth failure state")
    if show_title:
        fig.suptitle(
            "Failure-Conditioned Expert Routing and Source Allocation",
            fontsize=13,
        )

    colorbar = fig.colorbar(
        image,
        ax=axes,
        fraction=0.028,
        pad=0.03,
    )
    colorbar.set_label("Mean weight")

    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_entropy_supplement(
    entropy_values: np.ndarray,
    row_labels: Sequence[str],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    apply_paper_style()
    finite = entropy_values[np.isfinite(entropy_values)]
    maximum = float(np.max(finite)) if finite.size else 1.0
    upper = max(0.001, maximum * 1.30)

    x = np.arange(len(row_labels))
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    bars = ax.bar(x, entropy_values, edgecolor="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(row_labels)
    ax.set_ylim(0.0, upper)
    ax.set_ylabel("Normalized routing entropy")
    ax.set_title("Routing certainty by failure state")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    offset = upper * 0.025
    for bar, value in zip(bars, entropy_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + offset,
            f"{float(value):.4f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
        )

    fig.tight_layout()
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_detailed_qualitative(
    sample: Dict[str, Any],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    gt = sample["gt"]
    depth_values = gt[np.isfinite(gt) & (gt > 0)]
    vmin = float(np.percentile(depth_values, 2)) if depth_values.size else 0.0
    vmax = float(np.percentile(depth_values, 98)) if depth_values.size else 1.0

    panels = [
        (sample["rgb"], "RGB", None, None, None),
        (sample["raw"], "Raw depth", "viridis", vmin, vmax),
        (sample["gt"], "Ground truth", "viridis", vmin, vmax),
        (sample["raw_error"], "Raw absolute error", "magma", 0.0, None),
        (sample["target"], "Target failure state", "tab10", 0.0, 3.0),
        (sample["predicted"], "Predicted failure state", "tab10", 0.0, 3.0),
        (sample["pi"][0], "Route: missing expert", "viridis", 0.0, 1.0),
        (sample["pi"][1], "Route: biased expert", "viridis", 0.0, 1.0),
        (sample["pi"][2], "Route: boundary expert", "viridis", 0.0, 1.0),
        (sample["alpha"][0], "Source weight: raw", "viridis", 0.0, 1.0),
        (sample["alpha"][1], "Source weight: relative", "viridis", 0.0, 1.0),
        (sample["alpha"][2], "Source weight: expert", "viridis", 0.0, 1.0),
        (sample["entropy"], "Routing entropy", "viridis", 0.0, 1.0),
        (sample["candidate_error"], "Full candidate error", "magma", 0.0, None),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (image, title, cmap, lo, hi) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"Failure pixels={sample['failure_pixels']}, "
        f"mean failure error={sample['severity']:.4f}\n"
        f"{sample['source_shard']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_compact_qualitative(
    sample: Dict[str, Any],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_paper_style()

    raw_error = sample["raw_error"]
    error_values = raw_error[np.isfinite(raw_error)]
    error_max = (
        float(np.percentile(error_values, 98))
        if error_values.size
        else 1.0
    )
    error_max = max(error_max, 1.0e-6)

    panels = [
        (sample["rgb"], "RGB", None, None, None),
        (
            sample["raw_error"],
            "Raw absolute error",
            "magma",
            0.0,
            error_max,
        ),
        (
            sample["target"],
            "Target failure state",
            "tab10",
            0.0,
            3.0,
        ),
        (
            sample["predicted"],
            "Predicted failure state",
            "tab10",
            0.0,
            3.0,
        ),
        (
            sample["pi"][1],
            "Biased-expert routing",
            "viridis",
            0.0,
            1.0,
        ),
        (
            sample["pi"][2],
            "Boundary-expert routing",
            "viridis",
            0.0,
            1.0,
        ),
        (
            sample["alpha"][0],
            "Raw-source weight",
            "viridis",
            0.0,
            1.0,
        ),
        (
            sample["alpha"][2],
            "Expert-source weight",
            "viridis",
            0.0,
            1.0,
        ),
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12.8, 6.2),
        constrained_layout=True,
    )
    for ax, (image, title, cmap, lo, hi) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=10.0)
        ax.axis("off")

    # Compact categorical legend for the two failure-state panels.
    state_cmap = plt.get_cmap("tab10")
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=state_cmap(index / 10.0),
            markeredgecolor="none",
            markersize=8,
            label=name,
        )
        for index, name in enumerate(FAILURE_NAMES)
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.025),
    )

    fig.suptitle(
        "Failure-Conditioned Routing and Source Allocation",
        fontsize=12.5,
    )
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
