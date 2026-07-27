#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate the paper-ready main failure/difficulty-region figure for FAPR-Depth v6.

This script reads the CSV already produced by analyze_fapr_failure_regions.py.
It does not run model inference again.

Default input:
outputs/analysis\
02_failure_region_analysis\failure_region_gain_vs_backbone.csv

Outputs:
- full_gain_vs_backbone_paper.png
- full_gain_vs_backbone_paper.pdf
- full_gain_vs_backbone_paper_data.csv

Positive values mean lower RMSE than the Backbone Baseline.
Negative values mean degradation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_CSV = Path(
    r"outputs/analysis"
    r"\02_failure_region_analysis\failure_region_gain_vs_backbone.csv"
)

REGION_ORDER: List[str] = [
    "valid_state",
    "any_failure",
    "biased_failure",
    "boundary_failure",
    "boundary_ring",
    "hard_backbone_top20",
]

REGION_LABELS: Dict[str, str] = {
    "valid_state": "Valid\nState",
    "any_failure": "Any\nFailure",
    "biased_failure": "Biased\nFailure",
    "boundary_failure": "Boundary\nFailure",
    "boundary_ring": "Boundary\nRing",
    "hard_backbone_top20": "Hardest\n20%",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a paper-ready relative RMSE improvement figure from the "
            "FAPR failure-region CSV."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to failure_region_gain_vs_backbone.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the CSV directory.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="Full Candidate",
        help="Method row to plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Relative RMSE Improvement across Failure and Difficulty Regions",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the title for journal layouts that use only a caption.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--width", type=float, default=9.2)
    parser.add_argument("--height", type=float, default=4.8)
    return parser.parse_args()


def choose_serif_font() -> str:
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"):
        if candidate in installed:
            return candidate
    return "serif"


def load_plot_data(csv_path: Path, method: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {
        "region",
        "method",
        "pixels",
        "rmse",
        "rmse_relative_improvement_pct",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing_columns)
        )

    selected = frame[
        (frame["method"] == method)
        & (frame["region"].isin(REGION_ORDER))
    ].copy()

    if selected.empty:
        methods = sorted(frame["method"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"No rows found for method={method!r}. Available methods: {methods}"
        )

    selected["region"] = pd.Categorical(
        selected["region"],
        categories=REGION_ORDER,
        ordered=True,
    )
    selected = selected.sort_values("region").reset_index(drop=True)

    found = set(selected["region"].astype(str))
    missing_regions = [region for region in REGION_ORDER if region not in found]
    if missing_regions:
        raise ValueError(
            "The CSV does not contain all requested regions: "
            + ", ".join(missing_regions)
        )

    selected["improvement_pct"] = pd.to_numeric(
        selected["rmse_relative_improvement_pct"],
        errors="coerce",
    )
    selected["pixels"] = pd.to_numeric(selected["pixels"], errors="coerce")
    selected["rmse"] = pd.to_numeric(selected["rmse"], errors="coerce")

    invalid = selected["improvement_pct"].isna()
    if invalid.any():
        bad_regions = selected.loc[invalid, "region"].astype(str).tolist()
        raise ValueError(
            "Non-finite RMSE improvement values in regions: "
            + ", ".join(bad_regions)
        )

    selected["display_label"] = (
        selected["region"].astype(str).map(REGION_LABELS)
    )
    return selected


def add_value_labels(ax, bars, values: np.ndarray, y_span: float) -> None:
    offset = max(0.18, 0.025 * y_span)
    for bar, value in zip(bars, values):
        if value >= 0:
            y = value + offset
            va = "bottom"
        else:
            y = value - offset
            va = "top"
            bar.set_hatch("///")

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{value:+.1f}%",
            ha="center",
            va=va,
            fontsize=9.5,
        )


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_plot_data(csv_path, args.method)

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": choose_serif_font(),
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    values = data["improvement_pct"].to_numpy(dtype=np.float64)
    labels = data["display_label"].tolist()
    x = np.arange(len(data))

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    lower = min(-4.5, minimum - 1.2)
    upper = max(16.5, maximum + 1.6)
    y_span = upper - lower

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    bars = ax.bar(
        x,
        values,
        width=0.68,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative RMSE Improvement over Backbone (%)")
    ax.set_ylim(lower, upper)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)

    if not args.no_title:
        ax.set_title(args.title, pad=10)

    add_value_labels(ax, bars, values, y_span)

    ax.text(
        0.995,
        0.015,
        "Positive: improvement   Negative: degradation",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
    )

    fig.tight_layout()

    png_path = out_dir / "full_gain_vs_backbone_paper.png"
    pdf_path = out_dir / "full_gain_vs_backbone_paper.pdf"
    data_path = out_dir / "full_gain_vs_backbone_paper_data.csv"

    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    export = data[
        [
            "region",
            "display_label",
            "method",
            "pixels",
            "rmse",
            "improvement_pct",
        ]
    ].copy()
    export.to_csv(data_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("Paper-ready failure/difficulty-region figure saved")
    print("=" * 100)
    print(f"Input CSV : {csv_path}")
    print(f"Method    : {args.method}")
    print(f"PNG       : {png_path}")
    print(f"PDF       : {pdf_path}")
    print(f"Data      : {data_path}")
    print()
    print(export.to_string(index=False))


if __name__ == "__main__":
    main()
