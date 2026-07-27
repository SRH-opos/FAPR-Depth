#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create two paper-ready relative-prior robustness figures for FAPR-Depth v6.

This script reads the CSV already produced by:
    stress_test_fapr_relative_prior.py

It DOES NOT run model inference again.

Figure 1
--------
Safety compensation under representative relative-prior corruptions:
- Posterior Fusion w/o Safety Control
- Safe Posterior
- Full Candidate

Figure 2
--------
Full-Candidate relative Score degradation in a compact 2x3 family layout:
- random spatial dropout
- Gaussian noise
- bias
- scale
- spatial shift
- confidence reliability

Default input
-------------
outputs/analysis/
06_relative_prior_stress\\relative_prior_stress_results.csv

Default outputs
---------------
relative_prior_safety_compensation_paper.png
relative_prior_safety_compensation_paper.pdf
relative_prior_safety_compensation_paper_data.csv

relative_prior_family_sensitivity_paper.png
relative_prior_family_sensitivity_paper.pdf
relative_prior_family_sensitivity_paper_data.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_CSV = Path(
    r"outputs/analysis"
    r"\06_relative_prior_stress\relative_prior_stress_results.csv"
)

REPRESENTATIVE_CONDITIONS: List[str] = [
    "clean",
    "dropout_50pct",
    "dropout_100pct",
    "noise_sigma_0.040",
    "shift_8px",
    "confidence_zero",
    "confidence_inverted",
]

CONDITION_LABELS: Dict[str, str] = {
    "clean": "Clean",
    "dropout_25pct": "25%\nDropout",
    "dropout_50pct": "50%\nDropout",
    "dropout_75pct": "75%\nDropout",
    "dropout_100pct": "100%\nDropout",
    "noise_sigma_0.005": r"$\sigma$=.005 m",
    "noise_sigma_0.010": r"$\sigma$=.010 m",
    "noise_sigma_0.020": r"$\sigma$=.020 m",
    "noise_sigma_0.040": r"$\sigma$=.040 m",
    "bias_minus_0.030": "−3 cm",
    "bias_plus_0.030": "+3 cm",
    "bias_plus_0.060": "+6 cm",
    "scale_0.95": "×0.95",
    "scale_1.05": "×1.05",
    "shift_2px": "2 px",
    "shift_4px": "4 px",
    "shift_8px": "8 px",
    "confidence_zero": "Confidence\nZero",
    "confidence_inverted": "Confidence\nInverted",
}

OUTPUT_ORDER = [
    "Posterior Fusion w/o Safety Control",
    "Safe Posterior",
    "Full Candidate",
]

OUTPUT_LABELS = {
    "Posterior Fusion w/o Safety Control": "Posterior w/o Safety",
    "Safe Posterior": "Safe Posterior",
    "Full Candidate": "Full Candidate",
}

FAMILY_ORDER = [
    "dropout",
    "noise",
    "bias",
    "scale",
    "shift",
    "confidence",
]

FAMILY_TITLES = {
    "dropout": "(a) Random spatial dropout",
    "noise": "(b) Gaussian noise",
    "bias": "(c) Additive bias",
    "scale": "(d) Scale error",
    "shift": "(e) Spatial shift",
    "confidence": "(f) Confidence reliability",
}

FAMILY_CONDITIONS: Dict[str, List[str]] = {
    "dropout": [
        "dropout_25pct",
        "dropout_50pct",
        "dropout_75pct",
        "dropout_100pct",
    ],
    "noise": [
        "noise_sigma_0.005",
        "noise_sigma_0.010",
        "noise_sigma_0.020",
        "noise_sigma_0.040",
    ],
    "bias": [
        "bias_minus_0.030",
        "bias_plus_0.030",
        "bias_plus_0.060",
    ],
    "scale": [
        "scale_0.95",
        "scale_1.05",
    ],
    "shift": [
        "shift_2px",
        "shift_4px",
        "shift_8px",
    ],
    "confidence": [
        "confidence_zero",
        "confidence_inverted",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready relative-prior stress figures."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to relative_prior_stress_results.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the CSV directory.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Remove overall titles and rely on the paper caption.",
    )
    parser.add_argument(
        "--family-ymin",
        type=float,
        default=-1.5,
        help="Lower y limit for the family sensitivity figure.",
    )
    parser.add_argument(
        "--family-ymax",
        type=float,
        default=19.5,
        help="Upper y limit for the family sensitivity figure.",
    )
    parser.add_argument(
        "--compensation-width",
        type=float,
        default=11.2,
    )
    parser.add_argument(
        "--compensation-height",
        type=float,
        default=5.1,
    )
    parser.add_argument(
        "--family-width",
        type=float,
        default=11.2,
    )
    parser.add_argument(
        "--family-height",
        type=float,
        default=7.0,
    )
    return parser.parse_args()


def choose_serif_font() -> str:
    from matplotlib import font_manager

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "DejaVu Serif",
    ):
        if candidate in installed:
            return candidate
    return "serif"


def apply_paper_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": choose_serif_font(),
            "font.size": 10.5,
            "axes.labelsize": 11.0,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.3,
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {
        "condition",
        "family",
        "severity",
        "output",
        "score",
        "relative_score_degradation_pct",
        "delta_score_vs_clean",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
        )

    numeric_columns = [
        "severity",
        "score",
        "relative_score_degradation_pct",
        "delta_score_vs_clean",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[numeric_columns].isna().any().any():
        raise ValueError("CSV contains non-numeric values in required numeric columns.")

    duplicated = frame.duplicated(subset=["condition", "output"], keep=False)
    if duplicated.any():
        rows = frame.loc[duplicated, ["condition", "output"]]
        raise ValueError(
            "Duplicate condition/output rows found:\n"
            + rows.to_string(index=False)
        )

    return frame


def require_rows(
    frame: pd.DataFrame,
    conditions: Sequence[str],
    outputs: Sequence[str],
) -> None:
    missing: List[str] = []
    for condition in conditions:
        for output in outputs:
            match = frame[
                (frame["condition"] == condition)
                & (frame["output"] == output)
            ]
            if len(match) != 1:
                missing.append(f"{condition} / {output}")
    if missing:
        raise ValueError(
            "Required rows are missing from the CSV:\n  "
            + "\n  ".join(missing)
        )


def create_compensation_data(frame: pd.DataFrame) -> pd.DataFrame:
    require_rows(frame, REPRESENTATIVE_CONDITIONS, OUTPUT_ORDER)

    rows: List[Dict[str, float]] = []
    for condition in REPRESENTATIVE_CONDITIONS:
        condition_rows = frame[frame["condition"] == condition]
        lookup = {
            row["output"]: row
            for _, row in condition_rows.iterrows()
        }

        posterior = lookup["Posterior Fusion w/o Safety Control"]
        safe = lookup["Safe Posterior"]
        full = lookup["Full Candidate"]

        clean_posterior = float(
            frame[
                (frame["condition"] == "clean")
                & (
                    frame["output"]
                    == "Posterior Fusion w/o Safety Control"
                )
            ]["score"].iloc[0]
        )
        clean_safe = float(
            frame[
                (frame["condition"] == "clean")
                & (frame["output"] == "Safe Posterior")
            ]["score"].iloc[0]
        )

        posterior_extra = float(posterior["score"] - clean_posterior)
        safe_extra = float(safe["score"] - clean_safe)
        safe_recovery = (
            100.0 * (posterior_extra - safe_extra) / posterior_extra
            if condition != "clean" and posterior_extra > 0
            else np.nan
        )

        rows.append(
            {
                "condition": condition,
                "display_condition": CONDITION_LABELS[condition].replace("\n", " "),
                "posterior_score": float(posterior["score"]),
                "safe_score": float(safe["score"]),
                "full_score": float(full["score"]),
                "full_relative_degradation_pct": float(
                    full["relative_score_degradation_pct"]
                ),
                "posterior_delta_vs_clean": posterior_extra,
                "safe_delta_vs_clean": safe_extra,
                "safe_recovery_pct": safe_recovery,
            }
        )
    return pd.DataFrame(rows)


def plot_safety_compensation(
    data: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
    width: float,
    height: float,
    show_title: bool,
) -> None:
    import matplotlib.pyplot as plt

    apply_paper_style()

    x = np.arange(len(data))
    bar_width = 0.245

    fig, ax = plt.subplots(figsize=(width, height))
    bars_by_output = {}

    for index, output in enumerate(OUTPUT_ORDER):
        column = {
            "Posterior Fusion w/o Safety Control": "posterior_score",
            "Safe Posterior": "safe_score",
            "Full Candidate": "full_score",
        }[output]
        positions = x + (index - 1) * bar_width
        bars = ax.bar(
            positions,
            data[column].to_numpy(),
            width=bar_width,
            label=OUTPUT_LABELS[output],
            edgecolor="black",
            linewidth=0.55,
        )
        bars_by_output[output] = bars

    clean_full = float(
        data.loc[data["condition"] == "clean", "full_score"].iloc[0]
    )
    ax.axhline(
        clean_full,
        linestyle="--",
        linewidth=1.0,
        label="Clean Full Candidate",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS[value] for value in data["condition"]],
    )
    ax.set_ylabel("Score (lower is better)")
    if show_title:
        ax.set_title(
            "Safety Compensation under Relative-Prior Corruption",
            pad=10,
        )

    all_scores = data[
        ["posterior_score", "safe_score", "full_score"]
    ].to_numpy()
    lower = max(0.0, float(np.min(all_scores)) - 0.0012)
    upper = float(np.max(all_scores)) + 0.0022
    ax.set_ylim(lower, upper)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)

    # Annotate only the Full Candidate bars to avoid clutter.
    full_bars = bars_by_output["Full Candidate"]
    degradations = data["full_relative_degradation_pct"].to_numpy()
    offset = (upper - lower) * 0.018
    for bar, degradation in zip(full_bars, degradations):
        if abs(float(degradation)) < 0.005:
            label = "Clean"
        else:
            label = f"{float(degradation):+.1f}%"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8.8,
        )

    handles, labels = ax.get_legend_handles_labels()
    desired = [
        "Posterior w/o Safety",
        "Safe Posterior",
        "Full Candidate",
        "Clean Full Candidate",
    ]
    ordered_handles = [
        handles[labels.index(label)]
        for label in desired
        if label in labels
    ]
    ordered_labels = [
        label
        for label in desired
        if label in labels
    ]
    ax.legend(
        ordered_handles,
        ordered_labels,
        ncol=2,
        loc="upper left",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def create_family_data(frame: pd.DataFrame) -> pd.DataFrame:
    full = frame[frame["output"] == "Full Candidate"].copy()
    required_conditions = [
        condition
        for family in FAMILY_ORDER
        for condition in FAMILY_CONDITIONS[family]
    ]
    missing = [
        condition
        for condition in required_conditions
        if not (full["condition"] == condition).any()
    ]
    if missing:
        raise ValueError(
            "Missing Full Candidate stress conditions: "
            + ", ".join(missing)
        )

    rows: List[Dict[str, float]] = []
    for family in FAMILY_ORDER:
        for order, condition in enumerate(FAMILY_CONDITIONS[family]):
            row = full[full["condition"] == condition].iloc[0]
            rows.append(
                {
                    "family": family,
                    "condition": condition,
                    "order": order,
                    "display_condition": CONDITION_LABELS[condition].replace("\n", " "),
                    "severity": float(row["severity"]),
                    "score": float(row["score"]),
                    "relative_score_degradation_pct": float(
                        row["relative_score_degradation_pct"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_family_sensitivity(
    data: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
    width: float,
    height: float,
    ymin: float,
    ymax: float,
    show_title: bool,
) -> None:
    import matplotlib.pyplot as plt

    apply_paper_style()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(width, height),
        sharey=True,
        constrained_layout=True,
    )

    for ax, family in zip(axes.flat, FAMILY_ORDER):
        subset = (
            data[data["family"] == family]
            .sort_values("order")
            .reset_index(drop=True)
        )
        x = np.arange(len(subset))
        values = subset["relative_score_degradation_pct"].to_numpy()

        if family == "confidence":
            marks = ax.bar(
                x,
                values,
                width=0.62,
                edgecolor="black",
                linewidth=0.6,
            )
        else:
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=1.8,
            )
            marks = None

        ax.axhline(0.0, linewidth=0.9, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CONDITION_LABELS[value] for value in subset["condition"]],
        )
        ax.set_title(FAMILY_TITLES[family], pad=7)
        ax.set_ylim(float(ymin), float(ymax))
        ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.32)
        ax.set_axisbelow(True)

        span = float(ymax - ymin)
        for index, value in enumerate(values):
            offset = 0.025 * span
            y = float(value) + offset if value >= 0 else float(value) - offset
            va = "bottom" if value >= 0 else "top"
            ax.text(
                index,
                y,
                f"{float(value):+.1f}%",
                ha="center",
                va=va,
                fontsize=8.2,
            )
            if marks is not None and value < 0:
                marks[index].set_hatch("///")

    axes[0, 0].set_ylabel("Relative Score degradation (%)")
    axes[1, 0].set_ylabel("Relative Score degradation (%)")

    if show_title:
        fig.suptitle(
            "Full-Candidate Sensitivity to Relative-Prior Perturbations",
            fontsize=13,
        )

    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else csv_path.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_results(csv_path)

    compensation_data = create_compensation_data(frame)
    compensation_csv = (
        out_dir / "relative_prior_safety_compensation_paper_data.csv"
    )
    compensation_data.to_csv(
        compensation_csv,
        index=False,
        encoding="utf-8-sig",
    )

    compensation_png = (
        out_dir / "relative_prior_safety_compensation_paper.png"
    )
    compensation_pdf = (
        out_dir / "relative_prior_safety_compensation_paper.pdf"
    )
    plot_safety_compensation(
        compensation_data,
        compensation_png,
        compensation_pdf,
        dpi=args.dpi,
        width=args.compensation_width,
        height=args.compensation_height,
        show_title=not args.no_title,
    )

    family_data = create_family_data(frame)
    family_csv = (
        out_dir / "relative_prior_family_sensitivity_paper_data.csv"
    )
    family_data.to_csv(
        family_csv,
        index=False,
        encoding="utf-8-sig",
    )

    family_png = (
        out_dir / "relative_prior_family_sensitivity_paper.png"
    )
    family_pdf = (
        out_dir / "relative_prior_family_sensitivity_paper.pdf"
    )
    plot_family_sensitivity(
        family_data,
        family_png,
        family_pdf,
        dpi=args.dpi,
        width=args.family_width,
        height=args.family_height,
        ymin=args.family_ymin,
        ymax=args.family_ymax,
        show_title=not args.no_title,
    )

    print("=" * 104)
    print("Paper-ready relative-prior stress figures saved")
    print("=" * 104)
    print(f"Input CSV          : {csv_path}")
    print(f"Output directory   : {out_dir}")
    print(f"Compensation PNG   : {compensation_png}")
    print(f"Compensation PDF   : {compensation_pdf}")
    print(f"Compensation data  : {compensation_csv}")
    print(f"Family PNG         : {family_png}")
    print(f"Family PDF         : {family_pdf}")
    print(f"Family data        : {family_csv}")
    print()
    print("Representative-condition table")
    print(compensation_data.to_string(index=False))
    print()
    print("Family sensitivity table")
    print(family_data.to_string(index=False))


if __name__ == "__main__":
    main()
