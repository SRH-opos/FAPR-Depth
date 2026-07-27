#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Region- and failure-conditioned depth analysis for FAPR-Depth v6.

The script compares:
- Raw Depth
- Backbone Baseline
- Posterior Fusion w/o Safety Control
- Safe Posterior
- Full Candidate

across transparent pixels, inferred ground-truth sensor failure classes,
boundary/interior regions, and the hardest 20% of backbone pixels.

Outputs
-------
- failure_region_metrics.csv
- failure_region_gain_vs_backbone.csv
- failure_region_metrics.json
- region_rmse.png
- region_mae.png
- full_gain_vs_backbone.png
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from fapr_analysis_common import (
    PixelMetricAccumulator,
    add_common_args,
    bootstrap,
    failure_labels_and_regions,
    iter_forward,
    output_predictions,
    write_csv,
    write_json,
    write_run_manifest,
)


METHOD_ORDER = [
    "Raw Depth",
    "Backbone Baseline",
    "Posterior Fusion w/o Safety Control",
    "Safe Posterior",
    "Full Candidate",
]

REGION_ORDER = [
    "all_transparent",
    "valid_state",
    "any_failure",
    "missing_failure",
    "biased_failure",
    "boundary_failure",
    "boundary_ring",
    "interior",
    "raw_failure_threshold",
    "hard_backbone_top20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FAPR outputs in failure and difficulty regions."
    )
    add_common_args(parser, "02_failure_region_analysis", default_phase="joint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)

    accumulators: Dict[Tuple[str, str], PixelMetricAccumulator] = {}
    for method in METHOD_ORDER:
        for region in REGION_ORDER:
            accumulators[(method, region)] = PixelMetricAccumulator()

    progress = tqdm(
        iter_forward(ctx),
        total=len(ctx.shards),
        desc="Failure-region metrics",
        dynamic_ncols=True,
    )
    for inp, out, _ in progress:
        _, regions = failure_labels_and_regions(ctx.train_mod, inp, out)
        predictions = output_predictions(inp, out)
        for method in METHOD_ORDER:
            pred = predictions[method]
            for region in REGION_ORDER:
                accumulators[(method, region)].update(
                    pred,
                    inp["gt"],
                    regions[region],
                    min_depth=float(ctx.train_mod.MIN_DEPTH),
                )

    rows: List[Dict] = []
    by_key: Dict[Tuple[str, str], Dict] = {}
    for region in REGION_ORDER:
        for method in METHOD_ORDER:
            metrics = accumulators[(method, region)].result()
            row = {"region": region, "method": method, **metrics}
            rows.append(row)
            by_key[(method, region)] = row

    gain_rows: List[Dict] = []
    for region in REGION_ORDER:
        baseline = by_key[("Backbone Baseline", region)]
        for method in METHOD_ORDER:
            row = by_key[(method, region)]
            gain_rows.append(
                {
                    "region": region,
                    "method": method,
                    "pixels": row["pixels"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "mae_gain_vs_backbone": (
                        baseline["mae"] - row["mae"]
                        if np.isfinite(baseline["mae"]) and np.isfinite(row["mae"])
                        else float("nan")
                    ),
                    "rmse_gain_vs_backbone": (
                        baseline["rmse"] - row["rmse"]
                        if np.isfinite(baseline["rmse"]) and np.isfinite(row["rmse"])
                        else float("nan")
                    ),
                    "mae_relative_improvement_pct": (
                        100.0 * (baseline["mae"] - row["mae"]) / baseline["mae"]
                        if np.isfinite(baseline["mae"])
                        and baseline["mae"] > 0
                        and np.isfinite(row["mae"])
                        else float("nan")
                    ),
                    "rmse_relative_improvement_pct": (
                        100.0 * (baseline["rmse"] - row["rmse"]) / baseline["rmse"]
                        if np.isfinite(baseline["rmse"])
                        and baseline["rmse"] > 0
                        and np.isfinite(row["rmse"])
                        else float("nan")
                    ),
                }
            )

    write_csv(out_dir / "failure_region_metrics.csv", rows)
    write_csv(out_dir / "failure_region_gain_vs_backbone.csv", gain_rows)
    write_json(
        out_dir / "failure_region_metrics.json",
        {
            "method_order": METHOD_ORDER,
            "region_order": REGION_ORDER,
            "metrics": rows,
            "gain_vs_backbone": gain_rows,
        },
    )
    write_run_manifest(ctx, {"analysis": "failure_region_analysis"})

    frame = pd.DataFrame(rows)
    gain_frame = pd.DataFrame(gain_rows)
    plot_grouped_metric(frame, "rmse", "Region RMSE", out_dir / "region_rmse.png")
    plot_grouped_metric(frame, "mae", "Region MAE", out_dir / "region_mae.png")

    full_gain = gain_frame[gain_frame["method"] == "Full Candidate"].copy()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = np.arange(len(REGION_ORDER))
    values = [
        float(
            full_gain.loc[
                full_gain["region"] == region,
                "rmse_relative_improvement_pct",
            ].iloc[0]
        )
        for region in REGION_ORDER
    ]
    ax.bar(x, values)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER, rotation=35, ha="right")
    ax.set_ylabel("RMSE improvement over Backbone Baseline (%)")
    ax.set_title("Full Candidate improvement by failure/difficulty region")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "full_gain_vs_backbone.png", dpi=220)
    plt.close(fig)

    print("\nSaved region analysis to:", out_dir)
    print(
        frame[
            (frame["method"].isin(["Backbone Baseline", "Full Candidate"]))
            & (frame["region"].isin(REGION_ORDER))
        ][["region", "method", "pixels", "rmse", "mae", "rel"]].to_string(index=False)
    )


def plot_grouped_metric(
    frame: pd.DataFrame,
    metric: str,
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    width = 0.16
    x = np.arange(len(REGION_ORDER))
    fig, ax = plt.subplots(figsize=(15, 6.5))
    for index, method in enumerate(METHOD_ORDER):
        values = []
        for region in REGION_ORDER:
            selected = frame[
                (frame["region"] == region) & (frame["method"] == method)
            ]
            values.append(
                float(selected.iloc[0][metric]) if len(selected) else float("nan")
            )
        ax.bar(
            x + (index - (len(METHOD_ORDER) - 1) / 2.0) * width,
            values,
            width=width,
            label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER, rotation=35, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
