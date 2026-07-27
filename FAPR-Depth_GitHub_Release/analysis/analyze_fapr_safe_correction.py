#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Risk-controlled safe-fusion analysis for FAPR-Depth v6.

The script measures whether the anchor-to-posterior update is helpful or
harmful, conditioned on failure/difficulty regions and safe-gate bins.

Outputs
-------
- safe_region_outcomes.csv
- safe_gate_bins.csv
- safe_summary.json
- safe_region_improve_damage.png
- safe_gate_calibration.png
- safe_gain_by_gate.png
- qualitative/safe_improve_*.png
- qualitative/safe_damage_*.png

Definitions
-----------
true safe gain = |anchor - GT| - |safe posterior - GT|
positive gain  = the safe update improves the pixel
negative gain  = the safe update damages the pixel

Improve/damage ratios are reported among pixels with a non-negligible safe
update.  The script also reports statistics over all pixels in each region.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from fapr_analysis_common import (
    Reservoir,
    add_common_args,
    bootstrap,
    chw_rgb,
    failure_labels_and_regions,
    iter_forward,
    map2d,
    select_top_records,
    write_csv,
    write_json,
    write_run_manifest,
)


REGION_ORDER = [
    "all_transparent",
    "valid_state",
    "any_failure",
    "missing_failure",
    "biased_failure",
    "boundary_failure",
    "boundary_ring",
    "interior",
    "hard_backbone_top20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze safe fusion improvement, damage, gate and oracle gap."
    )
    add_common_args(parser, "04_safe_correction", default_phase="joint")
    parser.add_argument("--gate-bins", type=int, default=12)
    parser.add_argument("--update-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--neutral-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--num-visualizations", type=int, default=6)
    parser.add_argument("--reservoir-pixels", type=int, default=1_500_000)
    return parser.parse_args()


class OutcomeAccumulator:
    def __init__(self) -> None:
        self.pixels = 0
        self.updated = 0
        self.improved = 0
        self.damaged = 0
        self.neutral = 0
        self.sum_gain_all = 0.0
        self.sum_gain_updated = 0.0
        self.sum_abs_update_all = 0.0
        self.sum_abs_update_updated = 0.0
        self.sum_gate = 0.0
        self.sum_support = 0.0
        self.sum_oracle_gain = 0.0
        self.sum_anchor_error = 0.0
        self.sum_legacy_error = 0.0
        self.sum_safe_error = 0.0

    @torch.no_grad()
    def update(
        self,
        region: torch.Tensor,
        anchor_error: torch.Tensor,
        legacy_error: torch.Tensor,
        safe_error: torch.Tensor,
        safe_update: torch.Tensor,
        gate: torch.Tensor,
        support: torch.Tensor,
        update_epsilon: float,
        neutral_epsilon: float,
    ) -> None:
        mask = region > 0.5
        n = int(mask.sum().item())
        if n <= 0:
            return
        gain = anchor_error - safe_error
        abs_update = torch.abs(safe_update)
        updated = mask & (abs_update > float(update_epsilon))
        improve = updated & (gain > float(neutral_epsilon))
        damage = updated & (gain < -float(neutral_epsilon))
        neutral = updated & ~(improve | damage)
        oracle_gain = anchor_error - torch.minimum(anchor_error, legacy_error)

        self.pixels += n
        self.updated += int(updated.sum().item())
        self.improved += int(improve.sum().item())
        self.damaged += int(damage.sum().item())
        self.neutral += int(neutral.sum().item())
        self.sum_gain_all += float(gain[mask].sum().item())
        self.sum_abs_update_all += float(abs_update[mask].sum().item())
        self.sum_gate += float(gate[mask].sum().item())
        self.sum_support += float(support[mask].sum().item())
        self.sum_oracle_gain += float(oracle_gain[mask].sum().item())
        self.sum_anchor_error += float(anchor_error[mask].sum().item())
        self.sum_legacy_error += float(legacy_error[mask].sum().item())
        self.sum_safe_error += float(safe_error[mask].sum().item())
        if self.updated > 0 and int(updated.sum().item()) > 0:
            self.sum_gain_updated += float(gain[updated].sum().item())
            self.sum_abs_update_updated += float(abs_update[updated].sum().item())

    def result(self, name: str) -> Dict[str, float]:
        return {
            "region": name,
            "pixels": self.pixels,
            "updated_pixels": self.updated,
            "update_coverage": self.updated / self.pixels if self.pixels else float("nan"),
            "improve_ratio_updated": (
                self.improved / self.updated if self.updated else float("nan")
            ),
            "damage_ratio_updated": (
                self.damaged / self.updated if self.updated else float("nan")
            ),
            "neutral_ratio_updated": (
                self.neutral / self.updated if self.updated else float("nan")
            ),
            "mean_true_gain_all": (
                self.sum_gain_all / self.pixels if self.pixels else float("nan")
            ),
            "mean_true_gain_updated": (
                self.sum_gain_updated / self.updated if self.updated else float("nan")
            ),
            "mean_abs_update_all": (
                self.sum_abs_update_all / self.pixels if self.pixels else float("nan")
            ),
            "mean_abs_update_updated": (
                self.sum_abs_update_updated / self.updated if self.updated else float("nan")
            ),
            "mean_gate": self.sum_gate / self.pixels if self.pixels else float("nan"),
            "mean_support": (
                self.sum_support / self.pixels if self.pixels else float("nan")
            ),
            "mean_oracle_anchor_posterior_gain": (
                self.sum_oracle_gain / self.pixels if self.pixels else float("nan")
            ),
            "anchor_mae": (
                self.sum_anchor_error / self.pixels if self.pixels else float("nan")
            ),
            "legacy_mae": (
                self.sum_legacy_error / self.pixels if self.pixels else float("nan")
            ),
            "safe_mae": (
                self.sum_safe_error / self.pixels if self.pixels else float("nan")
            ),
        }


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)
    qualitative_dir = out_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    stats = {name: OutcomeAccumulator() for name in REGION_ORDER}
    reservoir = Reservoir(
        args.reservoir_pixels,
        columns=[
            "gate",
            "support",
            "legacy_better",
            "safe_gain",
            "legacy_gain",
            "abs_update",
            "anchor_error",
            "legacy_error",
            "safe_error",
        ],
        seed=args.seed + 211,
    )
    top_improve: List[Dict[str, Any]] = []
    top_damage: List[Dict[str, Any]] = []

    progress = tqdm(
        iter_forward(ctx),
        total=len(ctx.shards),
        desc="Safe correction",
        dynamic_ncols=True,
    )
    for inp, out, meta in progress:
        _, regions = failure_labels_and_regions(ctx.train_mod, inp, out)
        anchor_error = torch.abs(out["anchor_depth"] - inp["gt"])
        legacy_error = torch.abs(out["legacy_fused"] - inp["gt"])
        safe_error = torch.abs(out["safe_posterior"] - inp["gt"])
        safe_gain = anchor_error - safe_error
        legacy_gain = anchor_error - legacy_error

        for name in REGION_ORDER:
            stats[name].update(
                regions[name],
                anchor_error,
                legacy_error,
                safe_error,
                out["safe_update"],
                out["safe_gate"],
                out["safe_support"],
                args.update_epsilon,
                args.neutral_epsilon,
            )

        region = regions["all_transparent"] > 0.5
        reservoir.add_arrays(
            gate=out["safe_gate"][region].detach().float().cpu().numpy(),
            support=out["safe_support"][region].detach().float().cpu().numpy(),
            legacy_better=(legacy_gain[region] > 0).detach().float().cpu().numpy(),
            safe_gain=safe_gain[region].detach().float().cpu().numpy(),
            legacy_gain=legacy_gain[region].detach().float().cpu().numpy(),
            abs_update=torch.abs(out["safe_update"])[region].detach().float().cpu().numpy(),
            anchor_error=anchor_error[region].detach().float().cpu().numpy(),
            legacy_error=legacy_error[region].detach().float().cpu().numpy(),
            safe_error=safe_error[region].detach().float().cpu().numpy(),
        )

        for bi in range(inp["rgb"].shape[0]):
            sample_region = regions["all_transparent"][bi : bi + 1] > 0.5
            count = int(sample_region.sum().item())
            if count <= 0:
                continue
            sample_gain = float(
                safe_gain[bi : bi + 1][sample_region].mean().item()
            )
            sample_update = float(
                torch.abs(out["safe_update"][bi : bi + 1])[sample_region]
                .mean()
                .item()
            )
            record = {
                "sample_gain": sample_gain,
                "mean_abs_update": sample_update,
                "source_shard": meta["source_shard"],
                "rgb": chw_rgb(inp["rgb"], bi),
                "gt": map2d(inp["gt"], bi),
                "anchor": map2d(out["anchor_depth"], bi),
                "legacy": map2d(out["legacy_fused"], bi),
                "safe": map2d(out["safe_posterior"], bi),
                "anchor_error": map2d(anchor_error, bi),
                "legacy_error": map2d(legacy_error, bi),
                "safe_error": map2d(safe_error, bi),
                "gate": map2d(out["safe_gate"], bi),
                "support": map2d(out["safe_support"], bi),
                "update": map2d(out["safe_update"], bi),
                "gain": map2d(safe_gain, bi),
            }
            select_top_records(
                top_improve,
                record.copy(),
                key="sample_gain",
                k=args.num_visualizations,
                largest=True,
            )
            select_top_records(
                top_damage,
                record.copy(),
                key="sample_gain",
                k=args.num_visualizations,
                largest=False,
            )

    rows = [stats[name].result(name) for name in REGION_ORDER]
    frame = reservoir.frame()
    gate_rows = make_gate_bins(frame, args.gate_bins)

    summary = {
        "evaluation_phase": ctx.phase,
        "update_epsilon": args.update_epsilon,
        "neutral_epsilon": args.neutral_epsilon,
        "region_outcomes": rows,
        "gate_bins": gate_rows,
        "sampled_pixels": int(len(frame)),
    }
    write_csv(out_dir / "safe_region_outcomes.csv", rows)
    write_csv(out_dir / "safe_gate_bins.csv", gate_rows)
    write_json(out_dir / "safe_summary.json", summary)
    write_run_manifest(ctx, {"analysis": "safe_correction"})

    plot_region_outcomes(rows, out_dir / "safe_region_improve_damage.png")
    plot_gate_calibration(gate_rows, out_dir / "safe_gate_calibration.png")
    plot_gain_by_gate(gate_rows, out_dir / "safe_gain_by_gate.png")

    for index, record in enumerate(top_improve, start=1):
        save_qualitative(
            record,
            qualitative_dir / f"safe_improve_{index:02d}.png",
            "Strong safe-fusion improvement",
        )
    for index, record in enumerate(top_damage, start=1):
        save_qualitative(
            record,
            qualitative_dir / f"safe_damage_{index:02d}.png",
            "Strong safe-fusion damage",
        )

    print("\nSafe correction region summary")
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))
    print("Saved to:", out_dir)


def make_gate_bins(frame, bins: int) -> List[Dict[str, float]]:
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: List[Dict[str, float]] = []
    gate = frame["gate"].to_numpy()
    for index in range(int(bins)):
        lo, hi = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (gate >= lo) & (gate <= hi)
        else:
            mask = (gate >= lo) & (gate < hi)
        subset = frame.loc[mask]
        rows.append(
            {
                "bin": index,
                "lower": lo,
                "upper": hi,
                "pixels": int(len(subset)),
                "mean_gate": float(subset["gate"].mean()) if len(subset) else float("nan"),
                "legacy_better_frequency": (
                    float(subset["legacy_better"].mean())
                    if len(subset)
                    else float("nan")
                ),
                "mean_safe_gain": (
                    float(subset["safe_gain"].mean()) if len(subset) else float("nan")
                ),
                "mean_legacy_gain": (
                    float(subset["legacy_gain"].mean()) if len(subset) else float("nan")
                ),
                "mean_abs_update": (
                    float(subset["abs_update"].mean()) if len(subset) else float("nan")
                ),
            }
        )
    return rows


def plot_region_outcomes(rows: List[Dict[str, float]], path: Path) -> None:
    import matplotlib.pyplot as plt

    names = [row["region"] for row in rows]
    improve = [row["improve_ratio_updated"] for row in rows]
    damage = [row["damage_ratio_updated"] for row in rows]
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.bar(x - width / 2, improve, width=width, label="Improve ratio")
    ax.bar(x + width / 2, damage, width=width, label="Damage ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Ratio among updated pixels")
    ax.set_title("Safe-fusion outcomes by region")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_gate_calibration(rows: List[Dict[str, float]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(rows)
    valid = frame["pixels"] > 0
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ax.plot(
        frame.loc[valid, "mean_gate"],
        frame.loc[valid, "legacy_better_frequency"],
        marker="o",
        label="Observed",
    )
    ax.set_xlabel("Mean safe gate")
    ax.set_ylabel("Frequency that posterior beats anchor")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Safe-gate calibration")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_gain_by_gate(rows: List[Dict[str, float]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(rows)
    valid = frame["pixels"] > 0
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(
        frame.loc[valid, "mean_gate"],
        frame.loc[valid, "mean_safe_gain"],
        marker="o",
        label="Realized safe gain",
    )
    ax.plot(
        frame.loc[valid, "mean_gate"],
        frame.loc[valid, "mean_legacy_gain"],
        marker="s",
        label="Unconstrained posterior gain",
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Mean safe gate")
    ax.set_ylabel("Mean error reduction (m)")
    ax.set_title("Realized gain by safe-gate bin")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_qualitative(record: Dict[str, Any], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    gt = record["gt"]
    values = gt[np.isfinite(gt) & (gt > 0)]
    vmin = float(np.percentile(values, 2)) if values.size else 0.0
    vmax = float(np.percentile(values, 98)) if values.size else 1.0
    max_error = float(
        np.percentile(
            np.concatenate(
                [
                    record["anchor_error"].reshape(-1),
                    record["legacy_error"].reshape(-1),
                    record["safe_error"].reshape(-1),
                ]
            ),
            98,
        )
    )
    max_gain = float(np.percentile(np.abs(record["gain"]), 98))

    panels = [
        (record["rgb"], "RGB", None, None, None),
        (record["gt"], "Ground truth", "viridis", vmin, vmax),
        (record["anchor"], "Backbone Baseline", "viridis", vmin, vmax),
        (record["legacy"], "Unconstrained posterior", "viridis", vmin, vmax),
        (record["safe"], "Safe posterior", "viridis", vmin, vmax),
        (record["anchor_error"], "Backbone error", "magma", 0.0, max_error),
        (record["legacy_error"], "Posterior error", "magma", 0.0, max_error),
        (record["safe_error"], "Safe error", "magma", 0.0, max_error),
        (record["gate"], "Safe gate", "viridis", 0.0, 1.0),
        (record["support"], "Safe support", "viridis", 0.0, 1.0),
        (record["update"], "Applied safe update", "coolwarm", -0.08, 0.08),
        (record["gain"], "True safe gain", "coolwarm", -max_gain, max_gain),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    for ax, (image, name, cmap, lo, hi) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(
        f"{title}: mean gain={record['sample_gain']:.6f}, "
        f"mean |update|={record['mean_abs_update']:.6f}\n"
        f"{record['source_shard']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
