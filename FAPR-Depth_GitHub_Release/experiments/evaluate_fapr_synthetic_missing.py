#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Controlled Synthetic Missing Failure evaluation for FAPR-Depth v6.

Why this experiment
-------------------
The natural test split contains no supported Missing-failure pixels. This
script creates deterministic missing-depth masks at test time by setting
selected raw-depth pixels to zero while keeping ground-truth validity intact.

It evaluates:
- overall transparent-region performance;
- synthetic-missing-region performance;
- Missing-state precision / recall / F1;
- mean P(Missing);
- routing mass assigned to the Missing Expert;
- Raw / Relative / Expert source allocation in the missing region.

No retraining is required.

Profiles
--------
quick:
    clean, random_25, random_50, block_25, boundary_50

paper:
    clean,
    random_10/25/50/75/100,
    block_25/50,
    boundary_50/100

Outputs
-------
synthetic_missing_results.csv
synthetic_missing_results.json
synthetic_missing_performance.png
synthetic_missing_detection_routing.png
synthetic_missing_qualitative.png
synthetic_missing_qualitative.pdf
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from fapr_analysis_common import (
    OfficialMetricAccumulator,
    PixelMetricAccumulator,
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


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FAPR v6 on controlled synthetic missing-depth failures."
    )
    add_common_args(parser, "08_synthetic_missing", default_phase="joint")
    parser.add_argument(
        "--profile",
        choices=["quick", "paper"],
        default="paper",
    )
    parser.add_argument(
        "--block-kernel",
        type=int,
        default=41,
        help="Smoothing kernel used to create spatially contiguous missing blocks.",
    )
    parser.add_argument(
        "--qualitative-condition",
        type=str,
        default="block_50",
        help="Condition used for the qualitative example.",
    )
    parser.add_argument("--paper-dpi", type=int, default=600)
    return parser.parse_args()


def build_conditions(profile: str) -> List[Condition]:
    if profile == "quick":
        return [
            Condition("clean", "clean", 0.0),
            Condition("random_25", "random", 0.25),
            Condition("random_50", "random", 0.50),
            Condition("block_25", "block", 0.25),
            Condition("boundary_50", "boundary", 0.50),
        ]

    return [
        Condition("clean", "clean", 0.0),
        Condition("random_10", "random", 0.10),
        Condition("random_25", "random", 0.25),
        Condition("random_50", "random", 0.50),
        Condition("random_75", "random", 0.75),
        Condition("random_100", "random", 1.00),
        Condition("block_25", "block", 0.25),
        Condition("block_50", "block", 0.50),
        Condition("boundary_50", "boundary", 0.50),
        Condition("boundary_100", "boundary", 1.00),
    ]


def stable_seed(
    base_seed: int,
    condition: str,
    shard_index: int,
    sample_offset: int,
) -> int:
    digest = hashlib.sha1(
        f"{base_seed}|{condition}|{shard_index}|{sample_offset}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def exact_ratio_mask(
    score: torch.Tensor,
    eligible: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    """Select approximately ratio of eligible pixels independently per image."""
    result = torch.zeros_like(eligible, dtype=torch.bool)
    ratio = float(np.clip(ratio, 0.0, 1.0))

    for bi in range(score.shape[0]):
        region = eligible[bi]
        count = int(region.sum().item())
        if count <= 0 or ratio <= 0:
            continue
        if ratio >= 1.0:
            result[bi] = region
            continue

        values = score[bi][region]
        keep = max(1, int(round(ratio * count)))
        keep = min(keep, count)

        # top-k avoids quantile ties creating too many selected pixels.
        _, indices = torch.topk(values.reshape(-1), k=keep, largest=True)
        flat_region_indices = torch.nonzero(
            region.reshape(-1),
            as_tuple=False,
        ).reshape(-1)
        selected_flat = flat_region_indices[indices]

        flat_result = result[bi].reshape(-1)
        flat_result[selected_flat] = True

    return result


def make_missing_mask(
    inp: Mapping[str, torch.Tensor],
    condition: Condition,
    seed: int,
    block_kernel: int,
) -> torch.Tensor:
    base = (
        (inp["mask"] > 0.5)
        & (inp["valid"] > 0.5)
    )

    if condition.family == "clean":
        return torch.zeros_like(base, dtype=torch.bool)

    generator = torch.Generator(device=inp["raw"].device)
    generator.manual_seed(int(seed))

    random_score = torch.rand(
        inp["raw"].shape,
        generator=generator,
        device=inp["raw"].device,
        dtype=inp["raw"].dtype,
    )

    if condition.family == "random":
        return exact_ratio_mask(
            random_score,
            base,
            condition.ratio,
        )

    if condition.family == "block":
        kernel = max(3, int(block_kernel))
        if kernel % 2 == 0:
            kernel += 1
        smooth = F.avg_pool2d(
            random_score,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        smooth = F.avg_pool2d(
            smooth,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        return exact_ratio_mask(
            smooth,
            base,
            condition.ratio,
        )

    if condition.family == "boundary":
        boundary_region = base & (inp["boundary"] > 0.15)
        return exact_ratio_mask(
            random_score,
            boundary_region,
            condition.ratio,
        )

    raise ValueError(f"Unknown missing family: {condition.family}")


def make_transform(
    ctx,
    condition: Condition,
    block_kernel: int,
):
    def transform(
        inp: Dict[str, torch.Tensor],
        shard_index: int,
        sample_offset: int,
    ) -> Dict[str, torch.Tensor]:
        seed = stable_seed(
            int(ctx.args.seed),
            condition.name,
            shard_index,
            sample_offset,
        )
        missing_mask = make_missing_mask(
            inp,
            condition,
            seed,
            block_kernel,
        )

        out = dict(inp)
        raw = inp["raw"].clone()
        raw[missing_mask] = 0.0
        out["raw"] = raw
        out["_synthetic_missing_mask"] = missing_mask.float()
        return out

    return transform


class MissingDetectionAccumulator:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.support = 0
        self.sum_p_missing = 0.0
        self.sum_route_missing = 0.0
        self.sum_source_raw = 0.0
        self.sum_source_relative = 0.0
        self.sum_source_expert = 0.0

    @torch.no_grad()
    def update(
        self,
        inp: Mapping[str, torch.Tensor],
        out: Mapping[str, torch.Tensor],
    ) -> None:
        target = inp["_synthetic_missing_mask"] > 0.5
        transparent = (
            (inp["mask"] > 0.5)
            & (inp["valid"] > 0.5)
        )
        predicted = (
            torch.argmax(
                out["fail_prob"],
                dim=1,
                keepdim=True,
            )
            == 1
        )

        self.tp += int((predicted & target).sum().item())
        self.fp += int((predicted & transparent & (~target)).sum().item())
        self.fn += int(((~predicted) & target).sum().item())

        count = int(target.sum().item())
        self.support += count
        if count <= 0:
            return

        self.sum_p_missing += float(
            out["fail_prob"][:, 1:2][target].sum().item()
        )
        self.sum_route_missing += float(
            out["pi"][:, 0:1][target].sum().item()
        )
        self.sum_source_raw += float(
            out["alpha"][:, 0:1][target].sum().item()
        )
        self.sum_source_relative += float(
            out["alpha"][:, 1:2][target].sum().item()
        )
        self.sum_source_expert += float(
            out["alpha"][:, 2:3][target].sum().item()
        )

    def result(self) -> Dict[str, float]:
        precision = (
            self.tp / (self.tp + self.fp)
            if self.tp + self.fp > 0
            else float("nan")
        )
        recall = (
            self.tp / (self.tp + self.fn)
            if self.tp + self.fn > 0
            else float("nan")
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision)
            and np.isfinite(recall)
            and precision + recall > 0
            else float("nan")
        )

        return {
            "missing_support_pixels": int(self.support),
            "missing_precision": precision,
            "missing_recall": recall,
            "missing_f1": f1,
            "mean_p_missing": (
                self.sum_p_missing / self.support
                if self.support > 0
                else float("nan")
            ),
            "mean_route_missing_expert": (
                self.sum_route_missing / self.support
                if self.support > 0
                else float("nan")
            ),
            "mean_source_raw": (
                self.sum_source_raw / self.support
                if self.support > 0
                else float("nan")
            ),
            "mean_source_relative": (
                self.sum_source_relative / self.support
                if self.support > 0
                else float("nan")
            ),
            "mean_source_expert": (
                self.sum_source_expert / self.support
                if self.support > 0
                else float("nan")
            ),
        }


def evaluate_condition(
    ctx,
    condition: Condition,
    block_kernel: int,
    qualitative_condition: str,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    official = {
        "Backbone Baseline": OfficialMetricAccumulator(ctx.train_mod),
        "Safe Posterior": OfficialMetricAccumulator(ctx.train_mod),
        "Full Candidate": OfficialMetricAccumulator(ctx.train_mod),
    }
    regional = {
        "Backbone Baseline": PixelMetricAccumulator(),
        "Safe Posterior": PixelMetricAccumulator(),
        "Full Candidate": PixelMetricAccumulator(),
    }
    detection = MissingDetectionAccumulator()
    top_qualitative: List[Dict[str, Any]] = []

    transform = make_transform(
        ctx,
        condition,
        block_kernel,
    )

    progress = tqdm(
        iter_forward(ctx, input_transform=transform),
        total=len(ctx.shards),
        desc=f"Synthetic missing: {condition.name}",
        dynamic_ncols=True,
        leave=False,
    )

    for inp, out, meta in progress:
        raw = inp["raw"]
        gt = inp["gt"]
        mask = inp["mask"]
        valid = inp["valid"]
        missing_mask = inp["_synthetic_missing_mask"]

        predictions = {
            "Backbone Baseline": (
                out["anchor_depth"] * mask
                + raw * (1.0 - mask)
            ),
            "Safe Posterior": out["safe_benchmark"],
            "Full Candidate": out["candidate_benchmark"],
        }

        for method, prediction in predictions.items():
            official[method].update(
                prediction,
                raw,
                gt,
                mask,
                valid,
            )
            regional[method].update(
                prediction,
                gt,
                missing_mask,
                min_depth=float(ctx.train_mod.MIN_DEPTH),
            )

        detection.update(inp, out)

        if (
            condition.name == qualitative_condition
            and int(missing_mask.sum().item()) > 0
        ):
            for bi in range(inp["rgb"].shape[0]):
                region = missing_mask[bi : bi + 1] > 0.5
                count = int(region.sum().item())
                if count <= 0:
                    continue

                anchor_error = torch.abs(
                    out["anchor_depth"][bi : bi + 1]
                    - gt[bi : bi + 1]
                )
                full_error = torch.abs(
                    out["candidate"][bi : bi + 1]
                    - gt[bi : bi + 1]
                )
                gain = float(
                    (
                        anchor_error[region]
                        - full_error[region]
                    ).mean().item()
                )

                record = {
                    "gain": gain,
                    "source_shard": str(meta["source_shard"]),
                    "shard_index": int(meta["shard_index"]),
                    "sample_in_shard": int(meta["sample_offset"] + bi),
                    "rgb": chw_rgb(inp["rgb"], bi),
                    "corrupted_raw": map2d(raw, bi),
                    "gt": map2d(gt, bi),
                    "missing_mask": map2d(missing_mask, bi),
                    "p_missing": map2d(
                        out["fail_prob"][:, 1:2],
                        bi,
                    ),
                    "route_missing": map2d(
                        out["pi"][:, 0:1],
                        bi,
                    ),
                    "source_expert": map2d(
                        out["alpha"][:, 2:3],
                        bi,
                    ),
                    "anchor": map2d(out["anchor_depth"], bi),
                    "full": map2d(out["candidate"], bi),
                    "anchor_error": map2d(anchor_error, 0),
                    "full_error": map2d(full_error, 0),
                }
                select_top_records(
                    top_qualitative,
                    record,
                    key="gain",
                    k=1,
                    largest=True,
                )

    rows: List[Dict[str, Any]] = []
    detection_result = detection.result()

    for method in (
        "Backbone Baseline",
        "Safe Posterior",
        "Full Candidate",
    ):
        whole = official[method].result()
        missing = regional[method].result()
        rows.append(
            {
                "condition": condition.name,
                "family": condition.family,
                "missing_ratio": condition.ratio,
                "method": method,
                "overall_rmse": whole.get("rmse_mask", float("nan")),
                "overall_rel": whole.get("rel_mask", float("nan")),
                "overall_mae": whole.get("mae_mask", float("nan")),
                "overall_score": whole.get("score", float("nan")),
                "missing_region_pixels": missing.get("pixels", 0),
                "missing_region_rmse": missing.get("rmse", float("nan")),
                "missing_region_rel": missing.get("rel", float("nan")),
                "missing_region_mae": missing.get("mae", float("nan")),
                "missing_region_delta_105": missing.get(
                    "delta_105",
                    float("nan"),
                ),
                **detection_result,
            }
        )

    qualitative = top_qualitative[0] if top_qualitative else None
    return rows, qualitative


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(args.profile)
    condition_names = {condition.name for condition in conditions}
    if args.qualitative_condition not in condition_names:
        print(
            f"[WARNING] qualitative condition {args.qualitative_condition!r} "
            f"is not in profile={args.profile}; no qualitative figure will be saved."
        )

    all_rows: List[Dict[str, Any]] = []
    qualitative_record: Optional[Dict[str, Any]] = None

    for condition in conditions:
        rows, record = evaluate_condition(
            ctx,
            condition,
            block_kernel=args.block_kernel,
            qualitative_condition=args.qualitative_condition,
        )
        all_rows.extend(rows)
        if record is not None:
            qualitative_record = record

    # Add relative Score change from clean for each method.
    clean_score = {
        row["method"]: row["overall_score"]
        for row in all_rows
        if row["condition"] == "clean"
    }
    for row in all_rows:
        baseline = clean_score.get(row["method"], float("nan"))
        row["relative_score_change_vs_clean_pct"] = (
            100.0 * (row["overall_score"] - baseline) / baseline
            if np.isfinite(baseline) and baseline > 0
            else float("nan")
        )

    write_csv(out_dir / "synthetic_missing_results.csv", all_rows)
    write_json(
        out_dir / "synthetic_missing_results.json",
        {
            "profile": args.profile,
            "conditions": [condition.__dict__ for condition in conditions],
            "rows": all_rows,
        },
    )
    write_run_manifest(
        ctx,
        {
            "analysis": "synthetic_missing",
            "profile": args.profile,
            "conditions": [condition.__dict__ for condition in conditions],
        },
    )

    plot_performance(
        all_rows,
        out_dir / "synthetic_missing_performance.png",
        out_dir / "synthetic_missing_performance.pdf",
        dpi=args.paper_dpi,
    )
    plot_detection(
        all_rows,
        out_dir / "synthetic_missing_detection_routing.png",
        out_dir / "synthetic_missing_detection_routing.pdf",
        dpi=args.paper_dpi,
    )

    if qualitative_record is not None:
        save_qualitative(
            qualitative_record,
            args.qualitative_condition,
            out_dir / "synthetic_missing_qualitative.png",
            out_dir / "synthetic_missing_qualitative.pdf",
            dpi=args.paper_dpi,
        )

    print("\nSynthetic Missing Failure results saved to:", out_dir)
    print(
        "Run the quick profile first with --profile quick --max-shards 64, "
        "then use --profile paper for the final experiment."
    )


def condition_order(rows: Sequence[Dict[str, Any]]) -> List[str]:
    seen = []
    for row in rows:
        name = str(row["condition"])
        if name != "clean" and name not in seen:
            seen.append(name)
    return seen


def display_condition(name: str) -> str:
    mapping = {
        "random_10": "Random\n10%",
        "random_25": "Random\n25%",
        "random_50": "Random\n50%",
        "random_75": "Random\n75%",
        "random_100": "Random\n100%",
        "block_25": "Block\n25%",
        "block_50": "Block\n50%",
        "boundary_50": "Boundary\n50%",
        "boundary_100": "Boundary\n100%",
    }
    return mapping.get(name, name)


def plot_performance(
    rows: Sequence[Dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    conditions = condition_order(rows)
    methods = [
        "Backbone Baseline",
        "Safe Posterior",
        "Full Candidate",
    ]
    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11.2, 5.1))
    for method_index, method in enumerate(methods):
        values = []
        for condition in conditions:
            selected = [
                row
                for row in rows
                if row["condition"] == condition
                and row["method"] == method
            ]
            values.append(
                float(selected[0]["missing_region_rmse"])
                if selected
                else float("nan")
            )
        ax.bar(
            x + (method_index - 1) * width,
            values,
            width=width,
            label=method,
            edgecolor="black",
            linewidth=0.55,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_condition(condition) for condition in conditions]
    )
    ax.set_ylabel("RMSE on Synthetic Missing Region")
    ax.set_title("Depth Completion under Controlled Missing-Depth Failures")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(ncol=3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_detection(
    rows: Sequence[Dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    full_rows = [
        row
        for row in rows
        if row["method"] == "Full Candidate"
        and row["condition"] != "clean"
    ]
    conditions = [str(row["condition"]) for row in full_rows]
    x = np.arange(len(full_rows))

    fig, ax = plt.subplots(figsize=(11.2, 5.1))
    ax.plot(
        x,
        [row["missing_recall"] for row in full_rows],
        marker="o",
        label="Missing Recall",
    )
    ax.plot(
        x,
        [row["missing_f1"] for row in full_rows],
        marker="s",
        label="Missing F1",
    )
    ax.plot(
        x,
        [row["mean_p_missing"] for row in full_rows],
        marker="^",
        label="Mean P(Missing)",
    )
    ax.plot(
        x,
        [row["mean_route_missing_expert"] for row in full_rows],
        marker="D",
        label="Missing-Expert Routing",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [display_condition(condition) for condition in conditions]
    )
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Probability / Ratio")
    ax.set_title("Missing-State Recognition and Expert Routing")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_qualitative(
    record: Dict[str, Any],
    condition: str,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    gt = record["gt"]
    valid_values = gt[np.isfinite(gt) & (gt > 0)]
    vmin, vmax = (
        np.percentile(valid_values, [2, 98])
        if valid_values.size
        else (0.0, 1.0)
    )
    if vmax <= vmin:
        vmax = vmin + 1.0

    error_values = np.concatenate(
        [
            record["anchor_error"].reshape(-1),
            record["full_error"].reshape(-1),
        ]
    )
    error_values = error_values[np.isfinite(error_values)]
    error_max = (
        float(np.percentile(error_values, 98))
        if error_values.size
        else 0.05
    )

    panels = [
        (record["rgb"], "RGB", None, None, None),
        (
            record["corrupted_raw"],
            "Corrupted Raw Depth",
            "viridis",
            vmin,
            vmax,
        ),
        (record["gt"], "Ground Truth", "viridis", vmin, vmax),
        (
            record["missing_mask"],
            "Synthetic Missing Mask",
            "gray",
            0.0,
            1.0,
        ),
        (
            record["p_missing"],
            "P(Missing)",
            "viridis",
            0.0,
            1.0,
        ),
        (
            record["route_missing"],
            "Missing-Expert Routing",
            "viridis",
            0.0,
            1.0,
        ),
        (
            record["source_expert"],
            "Expert-Source Weight",
            "viridis",
            0.0,
            1.0,
        ),
        (
            record["anchor"],
            "Backbone Baseline",
            "viridis",
            vmin,
            vmax,
        ),
        (
            record["full"],
            "Full Candidate",
            "viridis",
            vmin,
            vmax,
        ),
        (
            record["anchor_error"],
            "Backbone Error",
            "magma",
            0.0,
            error_max,
        ),
        (
            record["full_error"],
            "Full Error",
            "magma",
            0.0,
            error_max,
        ),
        (
            record["anchor_error"] - record["full_error"],
            "True Error Reduction",
            "coolwarm",
            -error_max,
            error_max,
        ),
    ]

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(13.2, 9.2),
        constrained_layout=True,
    )
    for ax, (image, title, cmap, lo, hi) in zip(axes.flat, panels):
        if cmap is None:
            ax.imshow(image)
        else:
            ax.imshow(
                image,
                cmap=cmap,
                vmin=lo,
                vmax=hi,
            )
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"Synthetic Missing Failure: {condition} | "
        f"missing-region gain={record['gain']:+.4f} m",
        fontsize=13.0,
    )
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
