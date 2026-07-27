#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Expert-routing and source-fusion heatmaps for FAPR-Depth v6.

Outputs
-------
- routing_by_failure.csv
- source_fusion_by_failure.csv
- routing_entropy_by_failure.csv
- expert_routing_heatmap.png
- source_fusion_heatmap.png
- routing_entropy_by_failure.png
- qualitative/routing_sample_*.png

The main heatmap answers:
"Given the ground-truth sensor failure state, which correction expert receives
the largest routing mass?"

A second heatmap shows how the final posterior distributes weight among raw,
relative-prior, and expert candidates for each failure state.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

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
        description="Create expert-routing and source-fusion heatmaps."
    )
    add_common_args(parser, "03_expert_routing", default_phase="joint")
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument(
        "--min-failure-pixels",
        type=int,
        default=64,
        help="Minimum non-valid pixels for a sample to be considered for visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)
    qualitative_dir = out_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

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

        # Rank samples by the amount and severity of non-valid failure evidence.
        for bi in range(inp["rgb"].shape[0]):
            sample_region = region[bi : bi + 1]
            failure_pixels = int(
                ((labels[bi : bi + 1] > 0) & sample_region).sum().item()
            )
            if failure_pixels < int(args.min_failure_pixels):
                continue
            raw_error = torch.abs(inp["raw"][bi : bi + 1] - inp["gt"][bi : bi + 1])
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
                    torch.argmax(out["fail_prob"], dim=1, keepdim=True).float(),
                    bi,
                ),
                "pi": tensor_numpy(pi[bi]),
                "alpha": tensor_numpy(alpha[bi]),
                "entropy": map2d(entropy, bi),
                "candidate_error": map2d(
                    torch.abs(out["candidate"] - inp["gt"]), bi
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

    routing_rows = [
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
    source_rows = [
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
    entropy_rows = [
        {
            "failure_state": FAILURE_NAMES[i],
            "support_pixels": int(support[i]),
            "mean_normalized_route_entropy": float(entropy_mean[i]),
        }
        for i in range(4)
    ]

    write_csv(out_dir / "routing_by_failure.csv", routing_rows)
    write_csv(out_dir / "source_fusion_by_failure.csv", source_rows)
    write_csv(out_dir / "routing_entropy_by_failure.csv", entropy_rows)
    write_json(
        out_dir / "routing_analysis.json",
        {
            "failure_states": FAILURE_NAMES,
            "experts": EXPERT_NAMES,
            "sources": SOURCE_NAMES,
            "support_pixels": support,
            "routing_matrix": route_mean,
            "source_matrix": alpha_mean,
            "route_entropy": entropy_mean,
        },
    )
    write_run_manifest(ctx, {"analysis": "expert_routing"})

    save_heatmap(
        route_mean,
        FAILURE_NAMES,
        EXPERT_NAMES,
        "Failure-conditioned expert routing",
        out_dir / "expert_routing_heatmap.png",
        value_format=".3f",
        vmin=0.0,
        vmax=1.0,
    )
    save_heatmap(
        alpha_mean,
        FAILURE_NAMES,
        SOURCE_NAMES,
        "Source-fusion weights conditioned on failure state",
        out_dir / "source_fusion_heatmap.png",
        value_format=".3f",
        vmin=0.0,
        vmax=1.0,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(4)
    ax.bar(x, entropy_mean)
    ax.set_xticks(x)
    ax.set_xticklabels(FAILURE_NAMES)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Normalized routing entropy")
    ax.set_title("Routing certainty by failure state")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "routing_entropy_by_failure.png", dpi=220)
    plt.close(fig)

    for index, sample in enumerate(top_samples, start=1):
        save_qualitative(sample, qualitative_dir / f"routing_sample_{index:02d}.png")

    print("\nRouting matrix")
    print(np.array2string(route_mean, precision=4, suppress_small=True))
    print("\nSource-fusion matrix")
    print(np.array2string(alpha_mean, precision=4, suppress_small=True))
    print("Saved to:", out_dir)


def save_qualitative(sample: Dict[str, Any], path: Path) -> None:
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


if __name__ == "__main__":
    main()
