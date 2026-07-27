#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Controlled Synthetic Missing Failure evaluation for FAPR-Depth v6 — fixed v2.

Key protocol fix
----------------
The synthetic missing corruption is applied to the cached ``raw_depth`` tensor
BEFORE ``build_inputs()`` is called. Therefore, all raw-dependent features that
are computed inside the model—raw validity, raw gradients, discrepancy,
alignment anchors, expert inputs, source weights and safety inputs—are derived
from the corrupted observation rather than from the original raw depth.

The cached raw-reliability prior is also made consistent with the injected
failure by default:

    raw_prior <- max(raw_prior, synthetic_missing_mask)

This update uses only the known test-time corruption mask and does not use
ground-truth depth values. It can be disabled with ``--raw-prior-policy preserve``
for a diagnostic comparison.

The ground-truth validity map is intentionally preserved. Thus, pixels with
corrupted raw depth equal to zero and valid ground truth become Missing targets
under the original FAPR failure definition.

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
synthetic_missing_performance.csv
synthetic_missing_detection.csv
synthetic_missing_paper_table.csv
synthetic_missing_results.json
synthetic_missing_performance_paper.{png,pdf}
synthetic_missing_detection_routing_paper.{png,pdf}
synthetic_missing_qualitative_paper.{png,pdf}
run_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from fapr_analysis_common import (
    OfficialMetricAccumulator,
    PixelMetricAccumulator,
    Reservoir,
    add_common_args,
    batch_sample_count,
    binary_auc,
    binary_auprc,
    bootstrap,
    chw_rgb,
    make_loader,
    map2d,
    move_batch,
    select_top_records,
    slice_batch,
    write_csv,
    write_json,
    write_run_manifest,
)


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    ratio: float


METHODS = [
    "Backbone Baseline",
    "Safe Posterior",
    "Full Candidate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate FAPR v6 under controlled, sensor-consistent "
            "synthetic missing-depth failures."
        )
    )
    add_common_args(parser, "08_synthetic_missing_fixed_v2", default_phase="joint")
    parser.add_argument(
        "--profile",
        choices=["quick", "paper"],
        default="paper",
    )
    parser.add_argument(
        "--block-kernel",
        type=int,
        default=41,
        help="Smoothing kernel used to generate contiguous block-like masks.",
    )
    parser.add_argument(
        "--raw-prior-policy",
        choices=["mark_missing", "preserve", "zero_missing"],
        default="mark_missing",
        help=(
            "How to update the cached raw reliability prior on injected pixels. "
            "'mark_missing' is the coherent formal protocol; 'preserve' is a "
            "diagnostic ablation."
        ),
    )
    parser.add_argument(
        "--qualitative-condition",
        type=str,
        default="auto",
        help=(
            "Condition used for the qualitative example. 'auto' selects "
            "block_25 for quick and block_50 for paper."
        ),
    )
    parser.add_argument(
        "--reservoir-pixels",
        type=int,
        default=1_500_000,
        help="Maximum sampled transparent pixels per condition for AUROC/AUPRC.",
    )
    parser.add_argument("--paper-dpi", type=int, default=600)
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Remove overall figure titles.",
    )
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


def resolve_qualitative_condition(args: argparse.Namespace) -> str:
    if args.qualitative_condition != "auto":
        return str(args.qualitative_condition)
    return "block_25" if args.profile == "quick" else "block_50"


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
    """Select an exact approximate ratio independently for every image."""
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

        selected_count = max(1, int(round(ratio * count)))
        selected_count = min(selected_count, count)

        region_indices = torch.nonzero(
            region.reshape(-1),
            as_tuple=False,
        ).reshape(-1)
        values = score[bi].reshape(-1)[region_indices]
        _, top_indices = torch.topk(
            values,
            k=selected_count,
            largest=True,
        )

        flat_result = result[bi].reshape(-1)
        flat_result[region_indices[top_indices]] = True

    return result


def cached_map(
    train_mod,
    batch: Mapping[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"Cached batch is missing required key: {key}")
    return train_mod.force_4d_map(batch[key])


def make_missing_mask_from_cached_batch(
    ctx,
    batch: Mapping[str, torch.Tensor],
    condition: Condition,
    seed: int,
    block_kernel: int,
) -> torch.Tensor:
    """
    Construct the corruption mask from the uncorrupted cached batch.

    Eligible pixels:
    - transparent mask;
    - valid ground truth;
    - originally available raw depth.

    Requiring originally positive raw depth prevents existing sensor holes from
    being counted as newly injected failures.
    """
    train_mod = ctx.train_mod
    mask = cached_map(train_mod, batch, "mask") > 0.5
    valid = cached_map(train_mod, batch, "valid") > 0.5
    original_raw = cached_map(train_mod, batch, "raw_depth")
    raw_available = original_raw > float(train_mod.EPS)
    eligible = mask & valid & raw_available

    if condition.family == "clean":
        return torch.zeros_like(eligible, dtype=torch.bool)

    generator = torch.Generator(device=original_raw.device)
    generator.manual_seed(int(seed))
    random_score = torch.rand(
        original_raw.shape,
        generator=generator,
        device=original_raw.device,
        dtype=original_raw.dtype,
    )

    if condition.family == "random":
        return exact_ratio_mask(
            random_score,
            eligible,
            condition.ratio,
        )

    if condition.family == "block":
        kernel = max(3, int(block_kernel))
        if kernel % 2 == 0:
            kernel += 1
        smooth_score = F.avg_pool2d(
            random_score,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        smooth_score = F.avg_pool2d(
            smooth_score,
            kernel_size=kernel,
            stride=1,
            padding=kernel // 2,
        )
        return exact_ratio_mask(
            smooth_score,
            eligible,
            condition.ratio,
        )

    if condition.family == "boundary":
        if "boundary" in batch:
            boundary = cached_map(train_mod, batch, "boundary") > 0.15
        else:
            boundary = (
                train_mod.build_boundary_ring(
                    cached_map(train_mod, batch, "mask").clamp(0.0, 1.0)
                )
                > 0.15
            )
        return exact_ratio_mask(
            random_score,
            eligible & boundary,
            condition.ratio,
        )

    raise ValueError(f"Unknown missing family: {condition.family}")


def corrupt_cached_batch_before_build_inputs(
    ctx,
    batch: Dict[str, torch.Tensor],
    missing_mask: torch.Tensor,
    raw_prior_policy: str,
) -> Dict[str, torch.Tensor]:
    """
    Apply the sensor corruption to cached fields before build_inputs().

    ``valid`` and ``gt_depth`` are deliberately unchanged.
    """
    train_mod = ctx.train_mod
    corrupted = dict(batch)

    raw = cached_map(train_mod, batch, "raw_depth").clone()
    raw[missing_mask] = 0.0
    corrupted["raw_depth"] = raw

    if "raw_prior" in batch:
        raw_prior = cached_map(train_mod, batch, "raw_prior").clone().clamp(0.0, 1.0)
    else:
        raw_prior = torch.zeros_like(raw)

    if raw_prior_policy == "mark_missing":
        raw_prior = torch.maximum(
            raw_prior,
            missing_mask.float(),
        )
    elif raw_prior_policy == "zero_missing":
        raw_prior[missing_mask] = 0.0
    elif raw_prior_policy != "preserve":
        raise ValueError(f"Unsupported raw-prior policy: {raw_prior_policy}")

    corrupted["raw_prior"] = raw_prior
    return corrupted


@torch.inference_mode()
def iter_condition_forward(
    ctx,
    condition: Condition,
    block_kernel: int,
    raw_prior_policy: str,
) -> Iterator[
    Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Any],
    ]
]:
    """
    Correct inference order:

        cached microbatch
        -> construct synthetic mask
        -> corrupt raw_depth/raw_prior
        -> build_inputs
        -> model forward
    """
    loader = make_loader(ctx)
    microbatch = max(1, int(ctx.args.microbatch))

    for shard_index, cpu_batch in enumerate(loader):
        total = batch_sample_count(cpu_batch)

        for start in range(0, total, microbatch):
            end = min(total, start + microbatch)
            part = move_batch(
                slice_batch(cpu_batch, start, end, total),
                ctx.device,
            )

            seed = stable_seed(
                int(ctx.args.seed),
                condition.name,
                shard_index,
                start,
            )
            missing_mask = make_missing_mask_from_cached_batch(
                ctx,
                part,
                condition,
                seed,
                block_kernel,
            )
            corrupted_part = corrupt_cached_batch_before_build_inputs(
                ctx,
                part,
                missing_mask,
                raw_prior_policy,
            )

            inp = ctx.train_mod.build_inputs(corrupted_part)
            inp["_synthetic_missing_mask"] = missing_mask.float()

            with torch.autocast(
                device_type=ctx.device.type,
                dtype=(
                    torch.float16
                    if ctx.device.type == "cuda"
                    else torch.bfloat16
                ),
                enabled=ctx.use_amp,
            ):
                out = ctx.model(
                    inp,
                    phase=ctx.phase,
                    augment_safe=False,
                )

            meta = {
                "shard_index": shard_index,
                "sample_offset": start,
                "source_shard": str(ctx.shards[shard_index]),
            }
            yield inp, out, meta

            del part, corrupted_part, inp, out
            if (
                ctx.device.type == "cuda"
                and (shard_index + 1) % 100 == 0
            ):
                torch.cuda.empty_cache()


class MissingDetectionAccumulator:
    def __init__(
        self,
        reservoir_pixels: int,
        seed: int,
    ) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.support = 0
        self.nonmissing_support = 0

        self.sum_p_missing = 0.0
        self.sum_p_missing_nonmissing = 0.0
        self.sum_route_missing = 0.0
        self.sum_source_raw = 0.0
        self.sum_source_relative = 0.0
        self.sum_source_expert = 0.0

        self.reservoir = Reservoir(
            reservoir_pixels,
            columns=["p_missing", "target_missing"],
            seed=seed,
        )

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
        nonmissing = transparent & (~target)

        predicted_label = torch.argmax(
            out["fail_prob"],
            dim=1,
            keepdim=True,
        )
        predicted_missing = predicted_label == 1

        self.tp += int((predicted_missing & target).sum().item())
        self.fp += int((predicted_missing & nonmissing).sum().item())
        self.fn += int(((~predicted_missing) & target).sum().item())

        missing_count = int(target.sum().item())
        nonmissing_count = int(nonmissing.sum().item())
        self.support += missing_count
        self.nonmissing_support += nonmissing_count

        if missing_count > 0:
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

        if nonmissing_count > 0:
            self.sum_p_missing_nonmissing += float(
                out["fail_prob"][:, 1:2][nonmissing].sum().item()
            )

        p_missing = out["fail_prob"][:, 1:2][transparent]
        target_flat = target[transparent]
        self.reservoir.add_arrays(
            p_missing=p_missing.detach().float().cpu().numpy(),
            target_missing=target_flat.detach().float().cpu().numpy(),
        )

    def result(self) -> Dict[str, float]:
        precision = (
            self.tp / (self.tp + self.fp)
            if self.tp + self.fp > 0
            else 0.0
        )
        recall = (
            self.tp / (self.tp + self.fn)
            if self.tp + self.fn > 0
            else float("nan")
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(recall) and precision + recall > 0
            else 0.0
        )

        frame = self.reservoir.frame()
        if len(frame) > 0:
            scores = frame["p_missing"].to_numpy()
            labels = frame["target_missing"].to_numpy().astype(np.int64)
            auroc = binary_auc(scores, labels)
            auprc = binary_auprc(scores, labels)
        else:
            auroc = float("nan")
            auprc = float("nan")

        mean_missing = (
            self.sum_p_missing / self.support
            if self.support > 0
            else float("nan")
        )
        mean_nonmissing = (
            self.sum_p_missing_nonmissing / self.nonmissing_support
            if self.nonmissing_support > 0
            else float("nan")
        )

        return {
            "missing_support_pixels": int(self.support),
            "nonmissing_support_pixels": int(self.nonmissing_support),
            "missing_argmax_precision": precision,
            "missing_argmax_recall": recall,
            "missing_argmax_f1": f1,
            "missing_binary_auroc": auroc,
            "missing_binary_auprc": auprc,
            "mean_p_missing": mean_missing,
            "mean_p_missing_nonmissing": mean_nonmissing,
            "p_missing_lift": (
                mean_missing - mean_nonmissing
                if np.isfinite(mean_missing)
                and np.isfinite(mean_nonmissing)
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
            "calibration_reservoir_pixels": int(len(frame)),
        }


def evaluate_condition(
    ctx,
    condition: Condition,
    args: argparse.Namespace,
    qualitative_condition: str,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
    Optional[Dict[str, Any]],
]:
    official = {
        method: OfficialMetricAccumulator(ctx.train_mod)
        for method in METHODS
    }
    missing_region = {
        method: PixelMetricAccumulator()
        for method in METHODS
    }
    unchanged_region = {
        method: PixelMetricAccumulator()
        for method in METHODS
    }
    detection = MissingDetectionAccumulator(
        reservoir_pixels=args.reservoir_pixels,
        seed=args.seed + 4000 + sum(ord(c) for c in condition.name),
    )

    top_qualitative: List[Dict[str, Any]] = []

    progress = tqdm(
        iter_condition_forward(
            ctx,
            condition,
            block_kernel=args.block_kernel,
            raw_prior_policy=args.raw_prior_policy,
        ),
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
        synthetic_mask = inp["_synthetic_missing_mask"]
        unchanged_mask = (
            (mask > 0.5)
            & (valid > 0.5)
            & (synthetic_mask <= 0.5)
        ).float()

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
            missing_region[method].update(
                prediction,
                gt,
                synthetic_mask,
                min_depth=float(ctx.train_mod.MIN_DEPTH),
            )
            unchanged_region[method].update(
                prediction,
                gt,
                unchanged_mask,
                min_depth=float(ctx.train_mod.MIN_DEPTH),
            )

        detection.update(inp, out)

        if (
            condition.name == qualitative_condition
            and int(synthetic_mask.sum().item()) > 0
        ):
            for bi in range(inp["rgb"].shape[0]):
                region = synthetic_mask[bi : bi + 1] > 0.5
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
                gain_map = anchor_error - full_error
                gain = float(gain_map[region].mean().item())

                record = {
                    "gain": gain,
                    "source_shard": str(meta["source_shard"]),
                    "shard_index": int(meta["shard_index"]),
                    "sample_in_shard": int(meta["sample_offset"] + bi),
                    "rgb": chw_rgb(inp["rgb"], bi),
                    "corrupted_raw": map2d(raw, bi),
                    "gt": map2d(gt, bi),
                    "missing_mask": map2d(synthetic_mask, bi),
                    "p_missing": map2d(
                        out["fail_prob"][:, 1:2],
                        bi,
                    ),
                    "predicted_failure": map2d(
                        torch.argmax(
                            out["fail_prob"],
                            dim=1,
                            keepdim=True,
                        ).float(),
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
                    "gain_map": map2d(gain_map, 0),
                }
                select_top_records(
                    top_qualitative,
                    record,
                    key="gain",
                    k=1,
                    largest=True,
                )

    detection_result = detection.result()
    performance_rows: List[Dict[str, Any]] = []

    for method in METHODS:
        whole = official[method].result()
        missing = missing_region[method].result()
        unchanged = unchanged_region[method].result()

        performance_rows.append(
            {
                "condition": condition.name,
                "family": condition.family,
                "configured_ratio": condition.ratio,
                "raw_prior_policy": args.raw_prior_policy,
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
                "unchanged_region_pixels": unchanged.get("pixels", 0),
                "unchanged_region_rmse": unchanged.get("rmse", float("nan")),
                "unchanged_region_mae": unchanged.get("mae", float("nan")),
            }
        )

    detection_row = {
        "condition": condition.name,
        "family": condition.family,
        "configured_ratio": condition.ratio,
        "raw_prior_policy": args.raw_prior_policy,
        **detection_result,
    }
    qualitative = top_qualitative[0] if top_qualitative else None
    return performance_rows, detection_row, qualitative


def build_paper_table(
    performance_rows: Sequence[Dict[str, Any]],
    detection_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    performance_lookup = {
        (row["condition"], row["method"]): row
        for row in performance_rows
    }
    detection_lookup = {
        row["condition"]: row
        for row in detection_rows
    }

    conditions: List[str] = []
    for row in performance_rows:
        condition = str(row["condition"])
        if condition != "clean" and condition not in conditions:
            conditions.append(condition)

    rows: List[Dict[str, Any]] = []
    for condition in conditions:
        backbone = performance_lookup[(condition, "Backbone Baseline")]
        safe = performance_lookup[(condition, "Safe Posterior")]
        full = performance_lookup[(condition, "Full Candidate")]
        detection = detection_lookup[condition]

        backbone_rmse = float(backbone["missing_region_rmse"])
        full_rmse = float(full["missing_region_rmse"])
        full_gain = (
            100.0 * (backbone_rmse - full_rmse) / backbone_rmse
            if np.isfinite(backbone_rmse) and backbone_rmse > 0
            else float("nan")
        )

        rows.append(
            {
                "condition": condition,
                "family": full["family"],
                "ratio": full["configured_ratio"],
                "backbone_missing_rmse": backbone_rmse,
                "safe_missing_rmse": safe["missing_region_rmse"],
                "full_missing_rmse": full_rmse,
                "full_rmse_improvement_pct": full_gain,
                "missing_argmax_recall": detection["missing_argmax_recall"],
                "missing_argmax_f1": detection["missing_argmax_f1"],
                "missing_binary_auroc": detection["missing_binary_auroc"],
                "missing_binary_auprc": detection["missing_binary_auprc"],
                "mean_p_missing": detection["mean_p_missing"],
                "p_missing_lift": detection["p_missing_lift"],
                "route_missing_expert": detection["mean_route_missing_expert"],
                "source_expert": detection["mean_source_expert"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(args.profile)
    qualitative_condition = resolve_qualitative_condition(args)
    condition_names = {condition.name for condition in conditions}
    if qualitative_condition not in condition_names:
        raise ValueError(
            f"Qualitative condition {qualitative_condition!r} is not in "
            f"profile={args.profile}. Available: {sorted(condition_names)}"
        )

    all_performance: List[Dict[str, Any]] = []
    all_detection: List[Dict[str, Any]] = []
    qualitative_record: Optional[Dict[str, Any]] = None

    for condition in conditions:
        performance_rows, detection_row, record = evaluate_condition(
            ctx,
            condition,
            args,
            qualitative_condition,
        )
        all_performance.extend(performance_rows)
        all_detection.append(detection_row)
        if record is not None:
            qualitative_record = record

    clean_score = {
        row["method"]: float(row["overall_score"])
        for row in all_performance
        if row["condition"] == "clean"
    }
    for row in all_performance:
        baseline = clean_score.get(row["method"], float("nan"))
        row["relative_score_change_vs_clean_pct"] = (
            100.0 * (float(row["overall_score"]) - baseline) / baseline
            if np.isfinite(baseline) and baseline > 0
            else float("nan")
        )

    paper_rows = build_paper_table(
        all_performance,
        all_detection,
    )

    write_csv(
        out_dir / "synthetic_missing_performance.csv",
        all_performance,
    )
    write_csv(
        out_dir / "synthetic_missing_detection.csv",
        all_detection,
    )
    write_csv(
        out_dir / "synthetic_missing_paper_table.csv",
        paper_rows,
    )
    write_json(
        out_dir / "synthetic_missing_results.json",
        {
            "profile": args.profile,
            "raw_prior_policy": args.raw_prior_policy,
            "corruption_order": (
                "cached raw_depth corruption -> raw_prior consistency update "
                "-> build_inputs -> model forward"
            ),
            "conditions": [condition.__dict__ for condition in conditions],
            "performance": all_performance,
            "detection": all_detection,
            "paper_table": paper_rows,
        },
    )
    write_run_manifest(
        ctx,
        {
            "analysis": "synthetic_missing_fixed_v2",
            "profile": args.profile,
            "raw_prior_policy": args.raw_prior_policy,
            "qualitative_condition": qualitative_condition,
            "conditions": [condition.__dict__ for condition in conditions],
            "corruption_before_build_inputs": True,
        },
    )

    plot_performance(
        paper_rows,
        out_dir / "synthetic_missing_performance_paper.png",
        out_dir / "synthetic_missing_performance_paper.pdf",
        dpi=args.paper_dpi,
        show_title=not args.no_title,
    )
    plot_detection(
        paper_rows,
        out_dir / "synthetic_missing_detection_routing_paper.png",
        out_dir / "synthetic_missing_detection_routing_paper.pdf",
        dpi=args.paper_dpi,
        show_title=not args.no_title,
    )

    if qualitative_record is not None:
        save_qualitative(
            qualitative_record,
            qualitative_condition,
            out_dir / "synthetic_missing_qualitative_paper.png",
            out_dir / "synthetic_missing_qualitative_paper.pdf",
            dpi=args.paper_dpi,
            show_title=not args.no_title,
        )

    print("=" * 108)
    print("Synthetic Missing Failure fixed-v2 evaluation completed")
    print("=" * 108)
    print(f"Output directory       : {out_dir}")
    print(f"Profile                : {args.profile}")
    print(f"Raw-prior policy       : {args.raw_prior_policy}")
    print(f"Qualitative condition  : {qualitative_condition}")
    print("Corruption order       : raw_depth/raw_prior -> build_inputs -> model")
    print()
    for row in paper_rows:
        print(row)


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


def family_rows(
    rows: Sequence[Dict[str, Any]],
    family: str,
) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if row["family"] == family
    ]


def plot_performance(
    rows: Sequence[Dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
    show_title: bool,
) -> None:
    import matplotlib.pyplot as plt

    families = ["random", "block", "boundary"]
    titles = [
        "(a) Random missing",
        "(b) Block missing",
        "(c) Boundary-focused missing",
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.8, 4.5),
        sharey=True,
        constrained_layout=True,
    )

    width = 0.25
    for ax, family, title in zip(axes, families, titles):
        subset = family_rows(rows, family)
        x = np.arange(len(subset))

        values_by_method = {
            "Backbone Baseline": [
                row["backbone_missing_rmse"] for row in subset
            ],
            "Safe Posterior": [
                row["safe_missing_rmse"] for row in subset
            ],
            "Full Candidate": [
                row["full_missing_rmse"] for row in subset
            ],
        }

        for method_index, method in enumerate(METHODS):
            ax.bar(
                x + (method_index - 1) * width,
                values_by_method[method],
                width=width,
                label=method,
                edgecolor="black",
                linewidth=0.55,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [display_condition(row["condition"]) for row in subset]
        )
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        for index, row in enumerate(subset):
            value = float(row["full_missing_rmse"])
            gain = float(row["full_rmse_improvement_pct"])
            ax.text(
                index + width,
                value + 0.0008,
                f"{gain:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.3,
                rotation=90 if len(subset) >= 4 else 0,
            )

    axes[0].set_ylabel("RMSE on Synthetic Missing Region")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.015),
    )
    if show_title:
        fig.suptitle(
            "Depth Completion under Controlled Missing-Depth Failures",
            fontsize=13.0,
        )

    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_detection(
    rows: Sequence[Dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
    show_title: bool,
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(rows))
    labels = [display_condition(row["condition"]) for row in rows]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 4.6),
        constrained_layout=True,
    )

    axes[0].plot(
        x,
        [row["missing_binary_auroc"] for row in rows],
        marker="o",
        label="Missing AUROC",
    )
    axes[0].plot(
        x,
        [row["missing_binary_auprc"] for row in rows],
        marker="s",
        label="Missing AUPRC",
    )
    axes[0].plot(
        x,
        [row["missing_argmax_recall"] for row in rows],
        marker="^",
        label="Argmax Recall",
    )
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Recognition metric")
    axes[0].set_title("(a) Missing-state recognition")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(
        x,
        [row["mean_p_missing"] for row in rows],
        marker="o",
        label="Mean P(Missing)",
    )
    axes[1].plot(
        x,
        [row["route_missing_expert"] for row in rows],
        marker="s",
        label="Missing-Expert Routing",
    )
    axes[1].plot(
        x,
        [row["source_expert"] for row in rows],
        marker="^",
        label="Expert-Source Weight",
    )
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean probability / weight")
    axes[1].set_title("(b) Missing-conditioned control")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend()

    if show_title:
        fig.suptitle(
            "Missing-State Recognition and Expert Control",
            fontsize=13.0,
        )

    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def save_qualitative(
    record: Dict[str, Any],
    condition: str,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
    show_title: bool,
) -> None:
    import matplotlib.pyplot as plt

    gt = record["gt"]
    valid_values = gt[np.isfinite(gt) & (gt > 0)]
    if valid_values.size:
        vmin, vmax = np.percentile(valid_values, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    missing = record["missing_mask"] > 0.5
    error_values = np.concatenate(
        [
            record["anchor_error"][missing].reshape(-1),
            record["full_error"][missing].reshape(-1),
        ]
    ) if np.any(missing) else np.concatenate(
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
    error_max = max(error_max, 1.0e-5)

    gain_values = record["gain_map"][missing]
    gain_values = gain_values[np.isfinite(gain_values)]
    gain_max = (
        float(np.percentile(np.abs(gain_values), 98))
        if gain_values.size
        else 0.02
    )
    gain_max = max(gain_max, 1.0e-5)

    masked_gain = np.full_like(record["gain_map"], np.nan, dtype=np.float64)
    masked_gain[missing] = record["gain_map"][missing]

    panels = [
        (record["rgb"], "RGB", None, None, None),
        (
            record["corrupted_raw"],
            "Corrupted Raw",
            "viridis",
            vmin,
            vmax,
        ),
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
            record["predicted_failure"],
            "Predicted Failure State",
            "tab10",
            0.0,
            3.0,
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
            masked_gain,
            "Error Reduction in Missing Region",
            "coolwarm",
            -gain_max,
            gain_max,
        ),
    ]

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(13.2, 9.0),
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

    if show_title:
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
