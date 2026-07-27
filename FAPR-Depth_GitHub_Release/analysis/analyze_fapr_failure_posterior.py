#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Failure-posterior diagnostics for FAPR-Depth v6.

Outputs
-------
- failure_class_metrics.csv
- failure_confusion_counts.csv
- failure_confusion_row_normalized.csv
- failure_calibration_bins.csv
- failure_metrics.json
- failure_confusion_matrix.png
- failure_confusion_row_normalized.png
- failure_binary_reliability.png
- failure_probability_by_class.png

This script evaluates the learned four-state failure posterior:
valid / missing / biased / boundary.  It also reports binary failure
calibration because some classes, especially missing depth, may have little or
no support in a particular cached split.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from fapr_analysis_common import (
    Reservoir,
    add_common_args,
    binary_auc,
    binary_auprc,
    bootstrap,
    expected_calibration_error,
    iter_forward,
    save_heatmap,
    tensor_numpy,
    write_csv,
    write_json,
    write_run_manifest,
)


CLASS_NAMES = ["Valid", "Missing", "Biased", "Boundary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze FAPR v6 failure posterior and calibration."
    )
    add_common_args(parser, "01_failure_posterior", default_phase="joint")
    parser.add_argument(
        "--region",
        choices=["transparent", "all_valid"],
        default="transparent",
        help="Pixels used for failure diagnostics.",
    )
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument(
        "--reservoir-pixels",
        type=int,
        default=1_500_000,
        help="Maximum sampled pixels for calibration and probability plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)

    confusion = np.zeros((4, 4), dtype=np.int64)
    class_prob_sum = np.zeros((4, 4), dtype=np.float64)
    class_support = np.zeros(4, dtype=np.int64)

    reservoir = Reservoir(
        args.reservoir_pixels,
        columns=[
            "true_class",
            "pred_class",
            "top_confidence",
            "correct",
            "p_fail",
            "is_failure",
            "p_valid",
            "p_missing",
            "p_biased",
            "p_boundary",
        ],
        seed=args.seed + 101,
    )

    progress = tqdm(
        iter_forward(ctx),
        total=len(ctx.shards),
        desc="Failure posterior",
        dynamic_ncols=True,
    )
    for inp, out, _ in progress:
        labels, _ = ctx.train_mod.failure_targets(
            inp["raw"], inp["gt"], inp["valid"], inp["boundary"]
        )
        if args.region == "transparent":
            region = (inp["valid"] > 0.5) & (inp["mask"] > 0.5)
        else:
            region = inp["valid"] > 0.5

        probs = out["fail_prob"].float()
        pred = torch.argmax(probs, dim=1, keepdim=True)
        top_conf = torch.max(probs, dim=1, keepdim=True).values

        # labels/predictions use NCHW tensors with a singleton channel, while
        # class probabilities are converted to NHWC.  Therefore the boolean
        # mask used for the NHWC tensor must be [B, H, W], not [B, 1, H, W].
        region_nhw = region[:, 0]

        y = labels[region].detach().cpu().numpy().astype(np.int64)
        p = pred[region].detach().cpu().numpy().astype(np.int64)
        conf = top_conf[region].detach().cpu().numpy().astype(np.float64)
        correct = (p == y).astype(np.float64)
        flat_probs = (
            probs.permute(0, 2, 3, 1)[region_nhw]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        for true_cls in range(4):
            true_mask = y == true_cls
            class_support[true_cls] += int(true_mask.sum())
            if true_mask.any():
                class_prob_sum[true_cls] += flat_probs[true_mask].sum(axis=0)
            for pred_cls in range(4):
                confusion[true_cls, pred_cls] += int(
                    np.sum(true_mask & (p == pred_cls))
                )
        reservoir.add_arrays(
            true_class=y,
            pred_class=p,
            top_confidence=conf,
            correct=correct,
            p_fail=1.0 - flat_probs[:, 0],
            is_failure=(y > 0).astype(np.float64),
            p_valid=flat_probs[:, 0],
            p_missing=flat_probs[:, 1],
            p_biased=flat_probs[:, 2],
            p_boundary=flat_probs[:, 3],
        )

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else float("nan")
    row_sum = confusion.sum(axis=1, keepdims=True)
    row_norm = np.divide(
        confusion,
        row_sum,
        out=np.full_like(confusion, np.nan, dtype=np.float64),
        where=row_sum > 0,
    )

    class_rows: List[Dict[str, float]] = []
    recalls: List[float] = []
    f1_values: List[float] = []
    for cls, name in enumerate(CLASS_NAMES):
        tp = float(confusion[cls, cls])
        fp = float(confusion[:, cls].sum() - confusion[cls, cls])
        fn = float(confusion[cls, :].sum() - confusion[cls, cls])
        precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
        recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision)
            and np.isfinite(recall)
            and precision + recall > 0
            else float("nan")
        )
        mean_probs = (
            class_prob_sum[cls] / class_support[cls]
            if class_support[cls] > 0
            else np.full(4, np.nan)
        )
        class_rows.append(
            {
                "class_id": cls,
                "class": name,
                "support_pixels": int(class_support[cls]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_p_valid": mean_probs[0],
                "mean_p_missing": mean_probs[1],
                "mean_p_biased": mean_probs[2],
                "mean_p_boundary": mean_probs[3],
            }
        )
        if np.isfinite(recall):
            recalls.append(recall)
        if np.isfinite(f1):
            f1_values.append(f1)

    frame = reservoir.frame()
    top_ece, top_bins = expected_calibration_error(
        frame["top_confidence"].to_numpy(),
        frame["correct"].to_numpy(),
        bins=args.calibration_bins,
    )
    binary_ece, binary_bins = expected_calibration_error(
        frame["p_fail"].to_numpy(),
        frame["is_failure"].to_numpy(),
        bins=args.calibration_bins,
    )
    binary_labels = frame["is_failure"].to_numpy().astype(np.int64)
    p_fail = frame["p_fail"].to_numpy()
    binary_brier = float(np.mean((p_fail - binary_labels) ** 2))
    multiclass_probs = frame[
        ["p_valid", "p_missing", "p_biased", "p_boundary"]
    ].to_numpy()
    true_classes = frame["true_class"].to_numpy().astype(np.int64)
    one_hot = np.eye(4, dtype=np.float64)[true_classes]
    multiclass_brier = float(np.mean(np.sum((multiclass_probs - one_hot) ** 2, axis=1)))

    summary = {
        "region": args.region,
        "total_pixels": total,
        "accuracy": accuracy,
        "balanced_accuracy_present_classes": (
            float(np.mean(recalls)) if recalls else float("nan")
        ),
        "macro_f1_present_classes": (
            float(np.mean(f1_values)) if f1_values else float("nan")
        ),
        "top_label_ece": top_ece,
        "binary_failure_ece": binary_ece,
        "binary_failure_brier": binary_brier,
        "multiclass_brier": multiclass_brier,
        "binary_failure_auroc": binary_auc(p_fail, binary_labels),
        "binary_failure_auprc": binary_auprc(p_fail, binary_labels),
        "reservoir_pixels": int(len(frame)),
        "class_metrics": class_rows,
    }

    write_csv(out_dir / "failure_class_metrics.csv", class_rows)
    write_csv(
        out_dir / "failure_confusion_counts.csv",
        [
            {"true_class": CLASS_NAMES[i], **{
                f"pred_{CLASS_NAMES[j].lower()}": int(confusion[i, j])
                for j in range(4)
            }}
            for i in range(4)
        ],
    )
    write_csv(
        out_dir / "failure_confusion_row_normalized.csv",
        [
            {"true_class": CLASS_NAMES[i], **{
                f"pred_{CLASS_NAMES[j].lower()}": float(row_norm[i, j])
                for j in range(4)
            }}
            for i in range(4)
        ],
    )
    calibration_rows = [
        {"calibration": "top_label", **row} for row in top_bins
    ] + [
        {"calibration": "binary_failure", **row} for row in binary_bins
    ]
    write_csv(out_dir / "failure_calibration_bins.csv", calibration_rows)
    write_json(out_dir / "failure_metrics.json", summary)
    write_run_manifest(ctx, {"analysis": "failure_posterior", "region": args.region})

    save_heatmap(
        confusion.astype(np.float64),
        CLASS_NAMES,
        CLASS_NAMES,
        "Failure posterior confusion matrix (counts)",
        out_dir / "failure_confusion_matrix.png",
        value_format=".0f",
        cmap="Blues",
    )
    save_heatmap(
        row_norm,
        CLASS_NAMES,
        CLASS_NAMES,
        "Failure posterior confusion matrix (row normalized)",
        out_dir / "failure_confusion_row_normalized.png",
        value_format=".3f",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
    )

    import matplotlib.pyplot as plt

    # Binary failure reliability.
    reliability = pd.DataFrame(binary_bins)
    valid = reliability["count"] > 0
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ax.plot(
        reliability.loc[valid, "confidence"],
        reliability.loc[valid, "accuracy"],
        marker="o",
        label=f"FAPR (ECE={binary_ece:.4f})",
    )
    ax.set_xlabel("Predicted failure probability")
    ax.set_ylabel("Observed failure frequency")
    ax.set_title("Binary failure reliability")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "failure_binary_reliability.png", dpi=220)
    plt.close(fig)

    # Mean predicted class probabilities conditioned on the true class.
    probability_matrix = np.divide(
        class_prob_sum,
        class_support[:, None],
        out=np.full_like(class_prob_sum, np.nan),
        where=class_support[:, None] > 0,
    )
    save_heatmap(
        probability_matrix,
        CLASS_NAMES,
        [f"P({name})" for name in CLASS_NAMES],
        "Mean failure posterior conditioned on ground truth",
        out_dir / "failure_probability_by_class.png",
        value_format=".3f",
        vmin=0.0,
        vmax=1.0,
    )

    print("\nFailure posterior summary")
    print(json_like(summary))


def json_like(payload: Dict) -> str:
    import json
    return json.dumps(payload, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
