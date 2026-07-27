#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Relative-prior availability and corruption stress test for FAPR-Depth v6.

This is a test-time robustness experiment; it does not retrain the model.

The relative-prior input is perturbed by:
- random spatial dropout
- additive Gaussian noise
- global bias
- scale error
- pixel shift
- confidence removal/inversion

For every condition, the script evaluates:
- Posterior Fusion w/o Safety Control
- Safe Posterior
- Full Candidate
- Risk-Accepted Output

Outputs
-------
- relative_prior_stress_results.csv
- relative_prior_stress_results.json
- stress_score_by_family.png
- stress_rmse_by_family.png
- stress_posterior_vs_full.png
- robustness_summary.txt

Recommended paper protocol
--------------------------
python stress_test_fapr_relative_prior.py --profile paper --max-shards 512

Fast smoke test
---------------
python stress_test_fapr_relative_prior.py --profile quick --max-shards 64
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Tuple

import numpy as np
import torch
from tqdm import tqdm

from fapr_analysis_common import (
    OfficialMetricAccumulator,
    add_common_args,
    bootstrap,
    iter_forward,
    write_csv,
    write_json,
    write_run_manifest,
)


@dataclass(frozen=True)
class Condition:
    name: str
    family: str
    severity: float
    parameter: str


OUTPUT_NAMES = [
    "Posterior Fusion w/o Safety Control",
    "Safe Posterior",
    "Full Candidate",
    "Risk-Accepted Output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test FAPR v6 against missing or corrupted relative priors."
    )
    add_common_args(parser, "06_relative_prior_stress", default_phase="joint")
    parser.add_argument("--profile", choices=["quick", "paper"], default="paper")
    parser.add_argument(
        "--scope",
        choices=["transparent", "all"],
        default="transparent",
        help="Where to apply spatial perturbations.",
    )
    parser.add_argument(
        "--include-clean-visual-check",
        action="store_true",
        help="Reserved compatibility flag; metrics are always evaluated.",
    )
    return parser.parse_args()


def build_conditions(profile: str) -> List[Condition]:
    conditions = [Condition("clean", "clean", 0.0, "none")]

    if profile == "quick":
        conditions += [
            Condition("dropout_50pct", "dropout", 0.50, "dropout"),
            Condition("dropout_100pct", "dropout", 1.00, "dropout"),
            Condition("noise_sigma_0.010", "noise", 0.010, "noise"),
            Condition("noise_sigma_0.030", "noise", 0.030, "noise"),
            Condition("bias_plus_0.030", "bias", 0.030, "bias"),
            Condition("scale_1.05", "scale", 1.05, "scale"),
            Condition("shift_4px", "shift", 4.0, "shift"),
            Condition("confidence_zero", "confidence", 1.0, "confidence_zero"),
        ]
        return conditions

    conditions += [
        Condition("dropout_25pct", "dropout", 0.25, "dropout"),
        Condition("dropout_50pct", "dropout", 0.50, "dropout"),
        Condition("dropout_75pct", "dropout", 0.75, "dropout"),
        Condition("dropout_100pct", "dropout", 1.00, "dropout"),
        Condition("noise_sigma_0.005", "noise", 0.005, "noise"),
        Condition("noise_sigma_0.010", "noise", 0.010, "noise"),
        Condition("noise_sigma_0.020", "noise", 0.020, "noise"),
        Condition("noise_sigma_0.040", "noise", 0.040, "noise"),
        Condition("bias_minus_0.030", "bias", -0.030, "bias"),
        Condition("bias_plus_0.030", "bias", 0.030, "bias"),
        Condition("bias_plus_0.060", "bias", 0.060, "bias"),
        Condition("scale_0.95", "scale", 0.95, "scale"),
        Condition("scale_1.05", "scale", 1.05, "scale"),
        Condition("shift_2px", "shift", 2.0, "shift"),
        Condition("shift_4px", "shift", 4.0, "shift"),
        Condition("shift_8px", "shift", 8.0, "shift"),
        Condition("confidence_zero", "confidence", 1.0, "confidence_zero"),
        Condition("confidence_inverted", "confidence", 2.0, "confidence_inverted"),
    ]
    return conditions


def stable_seed(base_seed: int, condition: str, shard_index: int, offset: int) -> int:
    digest = hashlib.sha1(
        f"{base_seed}|{condition}|{shard_index}|{offset}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def make_transform(
    ctx,
    condition: Condition,
    scope: str,
) -> Callable:
    def transform(
        inp: Dict[str, torch.Tensor],
        shard_index: int,
        sample_offset: int,
    ) -> Dict[str, torch.Tensor]:
        if condition.family == "clean":
            return inp

        # Clone only fields that may be modified.
        out = dict(inp)
        rel = inp["rel"].clone()
        conf = inp["rel_conf"].clone()
        coverage = inp["rel_bg_coverage"].clone()
        bg_resid = inp["rel_bg_resid"].clone()

        if scope == "transparent":
            region = (inp["mask"] > 0.5).float()
        else:
            region = torch.ones_like(inp["mask"])

        seed = stable_seed(
            int(ctx.args.seed),
            condition.name,
            shard_index,
            sample_offset,
        )
        generator = torch.Generator(device=rel.device)
        generator.manual_seed(seed)

        if condition.parameter == "dropout":
            ratio = float(condition.severity)
            drop = (
                torch.rand(
                    rel.shape,
                    generator=generator,
                    device=rel.device,
                    dtype=rel.dtype,
                )
                < ratio
            ).float() * region
            rel = rel * (1.0 - drop) + inp["raw"] * drop
            conf = conf * (1.0 - drop)
            coverage = coverage * (1.0 - drop)
            bg_resid = bg_resid * (1.0 - drop)

        elif condition.parameter == "noise":
            sigma = float(condition.severity)
            noise = torch.randn(
                rel.shape,
                generator=generator,
                device=rel.device,
                dtype=rel.dtype,
            ) * sigma
            rel = rel + region * noise

        elif condition.parameter == "bias":
            rel = rel + region * float(condition.severity)

        elif condition.parameter == "scale":
            factor = float(condition.severity)
            rel = rel * (1.0 - region) + rel * factor * region

        elif condition.parameter == "shift":
            shift = int(round(condition.severity))
            shifted_rel = torch.roll(rel, shifts=(0, shift), dims=(-2, -1))
            shifted_conf = torch.roll(conf, shifts=(0, shift), dims=(-2, -1))
            shifted_coverage = torch.roll(coverage, shifts=(0, shift), dims=(-2, -1))
            shifted_bg_resid = torch.roll(bg_resid, shifts=(0, shift), dims=(-2, -1))

            # Replace wrapped columns with an unavailable source instead of circular content.
            if shift > 0:
                shifted_rel[..., :shift] = inp["raw"][..., :shift]
                shifted_conf[..., :shift] = 0.0
                shifted_coverage[..., :shift] = 0.0
                shifted_bg_resid[..., :shift] = 0.0

            rel = rel * (1.0 - region) + shifted_rel * region
            conf = conf * (1.0 - region) + shifted_conf * region
            coverage = coverage * (1.0 - region) + shifted_coverage * region
            bg_resid = bg_resid * (1.0 - region) + shifted_bg_resid * region

        elif condition.parameter == "confidence_zero":
            conf = conf * (1.0 - region)
            coverage = coverage * (1.0 - region)
            bg_resid = bg_resid * (1.0 - region)

        elif condition.parameter == "confidence_inverted":
            conf = conf * (1.0 - region) + (1.0 - conf) * region

        else:
            raise ValueError(f"Unknown perturbation: {condition.parameter}")

        out["rel"] = ctx.train_mod.safe_depth(rel)
        out["rel_conf"] = conf.clamp(0.0, 1.0)
        out["rel_bg_coverage"] = coverage.clamp(0.0, 1.0)
        out["rel_bg_resid"] = bg_resid
        return out

    return transform


def evaluate_condition(ctx, condition: Condition, scope: str) -> Dict[str, Dict[str, float]]:
    accumulators = {
        name: OfficialMetricAccumulator(ctx.train_mod)
        for name in OUTPUT_NAMES
    }
    aux_sums = {
        "raw_weight": 0.0,
        "relative_weight": 0.0,
        "expert_weight": 0.0,
        "safe_gate": 0.0,
        "safe_support": 0.0,
        "route_entropy": 0.0,
    }
    aux_count = 0

    transform = make_transform(ctx, condition, scope)
    progress = tqdm(
        iter_forward(ctx, input_transform=transform),
        total=len(ctx.shards),
        desc=f"Stress {condition.name}",
        dynamic_ncols=True,
        leave=False,
    )
    for inp, out, _ in progress:
        mask = inp["mask"]
        raw = inp["raw"]
        predictions = {
            "Posterior Fusion w/o Safety Control": (
                out["legacy_fused"] * mask + raw * (1.0 - mask)
            ),
            "Safe Posterior": out["safe_benchmark"],
            "Full Candidate": out["candidate_benchmark"],
            "Risk-Accepted Output": out["benchmark_output"],
        }
        for name, pred in predictions.items():
            accumulators[name].update(
                pred,
                inp["raw"],
                inp["gt"],
                inp["mask"],
                inp["valid"],
            )

        aux_sums["raw_weight"] += float(out["alpha"][:, 0:1].mean().item())
        aux_sums["relative_weight"] += float(out["alpha"][:, 1:2].mean().item())
        aux_sums["expert_weight"] += float(out["alpha"][:, 2:3].mean().item())
        aux_sums["safe_gate"] += float(out["safe_gate"].mean().item())
        aux_sums["safe_support"] += float(out["safe_support"].mean().item())
        aux_sums["route_entropy"] += float(out["route_entropy"].mean().item())
        aux_count += 1

    result = {name: acc.result() for name, acc in accumulators.items()}
    result["_aux"] = {
        key: value / max(aux_count, 1) for key, value in aux_sums.items()
    }
    return result


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)
    conditions = build_conditions(args.profile)

    rows: List[Dict] = []
    all_results: Dict[str, Dict] = {}
    for condition in conditions:
        result = evaluate_condition(ctx, condition, args.scope)
        all_results[condition.name] = result
        for output_name in OUTPUT_NAMES:
            row = {
                "condition": condition.name,
                "family": condition.family,
                "severity": condition.severity,
                "parameter": condition.parameter,
                "output": output_name,
                **result[output_name],
                **result["_aux"],
            }
            rows.append(row)

    clean_lookup = {
        row["output"]: row
        for row in rows
        if row["condition"] == "clean"
    }
    for row in rows:
        clean = clean_lookup[row["output"]]
        row["delta_score_vs_clean"] = row["score"] - clean["score"]
        row["relative_score_degradation_pct"] = (
            100.0 * (row["score"] - clean["score"]) / clean["score"]
            if clean["score"] > 0
            else float("nan")
        )
        row["delta_rmse_vs_clean"] = row["rmse_mask"] - clean["rmse_mask"]
        row["delta_mae_vs_clean"] = row["mae_mask"] - clean["mae_mask"]

    write_csv(out_dir / "relative_prior_stress_results.csv", rows)
    write_json(
        out_dir / "relative_prior_stress_results.json",
        {
            "profile": args.profile,
            "scope": args.scope,
            "conditions": [condition.__dict__ for condition in conditions],
            "results": rows,
        },
    )
    write_run_manifest(
        ctx,
        {
            "analysis": "relative_prior_stress",
            "profile": args.profile,
            "scope": args.scope,
            "condition_count": len(conditions),
        },
    )

    plot_by_family(
        rows,
        metric="score",
        ylabel="Score",
        title="Relative-prior stress: Score",
        path=out_dir / "stress_score_by_family.png",
    )
    plot_by_family(
        rows,
        metric="rmse_mask",
        ylabel="RMSE",
        title="Relative-prior stress: transparent-mask RMSE",
        path=out_dir / "stress_rmse_by_family.png",
    )
    plot_posterior_vs_full(
        rows,
        out_dir / "stress_posterior_vs_full.png",
    )

    report = build_report(rows)
    (out_dir / "robustness_summary.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("Saved to:", out_dir)


def plot_by_family(
    rows: List[Dict],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(rows)
    candidate = frame[frame["output"] == "Full Candidate"].copy()
    families = [
        family
        for family in candidate["family"].unique()
        if family != "clean"
    ]
    fig, axes = plt.subplots(
        len(families),
        1,
        figsize=(8.5, max(4.0, 3.4 * len(families))),
        squeeze=False,
    )
    clean_value = float(
        candidate[candidate["condition"] == "clean"][metric].iloc[0]
    )
    for ax, family in zip(axes.flat, families):
        subset = candidate[candidate["family"] == family].sort_values("severity")
        ax.plot(
            subset["severity"],
            subset[metric],
            marker="o",
            label="Full Candidate",
        )
        ax.axhline(clean_value, linestyle="--", label="Clean")
        ax.set_title(family)
        ax.set_xlabel("Perturbation severity")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_posterior_vs_full(rows: List[Dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(
        index="condition",
        columns="output",
        values="score",
        aggfunc="first",
    )
    order = (
        frame[["condition", "family", "severity"]]
        .drop_duplicates()
        .sort_values(["family", "severity", "condition"])["condition"]
        .tolist()
    )
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = np.arange(len(pivot))
    ax.plot(
        x,
        pivot["Posterior Fusion w/o Safety Control"],
        marker="o",
        label="Posterior w/o safety",
    )
    ax.plot(
        x,
        pivot["Full Candidate"],
        marker="s",
        label="Full Candidate",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Safety compensation under relative-prior corruption")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_report(rows: List[Dict]) -> str:
    import pandas as pd

    frame = pd.DataFrame(rows)
    candidate = frame[frame["output"] == "Full Candidate"].copy()
    posterior = frame[
        frame["output"] == "Posterior Fusion w/o Safety Control"
    ].copy()
    candidate = candidate.sort_values(
        "relative_score_degradation_pct",
        ascending=False,
    )
    posterior = posterior.sort_values(
        "relative_score_degradation_pct",
        ascending=False,
    )

    lines = [
        "=" * 100,
        "FAPR-Depth v6 relative-prior robustness summary",
        "=" * 100,
        "",
        "Largest Full-Candidate degradations:",
    ]
    for _, row in candidate.head(8).iterrows():
        lines.append(
            f"{row['condition']:<26} "
            f"Score={row['score']:.6f} "
            f"Δ={row['delta_score_vs_clean']:+.6f} "
            f"({row['relative_score_degradation_pct']:+.2f}%)"
        )
    lines += ["", "Largest unconstrained-posterior degradations:"]
    for _, row in posterior.head(8).iterrows():
        lines.append(
            f"{row['condition']:<26} "
            f"Score={row['score']:.6f} "
            f"Δ={row['delta_score_vs_clean']:+.6f} "
            f"({row['relative_score_degradation_pct']:+.2f}%)"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
