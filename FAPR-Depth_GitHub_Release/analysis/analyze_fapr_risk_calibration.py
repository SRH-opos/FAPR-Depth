#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Counterfactual refinement-risk calibration analysis for FAPR-Depth v6.

This script evaluates whether the proposal-risk head predicts the actual gain
from Safe Posterior -> Candidate, and whether its acceptance probability
suppresses harmful refinements.

Outputs
-------
- risk_summary.json
- risk_calibration_bins.csv
- risk_gain_quantiles.csv
- risk_region_outcomes.csv
- risk_reliability.png
- predicted_vs_true_gain.png
- risk_gain_quantiles.png
- qualitative/risk_improve_*.png
- qualitative/risk_damage_*.png

Important
---------
Use --phase joint (the default) so that risk-aware acceptance is active.  The
script warns when the loaded checkpoint appears to contain an untrained or
inactive risk head.
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
    binary_auc,
    binary_auprc,
    bootstrap,
    chw_rgb,
    expected_calibration_error,
    failure_labels_and_regions,
    iter_forward,
    map2d,
    pearsonr,
    select_top_records,
    spearmanr,
    write_csv,
    write_json,
    write_run_manifest,
)


REGION_ORDER = [
    "all_transparent",
    "valid_state",
    "any_failure",
    "biased_failure",
    "boundary_failure",
    "boundary_ring",
    "interior",
    "hard_backbone_top20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze FAPR proposal-risk calibration and accepted refinements."
    )
    add_common_args(parser, "05_risk_calibration", default_phase="joint")
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--quantile-bins", type=int, default=10)
    parser.add_argument("--proposal-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--neutral-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--accept-threshold", type=float, default=0.5)
    parser.add_argument("--reservoir-pixels", type=int, default=2_000_000)
    parser.add_argument("--num-visualizations", type=int, default=6)
    return parser.parse_args()


class RegionRiskAccumulator:
    def __init__(self) -> None:
        self.pixels = 0
        self.proposed = 0
        self.accepted = 0
        self.proposal_improved = 0
        self.proposal_damaged = 0
        self.accepted_improved = 0
        self.accepted_damaged = 0
        self.sum_true_gain = 0.0
        self.sum_predicted_gain = 0.0
        self.sum_final_gain = 0.0
        self.sum_abs_proposal = 0.0
        self.sum_abs_accepted = 0.0
        self.sum_acceptance = 0.0

    @torch.no_grad()
    def update(
        self,
        region: torch.Tensor,
        true_gain: torch.Tensor,
        predicted_gain: torch.Tensor,
        final_gain: torch.Tensor,
        proposal_update: torch.Tensor,
        accepted_update: torch.Tensor,
        acceptance: torch.Tensor,
        proposal_epsilon: float,
        neutral_epsilon: float,
        accept_threshold: float,
    ) -> None:
        mask = region > 0.5
        n = int(mask.sum().item())
        if n <= 0:
            return
        proposed = mask & (torch.abs(proposal_update) > proposal_epsilon)
        accepted = proposed & (acceptance >= accept_threshold)
        p_improve = proposed & (true_gain > neutral_epsilon)
        p_damage = proposed & (true_gain < -neutral_epsilon)
        a_improve = accepted & (true_gain > neutral_epsilon)
        a_damage = accepted & (true_gain < -neutral_epsilon)

        self.pixels += n
        self.proposed += int(proposed.sum().item())
        self.accepted += int(accepted.sum().item())
        self.proposal_improved += int(p_improve.sum().item())
        self.proposal_damaged += int(p_damage.sum().item())
        self.accepted_improved += int(a_improve.sum().item())
        self.accepted_damaged += int(a_damage.sum().item())
        self.sum_true_gain += float(true_gain[mask].sum().item())
        self.sum_predicted_gain += float(predicted_gain[mask].sum().item())
        self.sum_final_gain += float(final_gain[mask].sum().item())
        self.sum_abs_proposal += float(torch.abs(proposal_update)[mask].sum().item())
        self.sum_abs_accepted += float(torch.abs(accepted_update)[mask].sum().item())
        self.sum_acceptance += float(acceptance[mask].sum().item())

    def result(self, region: str) -> Dict[str, float]:
        return {
            "region": region,
            "pixels": self.pixels,
            "proposed_pixels": self.proposed,
            "accepted_pixels": self.accepted,
            "proposal_coverage": self.proposed / self.pixels if self.pixels else float("nan"),
            "acceptance_rate_proposed": self.accepted / self.proposed if self.proposed else float("nan"),
            "proposal_improve_ratio": (
                self.proposal_improved / self.proposed if self.proposed else float("nan")
            ),
            "proposal_damage_ratio": (
                self.proposal_damaged / self.proposed if self.proposed else float("nan")
            ),
            "accepted_improve_ratio": (
                self.accepted_improved / self.accepted if self.accepted else float("nan")
            ),
            "accepted_damage_ratio": (
                self.accepted_damaged / self.accepted if self.accepted else float("nan")
            ),
            "mean_true_candidate_gain": (
                self.sum_true_gain / self.pixels if self.pixels else float("nan")
            ),
            "mean_predicted_gain": (
                self.sum_predicted_gain / self.pixels if self.pixels else float("nan")
            ),
            "mean_final_gain": (
                self.sum_final_gain / self.pixels if self.pixels else float("nan")
            ),
            "mean_abs_proposal_update": (
                self.sum_abs_proposal / self.pixels if self.pixels else float("nan")
            ),
            "mean_abs_accepted_update": (
                self.sum_abs_accepted / self.pixels if self.pixels else float("nan")
            ),
            "mean_acceptance": (
                self.sum_acceptance / self.pixels if self.pixels else float("nan")
            ),
        }


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)
    qualitative_dir = out_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    reservoir = Reservoir(
        args.reservoir_pixels,
        columns=[
            "predicted_gain",
            "true_gain",
            "acceptance",
            "beneficial",
            "harmful",
            "risk_before",
            "risk_after",
            "safe_error",
            "candidate_error",
            "proposal_abs",
        ],
        seed=args.seed + 307,
    )
    regional = {name: RegionRiskAccumulator() for name in REGION_ORDER}
    top_improve: List[Dict[str, Any]] = []
    top_damage: List[Dict[str, Any]] = []

    acceptance_means: List[float] = []
    acceptance_stds: List[float] = []

    progress = tqdm(
        iter_forward(ctx),
        total=len(ctx.shards),
        desc="Risk calibration",
        dynamic_ncols=True,
    )
    for inp, out, meta in progress:
        _, regions = failure_labels_and_regions(ctx.train_mod, inp, out)
        safe_error = torch.abs(out["safe_posterior"] - inp["gt"])
        candidate_error = torch.abs(out["candidate"] - inp["gt"])
        final_error = torch.abs(out["final"] - inp["gt"])
        true_gain = safe_error - candidate_error
        final_gain = safe_error - final_error
        proposal_update = out["effective_candidate_update"]
        accepted_update = out["accepted_update"]
        acceptance = out["acceptance"]
        predicted_gain = out["predicted_gain"]

        acceptance_means.append(float(acceptance.mean().item()))
        acceptance_stds.append(float(acceptance.std().item()))

        for name in REGION_ORDER:
            regional[name].update(
                regions[name],
                true_gain,
                predicted_gain,
                final_gain,
                proposal_update,
                accepted_update,
                acceptance,
                args.proposal_epsilon,
                args.neutral_epsilon,
                args.accept_threshold,
            )

        region = (
            (regions["all_transparent"] > 0.5)
            & (torch.abs(proposal_update) > args.proposal_epsilon)
        )
        reservoir.add_arrays(
            predicted_gain=predicted_gain[region].detach().float().cpu().numpy(),
            true_gain=true_gain[region].detach().float().cpu().numpy(),
            acceptance=acceptance[region].detach().float().cpu().numpy(),
            beneficial=(true_gain[region] > args.neutral_epsilon)
            .detach()
            .float()
            .cpu()
            .numpy(),
            harmful=(true_gain[region] < -args.neutral_epsilon)
            .detach()
            .float()
            .cpu()
            .numpy(),
            risk_before=out["risk_before"][region].detach().float().cpu().numpy(),
            risk_after=out["risk_after"][region].detach().float().cpu().numpy(),
            safe_error=safe_error[region].detach().float().cpu().numpy(),
            candidate_error=candidate_error[region].detach().float().cpu().numpy(),
            proposal_abs=torch.abs(proposal_update)[region]
            .detach()
            .float()
            .cpu()
            .numpy(),
        )

        for bi in range(inp["rgb"].shape[0]):
            sample_region = regions["all_transparent"][bi : bi + 1] > 0.5
            proposed = sample_region & (
                torch.abs(proposal_update[bi : bi + 1]) > args.proposal_epsilon
            )
            count = int(proposed.sum().item())
            if count <= 0:
                continue
            accepted = proposed & (
                acceptance[bi : bi + 1] >= args.accept_threshold
            )
            accepted_count = int(accepted.sum().item())
            sample_gain = float(
                final_gain[bi : bi + 1][sample_region].mean().item()
            )
            accepted_gain = (
                float(true_gain[bi : bi + 1][accepted].mean().item())
                if accepted_count > 0
                else 0.0
            )
            record = {
                "sample_gain": sample_gain,
                "accepted_gain": accepted_gain,
                "proposed_pixels": count,
                "accepted_pixels": accepted_count,
                "source_shard": meta["source_shard"],
                "rgb": chw_rgb(inp["rgb"], bi),
                "gt": map2d(inp["gt"], bi),
                "safe": map2d(out["safe_posterior"], bi),
                "candidate": map2d(out["candidate"], bi),
                "final": map2d(out["final"], bi),
                "safe_error": map2d(safe_error, bi),
                "candidate_error": map2d(candidate_error, bi),
                "predicted_gain": map2d(predicted_gain, bi),
                "true_gain": map2d(true_gain, bi),
                "acceptance": map2d(acceptance, bi),
                "proposal_update": map2d(proposal_update, bi),
                "accepted_update": map2d(accepted_update, bi),
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

    frame = reservoir.frame()
    if len(frame) == 0:
        raise RuntimeError(
            "No proposal pixels were sampled. Check checkpoint, phase and proposal epsilon."
        )

    predicted = frame["predicted_gain"].to_numpy()
    actual = frame["true_gain"].to_numpy()
    acceptance = frame["acceptance"].to_numpy()
    beneficial = frame["beneficial"].to_numpy().astype(np.int64)
    harmful = frame["harmful"].to_numpy().astype(np.int64)

    ece, calibration_rows = expected_calibration_error(
        acceptance,
        beneficial,
        bins=args.calibration_bins,
    )
    quantile_rows = make_quantile_rows(frame, args.quantile_bins)
    region_rows = [regional[name].result(name) for name in REGION_ORDER]

    accepted_mask = acceptance >= args.accept_threshold
    summary = {
        "evaluation_phase": ctx.phase,
        "checkpoint_phase": ctx.checkpoint_phase,
        "sampled_proposal_pixels": int(len(frame)),
        "predicted_true_gain_mae": float(np.mean(np.abs(predicted - actual))),
        "predicted_true_gain_pearson": pearsonr(predicted, actual),
        "predicted_true_gain_spearman": spearmanr(predicted, actual),
        "beneficial_update_auroc": binary_auc(predicted, beneficial),
        "beneficial_update_auprc": binary_auprc(predicted, beneficial),
        "harmful_update_auroc_using_negative_gain": binary_auc(-predicted, harmful),
        "harmful_update_auprc_using_negative_gain": binary_auprc(-predicted, harmful),
        "acceptance_brier": float(np.mean((acceptance - beneficial) ** 2)),
        "acceptance_ece": ece,
        "acceptance_mean": float(np.mean(acceptance)),
        "acceptance_std": float(np.std(acceptance)),
        "candidate_improve_ratio": float(np.mean(beneficial)),
        "candidate_damage_ratio": float(np.mean(harmful)),
        "accepted_fraction": float(np.mean(accepted_mask)),
        "accepted_improve_ratio": (
            float(np.mean(beneficial[accepted_mask])) if accepted_mask.any() else float("nan")
        ),
        "accepted_damage_ratio": (
            float(np.mean(harmful[accepted_mask])) if accepted_mask.any() else float("nan")
        ),
        "mean_true_gain_all_proposals": float(np.mean(actual)),
        "mean_true_gain_accepted": (
            float(np.mean(actual[accepted_mask])) if accepted_mask.any() else float("nan")
        ),
        "calibration_bins": calibration_rows,
        "gain_quantiles": quantile_rows,
        "region_outcomes": region_rows,
    }

    write_json(out_dir / "risk_summary.json", summary)
    write_csv(out_dir / "risk_calibration_bins.csv", calibration_rows)
    write_csv(out_dir / "risk_gain_quantiles.csv", quantile_rows)
    write_csv(out_dir / "risk_region_outcomes.csv", region_rows)
    write_run_manifest(ctx, {"analysis": "risk_calibration"})

    plot_reliability(calibration_rows, ece, out_dir / "risk_reliability.png")
    plot_predicted_vs_true(frame, out_dir / "predicted_vs_true_gain.png")
    plot_quantiles(quantile_rows, out_dir / "risk_gain_quantiles.png")

    for index, record in enumerate(top_improve, start=1):
        save_qualitative(
            record,
            qualitative_dir / f"risk_improve_{index:02d}.png",
            "Strong accepted-refinement improvement",
        )
    for index, record in enumerate(top_damage, start=1):
        save_qualitative(
            record,
            qualitative_dir / f"risk_damage_{index:02d}.png",
            "Strong accepted-refinement damage",
        )

    if np.mean(acceptance_stds) < 1.0e-4:
        print(
            "\n[WARNING] Acceptance is nearly constant. "
            "The checkpoint may predate risk calibration, or the selected phase may be 'proposal'."
        )
    print("\nRisk summary")
    import json
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved to:", out_dir)


def make_quantile_rows(frame, bins: int) -> List[Dict[str, float]]:
    work = frame.sort_values("predicted_gain").reset_index(drop=True)
    groups = np.array_split(np.arange(len(work)), int(bins))
    rows: List[Dict[str, float]] = []
    for index, ids in enumerate(groups):
        subset = work.iloc[ids]
        rows.append(
            {
                "quantile": index + 1,
                "pixels": int(len(subset)),
                "mean_predicted_gain": float(subset["predicted_gain"].mean()),
                "mean_true_gain": float(subset["true_gain"].mean()),
                "beneficial_frequency": float(subset["beneficial"].mean()),
                "harmful_frequency": float(subset["harmful"].mean()),
                "mean_acceptance": float(subset["acceptance"].mean()),
                "mean_proposal_abs": float(subset["proposal_abs"].mean()),
            }
        )
    return rows


def plot_reliability(rows: List[Dict[str, float]], ece: float, path: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(rows)
    valid = frame["count"] > 0
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ax.plot(
        frame.loc[valid, "confidence"],
        frame.loc[valid, "accuracy"],
        marker="o",
        label=f"Risk acceptance (ECE={ece:.4f})",
    )
    ax.set_xlabel("Predicted acceptance probability")
    ax.set_ylabel("Observed beneficial-update frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Refinement-risk reliability")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_predicted_vs_true(frame, path: Path) -> None:
    import matplotlib.pyplot as plt

    x = frame["predicted_gain"].to_numpy()
    y = frame["true_gain"].to_numpy()
    limit = float(np.percentile(np.abs(np.concatenate([x, y])), 99))
    limit = max(limit, 1.0e-4)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    plot = ax.hexbin(x, y, gridsize=70, mincnt=1, bins="log")
    ax.plot([-limit, limit], [-limit, limit], linestyle="--")
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Predicted gain (m)")
    ax.set_ylabel("True gain (m)")
    ax.set_title("Predicted versus realized proposal gain")
    fig.colorbar(plot, ax=ax, label="log pixel count")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_quantiles(rows: List[Dict[str, float]], path: Path) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(1, len(rows) + 1)
    predicted = [row["mean_predicted_gain"] for row in rows]
    actual = [row["mean_true_gain"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(x, predicted, marker="o", label="Predicted gain")
    ax.plot(x, actual, marker="s", label="True gain")
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Predicted-gain quantile (low to high)")
    ax.set_ylabel("Mean gain (m)")
    ax.set_title("Risk ranking by predicted-gain quantile")
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
    error_max = float(
        np.percentile(
            np.concatenate(
                [
                    record["safe_error"].reshape(-1),
                    record["candidate_error"].reshape(-1),
                ]
            ),
            98,
        )
    )
    gain_max = float(np.percentile(np.abs(record["true_gain"]), 98))
    gain_max = max(gain_max, 1.0e-4)

    panels = [
        (record["rgb"], "RGB", None, None, None),
        (record["gt"], "Ground truth", "viridis", vmin, vmax),
        (record["safe"], "Safe posterior", "viridis", vmin, vmax),
        (record["candidate"], "Candidate", "viridis", vmin, vmax),
        (record["final"], "Risk-accepted output", "viridis", vmin, vmax),
        (record["safe_error"], "Safe error", "magma", 0.0, error_max),
        (record["candidate_error"], "Candidate error", "magma", 0.0, error_max),
        (record["predicted_gain"], "Predicted gain", "coolwarm", -gain_max, gain_max),
        (record["true_gain"], "True gain", "coolwarm", -gain_max, gain_max),
        (record["acceptance"], "Acceptance", "viridis", 0.0, 1.0),
        (record["proposal_update"], "Proposal update", "coolwarm", -0.05, 0.05),
        (record["accepted_update"], "Accepted update", "coolwarm", -0.05, 0.05),
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
        f"{title}: final mean gain={record['sample_gain']:.6f}, "
        f"accepted mean candidate gain={record['accepted_gain']:.6f}, "
        f"accepted/proposed={record['accepted_pixels']}/{record['proposed_pixels']}\n"
        f"{record['source_shard']}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
