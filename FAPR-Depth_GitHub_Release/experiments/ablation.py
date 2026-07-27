#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
FAPR-Depth v6: Low-Memory Controlled Component Ablation v4
==============================================================

This script performs paper-oriented ablation experiments for the final
FAPR-Depth v6 model:

    metric prior calibration
        -> latent failure-state inference
        -> failure-conditioned experts
        -> uncertainty-aware posterior fusion
        -> FDCT-anchored safe residual fusion
        -> bounded detail proposal

Scientific protocol
-------------------
1. Every trainable ablation starts from the same completed v5 checkpoint.
2. The original v6 six-epoch curriculum, optimizer, losses, data split, and
   random seed are reused.
3. Checkpoint selection is performed on the validation split only.
4. The fixed best candidate checkpoint is evaluated once on the test split.
5. The full model may reuse the already trained v6 best_candidate.pth.
6. "FDCT anchor", "Legacy posterior", and "w/o Detail Proposal" are exact
   output-path ablations from the same full checkpoint and do not require
   duplicate retraining.

Trainable structural variants
-----------------------------
full
    Original FAPR-Depth v6.

no_relative_prior
    Removes the monocular/relative-depth source from metric calibration,
    posterior fusion, and prior injection.

no_failure_conditioning
    Removes learned failure-state conditioning.  Failure probability becomes
    the transparent mask and expert routing becomes uniform.

single_expert
    Keeps learned failure support but replaces failure-conditioned expert
    routing with the uniform average of the three experts.

no_uncertainty_fusion
    Removes learned source-uncertainty weighting.  Raw, relative, and expert
    sources are fused with deterministic confidence weights.

no_safe_anchor
    Removes the FDCT-anchored safe residual gate.  The detail proposal operates
    directly on the legacy posterior candidate.

no_boundary_cue
    Removes boundary and signed-distance cues from all model inputs while
    retaining the same boundary-aware supervision.

Output-path rows generated from the full checkpoint
---------------------------------------------------
FDCT Anchor
Legacy Posterior
FAPR w/o Detail Proposal
FAPR-Depth v6 (Full)

Expected source files
---------------------
train.py
weights/fapr_v5_best_score.pth
weights/best_candidate.pth

Recommended workflow
--------------------
# 1. Small end-to-end smoke test
python train_and_test_fapr_depth_v6_ablation_lowmem_v4.py --smoke --variants no_relative_prior

# 2. Full experiment. Existing completed variants are resumed/skipped.
python train_and_test_fapr_depth_v6_ablation_8gb.py --variants all

# 3. Run one expensive variant at a time
python train_and_test_fapr_depth_v6_ablation_8gb.py --variants no_relative_prior
python train_and_test_fapr_depth_v6_ablation_8gb.py --variants no_failure_conditioning

Notes
-----
These are controlled v6-stage retraining experiments.  The frozen v5 posterior
parameters are inherited identically, while each ablated system retrains the
v6 safe/proposal/risk heads under the same curriculum.  This is more rigorous
than merely zeroing components at test time, while remaining faithful to the
actual v6 training route.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =============================================================================
# PATHS / FIXED EXPERIMENT IDENTITY
# =============================================================================
PROJECT_ROOT = Path(os.getenv("FAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent)))
V6_SOURCE = PROJECT_ROOT / "train_fapr_depth_safe_anchor_v6_8gb.py"

OUT_ROOT = PROJECT_ROOT / "outputs" / "fapr_depth_v6_component_ablation_v4"
AGGREGATE_DIR = OUT_ROOT / "aggregate"
AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)

EXISTING_FULL_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "fapr_depth_v6_safe_anchor"
    / "checkpoints"
    / "best_candidate.pth"
)

SEED = 6248
USE_CPU_ANCHOR = False

# Full-model reference from the accepted v6 complete test:
# best_candidate.pth -> Candidate benchmark.
OFFICIAL_FULL_REFERENCE = {
    "rmse_mask": 0.011437,
    "rel_mask": 0.015118,
    "mae_mask": 0.006967,
    "delta_105": 0.942700,
    "delta_110": 0.987840,
    "delta_125": 0.999210,
    "score": 0.021131,
}

TRAINABLE_VARIANTS = [
    "full",
    "no_relative_prior",
    "no_failure_conditioning",
    "single_expert",
    "no_uncertainty_fusion",
    "no_safe_anchor",
    "no_boundary_cue",
]

VARIANT_TITLES = {
    "full": "FAPR-Depth v6 (Full)",
    "no_relative_prior": "w/o Relative Prior",
    "no_failure_conditioning": "w/o Failure-State Conditioning",
    "single_expert": "w/o Failure-Conditioned Expert Routing",
    "no_uncertainty_fusion": "w/o Uncertainty-Aware Fusion",
    "no_safe_anchor": "w/o FDCT-Anchored Safe Fusion",
    "no_boundary_cue": "w/o Boundary Cue",
}

OUTPUT_PATH_TITLES = {
    "fdct_anchor": "FDCT Anchor",
    "legacy_posterior": "Legacy Posterior",
    "no_detail_proposal": "w/o Detail Proposal",
    "full": "FAPR-Depth v6 (Full)",
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================
def import_by_path(name: str, path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing Python source: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_dicts(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*[set(row.keys()) for row in rows]))
    out: Dict[str, float] = {}
    for key in keys:
        values = [finite_float(row.get(key)) for row in rows]
        values = [v for v in values if v is not None]
        if values:
            out[key] = float(np.mean(values))
    return out


def metric_row(
    title: str,
    variant_key: str,
    row: Mapping[str, Any],
    source: str,
    row_type: str,
) -> Dict[str, Any]:
    return {
        "method": title,
        "variant_key": variant_key,
        "row_type": row_type,
        "source": source,
        "RMSE": finite_float(row.get("rmse_mask")),
        "REL": finite_float(row.get("rel_mask")),
        "MAE": finite_float(row.get("mae_mask")),
        "delta_1_05": finite_float(row.get("delta_105")),
        "delta_1_10": finite_float(row.get("delta_110")),
        "delta_1_25": finite_float(row.get("delta_125")),
        "Boundary": finite_float(row.get("boundary")),
        "Score": finite_float(row.get("score")),
    }


def fmt(value: Any, digits: int = 4, percent: bool = False) -> str:
    x = finite_float(value)
    if x is None:
        return "-"
    if percent:
        return f"{100.0 * x:.2f}"
    return f"{x:.{digits}f}"


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Method | RMSE↓ | REL↓ | MAE↓ | δ1.05↑ (%) | δ1.10↑ (%) | δ1.25↑ (%) | Boundary↓ | Score↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {rmse} | {rel} | {mae} | {d105} | {d110} | {d125} | {boundary} | {score} |".format(
                method=row["method"],
                rmse=fmt(row.get("RMSE")),
                rel=fmt(row.get("REL")),
                mae=fmt(row.get("MAE")),
                d105=fmt(row.get("delta_1_05"), percent=True),
                d110=fmt(row.get("delta_1_10"), percent=True),
                d125=fmt(row.get("delta_1_25"), percent=True),
                boundary=fmt(row.get("Boundary")),
                score=fmt(row.get("Score")),
            )
        )
    return "\n".join(lines) + "\n"


def latex_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Component ablation of FAPR-Depth v6 on the TransCG test split.}",
        r"\label{tab:fapr_v6_ablation}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Method & RMSE$\downarrow$ & REL$\downarrow$ & MAE$\downarrow$ & "
        r"$\delta_{1.05}\uparrow$ & $\delta_{1.10}\uparrow$ & "
        r"$\delta_{1.25}\uparrow$ & Boundary$\downarrow$ & Score$\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        name = str(row["method"]).replace("_", r"\_")
        lines.append(
            f"{name} & {fmt(row.get('RMSE'))} & {fmt(row.get('REL'))} & "
            f"{fmt(row.get('MAE'))} & {fmt(row.get('delta_1_05'), percent=True)} & "
            f"{fmt(row.get('delta_1_10'), percent=True)} & "
            f"{fmt(row.get('delta_1_25'), percent=True)} & "
            f"{fmt(row.get('Boundary'))} & {fmt(row.get('Score'))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


# =============================================================================
# CONFIGURABLE ABLATION MODEL
# =============================================================================
def build_ablation_model_class(v6):
    class AblationFailureAwarePosteriorDepth(v6.FailureAwarePosteriorDepth):
        """Original v6 model with one controlled component intervention."""

        def __init__(self, base_mod, variant: str):
            if variant not in TRAINABLE_VARIANTS:
                raise ValueError(f"Unknown ablation variant: {variant}")
            super().__init__(base_mod)
            self.ablation_variant = variant

        @torch.no_grad()
        def forward_reference(
            self,
            rgb: torch.Tensor,
            depth: torch.Tensor,
        ) -> torch.Tensor:
            # Optional low-VRAM route: keep the frozen reference FDCT on CPU.
            ref_device = next(self.base_reference.parameters()).device
            if ref_device.type == "cpu" and rgb.device.type != "cpu":
                rgb_ref = rgb.detach().float().cpu()
                depth_ref = depth.detach().float().cpu()
                d = depth_ref[:, 0] if depth_ref.ndim == 4 else depth_ref
                out = self.base_reference(rgb_ref, d)
                out = out.unsqueeze(1) if out.ndim == 3 else out
                return v6.safe_depth(out).to(rgb.device)
            return super().forward_reference(rgb, depth)

        def _model_input(
            self,
            inp: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            out = dict(inp)
            if self.ablation_variant == "no_relative_prior":
                out["rel"] = inp["raw"]
                out["rel_conf"] = torch.zeros_like(inp["rel_conf"])
                out["rel_bg_resid"] = torch.zeros_like(inp["rel_bg_resid"])
                out["rel_bg_coverage"] = torch.zeros_like(inp["rel_bg_coverage"])
            if self.ablation_variant == "no_boundary_cue":
                out["boundary"] = torch.zeros_like(inp["boundary"])
            return out

        def forward_posterior(
            self,
            inp: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            # The complete model must execute the original v6 implementation
            # byte-for-byte at the Python level.  Earlier low-memory revisions
            # routed "full" through the ablation reimplementation, which changed
            # the accepted benchmark by about 2-3%.
            if self.ablation_variant == "full":
                return super().forward_posterior(inp)

            inp = self._model_input(inp)
            rgb, raw, rel = inp["rgb"], inp["raw"], inp["rel"]
            mask, valid = inp["mask"], inp["valid"]
            boundary = inp["boundary"]
            raw_prior, rel_conf = inp["raw_prior"], inp["rel_conf"]

            raw_valid = (raw > v6.EPS).float() * valid
            if self.ablation_variant == "no_boundary_cue":
                sdm = torch.zeros_like(mask)
            else:
                sdm = v6.approximate_signed_distance(mask)

            grad_raw = torch.clamp(v6.gradient_mag(raw) / 0.08, 0.0, 4.0)
            grad_rel = torch.clamp(v6.gradient_mag(rel) / 0.08, 0.0, 4.0)

            outside = (
                1.0 - v6.dilate_binary(mask, v6.ANCHOR_DILATE_KERNEL)
            ).clamp(0.0, 1.0)
            anchor = (
                raw_valid
                * outside
                * (grad_raw < v6.ANCHOR_GRAD_THR / 0.08).float()
            )

            if self.ablation_variant == "no_relative_prior":
                rel_global = raw
                rel_metric = raw
            else:
                if v6.INPUT_REL_ALREADY_METRIC_ALIGNED:
                    rel_global = rel
                else:
                    rel_global, _, _ = v6.robust_global_align(rel, raw, anchor)

                disc0 = (
                    torch.clamp((rel_global - raw) / 0.75, -1.0, 1.0)
                    * raw_valid
                )
                align_x = torch.cat(
                    [
                        rgb,
                        v6.norm_depth(raw),
                        v6.norm_depth(rel_global),
                        mask,
                        boundary,
                        raw_valid,
                        disc0,
                        grad_raw,
                        grad_rel,
                    ],
                    1,
                )
                rel_metric, _, _, _ = self.aligner(align_x, rel_global)

            discrepancy = (
                torch.clamp((rel_metric - raw) / 0.75, -1.0, 1.0)
                * raw_valid
            )

            fail_x = torch.cat(
                [
                    rgb,
                    v6.norm_depth(raw),
                    v6.norm_depth(rel_metric),
                    mask,
                    boundary,
                    sdm,
                    raw_valid,
                    discrepancy,
                    grad_raw,
                    grad_rel,
                    rel_conf,
                    raw_prior,
                ],
                1,
            )
            _, learned_fail_logits, source_logb = self.failure_net(fail_x)

            if self.ablation_variant == "no_failure_conditioning":
                # No learned latent failure state. The transparent mask provides
                # only a binary failure support prior; the three failure classes
                # are deliberately indistinguishable.
                p_valid = (1.0 - mask).clamp(0.0, 1.0)
                p_each_failure = mask / 3.0
                fail_prob = torch.cat(
                    [p_valid, p_each_failure, p_each_failure, p_each_failure],
                    1,
                )
                fail_prob = fail_prob / fail_prob.sum(
                    1, keepdim=True
                ).clamp_min(v6.EPS)
                fail_logits = torch.log(fail_prob.clamp_min(1.0e-6))
            else:
                fail_logits = learned_fail_logits
                fail_prob = F.softmax(fail_logits, 1)

            raw_logb = source_logb[:, 0:1]
            rel_logb = source_logb[:, 1:2]
            p_valid = fail_prob[:, 0:1]
            p_fail = 1.0 - p_valid

            priors = torch.cat(
                [
                    mask,
                    boundary,
                    sdm,
                    raw_valid,
                    rel_conf,
                    raw_prior,
                    v6.norm_depth(raw),
                    v6.norm_depth(rel_metric),
                    discrepancy,
                    grad_raw,
                    grad_rel,
                    inp["rel_bg_resid"],
                    inp["rel_bg_coverage"],
                ],
                1,
            )
            base_depth = self.forward_adapted_base(rgb, raw, priors)

            ctx = torch.cat([fail_x, v6.norm_depth(base_depth)], 1)
            shared = self.shared(ctx)
            dm, um = self.missing_expert(shared)
            dd, ud = self.biased_expert(shared)
            db, ub = self.boundary_expert(shared)

            learned_router_logits = self.router(
                torch.cat([shared, fail_prob], 1)
            )
            if self.ablation_variant in {
                "no_failure_conditioning",
                "single_expert",
            }:
                pi = torch.full_like(learned_router_logits, 1.0 / 3.0)
                router_logits = torch.zeros_like(learned_router_logits)
            else:
                router_logits = learned_router_logits
                pi = F.softmax(router_logits, 1)

            deltas = torch.cat([dm, dd, db], 1)
            expert_logbs = torch.cat([um, ud, ub], 1)
            mix_delta = (pi * deltas).sum(1, keepdim=True)
            expert = v6.safe_depth(base_depth + mix_delta)
            expert_var = (
                pi * (torch.exp(2.0 * expert_logbs) + deltas.square())
            ).sum(1, keepdim=True) - mix_delta.square()
            expert_logb = (
                0.5
                * torch.log(expert_var.clamp_min(1.0e-6))
            ).clamp(-6.0, 2.0)

            if self.ablation_variant == "no_uncertainty_fusion":
                w_raw = raw_valid * p_valid.clamp_min(0.02)
                w_rel = rel_conf.clamp_min(0.05)
                w_expert = 0.20 + 0.80 * p_fail
            else:
                w_raw = (
                    raw_valid
                    * p_valid.clamp_min(0.02)
                    * torch.exp(-raw_logb)
                )
                w_rel = rel_conf.clamp_min(0.05) * torch.exp(-rel_logb)
                w_expert = (
                    (0.20 + 0.80 * p_fail)
                    * torch.exp(-expert_logb)
                )

            if self.ablation_variant == "no_relative_prior":
                # Remove the relative source rather than leaving a duplicate raw
                # source with a non-zero confidence floor.
                w_rel = torch.zeros_like(w_rel)

            weights = torch.cat([w_raw, w_rel, w_expert], 1)
            alpha = weights / weights.sum(
                1, keepdim=True
            ).clamp_min(v6.EPS)

            candidates = torch.cat([raw, rel_metric, expert], 1)
            fused = v6.safe_depth(
                (alpha * candidates).sum(1, keepdim=True)
            )
            source_logbs = torch.cat(
                [raw_logb, rel_logb, expert_logb], 1
            )
            mix_var = (
                alpha
                * (
                    torch.exp(2.0 * source_logbs)
                    + candidates.square()
                )
            ).sum(1, keepdim=True) - fused.square()
            final_logb = (
                0.5 * torch.log(mix_var.clamp_min(1.0e-6))
            ).clamp(-6.0, 2.0)

            route_entropy = -(
                pi * torch.log(pi.clamp_min(v6.EPS))
            ).sum(1, keepdim=True) / math.log(3.0)

            return {
                "rel_metric": rel_metric,
                "base_depth": base_depth,
                "fail_logits": fail_logits,
                "fail_prob": fail_prob,
                "p_fail": p_fail,
                "raw_logb": raw_logb,
                "rel_logb": rel_logb,
                "router_logits": router_logits,
                "pi": pi,
                "alpha": alpha,
                "expert": expert,
                "expert_candidates": torch.cat(
                    [
                        v6.safe_depth(base_depth + dm),
                        v6.safe_depth(base_depth + dd),
                        v6.safe_depth(base_depth + db),
                    ],
                    1,
                ),
                "fused": fused,
                "final_logb": final_logb,
                "route_entropy": route_entropy,
                "sdm": sdm,
            }

        def forward(
            self,
            inp: Dict[str, torch.Tensor],
            phase: str = "joint",
            augment_safe: bool = False,
        ) -> Dict[str, torch.Tensor]:
            # Exact reference path for the non-ablated model.  This is essential
            # both for the full row and for the same-checkpoint Anchor/Legacy/
            # Safe output-path ablations.
            if self.ablation_variant == "full":
                return super().forward(
                    inp,
                    phase=phase,
                    augment_safe=augment_safe,
                )

            model_inp = self._model_input(inp)
            with torch.no_grad():
                posterior = self.forward_posterior(model_inp)
                anchor_depth = self.forward_reference(
                    model_inp["rgb"], model_inp["raw"]
                )

            rgb = model_inp["rgb"]
            raw = model_inp["raw"]
            mask = model_inp["mask"]
            boundary = model_inp["boundary"]
            legacy_fused = posterior["fused"]
            p_fail = posterior["p_fail"].detach()

            corruption_prob = 0.0
            if augment_safe and phase == "safe":
                corruption_prob = v6.SAFE_CORRUPTION_PROB_WARMUP
            elif augment_safe and phase == "joint":
                corruption_prob = v6.SAFE_CORRUPTION_PROB_JOINT

            legacy_candidate, corruption_mask = (
                self._corrupt_legacy_candidate(
                    anchor_depth,
                    legacy_fused,
                    mask,
                    boundary,
                    corruption_prob,
                )
            )
            legacy_residual = (
                legacy_candidate - anchor_depth
            ).clamp(
                -v6.SAFE_MAX_RESIDUAL,
                v6.SAFE_MAX_RESIDUAL,
            )

            safe_support = mask * torch.clamp(
                v6.SAFE_SUPPORT_FLOOR
                + (1.0 - v6.SAFE_SUPPORT_FLOOR) * p_fail
                + v6.SAFE_BOUNDARY_WEIGHT * boundary,
                0.0,
                1.0,
            )
            safe_support = safe_support.detach()

            safe_x = torch.cat(
                [
                    rgb,
                    v6.norm_depth(anchor_depth),
                    v6.norm_depth(legacy_candidate),
                    v6.norm_depth(raw),
                    v6.norm_depth(posterior["rel_metric"]),
                    (
                        legacy_residual
                        / max(v6.SAFE_MAX_RESIDUAL, v6.EPS)
                    ).clamp(-1.0, 1.0),
                    (
                        legacy_residual.abs()
                        / max(v6.SAFE_MAX_RESIDUAL, v6.EPS)
                    ).clamp(0.0, 1.0),
                    mask,
                    boundary,
                    posterior["sdm"],
                    p_fail,
                    posterior["final_logb"],
                    posterior["route_entropy"],
                    posterior["alpha"],
                    model_inp["raw_prior"],
                    model_inp["rel_conf"],
                ],
                1,
            )

            direct_gate_logit = self.safe_anchor.direct_gate_logit(safe_x)
            safe_risk_anchor, safe_risk_legacy = (
                self.safe_anchor.estimate_risk(safe_x.detach())
            )
            safe_predicted_gain = (
                safe_risk_anchor - safe_risk_legacy
            )
            use_safe_risk = phase in {"risk", "joint"}
            safe_gate_logit = direct_gate_logit
            if use_safe_risk:
                safe_gate_logit = (
                    safe_gate_logit
                    + safe_predicted_gain
                    / max(v6.SAFE_RISK_TEMPERATURE, v6.EPS)
                )
            safe_gate_logit = safe_gate_logit.clamp(-12.0, 12.0)
            safe_gate = torch.sigmoid(safe_gate_logit)

            if self.ablation_variant == "no_safe_anchor":
                # Keep a zero-valued graph connection to the safe head so the
                # original curriculum remains executable, but deploy the legacy
                # posterior directly as the proposal anchor.
                safe_gate = torch.ones_like(safe_gate)
                safe_update = legacy_candidate - anchor_depth
                safe_posterior = v6.safe_depth(
                    legacy_candidate + 0.0 * direct_gate_logit
                )
                safe_support = mask.detach()
            else:
                safe_update = (
                    safe_support * safe_gate * legacy_residual
                )
                safe_posterior = v6.safe_depth(
                    anchor_depth + safe_update
                )

            refine_support = mask * torch.clamp(
                v6.SAFE_SUPPORT_FLOOR
                + (1.0 - v6.SAFE_SUPPORT_FLOOR) * p_fail
                + v6.BOUNDARY_SUPPORT_WEIGHT * boundary,
                0.0,
                1.0,
            )
            refine_support = refine_support.detach()

            proposal_input = torch.cat(
                [
                    rgb,
                    v6.norm_depth(safe_posterior),
                    v6.norm_depth(raw),
                    v6.norm_depth(posterior["rel_metric"]),
                    v6.norm_depth(anchor_depth),
                    mask,
                    boundary,
                    posterior["sdm"],
                    p_fail,
                    posterior["final_logb"],
                    posterior["route_entropy"],
                    refine_support,
                    posterior["alpha"],
                ],
                1,
            )

            if phase == "safe":
                delta = torch.zeros_like(safe_posterior)
                candidate = safe_posterior
                risk_before = torch.zeros_like(safe_posterior)
                risk_after = torch.zeros_like(safe_posterior)
                predicted_gain = torch.zeros_like(safe_posterior)
                acceptance_logit = torch.zeros_like(safe_posterior)
                acceptance = torch.ones_like(safe_posterior)
                accepted_update = torch.zeros_like(safe_posterior)
                final = safe_posterior
            else:
                delta = self.risk_refiner.propose(proposal_input)
                candidate_update = refine_support * delta
                candidate = v6.safe_depth(
                    safe_posterior + candidate_update
                )

                risk_input = torch.cat(
                    [
                        proposal_input.detach(),
                        v6.norm_depth(candidate.detach()),
                        (
                            delta.detach()
                            / max(v6.MAX_RISK_DELTA, v6.EPS)
                        ).clamp(-1.0, 1.0),
                        torch.clamp(
                            v6.gradient_mag(
                                candidate_update.detach()
                            )
                            / max(v6.MAX_RISK_DELTA, v6.EPS),
                            0.0,
                            4.0,
                        ),
                    ],
                    1,
                )
                risk_before, risk_after = (
                    self.risk_refiner.estimate_risk(risk_input)
                )
                predicted_gain = risk_before - risk_after
                acceptance_logit = (
                    predicted_gain - v6.RISK_ACCEPT_MARGIN
                ) / max(v6.RISK_TEMPERATURE, v6.EPS)
                acceptance = torch.sigmoid(acceptance_logit)
                if phase == "proposal":
                    acceptance = torch.ones_like(acceptance)
                    acceptance_logit = torch.full_like(
                        acceptance_logit, 12.0
                    )
                accepted_update = (
                    refine_support * acceptance * delta
                )
                final = v6.safe_depth(
                    safe_posterior + accepted_update
                )

            safe_benchmark = (
                safe_posterior * mask + raw * (1.0 - mask)
            )
            candidate_benchmark = (
                candidate * mask + raw * (1.0 - mask)
            )
            benchmark_output = (
                final * mask + raw * (1.0 - mask)
            )

            return {
                **posterior,
                "anchor_depth": anchor_depth,
                "legacy_fused": legacy_fused,
                "legacy_candidate": legacy_candidate,
                "legacy_residual": legacy_residual,
                "corruption_mask": corruption_mask,
                "safe_support": safe_support,
                "safe_direct_logit": direct_gate_logit,
                "safe_gate_logit": safe_gate_logit,
                "safe_gate": safe_gate,
                "safe_update": safe_update,
                "safe_posterior": safe_posterior,
                "safe_benchmark": safe_benchmark,
                "safe_risk_anchor": safe_risk_anchor,
                "safe_risk_legacy": safe_risk_legacy,
                "safe_predicted_gain": safe_predicted_gain,
                "final": final,
                "benchmark_output": benchmark_output,
                "candidate": candidate,
                "candidate_benchmark": candidate_benchmark,
                "support": refine_support,
                "delta": delta,
                "effective_candidate_update": (
                    candidate - safe_posterior
                ),
                "accepted_update": accepted_update,
                "risk_before": risk_before,
                "risk_after": risk_after,
                "predicted_gain": predicted_gain,
                "acceptance_logit": acceptance_logit,
                "acceptance": acceptance,
                "ablation_variant": self.ablation_variant,
            }

    return AblationFailureAwarePosteriorDepth



def release_memory(*objects: Any) -> None:
    # Drop caller-owned temporary references by convention, then force Python
    # and CUDA allocators to release reusable blocks.
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def prepare_model_memory(model: nn.Module, v6) -> nn.Module:
    if USE_CPU_ANCHOR and v6.DEVICE == "cuda":
        model.base_reference.to("cpu")
        model.base_reference.eval()
        for parameter in model.base_reference.parameters():
            parameter.requires_grad_(False)
        print(
            "[Low-memory] Frozen FDCT reference anchor is on CPU; "
            "training will be slower but GPU memory is substantially lower."
        )
    return model


@torch.no_grad()
def low_memory_evaluate(
    v6,
    model: nn.Module,
    loader: DataLoader,
    phase: str,
    desc: str = "Val",
    use_amp: bool = False,
):
    """v6 evaluation that slices cache shards on CPU before CUDA transfer."""
    model.eval()
    names = [
        "Raw Depth",
        "Input relative prior",
        "Metric-calibrated prior",
        "Previous model result",
        "Base anchor",
        "Legacy posterior fusion",
        "Safe residual posterior",
        "Candidate correction",
        "Risk-accepted refinement",
        "Oracle anchor-posterior",
        "Oracle safe-candidate",
        "Safe benchmark",
        "Candidate benchmark",
        "Benchmark output",
    ]
    rows: Dict[str, List[Dict[str, float]]] = {name: [] for name in names}
    aux: List[Dict[str, float]] = []

    for loader_batch_cpu in tqdm(loader, desc=desc, leave=False):
        for batch_cpu, _, _ in v6.iter_microbatches(
            loader_batch_cpu,
            v6.VAL_MICROBATCH,
        ):
            batch = v6.to_device(batch_cpu)
            with torch.cuda.amp.autocast(
                enabled=bool(use_amp and v6.USE_AMP)
            ):
                _, stats, inp, out = v6.compute_loss(
                    model,
                    batch,
                    phase=phase,
                    return_outputs=True,
                )
            raw, gt = inp["raw"], inp["gt"]
            mask, valid = inp["mask"], inp["valid"]
            previous = inp["old_base"] * mask + raw * (1.0 - mask)
            anchor_benchmark = out["anchor_depth"] * mask + raw * (1.0 - mask)
            legacy_benchmark = out["legacy_fused"] * mask + raw * (1.0 - mask)

            anchor_err = torch.abs(out["anchor_depth"] - gt)
            legacy_err = torch.abs(out["legacy_fused"] - gt)
            oracle_anchor = torch.where(
                legacy_err < anchor_err,
                out["legacy_fused"],
                out["anchor_depth"],
            )
            oracle_anchor_benchmark = oracle_anchor * mask + raw * (1.0 - mask)

            safe_err = torch.abs(out["safe_posterior"] - gt)
            candidate_err = torch.abs(out["candidate"] - gt)
            oracle_refine = torch.where(
                candidate_err < safe_err,
                out["candidate"],
                out["safe_posterior"],
            )
            oracle_refine_benchmark = oracle_refine * mask + raw * (1.0 - mask)

            metric = v6.metric_values
            rows["Raw Depth"].append(metric(raw, raw, gt, mask, valid))
            rows["Input relative prior"].append(metric(inp["rel"], raw, gt, mask, valid))
            rows["Metric-calibrated prior"].append(metric(out["rel_metric"], raw, gt, mask, valid))
            rows["Previous model result"].append(metric(previous, raw, gt, mask, valid))
            rows["Base anchor"].append(metric(anchor_benchmark, raw, gt, mask, valid))
            rows["Legacy posterior fusion"].append(metric(legacy_benchmark, raw, gt, mask, valid))
            rows["Safe residual posterior"].append(metric(out["safe_posterior"], raw, gt, mask, valid))
            rows["Candidate correction"].append(metric(out["candidate"], raw, gt, mask, valid))
            rows["Risk-accepted refinement"].append(metric(out["final"], raw, gt, mask, valid))
            rows["Oracle anchor-posterior"].append(metric(oracle_anchor_benchmark, raw, gt, mask, valid))
            rows["Oracle safe-candidate"].append(metric(oracle_refine_benchmark, raw, gt, mask, valid))
            rows["Safe benchmark"].append(metric(out["safe_benchmark"], raw, gt, mask, valid))
            rows["Candidate benchmark"].append(metric(out["candidate_benchmark"], raw, gt, mask, valid))
            rows["Benchmark output"].append(metric(out["benchmark_output"], raw, gt, mask, valid))
            aux.append(stats)

            del batch, batch_cpu, inp, out
            if v6.DEVICE == "cuda":
                torch.cuda.empty_cache()

        del loader_batch_cpu

    avg = {name: v6.avg_dicts(values) for name, values in rows.items()}
    avg["_aux"] = v6.avg_dicts(aux)
    return avg


# =============================================================================
# DATA / OPTIMIZER / CHECKPOINT HELPERS
# =============================================================================
def build_loader(
    v6,
    split: str,
    max_shards: Optional[int],
    shuffle: bool,
    seed: int,
) -> DataLoader:
    shards = v6.load_split_shards(
        v6.CACHE_ROOT,
        split,
        max_shards,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        v6.CachedShardDataset(shards),
        batch_size=v6.LOADER_BATCH_SIZE,
        shuffle=shuffle,
        num_workers=v6.NUM_WORKERS,
        pin_memory=v6.PIN_MEMORY,
        collate_fn=v6.ragged_shard_collate,
        generator=generator if shuffle else None,
    )


def create_optimizer(v6, model: nn.Module):
    safe_gate_params = list(model.safe_anchor.gate_parameters())
    safe_risk_params = list(model.safe_anchor.risk_parameters())
    proposal_params = list(
        model.risk_refiner.proposal_parameters()
    )
    refine_risk_params = list(
        model.risk_refiner.risk_parameters()
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": safe_gate_params,
                "lr": v6.LR_SAFE_WARMUP,
                "name": "safe_gate",
            },
            {
                "params": safe_risk_params,
                "lr": 0.0,
                "name": "safe_risk",
            },
            {
                "params": proposal_params,
                "lr": 0.0,
                "name": "proposal",
            },
            {
                "params": refine_risk_params,
                "lr": 0.0,
                "name": "refine_risk",
            },
        ],
        weight_decay=v6.WEIGHT_DECAY,
    )
    return (
        optimizer,
        {
            "safe_gate": safe_gate_params,
            "safe_risk": safe_risk_params,
            "proposal": proposal_params,
            "refine_risk": refine_risk_params,
        },
    )


def save_ablation_checkpoint(
    path: Path,
    v6,
    variant: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    phase: str,
    rows: Dict[str, Dict[str, float]],
    history: List[Dict[str, Any]],
    best_score: float,
    best_safe_score: float,
    best_candidate_score: float,
    no_improve_joint: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ablation_variant": variant,
            "refine_epoch": int(epoch),
            "phase": phase,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": (
                scaler.state_dict()
                if scaler is not None
                else None
            ),
            "best_score": float(best_score),
            "best_safe_score": float(best_safe_score),
            "best_candidate_score": float(
                best_candidate_score
            ),
            "no_improve_joint": int(no_improve_joint),
            "history": history,
            "all_rows": {
                k: val
                for k, val in rows.items()
                if not k.startswith("_")
            },
            "aux": rows.get("_aux", {}),
            "source_checkpoint": str(v6.V5_CKPT),
            "protocol": {
                "seed": SEED,
                "train_split": "train",
                "selection_split": "val",
                "final_split": "test",
                "primary_output": "Candidate benchmark",
                "schedule": {
                    "safe": v6.SAFE_WARMUP_EPOCHS,
                    "proposal": v6.PROPOSAL_ADAPT_EPOCHS,
                    "risk": v6.RISK_CALIBRATION_EPOCHS,
                    "joint": v6.JOINT_EPOCHS,
                },
            },
        },
        str(path),
    )


def write_variant_history(
    variant_dir: Path,
    history: Sequence[Mapping[str, Any]],
) -> None:
    write_json(variant_dir / "training_history.json", list(history))
    write_csv(variant_dir / "training_history.csv", list(history))


def load_model_checkpoint(
    v6,
    model_cls,
    base_mod,
    variant: str,
    checkpoint: Path,
):
    model = prepare_model_memory(
        model_cls(base_mod, variant).to(v6.DEVICE),
        v6,
    )
    payload = safe_torch_load(checkpoint, map_location="cpu")
    state = payload.get(
        "model",
        payload.get("model_state_dict", payload),
    )
    clean = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(
        clean,
        strict=False,
    )
    metadata = {
        "phase": payload.get("phase", "joint"),
        "refine_epoch": payload.get("refine_epoch"),
        "ablation_variant": payload.get("ablation_variant", variant),
    }
    del clean, state, payload
    release_memory()

    if missing or unexpected:
        print(
            f"[Checkpoint load] {variant}: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print("  missing first:", missing[:10])
        if unexpected:
            print("  unexpected first:", unexpected[:10])
    return model, metadata


# =============================================================================
# TRAIN ONE VARIANT
# =============================================================================
def train_variant(
    v6,
    model_cls,
    base_mod,
    variant: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    force_retrain: bool,
) -> Path:
    variant_dir = OUT_ROOT / variant
    checkpoint_dir = variant_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_candidate_path = checkpoint_dir / "best_candidate.pth"
    last_path = checkpoint_dir / "last.pth"

    if (
        best_candidate_path.exists()
        and not force_retrain
        and not last_path.exists()
    ):
        print(
            f"[Skip completed] {variant}: "
            f"{best_candidate_path}"
        )
        return best_candidate_path

    set_seed(SEED)
    model = prepare_model_memory(model_cls(base_mod, variant).to(v6.DEVICE), v6)
    optimizer, param_groups = create_optimizer(v6, model)
    scaler = torch.cuda.amp.GradScaler(
        enabled=v6.USE_AMP
    )

    start_epoch = 1
    history: List[Dict[str, Any]] = []
    best_score = float("inf")
    best_safe_score = float("inf")
    best_candidate_score = float("inf")
    no_improve_joint = 0

    if last_path.exists() and not force_retrain:
        payload = safe_torch_load(last_path, map_location="cpu")
        model.load_state_dict(payload["model"], strict=False)
        optimizer.load_state_dict(payload["optimizer"])
        if (
            v6.USE_AMP
            and payload.get("scaler") is not None
        ):
            scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload.get("refine_epoch", 0)) + 1
        history = list(payload.get("history", []))
        best_score = float(
            payload.get("best_score", float("inf"))
        )
        best_safe_score = float(
            payload.get("best_safe_score", float("inf"))
        )
        best_candidate_score = float(
            payload.get(
                "best_candidate_score",
                float("inf"),
            )
        )
        no_improve_joint = int(
            payload.get("no_improve_joint", 0)
        )
        del payload
        release_memory()
        print(
            f"[Resume] {variant}: completed={start_epoch - 1}, "
            f"best_candidate={best_candidate_score:.6f}"
        )
    else:
        source_payload = model.load_v5_checkpoint(v6.V5_CKPT)
        source_epoch = source_payload.get("refine_epoch", "unknown")
        del source_payload
        release_memory()
        model.set_training_phase("safe")
        initial_rows = low_memory_evaluate(
            v6,
            model,
            val_loader,
            phase="safe",
            desc=f"{variant} initial val",
        )
        v6.print_summary(
            f"{variant} initialization",
            None,
            "safe",
            initial_rows,
        )

        best_score = v6.selection_score(
            initial_rows["Benchmark output"]
        )
        best_safe_score = v6.selection_score(
            initial_rows["Safe benchmark"]
        )
        best_candidate_score = v6.selection_score(
            initial_rows["Candidate benchmark"]
        )

        for name in (
            "initial_anchor.pth",
            "best_score.pth",
            "best_safe.pth",
            "best_candidate.pth",
        ):
            save_ablation_checkpoint(
                checkpoint_dir / name,
                v6,
                variant,
                model,
                optimizer,
                scaler,
                0,
                "safe",
                initial_rows,
                history,
                best_score,
                best_safe_score,
                best_candidate_score,
                no_improve_joint,
            )
        print(
            f"[Initialized] {variant} from v5 "
            f"epoch={source_epoch}"
        )

    if start_epoch > v6.REFINE_EPOCHS:
        print(
            f"[Already complete] {variant}: "
            f"{start_epoch - 1}/{v6.REFINE_EPOCHS}"
        )
        if last_path.exists():
            last_path.unlink(missing_ok=True)
        return best_candidate_path

    print(
        f"\n{'=' * 130}\n"
        f"TRAIN ABLATION: {variant} | {VARIANT_TITLES[variant]}\n"
        f"{'=' * 130}"
    )
    print(
        "Trainable parameter groups: "
        + ", ".join(
            f"{name}={sum(p.numel() for p in params):,}"
            for name, params in param_groups.items()
        )
    )

    for epoch in range(start_epoch, v6.REFINE_EPOCHS + 1):
        phase = v6.phase_for_refine_epoch(epoch)
        v6.configure_phase(model, optimizer, phase)
        model.train(True)

        active_params = [
            param
            for group in optimizer.param_groups
            for param in group["params"]
            if param.requires_grad
        ]
        epoch_stats: List[Dict[str, float]] = []
        progress = tqdm(
            train_loader,
            desc=(
                f"{variant} {epoch}/{v6.REFINE_EPOCHS} "
                f"| {phase}"
            ),
        )

        for step, loader_batch_cpu in enumerate(progress, 1):
            optimizer.zero_grad(set_to_none=True)
            micro_stats: List[Dict[str, float]] = []

            for batch_cpu, micro_n, total_n in v6.iter_microbatches(
                loader_batch_cpu,
                v6.TRAIN_MICROBATCH,
            ):
                # The original script moved the entire cache shard to CUDA before
                # slicing it. Here only one microbatch is transferred.
                batch = v6.to_device(batch_cpu)
                with torch.cuda.amp.autocast(
                    enabled=v6.USE_AMP
                ):
                    total, stats = v6.compute_loss(
                        model,
                        batch,
                        phase=phase,
                    )
                    scaled = total * (
                        float(micro_n) / float(total_n)
                    )
                scaler.scale(scaled).backward()
                micro_stats.append(stats)
                del batch, batch_cpu, total, scaled

            scaler.unscale_(optimizer)
            if active_params:
                torch.nn.utils.clip_grad_norm_(
                    active_params,
                    v6.CLIP_GRAD,
                )
            scaler.step(optimizer)
            scaler.update()
            del loader_batch_cpu

            step_stats = v6.avg_dicts(micro_stats)
            epoch_stats.append(step_stats)
            progress.set_postfix(
                loss=f"{step_stats.get('loss_total', 0):.4f}",
                gate=f"{step_stats.get('safe_gate_mean', 0):.2f}",
                safe=f"{step_stats.get('safe_mae_mask', 0):.5f}",
                cand=f"{step_stats.get('candidate_mae_mask', 0):.5f}",
                mb=v6.TRAIN_MICROBATCH,
            )

            if (
                v6.DEVICE == "cuda"
                and v6.EMPTY_CACHE_EVERY > 0
                and step % max(1, v6.EMPTY_CACHE_EVERY) == 0
            ):
                torch.cuda.empty_cache()

        train_loss = float(
            np.mean(
                [
                    row.get("loss_total", float("nan"))
                    for row in epoch_stats
                ]
            )
        )
        rows = low_memory_evaluate(
            v6,
            model,
            val_loader,
            phase=phase,
            desc=f"{variant} val epoch {epoch}",
        )
        v6.print_summary(
            f"{variant} epoch {epoch}",
            train_loss,
            phase,
            rows,
        )

        score = v6.selection_score(
            rows["Benchmark output"]
        )
        safe_score = v6.selection_score(
            rows["Safe benchmark"]
        )
        candidate_score = v6.selection_score(
            rows["Candidate benchmark"]
        )

        history_row: Dict[str, Any] = {
            "variant": variant,
            "refine_epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "score": score,
            "safe_score": safe_score,
            "candidate_score": candidate_score,
            **{
                f"candidate_{key}": value
                for key, value in rows[
                    "Candidate benchmark"
                ].items()
            },
            **{
                f"safe_{key}": value
                for key, value in rows[
                    "Safe benchmark"
                ].items()
            },
            **{
                f"aux_{key}": value
                for key, value in rows["_aux"].items()
            },
        }
        history.append(history_row)
        write_variant_history(variant_dir, history)

        if safe_score < best_safe_score - 1.0e-8:
            best_safe_score = safe_score
            save_ablation_checkpoint(
                checkpoint_dir / "best_safe.pth",
                v6,
                variant,
                model,
                optimizer,
                scaler,
                epoch,
                phase,
                rows,
                history,
                best_score,
                best_safe_score,
                best_candidate_score,
                no_improve_joint,
            )
            print(
                f"[Best safe] {variant}: "
                f"{best_safe_score:.6f}"
            )

        if candidate_score < best_candidate_score - 1.0e-8:
            best_candidate_score = candidate_score
            save_ablation_checkpoint(
                best_candidate_path,
                v6,
                variant,
                model,
                optimizer,
                scaler,
                epoch,
                phase,
                rows,
                history,
                best_score,
                best_safe_score,
                best_candidate_score,
                no_improve_joint,
            )
            print(
                f"[Best candidate] {variant}: "
                f"{best_candidate_score:.6f}"
            )

        if score < best_score - 1.0e-8:
            best_score = score
            no_improve_joint = 0
            save_ablation_checkpoint(
                checkpoint_dir / "best_score.pth",
                v6,
                variant,
                model,
                optimizer,
                scaler,
                epoch,
                phase,
                rows,
                history,
                best_score,
                best_safe_score,
                best_candidate_score,
                no_improve_joint,
            )
            print(
                f"[Best final] {variant}: "
                f"{best_score:.6f}"
            )
        elif phase == "joint":
            no_improve_joint += 1
            print(
                f"[Joint no improvement] {variant}: "
                f"{no_improve_joint}/"
                f"{v6.JOINT_EARLY_STOP_PATIENCE}"
            )

        save_ablation_checkpoint(
            last_path,
            v6,
            variant,
            model,
            optimizer,
            scaler,
            epoch,
            phase,
            rows,
            history,
            best_score,
            best_safe_score,
            best_candidate_score,
            no_improve_joint,
        )

        if (
            phase == "joint"
            and no_improve_joint
            >= v6.JOINT_EARLY_STOP_PATIENCE
        ):
            print(
                f"[Early stop] {variant}: "
                "joint validation plateau."
            )
            break

    # A missing last.pth is used as the completed marker.
    last_path.unlink(missing_ok=True)
    del model, optimizer, scaler
    release_memory()
    return best_candidate_path


# =============================================================================
# EVALUATE FIXED CHECKPOINT ON TEST
# =============================================================================
@torch.no_grad()
def evaluate_checkpoint(
    v6,
    model_cls,
    base_mod,
    variant: str,
    checkpoint: Path,
    test_loader: DataLoader,
) -> Dict[str, Any]:
    model, payload = load_model_checkpoint(
        v6,
        model_cls,
        base_mod,
        variant,
        checkpoint,
    )
    checkpoint_phase = str(payload.get("phase", "joint"))
    rows = low_memory_evaluate(
        v6,
        model,
        test_loader,
        phase=checkpoint_phase,
        desc=f"TEST {variant}",
        use_amp=v6.USE_AMP,
    )
    v6.print_summary(
        f"TEST {variant}",
        None,
        checkpoint_phase,
        rows,
    )

    if variant == "full":
        check_full_reference(
            v6,
            rows,
            strict=not USE_CPU_ANCHOR,
        )

    result = {
        "variant": variant,
        "title": VARIANT_TITLES[variant],
        "checkpoint": str(checkpoint),
        "checkpoint_phase": checkpoint_phase,
        "refine_epoch": payload.get("refine_epoch"),
        "protocol": {
            "cpu_anchor": bool(USE_CPU_ANCHOR),
            "test_amp": bool(v6.USE_AMP),
            "microbatch": int(v6.VAL_MICROBATCH),
        },
        "rows": rows,
    }
    variant_dir = OUT_ROOT / variant
    write_json(
        variant_dir / "test_summary.json",
        result,
    )
    write_csv(
        variant_dir / "test_summary.csv",
        [
            {
                "row": name,
                **metrics,
                "score": v6.selection_score(metrics),
            }
            for name, metrics in rows.items()
            if not name.startswith("_")
        ],
    )
    del model
    release_memory()
    return result



def check_full_reference(
    v6,
    rows: Mapping[str, Mapping[str, Any]],
    strict: bool,
) -> None:
    candidate = dict(rows["Candidate benchmark"])
    candidate["score"] = v6.selection_score(candidate)
    tolerances = {
        "rmse_mask": 2.0e-4,
        "rel_mask": 3.0e-4,
        "mae_mask": 1.5e-4,
        "delta_105": 3.0e-3,
        "delta_110": 2.0e-3,
        "delta_125": 5.0e-4,
        "score": 3.0e-4,
    }
    mismatches: List[str] = []
    for key, expected in OFFICIAL_FULL_REFERENCE.items():
        actual = finite_float(candidate.get(key))
        if actual is None:
            mismatches.append(f"{key}=missing")
            continue
        if abs(actual - expected) > tolerances[key]:
            mismatches.append(
                f"{key}: actual={actual:.6f}, expected={expected:.6f}, "
                f"|diff|={abs(actual - expected):.6f}"
            )

    if not mismatches:
        print(
            "[Full-reference check] PASS: current evaluation reproduces the "
            "accepted best_candidate benchmark within tolerance."
        )
        return

    message = (
        "Full-model evaluation does not reproduce the accepted v6 main result:\n- "
        + "\n- ".join(mismatches)
        + "\nDo not use this run for the paper ablation table. "
          "The usual cause is --cpu-anchor, a stale script/checkpoint, or routing the full model through an ablation forward implementation."
    )
    if strict:
        raise RuntimeError(message)
    print("[Full-reference warning] " + message)


# =============================================================================
# AGGREGATION
# =============================================================================
def aggregate_results(
    v6,
    results: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if "full" not in results:
        raise RuntimeError(
            "The aggregate ablation table requires the full model."
        )

    full = results["full"]
    full_rows = full["rows"]
    table: List[Dict[str, Any]] = []

    # Exact output-path ablations from the same full checkpoint.
    path_rows = [
        (
            "fdct_anchor",
            OUTPUT_PATH_TITLES["fdct_anchor"],
            "Base anchor",
        ),
        (
            "legacy_posterior",
            OUTPUT_PATH_TITLES["legacy_posterior"],
            "Legacy posterior fusion",
        ),
        (
            "no_detail_proposal",
            OUTPUT_PATH_TITLES["no_detail_proposal"],
            "Safe benchmark",
        ),
        (
            "full",
            OUTPUT_PATH_TITLES["full"],
            "Candidate benchmark",
        ),
    ]
    for key, title, row_name in path_rows:
        row = dict(full_rows[row_name])
        row["score"] = v6.selection_score(row)
        table.append(
            metric_row(
                title=title,
                variant_key=key,
                row=row,
                source=full["checkpoint"],
                row_type="same_checkpoint_output_path",
            )
        )

    for variant in TRAINABLE_VARIANTS:
        if variant == "full" or variant not in results:
            continue
        result = results[variant]
        row = dict(result["rows"]["Candidate benchmark"])
        row["score"] = v6.selection_score(row)
        table.append(
            metric_row(
                title=VARIANT_TITLES[variant],
                variant_key=variant,
                row=row,
                source=result["checkpoint"],
                row_type="retrained_v6_stage_ablation",
            )
        )

    full_score = next(
        row["Score"]
        for row in table
        if row["variant_key"] == "full"
    )
    full_rmse = next(
        row["RMSE"]
        for row in table
        if row["variant_key"] == "full"
    )
    full_mae = next(
        row["MAE"]
        for row in table
        if row["variant_key"] == "full"
    )

    for row in table:
        row["delta_score_vs_full"] = (
            None
            if row["Score"] is None or full_score is None
            else row["Score"] - full_score
        )
        row["delta_rmse_vs_full"] = (
            None
            if row["RMSE"] is None or full_rmse is None
            else row["RMSE"] - full_rmse
        )
        row["delta_mae_vs_full"] = (
            None
            if row["MAE"] is None or full_mae is None
            else row["MAE"] - full_mae
        )

    return table


def print_aggregate_table(rows: Sequence[Mapping[str, Any]]) -> None:
    print("\n" + "=" * 165)
    print("FAPR-DEPTH v6 COMPONENT ABLATION | TEST SPLIT")
    print("=" * 165)
    print(
        f"{'Method':<39} | {'RMSE':>8} | {'REL':>8} | "
        f"{'MAE':>8} | {'d1.05':>8} | {'d1.10':>8} | "
        f"{'d1.25':>8} | {'Boundary':>9} | {'Score':>8} | "
        f"{'ΔScore':>8}"
    )
    print("-" * 165)
    for row in rows:
        print(
            f"{row['method']:<39} | "
            f"{fmt(row.get('RMSE')):>8} | "
            f"{fmt(row.get('REL')):>8} | "
            f"{fmt(row.get('MAE')):>8} | "
            f"{fmt(row.get('delta_1_05'), percent=True):>8} | "
            f"{fmt(row.get('delta_1_10'), percent=True):>8} | "
            f"{fmt(row.get('delta_1_25'), percent=True):>8} | "
            f"{fmt(row.get('Boundary')):>9} | "
            f"{fmt(row.get('Score')):>8} | "
            f"{fmt(row.get('delta_score_vs_full')):>8}"
        )
    print("=" * 165)


# =============================================================================
# CLI / MAIN
# =============================================================================
def parse_variants(raw: str) -> List[str]:
    raw = raw.strip().lower()
    if raw == "all":
        return list(TRAINABLE_VARIANTS)
    requested = [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ]
    unknown = [
        item
        for item in requested
        if item not in TRAINABLE_VARIANTS
    ]
    if unknown:
        raise ValueError(
            f"Unknown variants: {unknown}. "
            f"Available: {TRAINABLE_VARIANTS}"
        )
    return requested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and test FAPR-Depth v6 component ablations."
        )
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="no_relative_prior",
        help=(
            "Comma-separated variants or 'all'. "
            f"Available: {', '.join(TRAINABLE_VARIANTS)}"
        ),
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing per-variant checkpoints.",
    )
    parser.add_argument(
        "--no-reuse-full",
        action="store_true",
        help=(
            "Retrain the full model instead of reusing the existing "
            "v6 best_candidate.pth."
        ),
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help=(
            "Train/select on validation only. Do not evaluate the "
            "fixed checkpoints on test."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast pipeline check: 8 train shards, 4 val shards, "
            "4 test shards, and a two-epoch safe/proposal schedule."
        ),
    )
    parser.add_argument(
        "--max-train-shards",
        type=int,
        default=-1,
        help="-1 uses the original v6 setting.",
    )
    parser.add_argument(
        "--max-val-shards",
        type=int,
        default=-1,
        help="-1 uses the original v6 setting.",
    )
    parser.add_argument(
        "--max-test-shards",
        type=int,
        default=0,
        help="0 evaluates the complete test split.",
    )
    parser.add_argument(
        "--train-microbatch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--val-microbatch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cpu-anchor",
        action="store_true",
        help=(
            "Emergency fallback only: keep the frozen FDCT reference-anchor "
            "stream on CPU. This lowers VRAM but can change the numerical output; "
            "do not use it for the final paper table unless the full-reference "
            "check remains consistent."
        ),
    )
    parser.add_argument(
        "--keep-pinned-memory",
        action="store_true",
        help="Keep the original CUDA pinned-memory loader behavior.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Do not train. Read the saved per-variant test_summary.json files "
            "and build the final ablation table."
        ),
    )
    return parser.parse_args()


def configure_runtime(v6, args: argparse.Namespace) -> None:
    global USE_CPU_ANCHOR
    USE_CPU_ANCHOR = bool(args.cpu_anchor)
    v6.NUM_WORKERS = 0
    v6.PIN_MEMORY = bool(args.keep_pinned_memory and v6.DEVICE == "cuda")
    v6.EMPTY_CACHE_EVERY = 10
    v6.TRAIN_MICROBATCH = max(
        1, int(args.train_microbatch)
    )
    v6.VAL_MICROBATCH = max(
        1, int(args.val_microbatch)
    )

    if args.max_train_shards >= 0:
        v6.MAX_TRAIN_SHARDS = (
            None
            if args.max_train_shards == 0
            else int(args.max_train_shards)
        )
    if args.max_val_shards >= 0:
        v6.MAX_VAL_SHARDS = (
            None
            if args.max_val_shards == 0
            else int(args.max_val_shards)
        )

    if USE_CPU_ANCHOR:
        print(
            "[WARNING] --cpu-anchor changes the device/precision path of the "
            "frozen reference FDCT. Use it only after the normal low-memory "
            "microbatch route still produces CUDA OOM."
        )

    if args.smoke:
        v6.MAX_TRAIN_SHARDS = 8
        v6.MAX_VAL_SHARDS = 4
        v6.SAFE_WARMUP_EPOCHS = 1
        v6.PROPOSAL_ADAPT_EPOCHS = 1
        v6.RISK_CALIBRATION_EPOCHS = 0
        v6.JOINT_EPOCHS = 0
        v6.REFINE_EPOCHS = 2
        v6.JOINT_EARLY_STOP_PATIENCE = 1



def load_saved_test_results(
    variants: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for variant in variants:
        path = OUT_ROOT / variant / "test_summary.json"
        if not path.exists():
            missing.append(f"{variant}: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        results[variant] = data
    if missing:
        raise RuntimeError(
            "Missing per-variant test summaries:\n- "
            + "\n- ".join(missing)
        )
    return results


def main() -> None:
    global OUT_ROOT, AGGREGATE_DIR

    args = parse_args()
    if args.smoke:
        # Smoke checkpoints must never be mistaken for completed formal
        # ablations. Keep the entire smoke pipeline in a separate tree.
        OUT_ROOT = (
            PROJECT_ROOT
            / "outputs"
            / "fapr_depth_v6_component_ablation_v4_smoke"
        )
        AGGREGATE_DIR = OUT_ROOT / "aggregate"
        AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)

    variants = parse_variants(args.variants)

    if not args.aggregate_only and len(variants) != 1:
        raise RuntimeError(
            "Low-memory mode trains exactly one variant per Python process. "
            "Run variants one at a time, then use --aggregate-only."
        )

    if not V6_SOURCE.exists():
        raise FileNotFoundError(
            f"Place the original v6 training script at: {V6_SOURCE}"
        )

    v6 = import_by_path(
        "fapr_v6_ablation_base",
        V6_SOURCE,
    )
    configure_runtime(v6, args)
    set_seed(SEED)

    if args.aggregate_only:
        aggregate_variants = (
            TRAINABLE_VARIANTS
            if args.variants.strip().lower() == "all"
            else variants
        )
        results = load_saved_test_results(aggregate_variants)
        table = aggregate_results(v6, results)
        print_aggregate_table(table)
        write_csv(
            AGGREGATE_DIR / "fapr_v6_component_ablation_test.csv",
            table,
        )
        write_json(
            AGGREGATE_DIR / "fapr_v6_component_ablation_test.json",
            {"rows": table},
        )
        (AGGREGATE_DIR / "fapr_v6_component_ablation_test.md").write_text(
            markdown_table(table),
            encoding="utf-8",
        )
        (AGGREGATE_DIR / "fapr_v6_component_ablation_test.tex").write_text(
            latex_table(table),
            encoding="utf-8",
        )
        print(f"[Saved aggregate] {AGGREGATE_DIR}")
        return

    model_cls = build_ablation_model_class(v6)
    base_mod = v6.load_base_source_module()

    print("=" * 150)
    print("FAPR-Depth v6 component ablation v4")
    print("=" * 150)
    print(f"DEVICE={v6.DEVICE}, AMP={v6.USE_AMP}")
    print(f"V6_SOURCE={V6_SOURCE}")
    print(f"V5_CKPT={v6.V5_CKPT}")
    print(f"OUT_ROOT={OUT_ROOT}")
    print(f"variants={variants}")
    print(
        f"memory mode=CPU-sliced microbatch; "
        f"reference anchor={'CPU fallback' if USE_CPU_ANCHOR else 'CUDA (paper protocol)'}"
    )
    print(
        f"schedule safe/proposal/risk/joint="
        f"{v6.SAFE_WARMUP_EPOCHS}/"
        f"{v6.PROPOSAL_ADAPT_EPOCHS}/"
        f"{v6.RISK_CALIBRATION_EPOCHS}/"
        f"{v6.JOINT_EPOCHS}"
    )
    print(
        f"train/val shards="
        f"{v6.MAX_TRAIN_SHARDS}/"
        f"{v6.MAX_VAL_SHARDS}"
    )
    if args.smoke:
        print("[WARNING] Smoke-test results are not paper results.")

    train_loader = build_loader(
        v6,
        split="train",
        max_shards=v6.MAX_TRAIN_SHARDS,
        shuffle=True,
        seed=SEED,
    )
    val_loader = build_loader(
        v6,
        split="val",
        max_shards=v6.MAX_VAL_SHARDS,
        shuffle=False,
        seed=SEED + 1,
    )

    test_max = (
        None
        if args.max_test_shards <= 0
        else int(args.max_test_shards)
    )
    if args.smoke:
        test_max = 4

    selected_checkpoints: Dict[str, Path] = {}

    for variant in variants:
        if (
            variant == "full"
            and not args.no_reuse_full
            and EXISTING_FULL_CKPT.exists()
            and not args.smoke
        ):
            selected_checkpoints[variant] = EXISTING_FULL_CKPT
            print(
                f"[Reuse full model] {EXISTING_FULL_CKPT}"
            )
            continue

        checkpoint = train_variant(
            v6=v6,
            model_cls=model_cls,
            base_mod=base_mod,
            variant=variant,
            train_loader=train_loader,
            val_loader=val_loader,
            force_retrain=args.force_retrain,
        )
        selected_checkpoints[variant] = checkpoint
        if v6.DEVICE == "cuda":
            torch.cuda.empty_cache()

    write_json(
        AGGREGATE_DIR / "selected_checkpoints.json",
        {
            key: str(path)
            for key, path in selected_checkpoints.items()
        },
    )

    if args.no_test:
        print(
            "\nTraining/validation completed. Test evaluation was skipped."
        )
        return

    test_loader = build_loader(
        v6,
        split="test",
        max_shards=test_max,
        shuffle=False,
        seed=SEED + 2,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        checkpoint = selected_checkpoints[variant]
        results[variant] = evaluate_checkpoint(
            v6=v6,
            model_cls=model_cls,
            base_mod=base_mod,
            variant=variant,
            checkpoint=checkpoint,
            test_loader=test_loader,
        )
        if v6.DEVICE == "cuda":
            torch.cuda.empty_cache()

    if "full" not in results:
        print(
            "\n[Variant complete] Per-variant test summary saved. "
            "Run the remaining variants in fresh processes, then execute "
            "--variants all --aggregate-only."
        )
        return

    table = aggregate_results(v6, results)
    print_aggregate_table(table)

    suffix = "smoke" if args.smoke else "test"
    csv_path = (
        AGGREGATE_DIR
        / f"fapr_v6_component_ablation_{suffix}.csv"
    )
    json_path = (
        AGGREGATE_DIR
        / f"fapr_v6_component_ablation_{suffix}.json"
    )
    md_path = (
        AGGREGATE_DIR
        / f"fapr_v6_component_ablation_{suffix}.md"
    )
    tex_path = (
        AGGREGATE_DIR
        / f"fapr_v6_component_ablation_{suffix}.tex"
    )

    write_csv(csv_path, table)
    write_json(
        json_path,
        {
            "protocol": {
                "seed": SEED,
                "selection_split": "val",
                "evaluation_split": "test",
                "test_max_shards": test_max,
                "primary_output": "Candidate benchmark",
                "full_checkpoint": str(
                    selected_checkpoints["full"]
                ),
                "smoke": bool(args.smoke),
            },
            "rows": table,
        },
    )
    md_path.write_text(
        markdown_table(table),
        encoding="utf-8",
    )
    tex_path.write_text(
        latex_table(table),
        encoding="utf-8",
    )

    print("\n[Saved]")
    print(csv_path)
    print(json_path)
    print(md_path)
    print(tex_path)


if __name__ == "__main__":
    main()
