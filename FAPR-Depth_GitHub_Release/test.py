# -*- coding: utf-8 -*-
r"""
FAPR-Depth v6 complete multi-checkpoint test
============================================

This script evaluates the completed v6 safe-anchor training run on the full
`test` split.  It imports the matching training script, so model construction,
preprocessing, constants, and checkpoint keys remain exactly aligned.

Default checkpoint roles
------------------------
* safe_warmup_complete.pth -> conservative safe-gate checkpoint (Safe benchmark)
* best_safe.pth            -> paper/deployment core model (Safe benchmark)
* best_candidate.pth       -> candidate-proposal ablation (Candidate benchmark)
* best_score.pth           -> full refinement ablation (Benchmark output)

The script does NOT choose a model after looking at test performance.  The roles
above are fixed before evaluation.  `best_safe.pth` is the primary v6 model;
`best_score.pth` is the full proposal+risk ablation.

Outputs per checkpoint
----------------------
1. Sample-mean metrics (same aggregation style as validation).
2. Global pixel-weighted diagnostic metrics.
3. Per-sample long-form metrics and primary comparisons.
4. Failure confusion/precision/recall/F1/Brier/ECE.
5. Safe-gate diagnostics, including gain and damage magnitudes.
6. Safe-gate coverage and refinement-risk coverage curves.
7. Optional qualitative panels.

Combined outputs
----------------
A fixed-role comparison table combines Base, Legacy posterior, safe warm-up,
best safe posterior, best candidate, best final, and oracle diagnostics.

Direct run
----------
    python test_fapr_depth_v6_complete.py

Faster main-only run
--------------------
    python test_fapr_depth_v6_complete.py --checkpoint-set main

Small pipeline check
--------------------
    python test_fapr_depth_v6_complete.py --max-shards 10
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =============================================================================
# DEFAULT CONFIG
# =============================================================================
PROJECT_ROOT = Path(os.getenv("FAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent)))
TRAIN_SCRIPT = Path(os.getenv("FAPR_TRAIN_SCRIPT", str(PROJECT_ROOT / "train.py")))
CACHE_ROOT = Path(os.getenv("FAPR_CACHE_ROOT", str(PROJECT_ROOT / "data" / "cache")))
CKPT_DIR = Path(os.getenv("FAPR_CKPT_DIR", str(PROJECT_ROOT / "outputs" / "fapr_depth_v6_safe_anchor" / "checkpoints")))
OUT_DIR = Path(os.getenv("FAPR_TEST_OUT_DIR", str(PROJECT_ROOT / "outputs" / "fapr_depth_v6_complete_test")))

TEST_SPLIT = "test"
MAX_TEST_SHARDS: Optional[int] = None
LOADER_BATCH_SIZE = 1
REQUESTED_MICROBATCH = 4
NUM_WORKERS = 0
SEED = 6248
EMPTY_CACHE_EVERY = 80
ECE_BINS = 15
COVERAGE_POINTS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00)
MAX_COVERAGE_PIXELS = 2_000_000

VARIANT_ORDER = [
    "Raw Depth",
    "Input relative prior",
    "Metric-calibrated prior",
    "Previous model result",
    "Base anchor",
    "Legacy posterior fusion",
    "Safe benchmark",
    "Candidate benchmark",
    "Benchmark output",
    "Oracle anchor-posterior",
    "Oracle safe-candidate",
]


# =============================================================================
# CLI / GENERAL UTILITIES
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete test for FAPR-Depth v6 safe-anchor checkpoints."
    )
    parser.add_argument("--train-script", type=Path, default=TRAIN_SCRIPT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, default=CKPT_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--split", type=str, default=TEST_SPLIT)
    parser.add_argument("--max-shards", type=int, default=MAX_TEST_SHARDS)
    parser.add_argument("--microbatch", type=int, default=REQUESTED_MICROBATCH)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--checkpoint-set",
        choices=("all", "main", "safe-only"),
        default="all",
        help=(
            "all: warmup+best_safe+best_candidate+best_score; "
            "main: best_safe+best_score; safe-only: best_safe only"
        ),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--visualizations",
        type=int,
        default=0,
        help="Number of qualitative panels saved per checkpoint; 0 disables them.",
    )
    parser.add_argument(
        "--coverage-pixels",
        type=int,
        default=MAX_COVERAGE_PIXELS,
        help="Maximum sampled support pixels retained for each coverage curve.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def import_training_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Matching v6 training script not found: {path}\n"
            "Place this test script beside the training script or pass --train-script."
        )
    name = "fapr_v6_complete_test_training_definition"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training module from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def score_from_row(train_mod, row: Dict[str, float]) -> float:
    return float(train_mod.selection_score(row))


def pct_change(new: float, old: float) -> float:
    if not math.isfinite(new) or not math.isfinite(old) or abs(old) < 1.0e-12:
        return float("nan")
    return 100.0 * (new - old) / old


def is_cuda_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "cuda" in text and "out of memory" in text


def slice_cpu_batch(batch: Dict[str, Any], start: int, end: int, n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True).float() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


# =============================================================================
# CHECKPOINT ROLES
# =============================================================================
@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    filename: str
    role: str
    primary_variant: str
    fixed_phase: Optional[str] = None


ALL_SPECS = [
    CheckpointSpec(
        key="safe_warmup",
        filename="safe_warmup_complete.pth",
        role="Conservative safe warm-up",
        primary_variant="Safe benchmark",
        fixed_phase="safe",
    ),
    CheckpointSpec(
        key="best_safe",
        filename="best_safe.pth",
        role="Primary safe-anchor model",
        primary_variant="Safe benchmark",
    ),
    CheckpointSpec(
        key="best_candidate",
        filename="best_candidate.pth",
        role="Candidate-proposal ablation",
        primary_variant="Candidate benchmark",
    ),
    CheckpointSpec(
        key="best_final",
        filename="best_score.pth",
        role="Full proposal+risk ablation",
        primary_variant="Benchmark output",
    ),
]


def select_specs(name: str) -> List[CheckpointSpec]:
    if name == "all":
        return list(ALL_SPECS)
    if name == "main":
        return [s for s in ALL_SPECS if s.key in {"best_safe", "best_final"}]
    return [s for s in ALL_SPECS if s.key == "best_safe"]


# =============================================================================
# VECTORIZED SAMPLE-MEAN METRICS
# =============================================================================
def masked_mean_per_sample(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    numerator = (x * mask).sum(dim=dims)
    denominator = mask.sum(dim=dims)
    result = numerator / denominator.clamp_min(1.0e-6)
    return torch.where(
        denominator > 0,
        result,
        torch.full_like(result, float("nan")),
    )


def metrics_per_sample(
    train_mod,
    pred: torch.Tensor,
    raw: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pred = train_mod.safe_depth(pred)
    raw = train_mod.safe_depth(raw)
    gt = train_mod.safe_depth(gt)
    mask = mask.float().clamp(0.0, 1.0)
    valid = valid.float().clamp(0.0, 1.0)

    region_all = valid
    region_mask = valid * mask
    boundary = train_mod.build_boundary_ring(mask) * valid
    reliable_bg = (
        valid
        * (1.0 - mask)
        * (torch.abs(raw - gt) <= float(train_mod.RELIABLE_BG_THR)).float()
    )

    abs_err = torch.abs(pred - gt)
    sq_err = (pred - gt).square()
    min_depth = float(train_mod.MIN_DEPTH)
    ratio = torch.maximum(
        pred / gt.clamp_min(min_depth),
        gt / pred.clamp_min(min_depth),
    )

    return {
        "mae_all": masked_mean_per_sample(abs_err, region_all),
        "rmse_all": torch.sqrt(
            masked_mean_per_sample(sq_err, region_all).clamp_min(1.0e-12)
        ),
        "mae_mask": masked_mean_per_sample(abs_err, region_mask),
        "rmse_mask": torch.sqrt(
            masked_mean_per_sample(sq_err, region_mask).clamp_min(1.0e-12)
        ),
        "rel_mask": masked_mean_per_sample(
            abs_err / gt.clamp_min(min_depth), region_mask
        ),
        "delta_105": masked_mean_per_sample((ratio < 1.05).float(), region_mask),
        "delta_110": masked_mean_per_sample((ratio < 1.10).float(), region_mask),
        "delta_125": masked_mean_per_sample((ratio < 1.25).float(), region_mask),
        "boundary": masked_mean_per_sample(abs_err, boundary),
        "reliable_bg_disturbance": masked_mean_per_sample(
            torch.abs(pred - raw), reliable_bg
        ),
    }


def sample_metric_dict(
    metric_tensors: Dict[str, torch.Tensor], index: int
) -> Dict[str, float]:
    return {
        key: float(value[index].detach().float().cpu())
        for key, value in metric_tensors.items()
    }


# =============================================================================
# GLOBAL PIXEL-WEIGHTED METRICS
# =============================================================================
@dataclass
class PixelMetricAccumulator:
    abs_all: float = 0.0
    sq_all: float = 0.0
    n_all: float = 0.0
    abs_mask: float = 0.0
    sq_mask: float = 0.0
    rel_mask: float = 0.0
    delta_105: float = 0.0
    delta_110: float = 0.0
    delta_125: float = 0.0
    n_mask: float = 0.0
    abs_boundary: float = 0.0
    n_boundary: float = 0.0
    bg_disturbance: float = 0.0
    n_bg: float = 0.0

    @torch.no_grad()
    def update(self, train_mod, pred, raw, gt, mask, valid) -> None:
        pred = train_mod.safe_depth(pred)
        raw = train_mod.safe_depth(raw)
        gt = train_mod.safe_depth(gt)
        mask = mask.float().clamp(0.0, 1.0)
        valid = valid.float().clamp(0.0, 1.0)
        region_mask = valid * mask
        boundary = train_mod.build_boundary_ring(mask) * valid
        reliable_bg = (
            valid
            * (1.0 - mask)
            * (torch.abs(raw - gt) <= float(train_mod.RELIABLE_BG_THR)).float()
        )
        err = torch.abs(pred - gt)
        sq = (pred - gt).square()
        ratio = torch.maximum(
            pred / gt.clamp_min(float(train_mod.MIN_DEPTH)),
            gt / pred.clamp_min(float(train_mod.MIN_DEPTH)),
        )

        self.abs_all += float((err * valid).sum().cpu())
        self.sq_all += float((sq * valid).sum().cpu())
        self.n_all += float(valid.sum().cpu())
        self.abs_mask += float((err * region_mask).sum().cpu())
        self.sq_mask += float((sq * region_mask).sum().cpu())
        self.rel_mask += float(
            ((err / gt.clamp_min(float(train_mod.MIN_DEPTH))) * region_mask).sum().cpu()
        )
        self.delta_105 += float(((ratio < 1.05).float() * region_mask).sum().cpu())
        self.delta_110 += float(((ratio < 1.10).float() * region_mask).sum().cpu())
        self.delta_125 += float(((ratio < 1.25).float() * region_mask).sum().cpu())
        self.n_mask += float(region_mask.sum().cpu())
        self.abs_boundary += float((err * boundary).sum().cpu())
        self.n_boundary += float(boundary.sum().cpu())
        self.bg_disturbance += float((torch.abs(pred - raw) * reliable_bg).sum().cpu())
        self.n_bg += float(reliable_bg.sum().cpu())

    def finalize(self) -> Dict[str, float]:
        def div(a: float, b: float) -> float:
            return a / max(b, 1.0)

        return {
            "mae_all": div(self.abs_all, self.n_all),
            "rmse_all": math.sqrt(div(self.sq_all, self.n_all)),
            "mae_mask": div(self.abs_mask, self.n_mask),
            "rmse_mask": math.sqrt(div(self.sq_mask, self.n_mask)),
            "rel_mask": div(self.rel_mask, self.n_mask),
            "delta_105": div(self.delta_105, self.n_mask),
            "delta_110": div(self.delta_110, self.n_mask),
            "delta_125": div(self.delta_125, self.n_mask),
            "boundary": div(self.abs_boundary, self.n_boundary),
            "reliable_bg_disturbance": div(self.bg_disturbance, self.n_bg),
            "count_all": self.n_all,
            "count_mask": self.n_mask,
            "count_boundary": self.n_boundary,
            "count_reliable_bg": self.n_bg,
        }


# =============================================================================
# ONLINE FAILURE METRICS (LOW MEMORY)
# =============================================================================
@dataclass
class FailureAccumulator:
    bins: int = ECE_BINS
    confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((4, 4), dtype=np.int64)
    )
    brier_sum: float = 0.0
    count: float = 0.0
    ece_count: np.ndarray = field(init=False)
    ece_conf_sum: np.ndarray = field(init=False)
    ece_target_sum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.ece_count = np.zeros(self.bins, dtype=np.float64)
        self.ece_conf_sum = np.zeros(self.bins, dtype=np.float64)
        self.ece_target_sum = np.zeros(self.bins, dtype=np.float64)

    @torch.no_grad()
    def update(self, train_mod, out, inp) -> None:
        valid = inp["valid"] > 0.5
        labels, _ = train_mod.failure_targets(
            inp["raw"], inp["gt"], inp["valid"], inp["boundary"] * inp["valid"]
        )
        pred = out["fail_logits"].argmax(1, keepdim=True)
        target_np = labels[valid].detach().cpu().numpy().astype(np.int64)
        pred_np = pred[valid].detach().cpu().numpy().astype(np.int64)
        np.add.at(self.confusion, (target_np, pred_np), 1)

        onehot = F.one_hot(labels[:, 0], num_classes=4).permute(0, 3, 1, 2).float()
        valid_f = inp["valid"].float()
        brier_map = (out["fail_prob"] - onehot).square().sum(1, keepdim=True)
        self.brier_sum += float((brier_map * valid_f).sum().cpu())
        self.count += float(valid_f.sum().cpu())

        conf = (1.0 - out["fail_prob"][:, 0:1])[valid].detach().float().cpu().numpy()
        target = (labels > 0)[valid].float().detach().cpu().numpy()
        if conf.size:
            ids = np.minimum((conf * self.bins).astype(np.int64), self.bins - 1)
            self.ece_count += np.bincount(ids, minlength=self.bins)
            self.ece_conf_sum += np.bincount(ids, weights=conf, minlength=self.bins)
            self.ece_target_sum += np.bincount(ids, weights=target, minlength=self.bins)

    def finalize(self) -> Dict[str, Any]:
        cm = self.confusion
        names = ["valid", "missing", "biased", "boundary"]
        per_class: Dict[str, Dict[str, float]] = {}
        recalls = []
        total = max(float(cm.sum()), 1.0)
        for cls, name in enumerate(names):
            tp = float(cm[cls, cls])
            fp = float(cm[:, cls].sum() - cm[cls, cls])
            fn = float(cm[cls, :].sum() - cm[cls, cls])
            support = float(cm[cls, :].sum())
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
            recalls.append(recall)
            per_class[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "prevalence": support / total,
            }

        ece = 0.0
        ece_rows = []
        ece_total = max(float(self.ece_count.sum()), 1.0)
        for index in range(self.bins):
            n = float(self.ece_count[index])
            lo, hi = index / self.bins, (index + 1) / self.bins
            if n <= 0:
                ece_rows.append(
                    {
                        "bin": index,
                        "lower": lo,
                        "upper": hi,
                        "count": 0,
                        "confidence": None,
                        "accuracy": None,
                        "gap": None,
                    }
                )
                continue
            mean_conf = float(self.ece_conf_sum[index] / n)
            mean_acc = float(self.ece_target_sum[index] / n)
            gap = abs(mean_conf - mean_acc)
            ece += (n / ece_total) * gap
            ece_rows.append(
                {
                    "bin": index,
                    "lower": lo,
                    "upper": hi,
                    "count": int(n),
                    "confidence": mean_conf,
                    "accuracy": mean_acc,
                    "gap": gap,
                }
            )

        return {
            "accuracy": float(np.trace(cm)) / total,
            "balanced_accuracy": float(np.mean(recalls)),
            "brier": self.brier_sum / max(self.count, 1.0),
            "ece_any_failure": float(ece),
            "per_class": per_class,
            "confusion_matrix_target_rows_pred_columns": cm.tolist(),
            "ece_bins": ece_rows,
        }


# =============================================================================
# WEIGHTED DIAGNOSTICS / HISTOGRAM QUANTILES
# =============================================================================
class WeightedDiagnostics:
    def __init__(self, histogram_bins: int = 200) -> None:
        self.numerator: Dict[str, float] = defaultdict(float)
        self.denominator: Dict[str, float] = defaultdict(float)
        self.hist_bins = int(histogram_bins)
        self.hist: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.hist_bins, dtype=np.float64)
        )

    @torch.no_grad()
    def add(self, key: str, value: torch.Tensor, weight: torch.Tensor) -> None:
        w = weight.float()
        self.numerator[key] += float((value.float() * w).sum().cpu())
        self.denominator[key] += float(w.sum().cpu())

    @torch.no_grad()
    def add_conditional(
        self, key: str, value: torch.Tensor, weight: torch.Tensor, condition: torch.Tensor
    ) -> None:
        self.add(key, value, weight * condition.float())

    @torch.no_grad()
    def add_hist(self, key: str, value: torch.Tensor, weight: torch.Tensor) -> None:
        selected = weight > 0
        if not selected.any():
            return
        v = value[selected].detach().float().cpu().numpy()
        w = weight[selected].detach().float().cpu().numpy()
        ids = np.minimum((np.clip(v, 0.0, 1.0) * self.hist_bins).astype(np.int64), self.hist_bins - 1)
        self.hist[key] += np.bincount(ids, weights=w, minlength=self.hist_bins)

    def _quantile(self, key: str, q: float) -> float:
        hist = self.hist.get(key)
        if hist is None or hist.sum() <= 0:
            return float("nan")
        cdf = np.cumsum(hist)
        index = int(np.searchsorted(cdf, q * cdf[-1], side="left"))
        return min(max((index + 0.5) / self.hist_bins, 0.0), 1.0)

    def summary(self) -> Dict[str, float]:
        result = {
            key: self.numerator[key] / max(self.denominator[key], 1.0e-12)
            for key in sorted(self.numerator)
        }
        for hist_key in sorted(self.hist):
            for q, label in ((0.10, "q10"), (0.25, "q25"), (0.50, "q50"), (0.75, "q75"), (0.90, "q90")):
                result[f"{hist_key}_{label}"] = self._quantile(hist_key, q)
        return result


# =============================================================================
# SAMPLED COVERAGE CURVES
# =============================================================================
class CoverageAccumulator:
    def __init__(self, max_pixels: int, seed: int) -> None:
        self.max_pixels = max(1, int(max_pixels))
        self.rng = np.random.default_rng(seed)
        self.score_parts: List[np.ndarray] = []
        self.gain_parts: List[np.ndarray] = []
        self.before_parts: List[np.ndarray] = []
        self.after_parts: List[np.ndarray] = []
        self.count = 0

    @torch.no_grad()
    def update(
        self,
        score: torch.Tensor,
        true_gain: torch.Tensor,
        before_error: torch.Tensor,
        after_error: torch.Tensor,
        region: torch.Tensor,
    ) -> None:
        selected = region > 0
        n = int(selected.sum().item())
        if n <= 0 or self.count >= self.max_pixels:
            return
        remaining = self.max_pixels - self.count
        score_np = score[selected].detach().float().cpu().numpy().astype(np.float32)
        gain_np = true_gain[selected].detach().float().cpu().numpy().astype(np.float32)
        before_np = before_error[selected].detach().float().cpu().numpy().astype(np.float32)
        after_np = after_error[selected].detach().float().cpu().numpy().astype(np.float32)
        if n > remaining:
            ids = self.rng.choice(n, size=remaining, replace=False)
            score_np, gain_np = score_np[ids], gain_np[ids]
            before_np, after_np = before_np[ids], after_np[ids]
        self.score_parts.append(score_np)
        self.gain_parts.append(gain_np)
        self.before_parts.append(before_np)
        self.after_parts.append(after_np)
        self.count += score_np.size

    def finalize(self, coverage_points: Sequence[float]) -> List[Dict[str, Any]]:
        if not self.score_parts:
            return []
        score = np.concatenate(self.score_parts)
        gain = np.concatenate(self.gain_parts)
        before = np.concatenate(self.before_parts)
        after = np.concatenate(self.after_parts)
        order = np.argsort(-score, kind="mergesort")
        total = score.size
        rows = []
        for coverage in coverage_points:
            k = max(1, min(total, int(round(float(coverage) * total))))
            ids = order[:k]
            positive = gain[ids] > 0
            negative = gain[ids] < 0
            rows.append(
                {
                    "coverage": float(coverage),
                    "selected_pixels": int(k),
                    "sampled_support_pixels": int(total),
                    "score_threshold": float(score[ids[-1]]),
                    "mean_score": float(score[ids].mean()),
                    "beneficial_rate": float(positive.mean()),
                    "harmful_rate": float(negative.mean()),
                    "mean_true_gain_m": float(gain[ids].mean()),
                    "mean_positive_gain_m": float(gain[ids][positive].mean()) if positive.any() else 0.0,
                    "mean_harmful_damage_m": float((-gain[ids][negative]).mean()) if negative.any() else 0.0,
                    "mae_before_m": float(before[ids].mean()),
                    "mae_after_m": float(after[ids].mean()),
                }
            )
        return rows


# =============================================================================
# TABLE / OPTIONAL VISUALIZATION
# =============================================================================
def table_text(train_mod, rows: Dict[str, Dict[str, float]], order: Sequence[str]) -> str:
    lines = [
        f"{'Variant':<30} | {'MAE_all':>9} | {'RMSE_all':>9} | {'MAE_mask':>9} | "
        f"{'RMSE_mask':>9} | {'REL':>9} | {'Boundary':>9} | {'BG':>9} | {'Score':>9}",
        "-" * 166,
    ]
    for name in order:
        if name not in rows:
            continue
        row = rows[name]
        lines.append(
            f"{name:<30} | {row.get('mae_all', float('nan')):>9.6f} | "
            f"{row.get('rmse_all', float('nan')):>9.6f} | "
            f"{row.get('mae_mask', float('nan')):>9.6f} | "
            f"{row.get('rmse_mask', float('nan')):>9.6f} | "
            f"{row.get('rel_mask', float('nan')):>9.6f} | "
            f"{row.get('boundary', float('nan')):>9.6f} | "
            f"{row.get('reliable_bg_disturbance', float('nan')):>9.6f} | "
            f"{score_from_row(train_mod, row):>9.6f}"
        )
    lines.extend(
        [
            "",
            f"{'Variant':<30} | {'delta_1.05':>11} | {'delta_1.10':>11} | {'delta_1.25':>11}",
            "-" * 74,
        ]
    )
    for name in order:
        if name not in rows:
            continue
        row = rows[name]
        lines.append(
            f"{name:<30} | {row.get('delta_105', float('nan')):>11.6f} | "
            f"{row.get('delta_110', float('nan')):>11.6f} | "
            f"{row.get('delta_125', float('nan')):>11.6f}"
        )
    return "\n".join(lines)


def save_visualization(path: Path, inp: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], title: str) -> None:
    import matplotlib.pyplot as plt

    def arr(x: torch.Tensor) -> np.ndarray:
        y = x.detach().float().cpu().numpy()
        if y.ndim == 4:
            y = y[0]
        if y.shape[0] in (1, 3):
            y = np.moveaxis(y, 0, -1)
        if y.ndim == 3 and y.shape[-1] == 1:
            y = y[..., 0]
        return y

    rgb = np.clip(arr(inp["rgb"]), 0.0, 1.0)
    gt = arr(inp["gt"])
    raw = arr(inp["raw"])
    anchor = arr(out["anchor_depth"])
    legacy = arr(out["legacy_fused"])
    safe = arr(out["safe_posterior"])
    candidate = arr(out["candidate"])
    final = arr(out["final"])
    gate = arr(out["safe_gate"])
    acceptance = arr(out["acceptance"])
    mask = arr(inp["mask"])
    valid_pixels = gt[inp["valid"][0, 0].detach().cpu().numpy() > 0.5]
    if valid_pixels.size:
        vmin, vmax = np.percentile(valid_pixels, [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    panels = [
        (rgb, "RGB", None),
        (raw, "Raw", "viridis"),
        (gt, "GT", "viridis"),
        (anchor, "Base anchor", "viridis"),
        (legacy, "Legacy posterior", "viridis"),
        (safe, "Safe posterior", "viridis"),
        (candidate, "Candidate", "viridis"),
        (final, "Risk final", "viridis"),
        (np.abs(safe - gt) * mask, "Safe abs error", "magma"),
        (gate, "Safe gate", "viridis"),
        (acceptance, "Refine acceptance", "viridis"),
        (mask, "Mask", "gray"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, (image, name, cmap) in zip(axes.flat, panels):
        if cmap == "viridis" and name not in {"Safe gate", "Refine acceptance"}:
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        elif name in {"Safe gate", "Refine acceptance", "Mask"}:
            ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image, cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MODEL / MICRO-BATCH HELPERS
# =============================================================================
def load_checkpoint_model(train_mod, checkpoint: Path, device: torch.device):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    base_mod = train_mod.load_base_source_module()
    model = train_mod.FailureAwarePosteriorDepth(base_mod).to(device)
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("model_state_dict", payload))
    clean = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {checkpoint}: missing={len(missing)}, "
            f"unexpected={len(unexpected)}\nmissing[:10]={missing[:10]}\n"
            f"unexpected[:10]={unexpected[:10]}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def choose_microbatch(
    train_mod,
    model,
    first_cpu_batch: Dict[str, Any],
    requested: int,
    phase: str,
    device: torch.device,
    use_amp: bool,
) -> int:
    n = int(train_mod.batch_sample_count(first_cpu_batch))
    value = max(1, min(int(requested), n))
    candidates: List[int] = []
    while value >= 1:
        if value not in candidates:
            candidates.append(value)
        if value == 1:
            break
        value = max(1, value // 2)

    with torch.inference_mode():
        for candidate in candidates:
            try:
                cpu_part = slice_cpu_batch(first_cpu_batch, 0, candidate, n)
                gpu_part = move_to_device(cpu_part, device)
                inp = train_mod.build_inputs(gpu_part)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(inp, phase=phase, augment_safe=False)
                _ = float(out["safe_gate"].mean().detach().cpu())
                del cpu_part, gpu_part, inp, out
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                return candidate
            except RuntimeError as error:
                if not is_cuda_oom(error):
                    raise
                print(f"[Microbatch] {candidate} caused CUDA OOM; trying smaller.")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    raise RuntimeError("No evaluation micro-batch fits on the available GPU.")


# =============================================================================
# ONE-CHECKPOINT EVALUATION
# =============================================================================
def evaluate_checkpoint(
    train_mod,
    spec: CheckpointSpec,
    checkpoint: Path,
    shards: Sequence[Path],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, Any]:
    checkpoint_out = args.out_dir / spec.key
    checkpoint_out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 170)
    print(f"Testing {spec.key}: {spec.role}")
    print(f"Checkpoint: {checkpoint}")

    model, payload = load_checkpoint_model(train_mod, checkpoint, device)
    checkpoint_phase = str(payload.get("phase", "joint"))
    phase = spec.fixed_phase or checkpoint_phase
    if phase not in {"safe", "proposal", "risk", "joint"}:
        phase = "joint"
    print(
        f"refine_epoch={payload.get('refine_epoch', -1)}, "
        f"checkpoint_phase={checkpoint_phase}, evaluation_phase={phase}, "
        f"primary={spec.primary_variant}"
    )

    loader = DataLoader(
        train_mod.CachedShardDataset(shards),
        batch_size=LOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
        collate_fn=train_mod.ragged_shard_collate,
        persistent_workers=int(args.num_workers) > 0,
    )
    first_cpu_batch = next(iter(loader))
    microbatch = choose_microbatch(
        train_mod, model, first_cpu_batch, args.microbatch, phase, device, use_amp
    )
    print(f"Selected evaluation microbatch={microbatch}")

    sample_metric_lists: Dict[str, Dict[str, List[float]]] = {
        name: defaultdict(list) for name in VARIANT_ORDER
    }
    global_acc: Dict[str, PixelMetricAccumulator] = {
        name: PixelMetricAccumulator() for name in VARIANT_ORDER
    }
    per_sample_long: List[Dict[str, Any]] = []
    per_sample_primary: List[Dict[str, Any]] = []
    failure_acc = FailureAccumulator(ECE_BINS)
    diagnostics = WeightedDiagnostics()
    safe_coverage = CoverageAccumulator(args.coverage_pixels, SEED + 11)
    refine_coverage = CoverageAccumulator(args.coverage_pixels, SEED + 29)

    sample_index = 0
    vis_saved = 0
    start_time = time.time()
    progress = tqdm(loader, desc=f"Test {spec.key}", dynamic_ncols=True)

    with torch.inference_mode():
        for shard_index, cpu_batch in enumerate(progress):
            n = int(train_mod.batch_sample_count(cpu_batch))
            source_shard = str(shards[shard_index]) if shard_index < len(shards) else ""
            for start in range(0, n, microbatch):
                end = min(n, start + microbatch)
                cpu_part = slice_cpu_batch(cpu_batch, start, end, n)
                part = move_to_device(cpu_part, device)
                inp = train_mod.build_inputs(part)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(inp, phase=phase, augment_safe=False)

                raw, gt, mask, valid = inp["raw"], inp["gt"], inp["mask"], inp["valid"]
                previous = inp["old_base"] * mask + raw * (1.0 - mask)
                base = out["anchor_depth"] * mask + raw * (1.0 - mask)
                legacy = out["legacy_fused"] * mask + raw * (1.0 - mask)

                anchor_err = torch.abs(out["anchor_depth"] - gt)
                legacy_err = torch.abs(out["legacy_fused"] - gt)
                safe_err = torch.abs(out["safe_posterior"] - gt)
                candidate_err = torch.abs(out["candidate"] - gt)
                final_err = torch.abs(out["final"] - gt)

                oracle_anchor_inside = torch.where(
                    legacy_err < anchor_err, out["legacy_fused"], out["anchor_depth"]
                )
                oracle_anchor = oracle_anchor_inside * mask + raw * (1.0 - mask)
                oracle_refine_inside = torch.where(
                    candidate_err < safe_err, out["candidate"], out["safe_posterior"]
                )
                oracle_refine = oracle_refine_inside * mask + raw * (1.0 - mask)

                variants = {
                    "Raw Depth": raw,
                    "Input relative prior": inp["rel"],
                    "Metric-calibrated prior": out["rel_metric"],
                    "Previous model result": previous,
                    "Base anchor": base,
                    "Legacy posterior fusion": legacy,
                    "Safe benchmark": out["safe_benchmark"],
                    "Candidate benchmark": out["candidate_benchmark"],
                    "Benchmark output": out["benchmark_output"],
                    "Oracle anchor-posterior": oracle_anchor,
                    "Oracle safe-candidate": oracle_refine,
                }

                batch_metrics: Dict[str, Dict[str, torch.Tensor]] = {}
                for name, prediction in variants.items():
                    metrics = metrics_per_sample(
                        train_mod, prediction, raw, gt, mask, valid
                    )
                    batch_metrics[name] = metrics
                    for key, tensor in metrics.items():
                        sample_metric_lists[name][key].extend(
                            tensor.detach().float().cpu().numpy().astype(float).tolist()
                        )
                    global_acc[name].update(
                        train_mod, prediction, raw, gt, mask, valid
                    )

                # Failure metrics are checkpoint invariant for frozen legacy modules,
                # but are written per checkpoint for a self-contained report.
                failure_acc.update(train_mod, out, inp)

                safe_region = valid * out["safe_support"].detach()
                if float(safe_region.sum()) <= 0:
                    safe_region = valid * mask
                refine_region = valid * out["support"].detach()
                if float(refine_region.sum()) <= 0:
                    refine_region = valid * mask

                true_safe_gain = anchor_err - legacy_err
                safe_target = torch.sigmoid(
                    true_safe_gain
                    / max(float(train_mod.SAFE_TARGET_TEMPERATURE), float(train_mod.EPS))
                )
                safe_delta_error = safe_err - anchor_err
                safe_benefit = (-safe_delta_error).clamp_min(0.0)
                safe_damage = safe_delta_error.clamp_min(0.0)
                safe_better = (safe_delta_error < 0).float()
                safe_worse = (safe_delta_error > 0).float()
                safe_guard_damage = (
                    safe_delta_error > float(train_mod.SAFE_GUARD_MARGIN)
                ).float()

                diagnostics.add("safe_support_mean", out["safe_support"], valid)
                diagnostics.add("safe_gate_mean", out["safe_gate"], safe_region)
                diagnostics.add("safe_gate_target_mean", safe_target, safe_region)
                diagnostics.add(
                    "safe_gate_mae", torch.abs(out["safe_gate"] - safe_target), safe_region
                )
                diagnostics.add(
                    "safe_update_abs", torch.abs(out["safe_posterior"] - out["anchor_depth"]), safe_region
                )
                diagnostics.add("legacy_improve_ratio", (true_safe_gain > 0).float(), safe_region)
                diagnostics.add("safe_improve_ratio", safe_better, safe_region)
                diagnostics.add("safe_worse_ratio", safe_worse, safe_region)
                diagnostics.add("safe_damage_ratio", safe_guard_damage, safe_region)
                diagnostics.add("safe_net_gain_m", anchor_err - safe_err, safe_region)
                diagnostics.add("safe_benefit_per_support_m", safe_benefit, safe_region)
                diagnostics.add("safe_damage_per_support_m", safe_damage, safe_region)
                diagnostics.add_conditional(
                    "safe_mean_beneficial_gain_m", safe_benefit, safe_region, safe_better
                )
                diagnostics.add_conditional(
                    "safe_mean_harmful_damage_m", safe_damage, safe_region, safe_worse
                )
                diagnostics.add(
                    "safe_damage_gt_1mm_ratio", (safe_damage > 0.001).float(), safe_region
                )
                diagnostics.add(
                    "safe_damage_gt_5mm_ratio", (safe_damage > 0.005).float(), safe_region
                )
                diagnostics.add(
                    "safe_oracle_gain_m", anchor_err - torch.minimum(anchor_err, legacy_err), safe_region
                )
                diagnostics.add(
                    "safe_risk_anchor_mae", torch.abs(out["safe_risk_anchor"] - anchor_err), safe_region
                )
                diagnostics.add(
                    "safe_risk_legacy_mae", torch.abs(out["safe_risk_legacy"] - legacy_err), safe_region
                )
                diagnostics.add(
                    "safe_risk_gain_mae", torch.abs(out["safe_predicted_gain"] - true_safe_gain), safe_region
                )
                diagnostics.add_hist("safe_gate", out["safe_gate"], safe_region)

                true_refine_gain = safe_err - candidate_err
                final_delta_error = final_err - safe_err
                final_benefit = (-final_delta_error).clamp_min(0.0)
                final_damage = final_delta_error.clamp_min(0.0)
                final_better = (final_delta_error < 0).float()
                final_worse = (final_delta_error > 0).float()
                accepted = (out["acceptance"] > 0.5).float()
                accepted_region = refine_region * accepted

                diagnostics.add(
                    "candidate_update_abs", torch.abs(out["candidate"] - out["safe_posterior"]), refine_region
                )
                diagnostics.add(
                    "accepted_update_abs", torch.abs(out["final"] - out["safe_posterior"]), refine_region
                )
                diagnostics.add(
                    "candidate_improve_ratio", (true_refine_gain > 0).float(), refine_region
                )
                diagnostics.add("final_improve_ratio", final_better, refine_region)
                diagnostics.add("final_worse_ratio", final_worse, refine_region)
                diagnostics.add("acceptance_mean", out["acceptance"], refine_region)
                diagnostics.add_hist("acceptance", out["acceptance"], refine_region)
                diagnostics.add_conditional(
                    "accepted_improve_ratio", final_better, refine_region, accepted
                )
                diagnostics.add("refine_net_gain_m", safe_err - final_err, refine_region)
                diagnostics.add("refine_benefit_per_support_m", final_benefit, refine_region)
                diagnostics.add("refine_damage_per_support_m", final_damage, refine_region)
                diagnostics.add_conditional(
                    "refine_mean_beneficial_gain_m", final_benefit, refine_region, final_better
                )
                diagnostics.add_conditional(
                    "refine_mean_harmful_damage_m", final_damage, refine_region, final_worse
                )
                diagnostics.add(
                    "refine_oracle_gain_m", safe_err - torch.minimum(safe_err, candidate_err), refine_region
                )
                diagnostics.add(
                    "risk_before_mae", torch.abs(out["risk_before"] - safe_err), refine_region
                )
                diagnostics.add(
                    "risk_after_mae", torch.abs(out["risk_after"] - candidate_err), refine_region
                )
                diagnostics.add(
                    "risk_gain_mae", torch.abs(out["predicted_gain"] - true_refine_gain), refine_region
                )

                safe_coverage.update(
                    out["safe_gate"], true_safe_gain, anchor_err, legacy_err, safe_region
                )
                refine_coverage.update(
                    out["acceptance"], true_refine_gain, safe_err, candidate_err, refine_region
                )

                batch_n = end - start
                for local in range(batch_n):
                    row_base: Dict[str, Any] = {
                        "sample_index": sample_index,
                        "shard_index": shard_index,
                        "sample_in_shard": start + local,
                        "source_shard": source_shard,
                        "checkpoint_key": spec.key,
                    }
                    primary_row = dict(row_base)
                    for name in VARIANT_ORDER:
                        values = sample_metric_dict(batch_metrics[name], local)
                        per_sample_long.append(
                            {
                                **row_base,
                                "variant": name,
                                **values,
                                "score": score_from_row(train_mod, values),
                            }
                        )
                        if name in {
                            "Base anchor",
                            "Legacy posterior fusion",
                            "Safe benchmark",
                            "Candidate benchmark",
                            "Benchmark output",
                            "Oracle anchor-posterior",
                            "Oracle safe-candidate",
                        }:
                            prefix = {
                                "Base anchor": "base",
                                "Legacy posterior fusion": "legacy",
                                "Safe benchmark": "safe",
                                "Candidate benchmark": "candidate",
                                "Benchmark output": "final",
                                "Oracle anchor-posterior": "oracle_anchor",
                                "Oracle safe-candidate": "oracle_refine",
                            }[name]
                            for key, value in values.items():
                                primary_row[f"{prefix}_{key}"] = value
                            primary_row[f"{prefix}_score"] = score_from_row(train_mod, values)
                    primary_row["safe_minus_base_score"] = (
                        primary_row["safe_score"] - primary_row["base_score"]
                    )
                    primary_row["candidate_minus_safe_score"] = (
                        primary_row["candidate_score"] - primary_row["safe_score"]
                    )
                    primary_row["final_minus_safe_score"] = (
                        primary_row["final_score"] - primary_row["safe_score"]
                    )
                    primary_row["safe_better_than_base"] = int(
                        primary_row["safe_score"] < primary_row["base_score"]
                    )
                    primary_row["primary_variant"] = spec.primary_variant
                    primary_row["primary_score"] = {
                        "Safe benchmark": primary_row["safe_score"],
                        "Candidate benchmark": primary_row["candidate_score"],
                        "Benchmark output": primary_row["final_score"],
                    }[spec.primary_variant]
                    per_sample_primary.append(primary_row)

                    if vis_saved < max(0, int(args.visualizations)):
                        sample_inp = {
                            k: v[local:local + 1]
                            for k, v in inp.items()
                            if torch.is_tensor(v)
                        }
                        sample_out = {
                            k: v[local:local + 1]
                            for k, v in out.items()
                            if torch.is_tensor(v)
                        }
                        save_visualization(
                            checkpoint_out / "visualizations" / f"sample_{sample_index:06d}.png",
                            sample_inp,
                            sample_out,
                            f"{spec.key} | sample {sample_index}",
                        )
                        vis_saved += 1
                    sample_index += 1

                del cpu_part, part, inp, out, variants, batch_metrics

            elapsed = max(time.time() - start_time, 1.0e-6)
            progress.set_postfix(samples=sample_index, mb=microbatch, sps=f"{sample_index/elapsed:.2f}")
            if (
                device.type == "cuda"
                and EMPTY_CACHE_EVERY > 0
                and (shard_index + 1) % EMPTY_CACHE_EVERY == 0
            ):
                torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    if sample_index <= 0:
        raise RuntimeError(f"No samples evaluated for {spec.key}")

    sample_summary: Dict[str, Dict[str, float]] = {}
    global_summary: Dict[str, Dict[str, float]] = {}
    for name in VARIANT_ORDER:
        sample_summary[name] = {
            key: finite_mean(values)
            for key, values in sample_metric_lists[name].items()
        }
        sample_summary[name]["score"] = score_from_row(train_mod, sample_summary[name])
        global_summary[name] = global_acc[name].finalize()
        global_summary[name]["score"] = score_from_row(train_mod, global_summary[name])

    failure_result = failure_acc.finalize()
    diagnostic_summary = diagnostics.summary()
    safe_curve = safe_coverage.finalize(COVERAGE_POINTS)
    refine_curve = refine_coverage.finalize(COVERAGE_POINTS)

    sample_better = sum(int(r["safe_better_than_base"]) for r in per_sample_primary)
    diagnostic_summary["samples_safe_better_than_base_fraction"] = sample_better / max(sample_index, 1)
    diagnostic_summary["samples_safe_worse_or_equal_fraction"] = 1.0 - diagnostic_summary["samples_safe_better_than_base_fraction"]

    report = "\n".join(
        [
            f"FAPR-Depth v6 complete test | {spec.key}",
            "=" * 100,
            f"Role: {spec.role}",
            f"Checkpoint: {checkpoint}",
            f"Checkpoint epoch/phase: {payload.get('refine_epoch', -1)} / {checkpoint_phase}",
            f"Evaluation phase: {phase}",
            f"Primary variant: {spec.primary_variant}",
            f"Split/shards/samples: {args.split} / {len(shards)} / {sample_index}",
            f"Elapsed seconds: {elapsed:.2f}",
            f"Samples/second: {sample_index/max(elapsed,1e-6):.3f}",
            f"Microbatch: {microbatch}",
            "",
            "Sample-mean metrics",
            table_text(train_mod, sample_summary, VARIANT_ORDER),
            "",
            "Global pixel-weighted metrics",
            table_text(train_mod, global_summary, VARIANT_ORDER),
            "",
            "Failure metrics",
            json.dumps(
                {
                    "accuracy": failure_result["accuracy"],
                    "balanced_accuracy": failure_result["balanced_accuracy"],
                    "brier": failure_result["brier"],
                    "ece_any_failure": failure_result["ece_any_failure"],
                    "per_class": failure_result["per_class"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "Safety / refinement diagnostics",
            json.dumps(diagnostic_summary, ensure_ascii=False, indent=2),
        ]
    )
    print("\n" + report)

    write_csv(
        checkpoint_out / "summary_sample_mean.csv",
        [{"variant": name, **sample_summary[name]} for name in VARIANT_ORDER],
    )
    write_json(checkpoint_out / "summary_sample_mean.json", sample_summary)
    write_csv(
        checkpoint_out / "summary_global_pixel.csv",
        [{"variant": name, **global_summary[name]} for name in VARIANT_ORDER],
    )
    write_json(checkpoint_out / "summary_global_pixel.json", global_summary)
    write_csv(checkpoint_out / "per_sample_metrics_long.csv", per_sample_long)
    write_csv(checkpoint_out / "per_sample_primary.csv", per_sample_primary)
    write_json(checkpoint_out / "failure_metrics.json", failure_result)
    write_csv(
        checkpoint_out / "failure_per_class.csv",
        [{"class": name, **row} for name, row in failure_result["per_class"].items()],
    )
    write_csv(checkpoint_out / "failure_ece_bins.csv", failure_result["ece_bins"])
    write_json(checkpoint_out / "safe_and_refinement_diagnostics.json", diagnostic_summary)
    write_csv(checkpoint_out / "safe_gate_coverage.csv", safe_curve)
    write_json(checkpoint_out / "safe_gate_coverage.json", safe_curve)
    write_csv(checkpoint_out / "refinement_risk_coverage.csv", refine_curve)
    write_json(checkpoint_out / "refinement_risk_coverage.json", refine_curve)
    (checkpoint_out / "test_report.txt").write_text(report, encoding="utf-8")
    write_json(
        checkpoint_out / "checkpoint_metadata.json",
        {
            "key": spec.key,
            "role": spec.role,
            "primary_variant": spec.primary_variant,
            "checkpoint": str(checkpoint),
            "refine_epoch": payload.get("refine_epoch"),
            "checkpoint_phase": checkpoint_phase,
            "evaluation_phase": phase,
            "recorded_best_score": payload.get("best_score"),
            "recorded_best_safe_score": payload.get("best_safe_score"),
            "recorded_best_candidate_score": payload.get("best_candidate_score"),
            "source_checkpoint": payload.get("source_checkpoint"),
            "split": args.split,
            "shards": len(shards),
            "samples": sample_index,
            "microbatch": microbatch,
            "amp": use_amp,
            "elapsed_seconds": elapsed,
        },
    )

    result = {
        "spec": spec,
        "checkpoint": checkpoint,
        "payload": payload,
        "phase": phase,
        "sample_summary": sample_summary,
        "global_summary": global_summary,
        "diagnostics": diagnostic_summary,
        "failure": failure_result,
        "samples": sample_index,
        "elapsed": elapsed,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


# =============================================================================
# FIXED-ROLE COMBINED TABLE
# =============================================================================
def combined_rows(train_mod, results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not results:
        return []
    reference_key = "best_safe" if "best_safe" in results else next(iter(results))
    reference = results[reference_key]["sample_summary"]
    rows: List[Dict[str, Any]] = []

    def add(label: str, checkpoint_key: str, variant: str, role: str) -> None:
        if checkpoint_key not in results:
            return
        source = results[checkpoint_key]["sample_summary"].get(variant)
        if source is None:
            return
        rows.append(
            {
                "method": label,
                "checkpoint_key": checkpoint_key,
                "variant": variant,
                "role": role,
                **source,
                "score": score_from_row(train_mod, source),
            }
        )

    # Baselines are checkpoint invariant; use the best-safe pass as source.
    for label in (
        "Raw Depth",
        "Previous model result",
        "Base anchor",
        "Legacy posterior fusion",
    ):
        source = reference[label]
        rows.append(
            {
                "method": label,
                "checkpoint_key": reference_key,
                "variant": label,
                "role": "baseline",
                **source,
                "score": score_from_row(train_mod, source),
            }
        )

    add("Safe warm-up", "safe_warmup", "Safe benchmark", "conservative checkpoint")
    add("FAPR-Depth v6 Safe (Ours)", "best_safe", "Safe benchmark", "primary model")
    add("+ Candidate proposal", "best_candidate", "Candidate benchmark", "ablation")
    add("+ Risk-accepted refinement", "best_final", "Benchmark output", "full ablation")

    oracle = reference["Oracle anchor-posterior"]
    rows.append(
        {
            "method": "Oracle anchor-posterior (diagnostic)",
            "checkpoint_key": reference_key,
            "variant": "Oracle anchor-posterior",
            "role": "diagnostic upper bound",
            **oracle,
            "score": score_from_row(train_mod, oracle),
        }
    )

    base_score = next((r["score"] for r in rows if r["method"] == "Base anchor"), float("nan"))
    for row in rows:
        row["delta_score_vs_base"] = row["score"] - base_score
        row["score_change_vs_base_percent"] = pct_change(row["score"], base_score)
    return rows


def combined_report(train_mod, rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "FAPR-Depth v6 fixed-role checkpoint comparison",
        "=" * 160,
        "The primary model is fixed as best_safe.pth / Safe benchmark before test evaluation.",
        "best_score.pth is reported as the full proposal+risk ablation.",
        "",
        f"{'Method':<40} | {'RMSE':>9} | {'REL':>9} | {'MAE':>9} | "
        f"{'d1.05(%)':>9} | {'d1.10(%)':>9} | {'d1.25(%)':>9} | {'Score':>9} | {'vs Base':>10}",
        "-" * 160,
    ]
    for row in rows:
        lines.append(
            f"{row['method']:<40} | {row.get('rmse_mask', float('nan')):>9.6f} | "
            f"{row.get('rel_mask', float('nan')):>9.6f} | "
            f"{row.get('mae_mask', float('nan')):>9.6f} | "
            f"{100*row.get('delta_105', float('nan')):>9.3f} | "
            f"{100*row.get('delta_110', float('nan')):>9.3f} | "
            f"{100*row.get('delta_125', float('nan')):>9.3f} | "
            f"{row.get('score', float('nan')):>9.6f} | "
            f"{row.get('delta_score_vs_base', float('nan')):>+10.6f}"
        )
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    set_seed(SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(device.type == "cuda" and not args.no_amp)

    print("=" * 170)
    print("FAPR-Depth v6 complete multi-checkpoint test")
    print("=" * 170)
    print(f"DEVICE={device}, AMP={use_amp}")
    print(f"TRAIN_SCRIPT={args.train_script}")
    print(f"CACHE_ROOT={args.cache_root}")
    print(f"CHECKPOINT_DIR={args.checkpoint_dir}")
    print(f"OUT_DIR={args.out_dir}")
    print(f"CHECKPOINT_SET={args.checkpoint_set}")

    train_mod = import_training_module(args.train_script)
    shards = train_mod.load_split_shards(args.cache_root, args.split, args.max_shards)
    if not shards:
        raise RuntimeError(f"No shards found for split={args.split}")
    print(f"Split={args.split}, shards={len(shards)}")

    specs = select_specs(args.checkpoint_set)
    available: List[Tuple[CheckpointSpec, Path]] = []
    missing: List[str] = []
    for spec in specs:
        path = args.checkpoint_dir / spec.filename
        if path.exists():
            available.append((spec, path))
        else:
            missing.append(str(path))
            print(f"[WARN] Missing checkpoint; skipping: {path}")
    if not available:
        raise FileNotFoundError("No requested checkpoints exist.")

    overall_start = time.time()
    results: Dict[str, Dict[str, Any]] = {}
    for spec, checkpoint in available:
        results[spec.key] = evaluate_checkpoint(
            train_mod,
            spec,
            checkpoint,
            shards,
            args,
            device,
            use_amp,
        )

    rows = combined_rows(train_mod, results)
    report = combined_report(train_mod, rows)
    print("\n" + report)
    write_csv(args.out_dir / "combined_checkpoint_comparison.csv", rows)
    write_json(args.out_dir / "combined_checkpoint_comparison.json", rows)
    (args.out_dir / "combined_checkpoint_comparison.txt").write_text(report, encoding="utf-8")

    role_manifest = {
        "primary_model": {
            "checkpoint": str(args.checkpoint_dir / "best_safe.pth"),
            "output": "Safe benchmark",
            "reason": "fixed primary safe-anchor model chosen from validation",
        },
        "full_ablation": {
            "checkpoint": str(args.checkpoint_dir / "best_score.pth"),
            "output": "Benchmark output",
            "reason": "proposal + refinement-risk ablation",
        },
        "candidate_ablation": {
            "checkpoint": str(args.checkpoint_dir / "best_candidate.pth"),
            "output": "Candidate benchmark",
        },
        "conservative_checkpoint": {
            "checkpoint": str(args.checkpoint_dir / "safe_warmup_complete.pth"),
            "output": "Safe benchmark",
        },
        "missing_requested_checkpoints": missing,
        "split": args.split,
        "shards": len(shards),
        "total_elapsed_seconds": time.time() - overall_start,
    }
    write_json(args.out_dir / "fixed_model_roles.json", role_manifest)

    print(f"\nDone. Results written to: {args.out_dir}")
    print("Primary paper/deployment result: best_safe.pth -> Safe benchmark")
    print("Full proposal+risk ablation: best_score.pth -> Benchmark output")


if __name__ == "__main__":
    main()
