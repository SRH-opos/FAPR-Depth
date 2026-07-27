# -*- coding: utf-8 -*-
r"""
FAPR-Depth v4.1: Failure-Aware Posterior Reconstruction with Risk-Controlled Refinement
=====================================================================================

This script is a dedicated refinement-stage trainer.  It starts from the best
posterior-fusion checkpoint produced by the previous training run, freezes the
entire posterior reconstruction system, and trains only a new risk-controlled
refinement module.

Core inference chain
--------------------
    metric prior calibration
        -> latent sensor-failure inference
        -> failure-conditioned experts
        -> uncertainty-aware posterior fusion
        -> failure-constrained risk-controlled refinement

The refinement update is deliberately conservative:

    D_refined = D_posterior + S_failure * A_risk * Delta

where S_failure restricts changes to transparent/failure support regions and
A_risk is derived from the predicted reduction in local reconstruction risk.
Reliable raw measurements are protected by a trust-region loss.

Usage
-----
1. Keep the existing Stage-2 best checkpoint at STAGE2_CKPT.
2. Edit PROJECT_ROOT, CACHE_ROOT, BASE_SOURCE_ROOT and STAGE2_CKPT if needed.
3. Run:
       python train_fapr_depth_risk_controlled_refinement_v4_1_ampfix_8gb.py

The script uses true-image micro-batches of one and is designed for an 8 GB GPU.
"""

from __future__ import annotations

from pathlib import Path
import os

# Reduce CUDA allocator fragmentation. Must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import csv
import importlib.util
import json
import math
import random
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(os.getenv("FAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent)))
CACHE_ROOT = Path(os.getenv("FAPR_CACHE_ROOT", str(PROJECT_ROOT / "data" / "cache")))

# Implementation source for the pretrained base completion stream.
# This is an implementation detail, not the identity of the proposed framework.
BASE_SOURCE_ROOT = Path(os.getenv("FAPR_BASE_SOURCE_ROOT", str(PROJECT_ROOT / "third_party" / "FDCT-main")))

# Best posterior-fusion checkpoint from the completed Stage-2 run (Epoch 6).
STAGE2_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "fdct_failure_aware_probabilistic_v3_calibrated_8gb"
    / "checkpoints"
    / "best_score.pth"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "fapr_depth_v4_risk_controlled_refinement"
CKPT_DIR = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Resume a partially completed v4/v4.1 refinement run when last.pth exists.
AUTO_RESUME = True
RESUME_CKPT = CKPT_DIR / "last.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
SEED = 6248

IMAGE_SIZE = (240, 320)
MAX_TRAIN_SHARDS: Optional[int] = 1964
MAX_VAL_SHARDS: Optional[int] = 512
LOADER_BATCH_SIZE = 1
TRAIN_MICROBATCH = 1
VAL_MICROBATCH = 1
NUM_WORKERS = 0
PIN_MEMORY = DEVICE == "cuda"
EMPTY_CACHE_EVERY = 100

# Base completion stream architecture. These values must match the Stage-2 checkpoint.
BASE_HIDDEN = 64
BASE_L = 5
BASE_K = 12
BASE_USE_DUC = True

# Dedicated risk-refinement optimization.
REFINE_EPOCHS = 4
LR_RISK_REFINER = 1.0e-5
WEIGHT_DECAY = 1.0e-4
CLIP_GRAD = 2.0
EARLY_STOP_PATIENCE = 2

MAX_DEPTH = 10.0
MIN_DEPTH = 0.03
DEPTH_NORM_SCALE = 5.0
EPS = 1.0e-6
BOUNDARY_KERNEL = 7
RELIABLE_RAW_THR = 0.010
HARD_RATIO = 0.20

# Geometry/failure settings inherited from the posterior checkpoint.
ANCHOR_DILATE_KERNEL = 15
ANCHOR_GRAD_THR = 0.035
MIN_ANCHOR_PIXELS = 128
SDM_STEPS = 12
FAIL_ABS_THR = 0.010
FAIL_REL_THR = 0.030
BOUNDARY_FAIL_THR = 0.0075
MAX_LOCAL_SCALE_LOG = 0.12
MAX_LOCAL_BIAS = 0.20
MAX_EXPERT_DELTA = 0.75
RELIABLE_BG_THR = 0.015
INPUT_REL_ALREADY_METRIC_ALIGNED = True
ADAPTER_FEATURE_SCALE = 0.16

# Risk-controlled refinement.
REFINE_STEPS = 1
MAX_RISK_DELTA = 0.050              # at most 5 cm candidate correction
SUPPORT_DILATE_KERNEL = 9           # transparent mask plus narrow boundary context
BOUNDARY_SUPPORT_WEIGHT = 0.50
RISK_TEMPERATURE = 0.005            # metres; converts predicted risk gain to acceptance
RISK_ACCEPT_MARGIN = 0.0
ACCEPT_TARGET_MARGIN = 1.0e-4       # candidate must improve by at least 0.1 mm
MONOTONIC_TOLERANCE = 0.0

# Loss weights for the new refinement stage only.
W_FINAL_MASK = 1.60
W_FINAL_ALL = 0.15
W_FINAL_RMSE = 0.80
W_BOUNDARY = 0.90
W_GRAD = 0.30
W_HARD = 0.25
W_MONOTONIC = 1.20
W_RISK_CALIBRATION = 0.30
W_ACCEPTANCE = 0.25
W_TRUST_REGION = 0.50
W_UPDATE_REG = 0.03

# Model selection score. Lower is better.
SCORE_BOUNDARY_WEIGHT = 0.50
SCORE_ALL_WEIGHT = 0.15


# =============================================================================
# BASE SOURCE IMPORT
# =============================================================================
def import_by_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_source_module():
    if not BASE_SOURCE_ROOT.exists():
        raise FileNotFoundError(f"BASE_SOURCE_ROOT not found: {BASE_SOURCE_ROOT}")
    for name in ("Model.py", "module.py"):
        if not (BASE_SOURCE_ROOT / name).exists():
            raise FileNotFoundError(f"Base source missing: {BASE_SOURCE_ROOT / name}")
    if str(BASE_SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(BASE_SOURCE_ROOT))
    return import_by_path("base_completion_source_for_fapr", BASE_SOURCE_ROOT / "Model.py")


# =============================================================================
# GENERAL UTILITIES
# =============================================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def safe_depth(d: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(d, nan=0.0, posinf=MAX_DEPTH, neginf=0.0).clamp(0.0, MAX_DEPTH)


def positive_depth(d: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(d, nan=MIN_DEPTH, posinf=MAX_DEPTH, neginf=MIN_DEPTH).clamp(MIN_DEPTH, MAX_DEPTH)


def norm_depth(d: torch.Tensor) -> torch.Tensor:
    return safe_depth(d).div(float(DEPTH_NORM_SCALE)).clamp(0.0, 2.0)


def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    return (x * m).sum() / m.sum().clamp_min(EPS)


def masked_rmse(pred: torch.Tensor, gt: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(masked_mean((pred - gt) ** 2, region) + 1.0e-12)


def charbonnier(pred: torch.Tensor, gt: torch.Tensor, region: torch.Tensor, eps: float = 1.0e-3) -> torch.Tensor:
    return masked_mean(torch.sqrt((pred - gt).square() + eps * eps), region)


def gradient_x(d: torch.Tensor) -> torch.Tensor:
    return d[:, :, :, 1:] - d[:, :, :, :-1]


def gradient_y(d: torch.Tensor) -> torch.Tensor:
    return d[:, :, 1:, :] - d[:, :, :-1, :]


def gradient_mag(d: torch.Tensor) -> torch.Tensor:
    gx = F.pad(torch.abs(gradient_x(d)), (0, 1, 0, 0))
    gy = F.pad(torch.abs(gradient_y(d)), (0, 0, 0, 1))
    return gx + gy


def gradient_l1(pred: torch.Tensor, gt: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    rx = region[:, :, :, 1:] * region[:, :, :, :-1]
    ry = region[:, :, 1:, :] * region[:, :, :-1, :]
    lx = masked_mean(torch.abs(gradient_x(pred) - gradient_x(gt)), rx)
    ly = masked_mean(torch.abs(gradient_y(pred) - gradient_y(gt)), ry)
    return 0.5 * (lx + ly)


def hard_pixel_rmse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    region: torch.Tensor,
    ratio: float = HARD_RATIO,
) -> torch.Tensor:
    err2 = ((pred - gt) ** 2).flatten(1)
    m = (region > 0.5).flatten(1)
    vals: List[torch.Tensor] = []
    for bi in range(pred.shape[0]):
        e = err2[bi][m[bi]]
        if e.numel() == 0:
            continue
        k = max(1, int(np.ceil(e.numel() * ratio)))
        vals.append(torch.topk(e, k=k, largest=True).values.mean())
    if not vals:
        return pred.new_tensor(0.0)
    return torch.sqrt(torch.stack(vals).mean() + 1.0e-12)


def force_4d_map(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        raise ValueError("force_4d_map got None")
    x = x.contiguous()
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        if x.shape[0] == 1:
            return x.unsqueeze(0)
        return x.unsqueeze(1)
    if x.ndim == 4:
        if x.shape[1] == 1:
            return x
        if x.shape[-1] == 1:
            return x.permute(0, 3, 1, 2).contiguous()
        return x.reshape(-1, 1, x.shape[-2], x.shape[-1])
    if x.ndim >= 5:
        if x.shape[-3] == 1:
            return x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        if x.shape[-1] == 1:
            y = x.reshape(-1, x.shape[-3], x.shape[-2], 1)
            return y.permute(0, 3, 1, 2).contiguous()
        return x.reshape(-1, 1, x.shape[-2], x.shape[-1])
    raise RuntimeError(f"Cannot interpret map shape={tuple(x.shape)}")


def force_4d_rgb(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        raise ValueError("force_4d_rgb got None")
    x = x.contiguous()
    if x.ndim == 3:
        if x.shape[0] == 3:
            return x.unsqueeze(0)
        if x.shape[-1] == 3:
            return x.permute(2, 0, 1).unsqueeze(0).contiguous()
    elif x.ndim == 4:
        if x.shape[1] == 3:
            return x
        if x.shape[-1] == 3:
            return x.permute(0, 3, 1, 2).contiguous()
    elif x.ndim >= 5:
        if x.shape[-3] == 3:
            return x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        if x.shape[-1] == 3:
            y = x.reshape(-1, x.shape[-3], x.shape[-2], 3)
            return y.permute(0, 3, 1, 2).contiguous()
    raise RuntimeError(f"Cannot interpret RGB shape={tuple(x.shape)}")


def build_boundary_ring(mask: torch.Tensor, kernel_size: int = BOUNDARY_KERNEL) -> torch.Tensor:
    mask = force_4d_map(mask).float().clamp(0.0, 1.0)
    pad = kernel_size // 2
    dil = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def erode_binary(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return -F.max_pool2d(-x, k, stride=1, padding=k // 2)


def dilate_binary(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    return F.max_pool2d(x, k, stride=1, padding=k // 2)


def approximate_signed_distance(mask: torch.Tensor, steps: int = SDM_STEPS) -> torch.Tensor:
    """GPU-friendly truncated signed distance: positive inside, negative outside."""
    m = (mask > 0.5).float()
    inside, outside = m, 1.0 - m
    di, do = torch.zeros_like(m), torch.zeros_like(m)
    ci, co = inside, outside
    for _ in range(steps):
        di = di + ci
        do = do + co
        ci = erode_binary(ci, 3)
        co = erode_binary(co, 3)
    return ((di - do) / float(max(steps, 1))).clamp(-1.0, 1.0)


def avg_dicts(items: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = sorted(set().union(*[d.keys() for d in items]))
    out: Dict[str, float] = {}
    for key in keys:
        vals = [d[key] for d in items if key in d and np.isfinite(d[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def fmt(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.6f}"


def selection_score(row: Dict[str, float]) -> float:
    return float(
        row.get("mae_mask", 0.0)
        + row.get("rmse_mask", 0.0)
        + SCORE_BOUNDARY_WEIGHT * row.get("boundary", 0.0)
        + SCORE_ALL_WEIGHT * row.get("rmse_all", 0.0)
    )


@torch.no_grad()
def metric_values(
    pred_final: torch.Tensor,
    raw: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
) -> Dict[str, float]:
    pred = safe_depth(pred_final)
    raw = safe_depth(raw)
    gt_pos = safe_depth(gt)
    mask = mask.float().clamp(0.0, 1.0)
    valid = valid.float().clamp(0.0, 1.0)
    region_all = valid
    region_mask = valid * mask
    boundary = build_boundary_ring(mask) * valid
    raw_err = torch.abs(raw - gt_pos)
    reliable_bg = valid * (1.0 - mask) * (raw_err <= RELIABLE_BG_THR).float()

    ratio = torch.maximum(pred / gt_pos.clamp_min(MIN_DEPTH), gt_pos / pred.clamp_min(MIN_DEPTH))
    return {
        "mae_all": float(masked_mean(torch.abs(pred - gt_pos), region_all).cpu()),
        "rmse_all": float(masked_rmse(pred, gt_pos, region_all).cpu()),
        "mae_mask": float(masked_mean(torch.abs(pred - gt_pos), region_mask).cpu()),
        "rmse_mask": float(masked_rmse(pred, gt_pos, region_mask).cpu()),
        "rel_mask": float(masked_mean(torch.abs(pred - gt_pos) / gt_pos.clamp_min(MIN_DEPTH), region_mask).cpu()),
        "delta_105": float(masked_mean((ratio < 1.05).float(), region_mask).cpu()),
        "delta_110": float(masked_mean((ratio < 1.10).float(), region_mask).cpu()),
        "delta_125": float(masked_mean((ratio < 1.25).float(), region_mask).cpu()),
        "boundary": float(masked_mean(torch.abs(pred - gt_pos), boundary).cpu()) if boundary.sum().item() > 0 else 0.0,
        "reliable_bg_disturbance": (
            float(masked_mean(torch.abs(pred - raw), reliable_bg).cpu())
            if reliable_bg.sum().item() > 0 else 0.0
        ),
    }


# =============================================================================
# CACHE DATASET
# =============================================================================
class CachedShardDataset(Dataset):
    def __init__(self, shards: Sequence[Path], image_size: Tuple[int, int] = IMAGE_SIZE):
        self.shards = [Path(p) for p in shards]
        self.image_size = image_size
        if not self.shards:
            raise RuntimeError("Empty shard list")

    def __len__(self) -> int:
        return len(self.shards)

    @staticmethod
    def _squeeze_tensor(x: torch.Tensor) -> torch.Tensor:
        if x.ndim >= 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        return x.float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.shards[idx]
        shard = torch.load(path, map_location="cpu", weights_only=False)
        required = ("rgb", "raw_depth", "gt_depth", "mask", "valid", "rel_aligned")
        for key in required:
            if key not in shard:
                raise KeyError(f"Cache shard missing required key '{key}': {path}")
        return {
            key: self._squeeze_tensor(value)
            for key, value in shard.items()
            if torch.is_tensor(value)
        }


def load_split_shards(cache_root: Path, split: str, max_n: Optional[int]) -> List[Path]:
    split_dir = cache_root / split
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = [split_dir / row["file"] for row in manifest["shards"]]
    rng = np.random.default_rng(SEED + (0 if split == "train" else 17))
    if max_n is not None and len(shards) > max_n:
        ids = rng.choice(len(shards), size=max_n, replace=False).tolist()
        shards = [shards[i] for i in ids]
    return shards


def ragged_shard_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}
    keys = sorted(set().union(*[set(x.keys()) for x in batch]))
    out: Dict[str, Any] = {}
    for key in keys:
        vals = [x[key] for x in batch if key in x]
        if not vals:
            continue
        if torch.is_tensor(vals[0]):
            vals = [v.float().contiguous() for v in vals]
            if vals[0].ndim <= 2:
                out[key] = torch.stack(vals, dim=0).contiguous()
            else:
                try:
                    out[key] = torch.cat(vals, dim=0).contiguous()
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"ragged_shard_collate failed for key={key}, "
                        f"shapes={[tuple(v.shape) for v in vals]}"
                    ) from exc
        else:
            out[key] = vals
    return out


def to_device(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v.to(DEVICE, non_blocking=True).float() if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def build_inputs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    rgb = force_4d_rgb(batch["rgb"]).clamp(0.0, 1.0)
    raw = safe_depth(force_4d_map(batch["raw_depth"]))
    gt = safe_depth(force_4d_map(batch["gt_depth"]))
    mask = force_4d_map(batch["mask"]).clamp(0.0, 1.0)
    valid = force_4d_map(batch["valid"]).clamp(0.0, 1.0)
    rel = safe_depth(force_4d_map(batch["rel_aligned"]))
    zeros = torch.zeros_like(mask)
    ones = torch.ones_like(mask)

    def opt(name: str, default: torch.Tensor, clamp: Optional[Tuple[float, float]] = (0.0, 1.0)):
        if name not in batch:
            return default
        value = force_4d_map(batch[name])
        if clamp is not None:
            value = value.clamp(*clamp)
        return value

    boundary = torch.maximum(opt("boundary", zeros), build_boundary_ring(mask))
    return {
        "rgb": rgb,
        "raw": raw,
        "gt": gt,
        "mask": mask,
        "valid": valid,
        "rel": rel,
        "rel_conf": opt("rel_conf", ones),
        "raw_prior": opt("raw_prior", zeros),
        "rel_bg_resid": opt("rel_bg_resid", zeros, clamp=None),
        "rel_bg_coverage": opt("rel_bg_coverage", zeros),
        "boundary": boundary,
        "old_base": safe_depth(force_4d_map(batch["base_final"])) if "base_final" in batch else raw,
    }


def batch_sample_count(batch: Dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise RuntimeError("Cannot determine batch sample count")


def iter_microbatches(batch: Dict[str, Any], microbatch_size: int):
    n = batch_sample_count(batch)
    microbatch_size = max(1, int(microbatch_size))
    for start in range(0, n, microbatch_size):
        end = min(n, start + microbatch_size)
        part: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n:
                part[key] = value[start:end]
            else:
                part[key] = value
        yield part, end - start, n


# =============================================================================
# POSTERIOR MODEL COMPONENTS
# =============================================================================
def zero_init_conv(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


@torch.no_grad()
def robust_global_align(rel: torch.Tensor, raw: torch.Tensor, anchor: torch.Tensor):
    aligned, scales, biases = [], [], []
    for i in range(rel.shape[0]):
        m = anchor[i, 0] > 0.5
        x, y = rel[i, 0][m], raw[i, 0][m]
        if x.numel() < MIN_ANCHOR_PIXELS:
            a, b = rel.new_tensor(1.0), rel.new_tensor(0.0)
        else:
            w = torch.ones_like(x)
            a, b = rel.new_tensor(1.0), rel.new_tensor(0.0)
            for _ in range(3):
                sw = w.sum().clamp_min(EPS)
                mx, my = (w * x).sum() / sw, (w * y).sum() / sw
                a = (w * (x - mx) * (y - my)).sum() / (w * (x - mx).square()).sum().clamp_min(EPS)
                b = my - a * mx
                a = a.clamp(0.35, 2.50)
                b = b.clamp(-1.5, 1.5)
                r = (y - (a * x + b)).abs()
                delta = (1.5 * r.median()).clamp_min(0.005)
                w = torch.where(r <= delta, torch.ones_like(r), delta / (r + EPS))
        aligned.append(a * rel[i:i + 1] + b)
        scales.append(a.reshape(1))
        biases.append(b.reshape(1))
    return safe_depth(torch.cat(aligned, 0)), torch.cat(scales), torch.cat(biases)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, dilation: int = 1):
        super().__init__()
        groups = min(8, cout)
        while cout % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalMetricAligner(nn.Module):
    def __init__(self, cin: int = 11, hidden: int = 32):
        super().__init__()
        self.body = nn.Sequential(ConvBlock(cin, hidden), ConvBlock(hidden, hidden, dilation=2))
        self.out = zero_init_conv(nn.Conv2d(hidden, 2, 3, padding=1))

    def forward(self, x: torch.Tensor, rel_global: torch.Tensor):
        z = self.out(self.body(x))
        ds = MAX_LOCAL_SCALE_LOG * torch.tanh(z[:, 0:1])
        db = MAX_LOCAL_BIAS * torch.tanh(z[:, 1:2])
        rel_metric = safe_depth(rel_global * torch.exp(ds) + db)
        align_reg = ds.abs().mean() + db.abs().mean() + 0.25 * (
            gradient_mag(ds).mean() + gradient_mag(db).mean()
        )
        return rel_metric, ds, db, align_reg


class FailureEstimator(nn.Module):
    def __init__(self, cin: int = 14, hidden: int = 48):
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(cin, hidden),
            ConvBlock(hidden, hidden, dilation=2),
            ConvBlock(hidden, hidden),
        )
        self.failure = nn.Conv2d(hidden, 4, 1)
        self.uncert = nn.Conv2d(hidden, 2, 1)

    def forward(self, x: torch.Tensor):
        feat = self.body(x)
        return feat, self.failure(feat), self.uncert(feat).clamp(-6.0, 2.0)


class ResidualExpert(nn.Module):
    def __init__(self, cin: int = 64, hidden: int = 48, dilation: int = 1):
        super().__init__()
        self.body = nn.Sequential(ConvBlock(cin, hidden, dilation=dilation), ConvBlock(hidden, hidden))
        self.out = nn.Conv2d(hidden, 2, 1)

    def forward(self, feat: torch.Tensor):
        z = self.out(self.body(feat))
        delta = MAX_EXPERT_DELTA * torch.tanh(z[:, 0:1])
        log_b = z[:, 1:2].clamp(-6.0, 2.0)
        return delta, log_b


class RiskControlledRefiner(nn.Module):
    """Predict a candidate correction and pre/post-update risks.

    The acceptance weight is not an independent unconstrained gate. It is derived
    from the predicted risk reduction, making the update decision interpretable.
    """

    def __init__(self, cin: int = 17, hidden: int = 48):
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(cin, hidden),
            ConvBlock(hidden, hidden, dilation=2),
            ConvBlock(hidden, hidden),
        )
        # candidate delta, pre-update risk logit, post-update risk logit
        self.out = zero_init_conv(nn.Conv2d(hidden, 3, 1))

    def forward(self, x: torch.Tensor):
        z = self.out(self.body(x))
        delta = MAX_RISK_DELTA * torch.tanh(z[:, 0:1])
        risk_before = F.softplus(z[:, 1:2])
        risk_after = F.softplus(z[:, 2:3])
        risk_gain = risk_before - risk_after
        # Keep the pre-sigmoid value for numerically stable BCEWithLogits training.
        # This also avoids torch.binary_cross_entropy being rejected inside AMP autocast.
        acceptance_logit = (risk_gain - RISK_ACCEPT_MARGIN) / max(RISK_TEMPERATURE, EPS)
        acceptance = torch.sigmoid(acceptance_logit)
        return delta, risk_before, risk_after, acceptance_logit, acceptance


class FailureAwarePosteriorDepth(nn.Module):
    """Failure-aware posterior reconstruction with a frozen base stream."""

    PRIOR_CH = 13

    def __init__(self, base_mod):
        super().__init__()
        # The source package exposes the base architecture under this constructor.
        self.base_stream = base_mod.FDCT(
            in_channels=4,
            hidden_channels=BASE_HIDDEN,
            L=BASE_L,
            k=BASE_K,
            use_DUC=BASE_USE_DUC,
        )
        self.base_reference = base_mod.FDCT(
            in_channels=4,
            hidden_channels=BASE_HIDDEN,
            L=BASE_L,
            k=BASE_K,
            use_DUC=BASE_USE_DUC,
        )

        h = BASE_HIDDEN
        for name in ("first", "e1", "e2", "e3", "e4", "d1", "d2", "d3", "out"):
            setattr(self, "adapt_" + name, nn.Conv2d(self.PRIOR_CH, h, 3, padding=1))

        self.aligner = LocalMetricAligner(11, 32)
        self.failure_net = FailureEstimator(14, 48)
        self.shared = nn.Sequential(ConvBlock(15, 64), ConvBlock(64, 64, dilation=2), ConvBlock(64, 64))
        self.missing_expert = ResidualExpert(64, 48, dilation=3)
        self.biased_expert = ResidualExpert(64, 48, dilation=1)
        self.boundary_expert = ResidualExpert(64, 48, dilation=2)
        self.router = nn.Sequential(ConvBlock(68, 48), nn.Conv2d(48, 3, 1))
        self.risk_refiner = RiskControlledRefiner(cin=17, hidden=48)

    @staticmethod
    def _map_stage2_key(key: str) -> Optional[str]:
        # Map legacy implementation names to neutral base-stream names.
        if key.startswith("fdct_ref."):
            return "base_reference." + key[len("fdct_ref."):]
        if key.startswith("fdct."):
            return "base_stream." + key[len("fdct."):]
        # The old unconstrained refiner is deliberately discarded.
        if key.startswith("refiner."):
            return None
        return key

    def load_stage2_checkpoint(self, ckpt_path: Path) -> Dict[str, Any]:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Stage-2 checkpoint not found: {ckpt_path}")
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        old_state = payload.get("model", payload.get("model_state_dict", payload))
        mapped: Dict[str, torch.Tensor] = {}
        for key, value in old_state.items():
            clean = key[7:] if key.startswith("module.") else key
            new_key = self._map_stage2_key(clean)
            if new_key is not None:
                mapped[new_key] = value

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        expected_missing = [k for k in missing if k.startswith("risk_refiner.")]
        other_missing = [k for k in missing if not k.startswith("risk_refiner.")]
        if other_missing or unexpected:
            print(
                f"[Posterior checkpoint] missing_other={len(other_missing)}, "
                f"missing_new_refiner={len(expected_missing)}, unexpected={len(unexpected)}"
            )
            if other_missing:
                print("  first missing:", other_missing[:10])
            if unexpected:
                print("  first unexpected:", unexpected[:10])
        else:
            print(
                f"[Posterior checkpoint] loaded successfully; "
                f"new refinement parameters={len(expected_missing)}"
            )
        return payload

    def freeze_posterior_train_refiner(self) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.risk_refiner.parameters():
            param.requires_grad_(True)

    def train(self, mode: bool = True):
        # Keep every frozen posterior module in eval mode. Only the new refiner trains.
        super().train(False)
        self.risk_refiner.train(mode)
        return self

    def prior_at(self, priors: torch.Tensor, feat: torch.Tensor, adapter: nn.Conv2d) -> torch.Tensor:
        p = F.interpolate(priors, size=feat.shape[-2:], mode="bilinear", align_corners=True)
        return adapter(p) * ADAPTER_FEATURE_SCALE

    @torch.no_grad()
    def forward_reference(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        d = depth[:, 0] if depth.ndim == 4 else depth
        out = self.base_reference(rgb, d)
        return safe_depth(out.unsqueeze(1) if out.ndim == 3 else out)

    def forward_adapted_base(self, rgb: torch.Tensor, depth: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
        f = self.base_stream
        dv = depth if depth.ndim == 4 else depth.unsqueeze(1)

        h = f.first(torch.cat((rgb, dv), 1))
        h = h + self.prior_at(priors, h, self.adapt_first)
        d1 = F.interpolate(dv, scale_factor=0.5, mode="bilinear", align_corners=True)
        h_d1s = h

        h = h + self.prior_at(priors, h, self.adapt_e1)
        h = f.dense1_conv1(torch.cat((h, d1), 1))
        h = f.dense1(h)
        h = f.dense1_conv2(h)

        d2 = F.interpolate(d1, scale_factor=0.5, mode="bilinear", align_corners=True)
        h_d2s = h
        h_d2d = f.skip_down2(torch.cat((h_d2s, f.skip_down1(h_d1s)), 1))

        h = h + self.prior_at(priors, h, self.adapt_e2)
        h = f.dense2_conv1(torch.cat((h, d2, f.down_res1(h_d1s)), 1))
        h = f.dense2(h)
        h = f.dense2_conv2(h)

        d3 = F.interpolate(d2, scale_factor=0.5, mode="bilinear", align_corners=True)
        h_d3s = h
        h_d3d = f.skip_down3(torch.cat((h_d3s, h_d2d), 1))

        h = h + self.prior_at(priors, h, self.adapt_e3)
        h = f.dense3_conv1(torch.cat((h, d3, f.down_res2(h_d2s)), 1))
        h = f.dense3(h)
        h = f.dense3_conv2(h)

        d4 = F.interpolate(d3, scale_factor=0.5, mode="bilinear", align_corners=True)
        h = h + self.prior_at(priors, h, self.adapt_e4)
        h = f.dense4_conv1(torch.cat((h, d4, f.down_res3(h_d3s)), 1))
        h = f.dense4(h)

        h = torch.cat((h, h_d3d), 1)
        h = f.cdown(h)
        h_skip3 = h

        h = h + self.prior_at(priors, h, self.adapt_d1)
        h = f.updense1_conv(torch.cat((h, d4), 1))
        h = f.updense1(h)
        h = f.updense1_duc(h)
        h_skip1 = h

        h = h + self.prior_at(priors, h, self.adapt_d2)
        h = torch.cat((h, h_d3s, d3, f.skip_up3(h_skip3)), 1)
        h = f.updense2_conv(h)
        h = f.updense2(h)
        h = f.updense2_duc(h)
        h_skip2 = h

        h = h + self.prior_at(priors, h, self.adapt_d3)
        h = torch.cat((h, h_d2s, d2, f.skip_up1(h_skip1)), 1)
        h = f.updense3_conv(h)
        h = f.updense3(h)
        h = f.updense3_duc(h)

        h = torch.cat((h, h_d1s, d1, f.skip_up2(h_skip2)), 1)
        h = f.updense4_conv(h)
        h = f.updense4(h)
        h = f.updense4_duc(h)
        h = h + self.prior_at(priors, h, self.adapt_out)
        return safe_depth(f.final(h))

    def forward_posterior(self, inp: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rgb, raw, rel = inp["rgb"], inp["raw"], inp["rel"]
        mask, valid = inp["mask"], inp["valid"]
        boundary, raw_prior, rel_conf = inp["boundary"], inp["raw_prior"], inp["rel_conf"]

        raw_valid = (raw > EPS).float() * valid
        sdm = approximate_signed_distance(mask)
        grad_raw = torch.clamp(gradient_mag(raw) / 0.08, 0.0, 4.0)
        grad_rel = torch.clamp(gradient_mag(rel) / 0.08, 0.0, 4.0)

        outside = (1.0 - dilate_binary(mask, ANCHOR_DILATE_KERNEL)).clamp(0.0, 1.0)
        anchor = raw_valid * outside * (grad_raw < ANCHOR_GRAD_THR / 0.08).float()
        if INPUT_REL_ALREADY_METRIC_ALIGNED:
            rel_global = rel
        else:
            rel_global, _, _ = robust_global_align(rel, raw, anchor)

        disc0 = torch.clamp((rel_global - raw) / 0.75, -1.0, 1.0) * raw_valid
        align_x = torch.cat(
            [
                rgb,
                norm_depth(raw),
                norm_depth(rel_global),
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
        discrepancy = torch.clamp((rel_metric - raw) / 0.75, -1.0, 1.0) * raw_valid

        fail_x = torch.cat(
            [
                rgb,
                norm_depth(raw),
                norm_depth(rel_metric),
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
        _, fail_logits, source_logb = self.failure_net(fail_x)
        fail_prob = F.softmax(fail_logits, 1)
        raw_logb, rel_logb = source_logb[:, 0:1], source_logb[:, 1:2]
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
                norm_depth(raw),
                norm_depth(rel_metric),
                discrepancy,
                grad_raw,
                grad_rel,
                inp["rel_bg_resid"],
                inp["rel_bg_coverage"],
            ],
            1,
        )
        base_depth = self.forward_adapted_base(rgb, raw, priors)

        ctx = torch.cat([fail_x, norm_depth(base_depth)], 1)
        shared = self.shared(ctx)
        dm, um = self.missing_expert(shared)
        dd, ud = self.biased_expert(shared)
        db, ub = self.boundary_expert(shared)
        router_logits = self.router(torch.cat([shared, fail_prob], 1))
        pi = F.softmax(router_logits, 1)

        deltas = torch.cat([dm, dd, db], 1)
        expert_logbs = torch.cat([um, ud, ub], 1)
        mix_delta = (pi * deltas).sum(1, keepdim=True)
        expert = safe_depth(base_depth + mix_delta)
        expert_var = (
            pi * (torch.exp(2.0 * expert_logbs) + deltas.square())
        ).sum(1, keepdim=True) - mix_delta.square()
        expert_logb = 0.5 * torch.log(expert_var.clamp_min(1.0e-6)).clamp(-6.0, 2.0)

        w_raw = raw_valid * p_valid.clamp_min(0.02) * torch.exp(-raw_logb)
        w_rel = rel_conf.clamp_min(0.05) * torch.exp(-rel_logb)
        w_expert = (0.20 + 0.80 * p_fail) * torch.exp(-expert_logb)
        weights = torch.cat([w_raw, w_rel, w_expert], 1)
        alpha = weights / weights.sum(1, keepdim=True).clamp_min(EPS)

        candidates = torch.cat([raw, rel_metric, expert], 1)
        fused = safe_depth((alpha * candidates).sum(1, keepdim=True))
        source_logbs = torch.cat([raw_logb, rel_logb, expert_logb], 1)
        mix_var = (
            alpha * (torch.exp(2.0 * source_logbs) + candidates.square())
        ).sum(1, keepdim=True) - fused.square()
        final_logb = 0.5 * torch.log(mix_var.clamp_min(1.0e-6)).clamp(-6.0, 2.0)
        route_entropy = -(
            pi * torch.log(pi.clamp_min(EPS))
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
                [safe_depth(base_depth + dm), safe_depth(base_depth + dd), safe_depth(base_depth + db)],
                1,
            ),
            "fused": fused,
            "final_logb": final_logb,
            "route_entropy": route_entropy,
            "sdm": sdm,
        }

    def forward(self, inp: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Frozen posterior inference does not build a training graph, saving memory.
        with torch.no_grad():
            posterior = self.forward_posterior(inp)

        rgb, raw, mask = inp["rgb"], inp["raw"], inp["mask"]
        boundary = inp["boundary"]
        fused = posterior["fused"]
        p_fail = posterior["p_fail"].detach()

        mask_support = dilate_binary(mask, SUPPORT_DILATE_KERNEL).clamp(0.0, 1.0)
        failure_support = torch.clamp(
            p_fail + BOUNDARY_SUPPORT_WEIGHT * boundary,
            0.0,
            1.0,
        )
        support = (mask_support * failure_support).detach()

        current = fused
        iter_preds: List[torch.Tensor] = []
        candidate = fused
        delta = torch.zeros_like(fused)
        risk_before = torch.zeros_like(fused)
        risk_after = torch.zeros_like(fused)
        acceptance_logit = torch.zeros_like(fused)
        acceptance = torch.zeros_like(fused)

        for _ in range(REFINE_STEPS):
            refine_input = torch.cat(
                [
                    rgb,
                    norm_depth(current),
                    norm_depth(raw),
                    norm_depth(posterior["rel_metric"]),
                    norm_depth(posterior["base_depth"]),
                    mask,
                    boundary,
                    posterior["sdm"],
                    p_fail,
                    posterior["final_logb"],
                    posterior["route_entropy"],
                    support,
                    posterior["alpha"],
                ],
                1,
            )
            delta, risk_before, risk_after, acceptance_logit, acceptance = self.risk_refiner(refine_input)
            candidate = safe_depth(current + support * delta)
            current = safe_depth(current + support * acceptance * delta)
            iter_preds.append(current)

        benchmark_output = current * mask + raw * (1.0 - mask)
        return {
            **posterior,
            "final": current,
            "benchmark_output": benchmark_output,
            "candidate": candidate,
            "iter_preds": iter_preds,
            "support": support,
            "delta": delta,
            "risk_before": risk_before,
            "risk_after": risk_after,
            "acceptance_logit": acceptance_logit,
            "acceptance": acceptance,
        }


# =============================================================================
# FAILURE TARGETS / DIAGNOSTICS
# =============================================================================
def failure_targets(
    raw: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor,
    boundary: torch.Tensor,
):
    err = (raw - gt).abs()
    missing = (raw <= EPS) & (valid > 0.5)
    fail = err > (FAIL_ABS_THR + FAIL_REL_THR * gt)
    boundary_fail = (
        (boundary > 0.15)
        & (err > BOUNDARY_FAIL_THR)
        & (~missing)
        & (valid > 0.5)
    )
    biased = fail & (~missing) & (~boundary_fail) & (valid > 0.5)
    labels = torch.zeros_like(raw, dtype=torch.long)
    labels[missing] = 1
    labels[biased] = 2
    labels[boundary_fail] = 3
    return labels, err


@torch.no_grad()
def failure_prf(
    pred_label: torch.Tensor,
    target_label: torch.Tensor,
    valid: torch.Tensor,
    cls: int,
):
    m = valid > 0.5
    p = pred_label == int(cls)
    y = target_label == int(cls)
    tp = (p & y & m).sum().float()
    fp = (p & (~y) & m).sum().float()
    fn = ((~p) & y & m).sum().float()
    support = (y & m).sum().float()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(EPS)
    return precision, recall, f1, support


# =============================================================================
# RISK-REFINEMENT LOSS
# =============================================================================
def compute_loss(
    model: FailureAwarePosteriorDepth,
    batch: Dict[str, Any],
    return_outputs: bool = False,
):
    inp = build_inputs(batch)
    out = model(inp)

    raw, gt = inp["raw"], safe_depth(inp["gt"])
    mask, valid = inp["mask"], inp["valid"]
    boundary = inp["boundary"] * valid
    region_mask = valid * mask
    support_region = valid * out["support"].detach()
    if support_region.sum().item() <= 0:
        support_region = region_mask

    final = out["final"]
    fused = out["fused"].detach()
    candidate = out["candidate"]

    final_err = torch.abs(final - gt)
    fused_err = torch.abs(fused - gt)
    candidate_err = torch.abs(candidate - gt)

    loss_mask = charbonnier(final, gt, region_mask)
    loss_all = charbonnier(final, gt, valid)
    loss_rmse = masked_rmse(final, gt, region_mask)
    loss_boundary = (
        masked_mean(final_err, boundary)
        if boundary.sum().item() > 0 else final.new_tensor(0.0)
    )
    loss_grad = gradient_l1(final, gt, region_mask)
    loss_hard = hard_pixel_rmse(final, gt, region_mask)

    # Actual update must not increase reconstruction risk relative to posterior fusion.
    loss_monotonic = masked_mean(
        F.relu(final_err - fused_err - MONOTONIC_TOLERANCE),
        support_region,
    )

    # Calibrate predicted pre/post risks against observed absolute errors.
    risk_before_target = fused_err.detach()
    risk_after_target = candidate_err.detach()
    risk_before_map = F.smooth_l1_loss(
        out["risk_before"], risk_before_target, reduction="none", beta=0.005
    )
    risk_after_map = F.smooth_l1_loss(
        out["risk_after"], risk_after_target, reduction="none", beta=0.005
    )
    loss_risk_calibration = 0.5 * (
        masked_mean(risk_before_map, support_region)
        + masked_mean(risk_after_map, support_region)
    )

    # Supervise the risk-derived acceptance decision.
    accept_target = (
        candidate_err.detach() + ACCEPT_TARGET_MARGIN < fused_err.detach()
    ).float()
    # BCEWithLogits is stable under AMP; plain BCE on sigmoid probabilities is not.
    acceptance_bce = F.binary_cross_entropy_with_logits(
        out["acceptance_logit"].float(),
        accept_target.float(),
        reduction="none",
    ).to(final.dtype)
    loss_acceptance = masked_mean(acceptance_bce, support_region)

    labels, raw_err = failure_targets(raw, gt, valid, boundary)
    reliable_region = (
        valid
        * (raw > EPS).float()
        * (raw_err <= RELIABLE_RAW_THR).float()
    )
    loss_trust = (
        masked_mean(torch.abs(final - raw), reliable_region)
        if reliable_region.sum().item() > 0 else final.new_tensor(0.0)
    )
    loss_update_reg = masked_mean(torch.abs(final - fused), support_region)

    total = (
        W_FINAL_MASK * loss_mask
        + W_FINAL_ALL * loss_all
        + W_FINAL_RMSE * loss_rmse
        + W_BOUNDARY * loss_boundary
        + W_GRAD * loss_grad
        + W_HARD * loss_hard
        + W_MONOTONIC * loss_monotonic
        + W_RISK_CALIBRATION * loss_risk_calibration
        + W_ACCEPTANCE * loss_acceptance
        + W_TRUST_REGION * loss_trust
        + W_UPDATE_REG * loss_update_reg
    )

    with torch.no_grad():
        pred_label = out["fail_logits"].argmax(1, keepdim=True)
        acc = masked_mean((pred_label == labels).float(), valid)
        recalls: List[torch.Tensor] = []
        cls_stats: Dict[str, float] = {}
        for cls, name in {0: "valid", 1: "missing", 2: "biased", 3: "boundary"}.items():
            precision, recall, f1, support_count = failure_prf(pred_label, labels, valid, cls)
            cls_stats[f"precision_{name}"] = float(precision)
            cls_stats[f"recall_{name}"] = float(recall)
            cls_stats[f"f1_{name}"] = float(f1)
            cls_stats[f"support_{name}"] = float(support_count)
            recalls.append(recall)

        monotonic_violation = masked_mean(
            (final_err > fused_err + 1.0e-6).float(), support_region
        )
        accepted = (out["acceptance"] > 0.5).float()
        accepted_region = support_region * accepted
        accepted_improvement = (
            masked_mean((final_err < fused_err).float(), accepted_region)
            if accepted_region.sum().item() > 0 else final.new_tensor(0.0)
        )

        biased_region = valid * (labels == 2).float()
        boundary_region = valid * (labels == 3).float()

        def region_source_weight(channel: int, region: torch.Tensor) -> float:
            if region.sum().item() <= 0:
                return 0.0
            return float(masked_mean(out["alpha"][:, channel:channel + 1], region))

        stats = {
            "loss_total": float(total),
            "mae_mask": float(masked_mean(final_err, region_mask)),
            "rmse_mask": float(masked_rmse(final, gt, region_mask)),
            "fused_mae_mask": float(masked_mean(fused_err, region_mask)),
            "fused_rmse_mask": float(masked_rmse(fused, gt, region_mask)),
            "loss_monotonic": float(loss_monotonic),
            "loss_risk_calibration": float(loss_risk_calibration),
            "loss_acceptance": float(loss_acceptance),
            "loss_trust": float(loss_trust),
            "support_mean": float(out["support"].mean()),
            "acceptance_mean": float(masked_mean(out["acceptance"], support_region)),
            "update_abs": float(masked_mean(torch.abs(final - fused), support_region)),
            "candidate_delta_abs": float(masked_mean(torch.abs(out["delta"]), support_region)),
            "monotonic_violation": float(monotonic_violation),
            "accepted_improvement": float(accepted_improvement),
            "risk_before_mae": float(masked_mean(torch.abs(out["risk_before"] - risk_before_target), support_region)),
            "risk_after_mae": float(masked_mean(torch.abs(out["risk_after"] - risk_after_target), support_region)),
            "fail_acc": float(acc),
            "balanced_acc": float(torch.stack(recalls).mean()),
            "route_entropy": float(out["route_entropy"].mean()),
            "raw_w": float(out["alpha"][:, 0:1].mean()),
            "rel_w": float(out["alpha"][:, 1:2].mean()),
            "expert_w": float(out["alpha"][:, 2:3].mean()),
            "expert_w_biased": region_source_weight(2, biased_region),
            "expert_w_boundary": region_source_weight(2, boundary_region),
        }
        stats.update(cls_stats)

    if return_outputs:
        return total, stats, inp, out
    return total, stats


# =============================================================================
# EVALUATION / LOGGING
# =============================================================================
@torch.no_grad()
def evaluate(
    model: FailureAwarePosteriorDepth,
    loader: DataLoader,
    desc: str = "Val",
):
    model.eval()
    names = [
        "Raw Depth",
        "Input relative prior",
        "Metric-calibrated prior",
        "Previous model result",
        "Base completion",
        "Posterior fusion",
        "Risk-controlled refinement",
        "Benchmark output",
    ]
    rows: Dict[str, List[Dict[str, float]]] = {name: [] for name in names}
    aux: List[Dict[str, float]] = []

    for loader_batch in tqdm(loader, desc=desc, leave=False):
        loader_batch = to_device(loader_batch)
        for batch, _, _ in iter_microbatches(loader_batch, VAL_MICROBATCH):
            _, stats, inp, out = compute_loss(model, batch, return_outputs=True)
            raw, gt, mask, valid = inp["raw"], inp["gt"], inp["mask"], inp["valid"]
            reference = model.forward_reference(inp["rgb"], raw)
            reference_benchmark = reference * mask + raw * (1.0 - mask)
            previous = inp["old_base"] * mask + raw * (1.0 - mask)

            rows["Raw Depth"].append(metric_values(raw, raw, gt, mask, valid))
            rows["Input relative prior"].append(metric_values(inp["rel"], raw, gt, mask, valid))
            rows["Metric-calibrated prior"].append(metric_values(out["rel_metric"], raw, gt, mask, valid))
            rows["Previous model result"].append(metric_values(previous, raw, gt, mask, valid))
            rows["Base completion"].append(metric_values(reference_benchmark, raw, gt, mask, valid))
            rows["Posterior fusion"].append(metric_values(out["fused"], raw, gt, mask, valid))
            rows["Risk-controlled refinement"].append(metric_values(out["final"], raw, gt, mask, valid))
            rows["Benchmark output"].append(metric_values(out["benchmark_output"], raw, gt, mask, valid))
            aux.append(stats)

    avg = {name: avg_dicts(values) for name, values in rows.items()}
    avg["_aux"] = avg_dicts(aux)
    return avg


def print_summary(label: str, train_loss: Optional[float], rows: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 170)
    if train_loss is None:
        print(label)
    else:
        print(f"{label} | train loss {train_loss:.6f}")

    print(
        f"{'Variant':<29} | {'MAE_all':>9} | {'RMSE_all':>9} | "
        f"{'MAE_mask':>9} | {'RMSE_mask':>9} | {'Boundary':>9} | "
        f"{'BG':>9} | {'Score':>9}"
    )
    ordered = [
        "Raw Depth",
        "Input relative prior",
        "Metric-calibrated prior",
        "Previous model result",
        "Base completion",
        "Posterior fusion",
        "Risk-controlled refinement",
        "Benchmark output",
    ]
    for name in ordered:
        row = rows[name]
        print(
            f"{name:<29} | {fmt(row.get('mae_all')):>9} | {fmt(row.get('rmse_all')):>9} | "
            f"{fmt(row.get('mae_mask')):>9} | {fmt(row.get('rmse_mask')):>9} | "
            f"{fmt(row.get('boundary')):>9} | {fmt(row.get('reliable_bg_disturbance')):>9} | "
            f"{selection_score(row):>9.6f}"
        )

    print("\nTransparent-mask benchmark metrics")
    print(
        f"{'Variant':<29} | {'REL':>9} | {'δ1.05':>9} | {'δ1.10':>9} | {'δ1.25':>9}"
    )
    for name in ("Previous model result", "Base completion", "Posterior fusion", "Risk-controlled refinement", "Benchmark output"):
        row = rows[name]
        print(
            f"{name:<29} | {fmt(row.get('rel_mask')):>9} | "
            f"{fmt(row.get('delta_105')):>9} | {fmt(row.get('delta_110')):>9} | "
            f"{fmt(row.get('delta_125')):>9}"
        )

    aux = rows["_aux"]
    print(
        f"[Risk] support={aux.get('support_mean', 0):.3f}, "
        f"accept={aux.get('acceptance_mean', 0):.3f}, "
        f"update={aux.get('update_abs', 0):.6f}, "
        f"violation={aux.get('monotonic_violation', 0):.4f}, "
        f"accepted_improve={aux.get('accepted_improvement', 0):.4f}, "
        f"risk_MAE(before/after)={aux.get('risk_before_mae', 0):.5f}/{aux.get('risk_after_mae', 0):.5f}"
    )
    print(
        f"[Failure] acc={aux.get('fail_acc', 0):.4f}, "
        f"balanced_acc={aux.get('balanced_acc', 0):.4f}, "
        f"F1(miss/bias/bnd)={aux.get('f1_missing', 0):.3f}/"
        f"{aux.get('f1_biased', 0):.3f}/{aux.get('f1_boundary', 0):.3f}"
    )
    print(
        f"[Posterior sources] raw/rel/expert={aux.get('raw_w', 0):.3f}/"
        f"{aux.get('rel_w', 0):.3f}/{aux.get('expert_w', 0):.3f}; "
        f"expert@biased={aux.get('expert_w_biased', 0):.3f}, "
        f"expert@boundary={aux.get('expert_w_boundary', 0):.3f}, "
        f"route_entropy={aux.get('route_entropy', 0):.3f}"
    )


def write_history(history: List[Dict[str, Any]]) -> None:
    (OUT_DIR / "risk_refinement_log.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not history:
        return
    keys = sorted(set().union(*[row.keys() for row in history]))
    with open(OUT_DIR / "risk_refinement_log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path: Path,
    model: FailureAwarePosteriorDepth,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: torch.cuda.amp.GradScaler,
    refine_epoch: int,
    rows: Dict[str, Dict[str, float]],
    history: List[Dict[str, Any]],
    best_score: float,
    no_improve: int,
) -> None:
    torch.save(
        {
            "refine_epoch": refine_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_score": float(best_score),
            "no_improve": int(no_improve),
            "history": history,
            "all_rows": {k: v for k, v in rows.items() if not k.startswith("_")},
            "aux": rows.get("_aux", {}),
            "stage2_checkpoint": str(STAGE2_CKPT),
            "config": {
                "MAX_RISK_DELTA": MAX_RISK_DELTA,
                "SUPPORT_DILATE_KERNEL": SUPPORT_DILATE_KERNEL,
                "BOUNDARY_SUPPORT_WEIGHT": BOUNDARY_SUPPORT_WEIGHT,
                "RISK_TEMPERATURE": RISK_TEMPERATURE,
                "LR_RISK_REFINER": LR_RISK_REFINER,
                "TRAIN_MICROBATCH": TRAIN_MICROBATCH,
                "VAL_MICROBATCH": VAL_MICROBATCH,
            },
        },
        str(path),
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    set_seed(SEED)
    print("=" * 150)
    print("FAPR-Depth v4.1 | Failure-Aware Posterior Reconstruction + Risk-Controlled Refinement")
    print("=" * 150)
    print(
        f"DEVICE={DEVICE}, AMP={USE_AMP}\n"
        f"CACHE_ROOT={CACHE_ROOT}\n"
        f"STAGE2_CKPT={STAGE2_CKPT}\n"
        f"OUT_DIR={OUT_DIR}"
    )

    base_mod = load_base_source_module()
    train_shards = load_split_shards(CACHE_ROOT, "train", MAX_TRAIN_SHARDS)
    val_shards = load_split_shards(CACHE_ROOT, "val", MAX_VAL_SHARDS)
    train_loader = DataLoader(
        CachedShardDataset(train_shards),
        batch_size=LOADER_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=ragged_shard_collate,
    )
    val_loader = DataLoader(
        CachedShardDataset(val_shards),
        batch_size=LOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=ragged_shard_collate,
    )

    first = to_device(next(iter(train_loader)))
    probe = build_inputs(first)
    print("[Probe]", {k: tuple(v.shape) for k, v in probe.items() if torch.is_tensor(v)})
    del first, probe
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    model = FailureAwarePosteriorDepth(base_mod).to(DEVICE)
    model.freeze_posterior_train_refiner()

    optimizer = torch.optim.AdamW(
        model.risk_refiner.parameters(),
        lr=LR_RISK_REFINER,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(REFINE_EPOCHS, 1),
        eta_min=LR_RISK_REFINER * 0.10,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    history: List[Dict[str, Any]] = []
    best = float("inf")
    no_improve = 0
    start_epoch = 1

    if AUTO_RESUME and RESUME_CKPT.exists():
        payload = torch.load(str(RESUME_CKPT), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if missing or unexpected:
            print(f"[Resume model] missing={len(missing)}, unexpected={len(unexpected)}")
        optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if USE_AMP and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload.get("refine_epoch", 0)) + 1
        history = list(payload.get("history", []))
        best = float(payload.get("best_score", float("inf")))
        no_improve = int(payload.get("no_improve", 0))
        print(
            f"[Resume] {RESUME_CKPT}\n"
            f"[Resume] completed refinement epoch={start_epoch - 1}, "
            f"next={start_epoch}, best={best:.6f}"
        )
    else:
        stage2_payload = model.load_stage2_checkpoint(STAGE2_CKPT)
        model.freeze_posterior_train_refiner()
        source_epoch = stage2_payload.get("epoch", "unknown")
        source_score = stage2_payload.get("best_score", stage2_payload.get("val", {}).get("score", "unknown"))
        print(f"[Initialization] posterior checkpoint epoch={source_epoch}, recorded best={source_score}")

        # With a zero-initialized delta head, this is exactly the Stage-2 posterior.
        initial_rows = evaluate(model, val_loader, desc="Initial posterior validation")
        print_summary("Initial posterior before risk refinement", None, initial_rows)
        best = selection_score(initial_rows["Benchmark output"])
        save_checkpoint(
            CKPT_DIR / "initial_posterior.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            0,
            initial_rows,
            history,
            best,
            no_improve,
        )
        save_checkpoint(
            CKPT_DIR / "best_score.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            0,
            initial_rows,
            history,
            best,
            no_improve,
        )
        print(f"[Initial best] score={best:.6f}")

    if start_epoch > REFINE_EPOCHS:
        print(f"Refinement already completed: {start_epoch - 1}/{REFINE_EPOCHS}")
        return

    trainable = [p for p in model.risk_refiner.parameters() if p.requires_grad]
    print(
        f"Trainable risk-refiner parameters: "
        f"{sum(p.numel() for p in trainable):,}"
    )

    for refine_epoch in range(start_epoch, REFINE_EPOCHS + 1):
        model.train(True)
        stats_all: List[Dict[str, float]] = []
        pbar = tqdm(
            train_loader,
            desc=f"Risk refinement {refine_epoch}/{REFINE_EPOCHS}",
        )

        for step, loader_batch in enumerate(pbar, 1):
            loader_batch = to_device(loader_batch)
            optimizer.zero_grad(set_to_none=True)
            micro_stats: List[Dict[str, float]] = []

            for batch, micro_n, total_n in iter_microbatches(loader_batch, TRAIN_MICROBATCH):
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    total, stats = compute_loss(model, batch)
                    scaled = total * (float(micro_n) / float(total_n))
                scaler.scale(scaled).backward()
                micro_stats.append(stats)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()

            step_stats = avg_dicts(micro_stats)
            stats_all.append(step_stats)
            pbar.set_postfix(
                loss=f"{step_stats.get('loss_total', 0):.4f}",
                rm=f"{step_stats.get('rmse_mask', 0):.5f}",
                acc=f"{step_stats.get('acceptance_mean', 0):.2f}",
                upd=f"{step_stats.get('update_abs', 0):.5f}",
                vio=f"{step_stats.get('monotonic_violation', 0):.3f}",
                mb=TRAIN_MICROBATCH,
            )

            del loader_batch, micro_stats
            if DEVICE == "cuda" and EMPTY_CACHE_EVERY > 0 and step % EMPTY_CACHE_EVERY == 0:
                torch.cuda.empty_cache()

        scheduler.step()
        train_loss = float(np.mean([row["loss_total"] for row in stats_all]))
        rows = evaluate(model, val_loader, desc=f"Val refinement {refine_epoch}")
        print_summary(f"Risk refinement epoch {refine_epoch}", train_loss, rows)

        benchmark = rows["Benchmark output"]
        refined = rows["Risk-controlled refinement"]
        fused = rows["Posterior fusion"]
        score = selection_score(benchmark)
        history_row: Dict[str, Any] = {
            "refine_epoch": refine_epoch,
            "train_loss": train_loss,
            "score": score,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"benchmark_{k}": v for k, v in benchmark.items()},
            **{f"refined_{k}": v for k, v in refined.items()},
            **{f"posterior_{k}": v for k, v in fused.items()},
            **{f"aux_{k}": v for k, v in rows["_aux"].items()},
        }
        history.append(history_row)
        write_history(history)

        improved = score < best - 1.0e-8
        if improved:
            best = score
            no_improve = 0
            save_checkpoint(
                CKPT_DIR / "best_score.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                refine_epoch,
                rows,
                history,
                best,
                no_improve,
            )
            print(f"[Best] score={best:.6f}")
        else:
            no_improve += 1
            print(f"[No improvement] {no_improve}/{EARLY_STOP_PATIENCE}; best={best:.6f}")

        save_checkpoint(
            CKPT_DIR / "last.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            refine_epoch,
            rows,
            history,
            best,
            no_improve,
        )

        if no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping after {no_improve} consecutive non-improving "
                f"refinement epochs."
            )
            break

    print("Done. Best checkpoint:", CKPT_DIR / "best_score.pth")


if __name__ == "__main__":
    main()
