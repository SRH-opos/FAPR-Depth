r"""
FAPR-Depth v6: Safe-Anchor Residual Posterior Reconstruction
============================================================

This stage starts from the completed v5 checkpoint and fixes the main failure
observed on the held-out test split: the learned posterior fusion can be worse
than the pretrained base completion stream.  The new output is therefore
anchored to the base stream and can only move through a bounded residual:

    base anchor
        -> legacy failure-aware posterior candidate
        -> counterfactual safety estimation
        -> bounded safe residual acceptance
        -> candidate proposal
        -> risk-controlled final acceptance

The safe gate is initialized near zero, so the initial deployable prediction is
approximately the strong base completion result rather than the weaker legacy
posterior.  A gate value of zero exactly recovers the base anchor; this provides
an architectural fallback that the previous absolute multi-source fusion did
not have.

Training curriculum
-------------------
A. safe-anchor gate warm-up (2 epochs)
B. candidate proposal adaptation on the safe anchor (1 epoch)
C. counterfactual risk calibration (1 epoch)
D. low-learning-rate joint optimization (2 epochs)

The script also reports an oracle between the base anchor and the legacy
posterior.  This diagnostic is essential: if that oracle is not better than the
base anchor, no learned gate can improve the base using only those two choices.

Direct run
----------
    python train_fapr_depth_safe_anchor_v6_8gb.py
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

# Completed v5 checkpoint.  Existing posterior, proposal and risk parameters are
# loaded; only the safe-anchor module is newly initialized.
V5_CKPT = Path(
    os.getenv(
        "FAPR_V5_CKPT",
        str(PROJECT_ROOT / "weights" / "fapr_v5_best_score.pth"),
    )
)

OUT_DIR = Path(os.getenv("FAPR_OUT_DIR", str(PROJECT_ROOT / "outputs" / "fapr_depth_v6_safe_anchor")))
CKPT_DIR = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Resume a partially completed v6 safe-anchor run when last.pth exists.
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

# Safe-anchor curriculum schedule.
SAFE_WARMUP_EPOCHS = 2
PROPOSAL_ADAPT_EPOCHS = 1
RISK_CALIBRATION_EPOCHS = 1
JOINT_EPOCHS = 2
REFINE_EPOCHS = (
    SAFE_WARMUP_EPOCHS
    + PROPOSAL_ADAPT_EPOCHS
    + RISK_CALIBRATION_EPOCHS
    + JOINT_EPOCHS
)

LR_SAFE_WARMUP = 2.0e-5
LR_SAFE_RISK = 2.0e-5
LR_PROPOSAL_ADAPT = 1.0e-5
LR_REFINE_RISK = 1.0e-5
LR_JOINT = 3.0e-6
WEIGHT_DECAY = 1.0e-4
CLIP_GRAD = 2.0
JOINT_EARLY_STOP_PATIENCE = 2

MAX_DEPTH = 10.0
MIN_DEPTH = 0.03
DEPTH_NORM_SCALE = 5.0
EPS = 1.0e-6
BOUNDARY_KERNEL = 7
RELIABLE_RAW_THR = 0.010
HARD_RATIO = 0.20

# Geometry/failure settings inherited from the frozen v5 system.
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

# Safe residual posterior.
SAFE_MAX_RESIDUAL = 0.080
SAFE_GATE_INIT_LOGIT = -8.0
SAFE_SUPPORT_FLOOR = 0.25
SAFE_BOUNDARY_WEIGHT = 0.25
SAFE_RISK_INIT_METERS = 0.010
SAFE_RISK_TEMPERATURE = 0.003
SAFE_TARGET_TEMPERATURE = 0.002
SAFE_RANK_TEMPERATURE = 0.003
SAFE_RANK_IGNORE = 1.0e-4
SAFE_GUARD_MARGIN = 5.0e-5
SAFE_EASY_ANCHOR_THR = 0.008

# Source-candidate corruption prevents the gate from learning an unconditional
# "always use posterior" rule on the easier validation distribution.
SAFE_CORRUPTION_PROB_WARMUP = 0.50
SAFE_CORRUPTION_PROB_JOINT = 0.20
SAFE_CORRUPTION_MAX_BIAS = 0.030
SAFE_CORRUPTION_MAX_NOISE = 0.015
SAFE_CORRUPTION_MIN_SCALE = 0.50
SAFE_CORRUPTION_MAX_SCALE = 1.50

# Candidate proposal and counterfactual-risk acceptance.
MAX_RISK_DELTA = 0.050
SUPPORT_DILATE_KERNEL = 9
BOUNDARY_SUPPORT_WEIGHT = 0.50
RISK_INIT_METERS = 0.010
RISK_TEMPERATURE = 0.003
RISK_ACCEPT_MARGIN = 0.0
ACCEPT_TARGET_MARGIN = 0.0
ACCEPT_TARGET_TEMPERATURE = 0.002
RISK_RANK_TEMPERATURE = 0.003
RISK_RANK_IGNORE = 1.0e-4
MONOTONIC_TOLERANCE = 1.0e-4

# Safe-anchor objective.
W_SAFE_MASK = 1.60
W_SAFE_RMSE = 0.85
W_SAFE_BOUNDARY = 0.90
W_SAFE_GRAD = 0.25
W_SAFE_HARD = 0.30
W_SAFE_GUARD = 3.00
W_SAFE_PRESERVE_EASY = 1.40
W_SAFE_GATE_BCE = 0.65
W_SAFE_UPDATE_REG = 0.03
W_SAFE_RISK_CALIBRATION = 0.80
W_SAFE_RISK_GAIN = 0.70
W_SAFE_RISK_RANK = 0.25

# Candidate adaptation objective.
W_CANDIDATE_MASK = 1.60
W_CANDIDATE_RMSE = 0.80
W_CANDIDATE_BOUNDARY = 0.90
W_CANDIDATE_GRAD = 0.25
W_CANDIDATE_HARD = 0.25
W_DELTA_TARGET = 0.70
W_CANDIDATE_MONOTONIC = 0.80
W_CANDIDATE_ANCHOR_GUARD = 1.80
W_CANDIDATE_TRUST = 0.45
W_CANDIDATE_UPDATE_REG = 0.02

# Refinement-risk objective.
W_RISK_CALIBRATION = 0.80
W_RISK_GAIN = 0.70
W_RISK_RANK = 0.25
W_ACCEPTANCE = 0.55

# Joint accepted-output objective.
W_FINAL_MASK = 1.60
W_FINAL_ALL = 0.15
W_FINAL_RMSE = 0.80
W_FINAL_BOUNDARY = 0.90
W_FINAL_GRAD = 0.30
W_FINAL_HARD = 0.25
W_FINAL_MONOTONIC = 1.20
W_FINAL_ANCHOR_GUARD = 2.50
W_FINAL_TRUST = 0.50
W_FINAL_UPDATE_REG = 0.03
W_JOINT_SAFE = 0.40
W_JOINT_CANDIDATE = 0.35
W_JOINT_RISK = 0.55

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


class SafeAnchorPosteriorHead(nn.Module):
    """Direct safety gate plus counterfactual anchor/candidate risk estimation."""

    def __init__(self, cin: int = 20, hidden: int = 48):
        super().__init__()
        self.gate_body = nn.Sequential(
            ConvBlock(cin, hidden),
            ConvBlock(hidden, hidden, dilation=2),
            ConvBlock(hidden, hidden),
        )
        self.gate_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, SAFE_GATE_INIT_LOGIT)

        self.risk_body = nn.Sequential(
            ConvBlock(cin, hidden),
            ConvBlock(hidden, hidden, dilation=2),
            ConvBlock(hidden, hidden),
        )
        self.risk_head = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.risk_head.weight)
        init_bias = math.log(math.expm1(max(SAFE_RISK_INIT_METERS, 1.0e-4)))
        nn.init.constant_(self.risk_head.bias, init_bias)

    def gate_parameters(self):
        yield from self.gate_body.parameters()
        yield from self.gate_head.parameters()

    def risk_parameters(self):
        yield from self.risk_body.parameters()
        yield from self.risk_head.parameters()

    def direct_gate_logit(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate_head(self.gate_body(x))

    def estimate_risk(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.risk_head(self.risk_body(x))
        return F.softplus(z[:, 0:1]), F.softplus(z[:, 1:2])


class CurriculumRiskRefiner(nn.Module):
    """Candidate proposal plus a separate counterfactual risk estimator.

    The proposal and risk subnetworks are deliberately separated.  This allows
    the proposal to learn useful corrections before the risk estimator is
    trained, preventing the all-reject collapse caused by zero-improvement
    labels at initialization.
    """

    def __init__(self, proposal_cin: int = 17, risk_cin: int = 20, hidden: int = 48):
        super().__init__()
        self.proposal_body = nn.Sequential(
            ConvBlock(proposal_cin, hidden),
            ConvBlock(hidden, hidden, dilation=2),
            ConvBlock(hidden, hidden),
        )
        self.delta_head = zero_init_conv(nn.Conv2d(hidden, 1, 1))

        self.risk_body = nn.Sequential(
            ConvBlock(risk_cin, hidden),
            ConvBlock(hidden, hidden, dilation=2),
            ConvBlock(hidden, hidden),
        )
        self.risk_head = nn.Conv2d(hidden, 2, 1)
        nn.init.zeros_(self.risk_head.weight)
        init_bias = math.log(math.expm1(max(RISK_INIT_METERS, 1.0e-4)))
        nn.init.constant_(self.risk_head.bias, init_bias)

    def proposal_parameters(self):
        yield from self.proposal_body.parameters()
        yield from self.delta_head.parameters()

    def risk_parameters(self):
        yield from self.risk_body.parameters()
        yield from self.risk_head.parameters()

    def propose(self, x: torch.Tensor) -> torch.Tensor:
        z = self.delta_head(self.proposal_body(x))
        return MAX_RISK_DELTA * torch.tanh(z)

    def estimate_risk(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.risk_head(self.risk_body(x))
        risk_before = F.softplus(z[:, 0:1])
        risk_after = F.softplus(z[:, 1:2])
        return risk_before, risk_after


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
        self.safe_anchor = SafeAnchorPosteriorHead(cin=20, hidden=48)
        self.risk_refiner = CurriculumRiskRefiner(proposal_cin=17, risk_cin=20, hidden=48)

    def load_v5_checkpoint(self, ckpt_path: Path) -> Dict[str, Any]:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"v5 checkpoint not found: {ckpt_path}")
        payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = payload.get("model", payload.get("model_state_dict", payload))
        clean = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        missing, unexpected = self.load_state_dict(clean, strict=False)
        expected_missing = [k for k in missing if k.startswith("safe_anchor.")]
        other_missing = [k for k in missing if not k.startswith("safe_anchor.")]
        if other_missing or unexpected:
            print(
                f"[v5 checkpoint] missing_other={len(other_missing)}, "
                f"missing_safe_anchor={len(expected_missing)}, unexpected={len(unexpected)}"
            )
            if other_missing:
                print("  first missing:", other_missing[:10])
            if unexpected:
                print("  first unexpected:", unexpected[:10])
        else:
            print(
                f"[v5 checkpoint] loaded successfully; "
                f"new safe-anchor tensors={len(expected_missing)}"
            )
        return payload

    def set_training_phase(self, phase: str) -> None:
        for param in self.parameters():
            param.requires_grad_(False)
        if phase in {"safe", "joint"}:
            for param in self.safe_anchor.gate_parameters():
                param.requires_grad_(True)
        if phase in {"risk", "joint"}:
            for param in self.safe_anchor.risk_parameters():
                param.requires_grad_(True)
            for param in self.risk_refiner.risk_parameters():
                param.requires_grad_(True)
        if phase in {"proposal", "joint"}:
            for param in self.risk_refiner.proposal_parameters():
                param.requires_grad_(True)

    def train(self, mode: bool = True):
        # Frozen legacy reconstruction stays in eval mode.  Only the phase-specific
        # safe-anchor/proposal/risk subnetworks are trainable.
        super().train(False)
        self.safe_anchor.train(mode)
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

    def _corrupt_legacy_candidate(
        self,
        anchor: torch.Tensor,
        legacy: torch.Tensor,
        mask: torch.Tensor,
        boundary: torch.Tensor,
        probability: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if probability <= 0.0:
            return legacy, torch.zeros_like(mask)
        b, _, h, w = legacy.shape
        apply = (torch.rand(b, 1, 1, 1, device=legacy.device) < probability).float()
        scale = torch.empty(b, 1, 1, 1, device=legacy.device).uniform_(
            SAFE_CORRUPTION_MIN_SCALE, SAFE_CORRUPTION_MAX_SCALE
        )
        low_h, low_w = max(2, h // 24), max(2, w // 24)
        smooth = torch.randn(b, 1, low_h, low_w, device=legacy.device)
        smooth = F.interpolate(smooth, size=(h, w), mode="bilinear", align_corners=True)
        smooth = smooth / smooth.flatten(1).std(1, keepdim=True).view(b, 1, 1, 1).clamp_min(0.25)
        bias_amp = torch.empty(b, 1, 1, 1, device=legacy.device).uniform_(
            -SAFE_CORRUPTION_MAX_BIAS, SAFE_CORRUPTION_MAX_BIAS
        )
        noise_amp = torch.empty(b, 1, 1, 1, device=legacy.device).uniform_(
            0.0, SAFE_CORRUPTION_MAX_NOISE
        )
        residual = legacy - anchor
        corrupted_residual = scale * residual + mask * (bias_amp + noise_amp * smooth)
        corrupted_residual = corrupted_residual + boundary * noise_amp * 0.5 * torch.randn_like(boundary)
        corrupted_residual = corrupted_residual.clamp(-SAFE_MAX_RESIDUAL, SAFE_MAX_RESIDUAL)
        corrupted = safe_depth(anchor + corrupted_residual)
        return apply * corrupted + (1.0 - apply) * legacy, apply.expand_as(mask)

    def forward(
        self,
        inp: Dict[str, torch.Tensor],
        phase: str = "joint",
        augment_safe: bool = False,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            posterior = self.forward_posterior(inp)
            anchor_depth = self.forward_reference(inp["rgb"], inp["raw"])

        rgb, raw, mask = inp["rgb"], inp["raw"], inp["mask"]
        boundary = inp["boundary"]
        legacy_fused = posterior["fused"]
        p_fail = posterior["p_fail"].detach()

        corruption_prob = 0.0
        if augment_safe and phase == "safe":
            corruption_prob = SAFE_CORRUPTION_PROB_WARMUP
        elif augment_safe and phase == "joint":
            corruption_prob = SAFE_CORRUPTION_PROB_JOINT
        legacy_candidate, corruption_mask = self._corrupt_legacy_candidate(
            anchor_depth, legacy_fused, mask, boundary, corruption_prob
        )

        legacy_residual = (legacy_candidate - anchor_depth).clamp(
            -SAFE_MAX_RESIDUAL, SAFE_MAX_RESIDUAL
        )
        safe_support = mask * torch.clamp(
            SAFE_SUPPORT_FLOOR
            + (1.0 - SAFE_SUPPORT_FLOOR) * p_fail
            + SAFE_BOUNDARY_WEIGHT * boundary,
            0.0,
            1.0,
        )
        safe_support = safe_support.detach()

        safe_x = torch.cat(
            [
                rgb,
                norm_depth(anchor_depth),
                norm_depth(legacy_candidate),
                norm_depth(raw),
                norm_depth(posterior["rel_metric"]),
                (legacy_residual / max(SAFE_MAX_RESIDUAL, EPS)).clamp(-1.0, 1.0),
                (legacy_residual.abs() / max(SAFE_MAX_RESIDUAL, EPS)).clamp(0.0, 1.0),
                mask,
                boundary,
                posterior["sdm"],
                p_fail,
                posterior["final_logb"],
                posterior["route_entropy"],
                posterior["alpha"],
                inp["raw_prior"],
                inp["rel_conf"],
            ],
            1,
        )
        direct_gate_logit = self.safe_anchor.direct_gate_logit(safe_x)
        safe_risk_anchor, safe_risk_legacy = self.safe_anchor.estimate_risk(safe_x.detach())
        safe_predicted_gain = safe_risk_anchor - safe_risk_legacy
        use_safe_risk = phase in {"risk", "joint"}
        safe_gate_logit = direct_gate_logit
        if use_safe_risk:
            safe_gate_logit = safe_gate_logit + safe_predicted_gain / max(
                SAFE_RISK_TEMPERATURE, EPS
            )
        safe_gate_logit = safe_gate_logit.clamp(-12.0, 12.0)
        safe_gate = torch.sigmoid(safe_gate_logit)
        safe_update = safe_support * safe_gate * legacy_residual
        safe_posterior = safe_depth(anchor_depth + safe_update)

        refine_support = mask * torch.clamp(
            SAFE_SUPPORT_FLOOR
            + (1.0 - SAFE_SUPPORT_FLOOR) * p_fail
            + BOUNDARY_SUPPORT_WEIGHT * boundary,
            0.0,
            1.0,
        )
        refine_support = refine_support.detach()

        proposal_input = torch.cat(
            [
                rgb,
                norm_depth(safe_posterior),
                norm_depth(raw),
                norm_depth(posterior["rel_metric"]),
                norm_depth(anchor_depth),
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
            candidate = safe_depth(safe_posterior + candidate_update)
            risk_input = torch.cat(
                [
                    proposal_input.detach(),
                    norm_depth(candidate.detach()),
                    (delta.detach() / max(MAX_RISK_DELTA, EPS)).clamp(-1.0, 1.0),
                    torch.clamp(
                        gradient_mag(candidate_update.detach()) / max(MAX_RISK_DELTA, EPS),
                        0.0,
                        4.0,
                    ),
                ],
                1,
            )
            risk_before, risk_after = self.risk_refiner.estimate_risk(risk_input)
            predicted_gain = risk_before - risk_after
            acceptance_logit = (
                predicted_gain - RISK_ACCEPT_MARGIN
            ) / max(RISK_TEMPERATURE, EPS)
            acceptance = torch.sigmoid(acceptance_logit)
            if phase == "proposal":
                acceptance = torch.ones_like(acceptance)
                acceptance_logit = torch.full_like(acceptance_logit, 12.0)
            accepted_update = refine_support * acceptance * delta
            final = safe_depth(safe_posterior + accepted_update)

        safe_benchmark = safe_posterior * mask + raw * (1.0 - mask)
        candidate_benchmark = candidate * mask + raw * (1.0 - mask)
        benchmark_output = final * mask + raw * (1.0 - mask)

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
            "effective_candidate_update": candidate - safe_posterior,
            "accepted_update": accepted_update,
            "risk_before": risk_before,
            "risk_after": risk_after,
            "predicted_gain": predicted_gain,
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
# SAFE-ANCHOR CURRICULUM LOSS
# =============================================================================
def phase_for_refine_epoch(epoch: int) -> str:
    if epoch <= SAFE_WARMUP_EPOCHS:
        return "safe"
    if epoch <= SAFE_WARMUP_EPOCHS + PROPOSAL_ADAPT_EPOCHS:
        return "proposal"
    if epoch <= SAFE_WARMUP_EPOCHS + PROPOSAL_ADAPT_EPOCHS + RISK_CALIBRATION_EPOCHS:
        return "risk"
    return "joint"


def phase_label(phase: str) -> str:
    return {
        "safe": "A: safe-anchor residual gate warm-up",
        "proposal": "B: candidate proposal adaptation",
        "risk": "C: counterfactual risk calibration",
        "joint": "D: joint safe reconstruction",
    }[phase]


def configure_phase(
    model: FailureAwarePosteriorDepth,
    optimizer: torch.optim.Optimizer,
    phase: str,
) -> None:
    model.set_training_phase(phase)
    lrs = {
        "safe": (LR_SAFE_WARMUP, 0.0, 0.0, 0.0),
        "proposal": (0.0, 0.0, LR_PROPOSAL_ADAPT, 0.0),
        "risk": (0.0, LR_SAFE_RISK, 0.0, LR_REFINE_RISK),
        "joint": (LR_JOINT, LR_JOINT, LR_JOINT, LR_JOINT),
    }[phase]
    for group, lr in zip(optimizer.param_groups, lrs):
        group["lr"] = lr


def compute_loss(
    model: FailureAwarePosteriorDepth,
    batch: Dict[str, Any],
    phase: str,
    return_outputs: bool = False,
):
    inp = build_inputs(batch)
    out = model(inp, phase=phase, augment_safe=not return_outputs)

    raw, gt = inp["raw"], safe_depth(inp["gt"])
    mask, valid = inp["mask"], inp["valid"]
    boundary = inp["boundary"] * valid
    region_mask = valid * mask
    safe_region = valid * out["safe_support"].detach()
    refine_region = valid * out["support"].detach()
    if safe_region.sum().item() <= 0:
        safe_region = region_mask
    if refine_region.sum().item() <= 0:
        refine_region = region_mask

    labels, _ = failure_targets(raw, gt, valid, boundary)
    anchor = out["anchor_depth"]
    legacy = out["legacy_candidate"]
    safe_pred = out["safe_posterior"]
    candidate = out["candidate"]
    final = out["final"]

    anchor_err = torch.abs(anchor - gt)
    legacy_err = torch.abs(legacy - gt)
    safe_err = torch.abs(safe_pred - gt)
    candidate_err = torch.abs(candidate - gt)
    final_err = torch.abs(final - gt)

    raw_reliable = (raw > EPS).float() * (torch.abs(raw - gt) <= RELIABLE_RAW_THR).float()
    reliable_region = valid * raw_reliable * (1.0 - mask)
    easy_anchor = region_mask * (anchor_err <= SAFE_EASY_ANCHOR_THR).float()

    # A. Safe-anchor residual gate.
    true_safe_gain = (anchor_err - legacy_err).detach()
    safe_gate_target = torch.sigmoid(
        true_safe_gain / max(SAFE_TARGET_TEMPERATURE, EPS)
    )
    safe_gate_bce = F.binary_cross_entropy_with_logits(
        out["safe_gate_logit"].float(),
        safe_gate_target.float(),
        reduction="none",
    ).to(final.dtype)
    loss_safe_gate = masked_mean(safe_gate_bce, safe_region)

    loss_safe_mask = charbonnier(safe_pred, gt, region_mask)
    loss_safe_rmse = masked_rmse(safe_pred, gt, region_mask)
    loss_safe_boundary = (
        masked_mean(safe_err, boundary)
        if boundary.sum().item() > 0 else final.new_tensor(0.0)
    )
    loss_safe_grad = gradient_l1(safe_pred, gt, region_mask)
    loss_safe_hard = hard_pixel_rmse(safe_pred, gt, region_mask)
    loss_safe_guard = masked_mean(
        F.relu(safe_err - anchor_err - SAFE_GUARD_MARGIN), region_mask
    )
    loss_safe_preserve_easy = (
        masked_mean(torch.abs(safe_pred - anchor), easy_anchor)
        if easy_anchor.sum().item() > 0 else final.new_tensor(0.0)
    )
    loss_safe_update_reg = masked_mean(torch.abs(safe_pred - anchor), safe_region)

    safe_objective = (
        W_SAFE_MASK * loss_safe_mask
        + W_SAFE_RMSE * loss_safe_rmse
        + W_SAFE_BOUNDARY * loss_safe_boundary
        + W_SAFE_GRAD * loss_safe_grad
        + W_SAFE_HARD * loss_safe_hard
        + W_SAFE_GUARD * loss_safe_guard
        + W_SAFE_PRESERVE_EASY * loss_safe_preserve_easy
        + W_SAFE_GATE_BCE * loss_safe_gate
        + W_SAFE_UPDATE_REG * loss_safe_update_reg
    )

    # Counterfactual anchor/legacy risk.
    safe_risk_anchor_map = F.smooth_l1_loss(
        out["safe_risk_anchor"], anchor_err.detach(), reduction="none", beta=0.003
    )
    safe_risk_legacy_map = F.smooth_l1_loss(
        out["safe_risk_legacy"], legacy_err.detach(), reduction="none", beta=0.003
    )
    loss_safe_risk_cal = 0.5 * (
        masked_mean(safe_risk_anchor_map, safe_region)
        + masked_mean(safe_risk_legacy_map, safe_region)
    )
    safe_gain_map = F.smooth_l1_loss(
        out["safe_predicted_gain"], true_safe_gain, reduction="none", beta=0.001
    )
    loss_safe_risk_gain = masked_mean(safe_gain_map, safe_region)
    safe_rank_region = safe_region * (true_safe_gain.abs() > SAFE_RANK_IGNORE).float()
    safe_rank_map = F.softplus(
        -torch.sign(true_safe_gain)
        * out["safe_predicted_gain"]
        / max(SAFE_RANK_TEMPERATURE, EPS)
    )
    loss_safe_risk_rank = (
        masked_mean(safe_rank_map, safe_rank_region)
        if safe_rank_region.sum().item() > 0 else final.new_tensor(0.0)
    )
    safe_risk_objective = (
        W_SAFE_RISK_CALIBRATION * loss_safe_risk_cal
        + W_SAFE_RISK_GAIN * loss_safe_risk_gain
        + W_SAFE_RISK_RANK * loss_safe_risk_rank
        + W_SAFE_GATE_BCE * loss_safe_gate
    )

    # B. Candidate proposal on the safe posterior.
    target_delta = torch.clamp(
        gt - safe_pred.detach(), -MAX_RISK_DELTA, MAX_RISK_DELTA
    )
    delta_map = F.smooth_l1_loss(
        out["delta"], target_delta, reduction="none", beta=0.003
    )
    loss_delta_target = masked_mean(delta_map, refine_region)
    loss_candidate_mask = charbonnier(candidate, gt, region_mask)
    loss_candidate_rmse = masked_rmse(candidate, gt, region_mask)
    loss_candidate_boundary = (
        masked_mean(candidate_err, boundary)
        if boundary.sum().item() > 0 else final.new_tensor(0.0)
    )
    loss_candidate_grad = gradient_l1(candidate, gt, region_mask)
    loss_candidate_hard = hard_pixel_rmse(candidate, gt, region_mask)
    loss_candidate_monotonic = masked_mean(
        F.relu(candidate_err - safe_err - MONOTONIC_TOLERANCE), refine_region
    )
    loss_candidate_anchor_guard = masked_mean(
        F.relu(candidate_err - anchor_err - SAFE_GUARD_MARGIN), region_mask
    )
    loss_candidate_trust = (
        masked_mean(torch.abs(candidate - raw), reliable_region)
        if reliable_region.sum().item() > 0 else final.new_tensor(0.0)
    )
    loss_candidate_update_reg = masked_mean(
        torch.abs(candidate - safe_pred), refine_region
    )
    candidate_objective = (
        W_CANDIDATE_MASK * loss_candidate_mask
        + W_CANDIDATE_RMSE * loss_candidate_rmse
        + W_CANDIDATE_BOUNDARY * loss_candidate_boundary
        + W_CANDIDATE_GRAD * loss_candidate_grad
        + W_CANDIDATE_HARD * loss_candidate_hard
        + W_DELTA_TARGET * loss_delta_target
        + W_CANDIDATE_MONOTONIC * loss_candidate_monotonic
        + W_CANDIDATE_ANCHOR_GUARD * loss_candidate_anchor_guard
        + W_CANDIDATE_TRUST * loss_candidate_trust
        + W_CANDIDATE_UPDATE_REG * loss_candidate_update_reg
    )

    # C. Candidate-before/after refinement risk.
    true_refine_gain = (safe_err.detach() - candidate_err.detach())
    risk_before_map = F.smooth_l1_loss(
        out["risk_before"], safe_err.detach(), reduction="none", beta=0.003
    )
    risk_after_map = F.smooth_l1_loss(
        out["risk_after"], candidate_err.detach(), reduction="none", beta=0.003
    )
    loss_risk_calibration = 0.5 * (
        masked_mean(risk_before_map, refine_region)
        + masked_mean(risk_after_map, refine_region)
    )
    gain_map = F.smooth_l1_loss(
        out["predicted_gain"], true_refine_gain, reduction="none", beta=0.001
    )
    loss_risk_gain = masked_mean(gain_map, refine_region)
    rank_region = refine_region * (true_refine_gain.abs() > RISK_RANK_IGNORE).float()
    rank_map = F.softplus(
        -torch.sign(true_refine_gain)
        * out["predicted_gain"]
        / max(RISK_RANK_TEMPERATURE, EPS)
    )
    loss_risk_rank = (
        masked_mean(rank_map, rank_region)
        if rank_region.sum().item() > 0 else final.new_tensor(0.0)
    )
    accept_target = torch.sigmoid(
        (true_refine_gain - ACCEPT_TARGET_MARGIN)
        / max(ACCEPT_TARGET_TEMPERATURE, EPS)
    )
    acceptance_bce = F.binary_cross_entropy_with_logits(
        out["acceptance_logit"].float(), accept_target.float(), reduction="none"
    ).to(final.dtype)
    loss_acceptance = masked_mean(acceptance_bce, refine_region)
    refine_risk_objective = (
        W_RISK_CALIBRATION * loss_risk_calibration
        + W_RISK_GAIN * loss_risk_gain
        + W_RISK_RANK * loss_risk_rank
        + W_ACCEPTANCE * loss_acceptance
    )

    # D. Deployable final output.
    loss_final_mask = charbonnier(final, gt, region_mask)
    loss_final_all = charbonnier(final, gt, valid)
    loss_final_rmse = masked_rmse(final, gt, region_mask)
    loss_final_boundary = (
        masked_mean(final_err, boundary)
        if boundary.sum().item() > 0 else final.new_tensor(0.0)
    )
    loss_final_grad = gradient_l1(final, gt, region_mask)
    loss_final_hard = hard_pixel_rmse(final, gt, region_mask)
    loss_final_monotonic = masked_mean(
        F.relu(final_err - safe_err - MONOTONIC_TOLERANCE), refine_region
    )
    loss_final_anchor_guard = masked_mean(
        F.relu(final_err - anchor_err - SAFE_GUARD_MARGIN), region_mask
    )
    loss_final_trust = (
        masked_mean(torch.abs(final - raw), reliable_region)
        if reliable_region.sum().item() > 0 else final.new_tensor(0.0)
    )
    loss_final_update_reg = masked_mean(torch.abs(final - safe_pred), refine_region)
    final_objective = (
        W_FINAL_MASK * loss_final_mask
        + W_FINAL_ALL * loss_final_all
        + W_FINAL_RMSE * loss_final_rmse
        + W_FINAL_BOUNDARY * loss_final_boundary
        + W_FINAL_GRAD * loss_final_grad
        + W_FINAL_HARD * loss_final_hard
        + W_FINAL_MONOTONIC * loss_final_monotonic
        + W_FINAL_ANCHOR_GUARD * loss_final_anchor_guard
        + W_FINAL_TRUST * loss_final_trust
        + W_FINAL_UPDATE_REG * loss_final_update_reg
    )

    if phase == "safe":
        total = safe_objective
    elif phase == "proposal":
        total = candidate_objective
    elif phase == "risk":
        total = safe_risk_objective + refine_risk_objective
    elif phase == "joint":
        total = (
            final_objective
            + W_JOINT_SAFE * safe_objective
            + W_JOINT_CANDIDATE * candidate_objective
            + W_JOINT_RISK * (safe_risk_objective + refine_risk_objective)
        )
    else:
        raise ValueError(f"Unknown phase: {phase}")

    with torch.no_grad():
        pred_label = out["fail_logits"].argmax(1, keepdim=True)
        acc = masked_mean((pred_label == labels).float(), valid)
        recalls = []
        cls_stats: Dict[str, float] = {}
        for cls, name in {0: "valid", 1: "missing", 2: "biased", 3: "boundary"}.items():
            precision, recall, f1, support_count = failure_prf(
                pred_label, labels, valid, cls
            )
            cls_stats[f"precision_{name}"] = float(precision)
            cls_stats[f"recall_{name}"] = float(recall)
            cls_stats[f"f1_{name}"] = float(f1)
            cls_stats[f"support_{name}"] = float(support_count)
            recalls.append(recall)

        legacy_better = (legacy_err < anchor_err).float()
        safe_better = (safe_err < anchor_err).float()
        candidate_better = (candidate_err < safe_err).float()
        final_better = (final_err < safe_err).float()
        accepted = (out["acceptance"] > 0.5).float()
        accepted_region = refine_region * accepted
        accepted_improvement = (
            masked_mean(final_better, accepted_region)
            if accepted_region.sum().item() > 0 else final.new_tensor(0.0)
        )

        oracle_anchor_legacy = torch.where(legacy_err < anchor_err, legacy, anchor)
        oracle_safe_candidate = torch.where(candidate_err < safe_err, candidate, safe_pred)
        oracle_anchor_err = torch.abs(oracle_anchor_legacy - gt)
        oracle_refine_err = torch.abs(oracle_safe_candidate - gt)

        biased_region = valid * (labels == 2).float()
        boundary_region = valid * (labels == 3).float()

        def region_source_weight(channel: int, region: torch.Tensor) -> float:
            if region.sum().item() <= 0:
                return 0.0
            return float(masked_mean(out["alpha"][:, channel:channel + 1], region))

        stats = {
            "loss_total": float(total),
            "loss_safe": float(safe_objective),
            "loss_candidate": float(candidate_objective),
            "loss_safe_risk": float(safe_risk_objective),
            "loss_refine_risk": float(refine_risk_objective),
            "loss_final": float(final_objective),
            "anchor_mae_mask": float(masked_mean(anchor_err, region_mask)),
            "legacy_mae_mask": float(masked_mean(legacy_err, region_mask)),
            "safe_mae_mask": float(masked_mean(safe_err, region_mask)),
            "candidate_mae_mask": float(masked_mean(candidate_err, region_mask)),
            "final_mae_mask": float(masked_mean(final_err, region_mask)),
            "loss_safe_guard": float(loss_safe_guard),
            "loss_safe_gate": float(loss_safe_gate),
            "loss_candidate_anchor_guard": float(loss_candidate_anchor_guard),
            "loss_final_anchor_guard": float(loss_final_anchor_guard),
            "safe_support_mean": float(out["safe_support"].mean()),
            "safe_gate_mean": float(masked_mean(out["safe_gate"], safe_region)),
            "safe_gate_target_mean": float(masked_mean(safe_gate_target, safe_region)),
            "safe_gate_mae": float(masked_mean(torch.abs(out["safe_gate"] - safe_gate_target), safe_region)),
            "safe_update_abs": float(masked_mean(torch.abs(safe_pred - anchor), safe_region)),
            "legacy_improve_ratio": float(masked_mean(legacy_better, safe_region)),
            "safe_improve_ratio": float(masked_mean(safe_better, safe_region)),
            "safe_damage_ratio": float(masked_mean((safe_err > anchor_err + SAFE_GUARD_MARGIN).float(), safe_region)),
            "safe_oracle_gain": float(masked_mean(anchor_err - oracle_anchor_err, safe_region)),
            "safe_risk_anchor_mae": float(masked_mean(torch.abs(out["safe_risk_anchor"] - anchor_err), safe_region)),
            "safe_risk_legacy_mae": float(masked_mean(torch.abs(out["safe_risk_legacy"] - legacy_err), safe_region)),
            "safe_risk_gain_mae": float(masked_mean(torch.abs(out["safe_predicted_gain"] - true_safe_gain), safe_region)),
            "corruption_fraction": float(out["corruption_mask"].mean()),
            "candidate_update_abs": float(masked_mean(torch.abs(candidate - safe_pred), refine_region)),
            "accepted_update_abs": float(masked_mean(torch.abs(final - safe_pred), refine_region)),
            "candidate_improve_ratio": float(masked_mean(candidate_better, refine_region)),
            "final_improve_ratio": float(masked_mean(final_better, refine_region)),
            "acceptance_mean": float(masked_mean(out["acceptance"], refine_region)),
            "accept_target_mean": float(masked_mean(accept_target, refine_region)),
            "acceptance_mae": float(masked_mean(torch.abs(out["acceptance"] - accept_target), refine_region)),
            "accepted_improvement": float(accepted_improvement),
            "refine_oracle_gain": float(masked_mean(safe_err - oracle_refine_err, refine_region)),
            "risk_before_mae": float(masked_mean(torch.abs(out["risk_before"] - safe_err), refine_region)),
            "risk_after_mae": float(masked_mean(torch.abs(out["risk_after"] - candidate_err), refine_region)),
            "risk_gain_mae": float(masked_mean(torch.abs(out["predicted_gain"] - true_refine_gain), refine_region)),
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
    phase: str,
    desc: str = "Val",
):
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

    for loader_batch in tqdm(loader, desc=desc, leave=False):
        loader_batch = to_device(loader_batch)
        for batch, _, _ in iter_microbatches(loader_batch, VAL_MICROBATCH):
            _, stats, inp, out = compute_loss(
                model, batch, phase=phase, return_outputs=True
            )
            raw, gt, mask, valid = inp["raw"], inp["gt"], inp["mask"], inp["valid"]
            previous = inp["old_base"] * mask + raw * (1.0 - mask)
            anchor_benchmark = out["anchor_depth"] * mask + raw * (1.0 - mask)
            legacy_benchmark = out["legacy_fused"] * mask + raw * (1.0 - mask)

            anchor_err = torch.abs(out["anchor_depth"] - gt)
            legacy_err = torch.abs(out["legacy_fused"] - gt)
            oracle_anchor = torch.where(
                legacy_err < anchor_err, out["legacy_fused"], out["anchor_depth"]
            )
            oracle_anchor_benchmark = oracle_anchor * mask + raw * (1.0 - mask)

            safe_err = torch.abs(out["safe_posterior"] - gt)
            candidate_err = torch.abs(out["candidate"] - gt)
            oracle_refine = torch.where(
                candidate_err < safe_err, out["candidate"], out["safe_posterior"]
            )
            oracle_refine_benchmark = oracle_refine * mask + raw * (1.0 - mask)

            rows["Raw Depth"].append(metric_values(raw, raw, gt, mask, valid))
            rows["Input relative prior"].append(metric_values(inp["rel"], raw, gt, mask, valid))
            rows["Metric-calibrated prior"].append(metric_values(out["rel_metric"], raw, gt, mask, valid))
            rows["Previous model result"].append(metric_values(previous, raw, gt, mask, valid))
            rows["Base anchor"].append(metric_values(anchor_benchmark, raw, gt, mask, valid))
            rows["Legacy posterior fusion"].append(metric_values(legacy_benchmark, raw, gt, mask, valid))
            rows["Safe residual posterior"].append(metric_values(out["safe_posterior"], raw, gt, mask, valid))
            rows["Candidate correction"].append(metric_values(out["candidate"], raw, gt, mask, valid))
            rows["Risk-accepted refinement"].append(metric_values(out["final"], raw, gt, mask, valid))
            rows["Oracle anchor-posterior"].append(metric_values(oracle_anchor_benchmark, raw, gt, mask, valid))
            rows["Oracle safe-candidate"].append(metric_values(oracle_refine_benchmark, raw, gt, mask, valid))
            rows["Safe benchmark"].append(metric_values(out["safe_benchmark"], raw, gt, mask, valid))
            rows["Candidate benchmark"].append(metric_values(out["candidate_benchmark"], raw, gt, mask, valid))
            rows["Benchmark output"].append(metric_values(out["benchmark_output"], raw, gt, mask, valid))
            aux.append(stats)

    avg = {name: avg_dicts(values) for name, values in rows.items()}
    avg["_aux"] = avg_dicts(aux)
    return avg


def print_summary(
    label: str,
    train_loss: Optional[float],
    phase: str,
    rows: Dict[str, Dict[str, float]],
) -> None:
    print("\n" + "=" * 184)
    phase_text = phase_label(phase)
    if train_loss is None:
        print(f"{label} | {phase_text}")
    else:
        print(f"{label} | {phase_text} | train loss {train_loss:.6f}")

    print(
        f"{'Variant':<30} | {'MAE_all':>9} | {'RMSE_all':>9} | "
        f"{'MAE_mask':>9} | {'RMSE_mask':>9} | {'Boundary':>9} | "
        f"{'BG':>9} | {'Score':>9}"
    )
    ordered = [k for k in rows.keys() if not k.startswith("_")]
    for name in ordered:
        row = rows[name]
        print(
            f"{name:<30} | {fmt(row.get('mae_all')):>9} | {fmt(row.get('rmse_all')):>9} | "
            f"{fmt(row.get('mae_mask')):>9} | {fmt(row.get('rmse_mask')):>9} | "
            f"{fmt(row.get('boundary')):>9} | {fmt(row.get('reliable_bg_disturbance')):>9} | "
            f"{selection_score(row):>9.6f}"
        )

    print("\nTransparent-mask benchmark metrics")
    print(f"{'Variant':<30} | {'REL':>9} | {'δ1.05':>9} | {'δ1.10':>9} | {'δ1.25':>9}")
    for name in (
        "Previous model result", "Base anchor", "Legacy posterior fusion",
        "Safe residual posterior", "Candidate correction", "Risk-accepted refinement",
        "Oracle anchor-posterior", "Oracle safe-candidate", "Benchmark output",
    ):
        row = rows[name]
        print(
            f"{name:<30} | {fmt(row.get('rel_mask')):>9} | "
            f"{fmt(row.get('delta_105')):>9} | {fmt(row.get('delta_110')):>9} | "
            f"{fmt(row.get('delta_125')):>9}"
        )

    aux = rows["_aux"]
    print(
        f"[Safe anchor] support={aux.get('safe_support_mean', 0):.3f}, "
        f"gate={aux.get('safe_gate_mean', 0):.3f}, "
        f"target={aux.get('safe_gate_target_mean', 0):.3f}, "
        f"gate_MAE={aux.get('safe_gate_mae', 0):.4f}, "
        f"update={aux.get('safe_update_abs', 0):.6f}"
    )
    print(
        f"[Safe outcome] legacy_improve={aux.get('legacy_improve_ratio', 0):.4f}, "
        f"safe_improve={aux.get('safe_improve_ratio', 0):.4f}, "
        f"damage={aux.get('safe_damage_ratio', 0):.4f}, "
        f"oracle_gain={aux.get('safe_oracle_gain', 0):.6f}, "
        f"corruption={aux.get('corruption_fraction', 0):.3f}"
    )
    print(
        f"[Safe risk] MAE(anchor/legacy/gain)="
        f"{aux.get('safe_risk_anchor_mae', 0):.5f}/"
        f"{aux.get('safe_risk_legacy_mae', 0):.5f}/"
        f"{aux.get('safe_risk_gain_mae', 0):.5f}"
    )
    print(
        f"[Refinement] candidate_update={aux.get('candidate_update_abs', 0):.6f}, "
        f"candidate_improve={aux.get('candidate_improve_ratio', 0):.4f}, "
        f"accept={aux.get('acceptance_mean', 0):.3f}, "
        f"accepted_improve={aux.get('accepted_improvement', 0):.4f}, "
        f"oracle_gain={aux.get('refine_oracle_gain', 0):.6f}"
    )
    print(
        f"[Failure] acc={aux.get('fail_acc', 0):.4f}, "
        f"balanced_acc={aux.get('balanced_acc', 0):.4f}, "
        f"F1(miss/bias/bnd)={aux.get('f1_missing', 0):.3f}/"
        f"{aux.get('f1_biased', 0):.3f}/{aux.get('f1_boundary', 0):.3f}"
    )
    print(
        f"[Legacy sources] raw/rel/expert={aux.get('raw_w', 0):.3f}/"
        f"{aux.get('rel_w', 0):.3f}/{aux.get('expert_w', 0):.3f}; "
        f"route_entropy={aux.get('route_entropy', 0):.3f}"
    )


def write_history(history: List[Dict[str, Any]]) -> None:
    (OUT_DIR / "safe_anchor_train_log.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not history:
        return
    keys = sorted(set().union(*[row.keys() for row in history]))
    with open(OUT_DIR / "safe_anchor_train_log.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path: Path,
    model: FailureAwarePosteriorDepth,
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
    torch.save(
        {
            "refine_epoch": epoch,
            "phase": phase,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_score": float(best_score),
            "best_safe_score": float(best_safe_score),
            "best_candidate_score": float(best_candidate_score),
            "no_improve_joint": int(no_improve_joint),
            "history": history,
            "all_rows": {k: v for k, v in rows.items() if not k.startswith("_")},
            "aux": rows.get("_aux", {}),
            "source_checkpoint": str(V5_CKPT),
            "config": {
                "schedule": {
                    "safe": SAFE_WARMUP_EPOCHS,
                    "proposal": PROPOSAL_ADAPT_EPOCHS,
                    "risk": RISK_CALIBRATION_EPOCHS,
                    "joint": JOINT_EPOCHS,
                },
                "SAFE_MAX_RESIDUAL": SAFE_MAX_RESIDUAL,
                "SAFE_GATE_INIT_LOGIT": SAFE_GATE_INIT_LOGIT,
                "SAFE_CORRUPTION_PROB_WARMUP": SAFE_CORRUPTION_PROB_WARMUP,
                "MAX_RISK_DELTA": MAX_RISK_DELTA,
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
    print("=" * 164)
    print("FAPR-Depth v6 | Safe-Anchor Residual Posterior + Risk-Controlled Refinement")
    print("=" * 164)
    print(
        f"DEVICE={DEVICE}, AMP={USE_AMP}\n"
        f"CACHE_ROOT={CACHE_ROOT}\n"
        f"V5_CKPT={V5_CKPT}\n"
        f"OUT_DIR={OUT_DIR}"
    )

    base_mod = load_base_source_module()
    train_shards = load_split_shards(CACHE_ROOT, "train", MAX_TRAIN_SHARDS)
    val_shards = load_split_shards(CACHE_ROOT, "val", MAX_VAL_SHARDS)
    train_loader = DataLoader(
        CachedShardDataset(train_shards), batch_size=LOADER_BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=ragged_shard_collate,
    )
    val_loader = DataLoader(
        CachedShardDataset(val_shards), batch_size=LOADER_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        collate_fn=ragged_shard_collate,
    )

    probe = to_device(next(iter(train_loader)))
    probe_in = build_inputs(next(iter(iter_microbatches(probe, TRAIN_MICROBATCH)))[0])
    print("[Probe]", {k: tuple(v.shape) for k, v in probe_in.items() if torch.is_tensor(v)})
    del probe, probe_in
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    model = FailureAwarePosteriorDepth(base_mod).to(DEVICE)
    safe_gate_params = list(model.safe_anchor.gate_parameters())
    safe_risk_params = list(model.safe_anchor.risk_parameters())
    proposal_params = list(model.risk_refiner.proposal_parameters())
    refine_risk_params = list(model.risk_refiner.risk_parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": safe_gate_params, "lr": LR_SAFE_WARMUP, "name": "safe_gate"},
            {"params": safe_risk_params, "lr": 0.0, "name": "safe_risk"},
            {"params": proposal_params, "lr": 0.0, "name": "proposal"},
            {"params": refine_risk_params, "lr": 0.0, "name": "refine_risk"},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    start_epoch = 1
    history: List[Dict[str, Any]] = []
    best = float("inf")
    best_safe = float("inf")
    best_candidate = float("inf")
    no_improve_joint = 0

    if AUTO_RESUME and RESUME_CKPT.exists():
        payload = torch.load(str(RESUME_CKPT), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if missing or unexpected:
            print(f"[Resume model] missing={len(missing)}, unexpected={len(unexpected)}")
        optimizer.load_state_dict(payload["optimizer"])
        if USE_AMP and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload.get("refine_epoch", 0)) + 1
        history = list(payload.get("history", []))
        best = float(payload.get("best_score", float("inf")))
        best_safe = float(payload.get("best_safe_score", float("inf")))
        best_candidate = float(payload.get("best_candidate_score", float("inf")))
        no_improve_joint = int(payload.get("no_improve_joint", 0))
        print(
            f"[Resume] completed epoch={start_epoch - 1}, next={start_epoch}, "
            f"best_final={best:.6f}, best_safe={best_safe:.6f}, "
            f"best_candidate={best_candidate:.6f}"
        )
    else:
        source_payload = model.load_v5_checkpoint(V5_CKPT)
        model.set_training_phase("safe")
        print(
            f"[Initialization] v5 refine_epoch={source_payload.get('refine_epoch', 'unknown')}, "
            f"recorded best={source_payload.get('best_score', 'unknown')}"
        )
        initial_rows = evaluate(model, val_loader, phase="safe", desc="Initial safe-anchor validation")
        print_summary("Initial v6 anchor", None, "safe", initial_rows)
        best = selection_score(initial_rows["Benchmark output"])
        best_safe = selection_score(initial_rows["Safe benchmark"])
        best_candidate = selection_score(initial_rows["Candidate benchmark"])
        for name in ("initial_anchor.pth", "best_score.pth", "best_safe.pth", "best_candidate.pth"):
            save_checkpoint(
                CKPT_DIR / name, model, optimizer, scaler, 0, "safe",
                initial_rows, history, best, best_safe, best_candidate, no_improve_joint,
            )
        anchor_score = selection_score(initial_rows["Base anchor"])
        oracle_score = selection_score(initial_rows["Oracle anchor-posterior"])
        print(
            f"[Safe oracle diagnostic] anchor={anchor_score:.6f}, "
            f"oracle(anchor,posterior)={oracle_score:.6f}, "
            f"available_gain={anchor_score-oracle_score:+.6f}"
        )
        if oracle_score >= anchor_score:
            print(
                "[WARNING] The anchor/posterior oracle is not better than the anchor. "
                "A gate cannot beat the anchor using only this residual candidate."
            )

    if start_epoch > REFINE_EPOCHS:
        print(f"Safe-anchor training already completed: {start_epoch - 1}/{REFINE_EPOCHS}")
        return

    print(
        f"Trainable modules: safe_gate={sum(p.numel() for p in safe_gate_params):,}, "
        f"safe_risk={sum(p.numel() for p in safe_risk_params):,}, "
        f"proposal={sum(p.numel() for p in proposal_params):,}, "
        f"refine_risk={sum(p.numel() for p in refine_risk_params):,}"
    )

    for epoch in range(start_epoch, REFINE_EPOCHS + 1):
        phase = phase_for_refine_epoch(epoch)
        configure_phase(model, optimizer, phase)
        model.train(True)
        active_params = [
            p for group in optimizer.param_groups for p in group["params"]
            if p.requires_grad
        ]
        stats_all: List[Dict[str, float]] = []
        pbar = tqdm(train_loader, desc=f"Safe curriculum {epoch}/{REFINE_EPOCHS} | {phase}")

        for step, loader_batch in enumerate(pbar, 1):
            loader_batch = to_device(loader_batch)
            optimizer.zero_grad(set_to_none=True)
            micro_stats = []
            for batch, micro_n, total_n in iter_microbatches(loader_batch, TRAIN_MICROBATCH):
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    total, stats = compute_loss(model, batch, phase=phase)
                    scaled = total * (float(micro_n) / float(total_n))
                scaler.scale(scaled).backward()
                micro_stats.append(stats)
            scaler.unscale_(optimizer)
            if active_params:
                torch.nn.utils.clip_grad_norm_(active_params, CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()

            step_stats = avg_dicts(micro_stats)
            stats_all.append(step_stats)
            pbar.set_postfix(
                loss=f"{step_stats.get('loss_total', 0):.4f}",
                gate=f"{step_stats.get('safe_gate_mean', 0):.2f}",
                safe=f"{step_stats.get('safe_mae_mask', 0):.5f}",
                cand=f"{step_stats.get('candidate_mae_mask', 0):.5f}",
                acc=f"{step_stats.get('acceptance_mean', 0):.2f}",
                mb=TRAIN_MICROBATCH,
            )
            del loader_batch, micro_stats
            if DEVICE == "cuda" and EMPTY_CACHE_EVERY > 0 and step % EMPTY_CACHE_EVERY == 0:
                torch.cuda.empty_cache()

        train_loss = float(np.mean([row["loss_total"] for row in stats_all]))
        rows = evaluate(model, val_loader, phase=phase, desc=f"Val safe curriculum {epoch}")
        print_summary(f"Safe-anchor epoch {epoch}", train_loss, phase, rows)

        score = selection_score(rows["Benchmark output"])
        safe_score = selection_score(rows["Safe benchmark"])
        candidate_score = selection_score(rows["Candidate benchmark"])
        history_row: Dict[str, Any] = {
            "refine_epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "score": score,
            "safe_score": safe_score,
            "candidate_score": candidate_score,
            **{f"final_{k}": v for k, v in rows["Benchmark output"].items()},
            **{f"safe_{k}": v for k, v in rows["Safe benchmark"].items()},
            **{f"candidate_{k}": v for k, v in rows["Candidate benchmark"].items()},
            **{f"aux_{k}": v for k, v in rows["_aux"].items()},
        }
        history.append(history_row)
        write_history(history)

        if safe_score < best_safe - 1.0e-8:
            best_safe = safe_score
            save_checkpoint(
                CKPT_DIR / "best_safe.pth", model, optimizer, scaler, epoch, phase,
                rows, history, best, best_safe, best_candidate, no_improve_joint,
            )
            print(f"[Best safe posterior] score={best_safe:.6f}")
        if candidate_score < best_candidate - 1.0e-8:
            best_candidate = candidate_score
            save_checkpoint(
                CKPT_DIR / "best_candidate.pth", model, optimizer, scaler, epoch, phase,
                rows, history, best, best_safe, best_candidate, no_improve_joint,
            )
            print(f"[Best candidate] score={best_candidate:.6f}")

        if score < best - 1.0e-8:
            best = score
            no_improve_joint = 0
            save_checkpoint(
                CKPT_DIR / "best_score.pth", model, optimizer, scaler, epoch, phase,
                rows, history, best, best_safe, best_candidate, no_improve_joint,
            )
            print(f"[Best final] score={best:.6f}")
        elif phase == "joint":
            no_improve_joint += 1
            print(f"[Joint no improvement] {no_improve_joint}/{JOINT_EARLY_STOP_PATIENCE}; best={best:.6f}")
        else:
            print(
                f"[Phase diagnostic] best_final={best:.6f}, "
                f"best_safe={best_safe:.6f}, best_candidate={best_candidate:.6f}"
            )

        save_checkpoint(
            CKPT_DIR / "last.pth", model, optimizer, scaler, epoch, phase,
            rows, history, best, best_safe, best_candidate, no_improve_joint,
        )
        if phase == "safe" and epoch == SAFE_WARMUP_EPOCHS:
            save_checkpoint(
                CKPT_DIR / "safe_warmup_complete.pth", model, optimizer, scaler,
                epoch, phase, rows, history, best, best_safe, best_candidate,
                no_improve_joint,
            )
        if phase == "risk":
            save_checkpoint(
                CKPT_DIR / "risk_calibration_complete.pth", model, optimizer, scaler,
                epoch, phase, rows, history, best, best_safe, best_candidate,
                no_improve_joint,
            )
        if phase == "joint" and no_improve_joint >= JOINT_EARLY_STOP_PATIENCE:
            print("Early stopping after consecutive non-improving joint epochs.")
            break

    print("Done. Best final checkpoint:", CKPT_DIR / "best_score.pth")
    print("Best safe-posterior checkpoint:", CKPT_DIR / "best_safe.pth")
    print("Best candidate checkpoint:", CKPT_DIR / "best_candidate.pth")


if __name__ == "__main__":
    main()
