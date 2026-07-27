# -*- coding: utf-8 -*-
r"""
train_fdct_relative_prior_adapter_strong_base_v8_refhead_metric.py

FDCT-RP Adapter: FDCT backbone + ReMake-style relative-prior adapters.

Purpose
-------
This script is the first route that directly targets your requirement:
  - the new Base final starts from the official FDCT backbone / checkpoint;
  - it is NOT old SV-TDEI Base final post-processing;
  - it is NOT selector / gate / ODelta;
  - old Base final is not used as input, teacher, anchor, or default output;
  - relative prior / mask / raw-error cues enter the FDCT feature hierarchy via
    zero-initialized adapters, so the model starts exactly from official FDCT and
    learns only improvements.

Core idea
---------
Official FDCT forward:
    RGB + raw depth -> FDCT completed depth

FDCT-RP forward:
    RGB + raw depth -> FDCT hierarchy
    relative/mask/raw-prior cues -> zero-init multi-scale adapters -> FDCT hierarchy
    output: completed depth

At initialization, all prior adapters are zero, so FDCT-RP == official FDCT. v5 keeps the frozen FDCT backbone in eval mode during training, freezes BN running stats, lowers adapter LR, and adds FDCT-preservation regularization for stable adapter learning.
This is important: the starting Base final is already FDCT-level, not old SV-TDEI-level.

Direct run:
    python train_fdct_relative_prior_adapter_strong_base.py

Default mode is expanded full-resolution metric probe:
    HxW=240x320, train=1964 shards, val<=512 shards, epochs=4, batch=2.
    v9 starts from the best v8 checkpoint and performs a short boundary/MAE-oriented
    fine-tune. It keeps the same architecture and only changes the optimization target.
Use this after v5 stable probe has shown FDCT-RP > FDCT on subset.
"""

from pathlib import Path
import os
# Reduce CUDA allocator fragmentation. Must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
import sys
import json
import csv
import random
import importlib.util
from typing import Dict, List, Any, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# HARD-CODED CONFIG
# =========================================================
PROJECT_ROOT = Path(os.getenv("FAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent)))
CACHE_ROOT = Path(os.getenv("FAPR_CACHE_ROOT", str(PROJECT_ROOT / "data" / "cache")))

FDCT_ROOT = Path(os.getenv("FAPR_BASE_SOURCE_ROOT", str(PROJECT_ROOT / "third_party" / "FDCT-main")))
FDCT_CKPT = FDCT_ROOT / "checkpoints" / "TransCG.tar"

OUT_DIR = PROJECT_ROOT / "outputs" / "fdct_relative_prior_adapter_strong_base_v9_boundary_mae_finetune"
CKPT_DIR = OUT_DIR / "checkpoints"
V8_RESUME_CKPT = PROJECT_ROOT / "outputs" / "fdct_relative_prior_adapter_strong_base_v8_refhead_metric" / "checkpoints" / "best_score.pth"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
SEED = 6248

# Expanded full-resolution probe. Still faster than full all-shard training, but no longer a tiny sanity check.
IMAGE_SIZE = (240, 320)
MAX_TRAIN_SHARDS = 1964
MAX_VAL_SHARDS = 512
EPOCHS = 3
BATCH_SIZE = 2
NUM_WORKERS = 0

# Official FDCT settings.
FDCT_HIDDEN = 64
FDCT_L = 5
FDCT_K = 12
FDCT_USE_DUC = True

LR_ADAPTER = 2.0e-5
LR_UNFROZEN = 5.0e-6
WEIGHT_DECAY = 1.0e-4
CLIP_GRAD = 5.0

# Freeze FDCT backbone in probe. Only train zero-init relative-prior adapters.
# If adapters improve over FDCT, later set this False and fine-tune decoder with low LR.
FREEZE_FDCT_BACKBONE = True
UNFREEZE_FINAL_HEAD = True

MAX_DEPTH = 10.0
DEPTH_NORM_SCALE = 5.0
EPS = 1e-6
BOUNDARY_KERNEL = 7
RELIABLE_BG_THR = 0.015
HARD_RATIO = 0.20

# Loss weights. Main objective is GT metrics; FDCT is only a guard/baseline.
W_L1_MASK = 1.70
W_RMSE_MASK = 0.92
W_RMSE_ALL = 0.45
W_HARD_MASK = 0.65
W_BOUNDARY = 0.95
W_GRAD = 0.52
W_BG = 0.05
W_FDCT_GUARD = 2.45
W_FDCT_PRESERVE_EASY = 1.00
ADAPTER_FEATURE_SCALE = 0.16
W_HIGH_FDCT = 1.00  # add weight on FDCT-hard pixels; total weight = 1 + W_HIGH_FDCT * hard_label
FDCT_HARD_THR = 0.010  # meters, used only for training loss weighting
FDCT_GUARD_MARGIN = 0.00035

# Evaluation score: same style as previous scripts.
# Keep it transparent: this score is for model selection only.


# =========================================================
# Import official FDCT source
# =========================================================
def import_by_path(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_fdct_module():
    if not FDCT_ROOT.exists():
        raise FileNotFoundError(f"FDCT_ROOT not found: {FDCT_ROOT}")
    if not (FDCT_ROOT / "Model.py").exists():
        raise FileNotFoundError(f"FDCT Model.py not found: {FDCT_ROOT / 'Model.py'}")
    if not (FDCT_ROOT / "module.py").exists():
        raise FileNotFoundError(f"FDCT module.py not found: {FDCT_ROOT / 'module.py'}")
    if not FDCT_CKPT.exists():
        raise FileNotFoundError(f"FDCT checkpoint not found: {FDCT_CKPT}")
    # FDCT Model.py does `from module import ...`, so FDCT_ROOT must be in sys.path.
    if str(FDCT_ROOT) not in sys.path:
        sys.path.insert(0, str(FDCT_ROOT))
    return import_by_path("fdct_model_for_rp", FDCT_ROOT / "Model.py")


# =========================================================
# Utilities
# =========================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def safe_depth(d: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(d, nan=0.0, posinf=MAX_DEPTH, neginf=0.0).clamp(0.0, MAX_DEPTH)


def robust_norm_depth(d: torch.Tensor, scale: float = DEPTH_NORM_SCALE) -> torch.Tensor:
    return torch.clamp(safe_depth(d) / float(scale), 0.0, 2.0)


def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    return (x * m).sum() / m.sum().clamp_min(EPS)


def masked_rmse(pred: torch.Tensor, gt: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(masked_mean((pred - gt) ** 2, region) + 1e-12)


def gradient_x(d: torch.Tensor) -> torch.Tensor:
    return d[:, :, :, 1:] - d[:, :, :, :-1]


def gradient_y(d: torch.Tensor) -> torch.Tensor:
    return d[:, :, 1:, :] - d[:, :, :-1, :]


def gradient_l1(pred: torch.Tensor, gt: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    rx = region[:, :, :, 1:] * region[:, :, :, :-1]
    ry = region[:, :, 1:, :] * region[:, :, :-1, :]
    lx = masked_mean(torch.abs(gradient_x(pred) - gradient_x(gt)), rx)
    ly = masked_mean(torch.abs(gradient_y(pred) - gradient_y(gt)), ry)
    return 0.5 * (lx + ly)


def gradient_mag(d: torch.Tensor) -> torch.Tensor:
    gx = F.pad(torch.abs(d[:, :, :, 1:] - d[:, :, :, :-1]), (0, 1, 0, 0))
    gy = F.pad(torch.abs(d[:, :, 1:, :] - d[:, :, :-1, :]), (0, 0, 0, 1))
    return gx + gy


def force_4d_map(x: torch.Tensor) -> torch.Tensor:
    """Return scalar map as [B,1,H,W].

    The cache stores mini-batches inside each shard. DataLoader then adds another
    batch dimension, so shapes like [loader_B, shard_B, H, W] or
    [loader_B, shard_B, 1, H, W] are normal.  We flatten all leading sample
    dimensions into one batch dimension.
    """
    if x is None:
        raise ValueError("force_4d_map got None")
    x = x.contiguous()

    if x.ndim == 2:                     # [H,W]
        return x.unsqueeze(0).unsqueeze(0).contiguous()

    if x.ndim == 3:
        # [1,H,W] single scalar map, otherwise [B,H,W]
        if x.shape[0] == 1:
            return x.unsqueeze(0).contiguous()
        return x.unsqueeze(1).contiguous()

    if x.ndim == 4:
        # [B,1,H,W]
        if x.shape[1] == 1:
            return x.contiguous()
        # [B,H,W,1]
        if x.shape[-1] == 1:
            return x.permute(0, 3, 1, 2).contiguous()
        # [B,N,H,W] from nested shard batches -> [B*N,1,H,W]
        return x.reshape(-1, 1, x.shape[-2], x.shape[-1]).contiguous()

    if x.ndim >= 5:
        # [...,1,H,W] -> [prod(...),1,H,W]
        if x.shape[-3] == 1:
            return x.reshape(-1, 1, x.shape[-2], x.shape[-1]).contiguous()
        # [...,H,W,1] -> [prod(...),1,H,W]
        if x.shape[-1] == 1:
            y = x.reshape(-1, x.shape[-3], x.shape[-2], 1)
            return y.permute(0, 3, 1, 2).contiguous()
        # [...,H,W] represented with extra leading dims -> [prod(...),1,H,W]
        return x.reshape(-1, 1, x.shape[-2], x.shape[-1]).contiguous()

    raise RuntimeError(f"Expected map tensor with >=2D, got shape={tuple(x.shape)}")


def force_4d_rgb(x: torch.Tensor) -> torch.Tensor:
    """Return RGB as [B,3,H,W].

    Handles single images [3,H,W]/[H,W,3], normal batches [B,3,H,W], and
    nested cache batches [loader_B, shard_B, 3, H, W].
    """
    if x is None:
        raise ValueError("force_4d_rgb got None")
    x = x.contiguous()

    if x.ndim == 3:
        if x.shape[0] == 3:             # [3,H,W]
            return x.unsqueeze(0).contiguous()
        if x.shape[-1] == 3:            # [H,W,3]
            return x.permute(2, 0, 1).unsqueeze(0).contiguous()
        raise RuntimeError(f"Cannot interpret RGB tensor shape={tuple(x.shape)}")

    if x.ndim == 4:
        if x.shape[1] == 3:             # [B,3,H,W]
            return x.contiguous()
        if x.shape[-1] == 3:            # [B,H,W,3]
            return x.permute(0, 3, 1, 2).contiguous()
        raise RuntimeError(f"Cannot interpret RGB tensor shape={tuple(x.shape)}")

    if x.ndim >= 5:
        if x.shape[-3] == 3:            # [...,3,H,W]
            return x.reshape(-1, 3, x.shape[-2], x.shape[-1]).contiguous()
        if x.shape[-1] == 3:            # [...,H,W,3]
            y = x.reshape(-1, x.shape[-3], x.shape[-2], 3)
            return y.permute(0, 3, 1, 2).contiguous()
        raise RuntimeError(f"Cannot interpret RGB tensor shape={tuple(x.shape)}")

    raise RuntimeError(f"Cannot interpret RGB tensor shape={tuple(x.shape)}")


def build_boundary_ring(mask: torch.Tensor, kernel_size: int = BOUNDARY_KERNEL) -> torch.Tensor:
    mask = force_4d_map(mask).float()
    pad = kernel_size // 2
    dil = F.max_pool2d(mask, kernel_size, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0)


def hard_pixel_rmse(pred: torch.Tensor, gt: torch.Tensor, region: torch.Tensor, ratio: float = HARD_RATIO) -> torch.Tensor:
    err2 = ((pred - gt) ** 2).flatten(1)
    m = (region > 0.5).flatten(1)
    vals = []
    for b in range(pred.shape[0]):
        e = err2[b][m[b]]
        if e.numel() == 0:
            continue
        k = max(1, int(np.ceil(e.numel() * ratio)))
        vals.append(torch.topk(e, k=k, largest=True).values.mean())
    if not vals:
        return pred.new_tensor(0.0)
    return torch.sqrt(torch.stack(vals).mean() + 1e-12)


def avg_dicts(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = sorted(set().union(*[d.keys() for d in items]))
    out = {}
    for k in keys:
        vals = [d[k] for d in items if k in d and d[k] is not None and np.isfinite(d[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def fmt(x):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:.6f}"


def selection_score(row: Dict[str, float]) -> float:
    # Transparent metric score for model selection. Lower is better.
    # Matches previous practical selection style: emphasize mask MAE/RMSE and boundary.
    return float(
        row.get("mae_mask", 0.0)
        + row.get("rmse_mask", 0.0)
        + 0.5 * row.get("boundary", 0.0)
        + 0.15 * row.get("rmse_all", 0.0)
    )


@torch.no_grad()
def metric_values(pred_final: torch.Tensor, raw: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    pred = safe_depth(pred_final)
    raw = safe_depth(raw)
    gt = safe_depth(gt)
    mask = mask.float().clamp(0.0, 1.0)
    valid = valid.float().clamp(0.0, 1.0)
    region_all = valid
    region_mask = valid * mask
    boundary = build_boundary_ring(mask) * valid
    raw_err = torch.abs(raw - gt)
    reliable_bg = valid * (1.0 - mask) * (raw_err <= RELIABLE_BG_THR).float()
    return {
        "mae_all": float(masked_mean(torch.abs(pred - gt), region_all).detach().cpu()),
        "rmse_all": float(masked_rmse(pred, gt, region_all).detach().cpu()),
        "mae_mask": float(masked_mean(torch.abs(pred - gt), region_mask).detach().cpu()),
        "rmse_mask": float(masked_rmse(pred, gt, region_mask).detach().cpu()),
        "boundary": float(masked_mean(torch.abs(pred - gt), boundary).detach().cpu()) if boundary.sum().item() > 0 else 0.0,
        "reliable_bg_disturbance": float(masked_mean(torch.abs(pred - raw), reliable_bg).detach().cpu()) if reliable_bg.sum().item() > 0 else 0.0,
    }


# =========================================================
# Cache dataset
# =========================================================
class CachedShardDataset(Dataset):
    def __init__(self, shards: List[Path], image_size: Tuple[int, int] = IMAGE_SIZE):
        self.shards = [Path(p) for p in shards]
        self.image_size = image_size
        if not self.shards:
            raise RuntimeError("Empty shard list.")

    def __len__(self):
        return len(self.shards)

    def _squeeze_tensor(self, x: torch.Tensor) -> torch.Tensor:
        # Cached files often store a leading batch dimension of 1.
        if x.ndim >= 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        return x.float()

    def __getitem__(self, idx):
        shard = torch.load(self.shards[idx], map_location="cpu")
        out = {}
        required = ["rgb", "raw_depth", "gt_depth", "mask", "valid", "rel_aligned", "rel_conf", "raw_prior", "base_final"]
        for k in required:
            if k not in shard:
                raise KeyError(f"Cache shard missing required key: {k}; file={self.shards[idx]}")
        optional = ["rel_bg_resid", "rel_bg_coverage", "boundary"]
        for k, v in shard.items():
            if torch.is_tensor(v):
                out[k] = self._squeeze_tensor(v)
        for k in optional:
            if k not in out:
                ref = out["mask"]
                out[k] = torch.zeros_like(ref)
        return out


def load_split_shards(cache_root: Path, split: str, max_n: int = None) -> List[Path]:
    split_dir = cache_root / split
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = [split_dir / s["file"] for s in manifest["shards"]]
    rng = np.random.default_rng(SEED + (0 if split == "train" else 17))
    if max_n is not None and len(shards) > max_n:
        idx = rng.choice(len(shards), size=max_n, replace=False).tolist()
        shards = [shards[i] for i in idx]
    return shards


def to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.to(DEVICE, non_blocking=True).float() if torch.is_tensor(v) else v for k, v in batch.items()}



def ragged_shard_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate cached shards with variable internal sample counts.

    Some cache shards contain 8 samples, while the last/partial shards can contain
    fewer samples such as 5.  PyTorch's default_collate tries to stack them into
    [loader_B, shard_B, ...] and crashes when shard_B differs.  For this task the
    correct operation is concatenating the shard-internal sample dimension:

        [8, 3, H, W] + [5, 3, H, W] -> [13, 3, H, W]

    The downstream force_4d_* functions already expect a flat sample batch.
    """
    if not batch:
        return {}
    keys = sorted(set().union(*[set(b.keys()) for b in batch]))
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if not vals:
            continue
        if torch.is_tensor(vals[0]):
            vals = [v.float().contiguous() for v in vals]
            # Normal cache tensors are [N,...]. Concatenate N across shards.
            # If a tensor is a single map [H,W], stack it as a batch.
            if vals[0].ndim <= 2:
                out[k] = torch.stack(vals, dim=0).contiguous()
            else:
                try:
                    out[k] = torch.cat(vals, dim=0).contiguous()
                except RuntimeError as e:
                    shapes = [tuple(v.shape) for v in vals]
                    raise RuntimeError(f"ragged_shard_collate failed for key={k}, shapes={shapes}") from e
        else:
            out[k] = vals
    return out



# =========================================================
# FAILURE-AWARE PROBABILISTIC CONFIG OVERRIDES
# =========================================================
OUT_DIR = PROJECT_ROOT / "outputs" / "fdct_failure_aware_probabilistic_v3_calibrated_8gb"
CKPT_DIR = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 12
STAGE1_EPOCHS = 2
STAGE2_EPOCHS = 4
LR_NEW = 1.0e-4
LR_ADAPTER = 3.0e-5
LR_FDCT_HEAD = 5.0e-6
FREEZE_FDCT_BACKBONE = True
UNFREEZE_FINAL_HEAD_IN_STAGE3 = False
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
MAX_REFINE_DELTA = 0.20
FDCT_GUARD_MARGIN = 0.0005
RELIABLE_RAW_THR = 0.010

# The current cache key is `rel_aligned`, which is already metric aligned.
# Keep True for the existing cache to avoid destructive second global alignment.
# Set False only after replacing the cache input with the original unaligned
# monocular depth / relative inverse-depth prediction.
INPUT_REL_ALREADY_METRIC_ALIGNED = True

# Initial Laplace scales (metres) for raw and monocular/relative depth.
RAW_UNCERT_INIT_B = 0.015
REL_UNCERT_INIT_B = 0.030

# Loss weights.
W_DEPTH_MASK = 1.50
W_DEPTH_ALL = 0.35
W_RMSE_MASK = 0.70
W_BOUNDARY_NEW = 0.80
W_GRAD_NEW = 0.35
W_HARD_NEW = 0.25
W_FAILURE = 0.85
W_BRIER = 0.20
W_ALIGN = 0.40
W_ALIGN_REG = 0.03
W_EXPERT = 0.35
W_ROUTER = 0.20
W_BALANCE = 0.01
W_NLL = 0.08
# Input-source uncertainty supervision. A conservative weight is used because
# the shifted Laplace NLL is numerically much larger than focal/Brier losses.
W_INPUT_NLL = 0.05
W_PRESERVE = 0.15
W_FDCT_GUARD_NEW = 0.40
W_ITER = 0.10

# 8 GB GPU-safe execution. A cache shard may contain several images; therefore
# DataLoader batch_size is not the actual network batch size. We split the
# concatenated shard batch into true image micro-batches and accumulate gradients.
BATCH_SIZE = 1
TRAIN_MICROBATCH = 1
VAL_MICROBATCH = 1
EMPTY_CACHE_EVERY = 100


# =========================================================
# Geometry / probability helpers
# =========================================================
def zero_init_conv(conv: nn.Conv2d):
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


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


@torch.no_grad()
def robust_global_align(rel: torch.Tensor, raw: torch.Tensor, anchor: torch.Tensor):
    """Per-image Huber-IRLS affine fit raw ~= a*rel+b using only anchor pixels."""
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
                mx, my = (w*x).sum()/sw, (w*y).sum()/sw
                a = (w*(x-mx)*(y-my)).sum() / (w*(x-mx).square()).sum().clamp_min(EPS)
                b = my - a*mx
                a = a.clamp(0.35, 2.50)
                b = b.clamp(-1.5, 1.5)
                r = (y - (a*x+b)).abs()
                delta = (1.5*r.median()).clamp_min(0.005)
                w = torch.where(r <= delta, torch.ones_like(r), delta/(r+EPS))
        aligned.append(a*rel[i:i+1] + b)
        scales.append(a.reshape(1)); biases.append(b.reshape(1))
    return safe_depth(torch.cat(aligned, 0)), torch.cat(scales), torch.cat(biases)


def charbonnier(pred, gt, region, eps=1e-3):
    return masked_mean(torch.sqrt((pred-gt).square()+eps*eps), region)


def laplace_nll(pred, gt, log_b, region):
    log_b = log_b.clamp(-6.0, 2.0)
    return masked_mean((pred-gt).abs()*torch.exp(-log_b)+log_b, region)


def shifted_laplace_nll(pred, gt, log_b, region):
    """Laplace NLL plus a constant; gradients are unchanged, logs stay non-negative."""
    return laplace_nll(pred, gt, log_b, region) + 6.0


@torch.no_grad()
def failure_prf(pred_label, target_label, valid, cls: int):
    """Return precision/recall/F1/support for one failure class."""
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


def dynamic_focal_loss(logits, labels, valid, gamma=2.0):
    # logits [B,C,H,W], labels [B,1,H,W]
    c = logits.shape[1]
    y = labels[:, 0].long()
    ce = F.cross_entropy(logits, y, reduction='none')
    pt = torch.exp(-ce)
    with torch.no_grad():
        counts = torch.stack([((y == k).float()*valid[:,0]).sum() for k in range(c)])
        alpha = (counts.sum()+c)/(counts+c)
        alpha = alpha/alpha.mean().clamp_min(EPS)
    a = alpha[y]
    return masked_mean(a*((1-pt)**gamma)*ce, valid[:,0])


def failure_targets(raw, gt, valid, boundary):
    err = (raw-gt).abs()
    missing = (raw <= EPS) & (valid > 0.5)
    fail = err > (FAIL_ABS_THR + FAIL_REL_THR*gt)
    bfail = (boundary > 0.15) & (err > BOUNDARY_FAIL_THR) & (~missing) & (valid > 0.5)
    biased = fail & (~missing) & (~bfail) & (valid > 0.5)
    labels = torch.zeros_like(raw, dtype=torch.long)
    labels[missing] = 1
    labels[biased] = 2
    labels[bfail] = 3
    return labels, err


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(max(1, min(8, cout)), cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, cout)), cout), nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class LocalMetricAligner(nn.Module):
    def __init__(self, cin=11, hidden=32):
        super().__init__()
        self.body = nn.Sequential(ConvBlock(cin, hidden), ConvBlock(hidden, hidden, dilation=2))
        self.out = zero_init_conv(nn.Conv2d(hidden, 2, 3, padding=1))
    def forward(self, x, rel_global):
        z = self.out(self.body(x))
        ds = MAX_LOCAL_SCALE_LOG*torch.tanh(z[:,0:1])
        db = MAX_LOCAL_BIAS*torch.tanh(z[:,1:2])
        rel_metric = safe_depth(rel_global*torch.exp(ds)+db)
        reg = ds.abs().mean()+db.abs().mean()+0.25*(gradient_mag(ds).mean()+gradient_mag(db).mean())
        return rel_metric, ds, db, reg


class FailureEstimator(nn.Module):
    def __init__(self, cin=14, hidden=48):
        super().__init__()
        self.body = nn.Sequential(ConvBlock(cin, hidden), ConvBlock(hidden, hidden, dilation=2), ConvBlock(hidden, hidden))
        self.failure = nn.Conv2d(hidden, 4, 1)
        self.uncert = zero_init_conv(nn.Conv2d(hidden, 2, 1))  # raw log-b, relative-prior log-b
        with torch.no_grad():
            # Start from physically reasonable error scales instead of random
            # source confidences, avoiding source-weight collapse at Stage 2.
            self.uncert.bias[0] = float(np.log(RAW_UNCERT_INIT_B))
            self.uncert.bias[1] = float(np.log(REL_UNCERT_INIT_B))
    def forward(self, x):
        f = self.body(x)
        return f, self.failure(f), self.uncert(f).clamp(-6.0, 2.0)


class ResidualExpert(nn.Module):
    def __init__(self, cin=64, hidden=48, dilation=1):
        super().__init__()
        self.body = nn.Sequential(ConvBlock(cin, hidden, dilation=dilation), ConvBlock(hidden, hidden))
        self.out = nn.Conv2d(hidden, 2, 1)  # normalized delta and log-b
        zero_init_conv(self.out)
    def forward(self, feat):
        z = self.out(self.body(feat))
        delta = MAX_EXPERT_DELTA*torch.tanh(z[:,0:1])
        log_b = z[:,1:2].clamp(-6.0, 2.0)
        return delta, log_b


class SharedRefiner(nn.Module):
    def __init__(self, cin=15, hidden=48):
        super().__init__()
        self.body = nn.Sequential(ConvBlock(cin, hidden), ConvBlock(hidden, hidden, dilation=2))
        self.out = nn.Conv2d(hidden, 2, 1)
        zero_init_conv(self.out)
        with torch.no_grad(): self.out.bias[1] = -2.0
    def forward(self, x):
        z = self.out(self.body(x))
        delta = MAX_REFINE_DELTA*torch.tanh(z[:,0:1])
        gate = torch.sigmoid(z[:,1:2])
        return delta, gate


# =========================================================
# FDCT + adapters + failure-aware probabilistic model
# =========================================================
class FailureAwareProbabilisticFDCT(nn.Module):
    PRIOR_CH = 13
    def __init__(self, fdct_mod):
        super().__init__()
        self.fdct = fdct_mod.FDCT(in_channels=4, hidden_channels=FDCT_HIDDEN, L=FDCT_L, k=FDCT_K, use_DUC=FDCT_USE_DUC)
        self.fdct_ref = fdct_mod.FDCT(in_channels=4, hidden_channels=FDCT_HIDDEN, L=FDCT_L, k=FDCT_K, use_DUC=FDCT_USE_DUC)
        h = FDCT_HIDDEN
        for name in ['first','e1','e2','e3','e4','d1','d2','d3','out']:
            setattr(self, 'adapt_'+name, zero_init_conv(nn.Conv2d(self.PRIOR_CH, h, 3, padding=1)))
        self.aligner = LocalMetricAligner(11, 32)
        self.failure_net = FailureEstimator(14, 48)
        self.shared = nn.Sequential(ConvBlock(15, 64), ConvBlock(64, 64, dilation=2), ConvBlock(64, 64))
        self.missing_expert = ResidualExpert(64, 48, dilation=3)
        self.biased_expert = ResidualExpert(64, 48, dilation=1)
        self.boundary_expert = ResidualExpert(64, 48, dilation=2)
        self.router = nn.Sequential(ConvBlock(68, 48), nn.Conv2d(48, 3, 1))
        self.refiner = SharedRefiner(13, 48)
        self.stage = 3

    def load_official_fdct(self, ckpt_path: Path):
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt.get('model', ckpt)))
        remap = {}
        for p in ['skip_down1','skip_down2','skip_down3']:
            remap[f'{p}.2']=f'{p}.1'; remap[f'{p}.4']=f'{p}.3'; remap[f'{p}.5']=f'{p}.4'
        clean={}
        for k,v in state.items():
            nk=k[7:] if k.startswith('module.') else k
            for old,new in remap.items():
                if nk.startswith(old+'.'): nk=new+nk[len(old):]; break
            clean[nk]=v
        m1,u1=self.fdct.load_state_dict(clean, strict=False)
        m2,u2=self.fdct_ref.load_state_dict(clean, strict=False)
        print(f'[FDCT load] train missing={len(m1)} unexpected={len(u1)}; ref missing={len(m2)} unexpected={len(u2)}')
        for p in self.fdct_ref.parameters(): p.requires_grad_(False)
        self.fdct_ref.eval()
        return ckpt

    def train(self, mode=True):
        super().train(mode)
        self.fdct_ref.eval()
        if FREEZE_FDCT_BACKBONE: self.fdct.eval()
        return self

    def set_stage(self, stage: int):
        self.stage=stage
        for n,p in self.named_parameters():
            if n.startswith('fdct_ref.'):
                p.requires_grad_(False)
            elif n.startswith('fdct.'):
                p.requires_grad_(stage==3 and UNFREEZE_FINAL_HEAD_IN_STAGE3 and n.startswith('fdct.final.'))
            elif n.startswith(('aligner.','failure_net.')):
                p.requires_grad_(True)
            elif n.startswith(('adapt_','shared.','missing_expert.','biased_expert.','boundary_expert.','router.')):
                p.requires_grad_(stage>=2)
            elif n.startswith('refiner.'):
                p.requires_grad_(stage>=3)

    def prior_at(self, priors, feat, adapter):
        p=F.interpolate(priors, size=feat.shape[-2:], mode='bilinear', align_corners=True)
        return adapter(p)*ADAPTER_FEATURE_SCALE

    def forward_base(self, rgb, depth):
        d=depth[:,0] if depth.ndim==4 else depth
        out=self.fdct_ref(rgb,d)
        return safe_depth(out.unsqueeze(1) if out.ndim==3 else out)

    def forward_adapted_fdct(self, rgb, depth, priors):
        f=self.fdct; dv=depth if depth.ndim==4 else depth.unsqueeze(1); d3=dv[:,0]
        h=f.first(torch.cat((rgb,dv),1)); h=h+self.prior_at(priors,h,self.adapt_first)
        d1=F.interpolate(dv,scale_factor=.5,mode='bilinear',align_corners=True); h_d1s=h
        h=h+self.prior_at(priors,h,self.adapt_e1); h=f.dense1_conv1(torch.cat((h,d1),1)); h=f.dense1(h); h=f.dense1_conv2(h)
        d2=F.interpolate(d1,scale_factor=.5,mode='bilinear',align_corners=True); h_d2s=h; h_d2d=f.skip_down2(torch.cat((h_d2s,f.skip_down1(h_d1s)),1))
        h=h+self.prior_at(priors,h,self.adapt_e2); h=f.dense2_conv1(torch.cat((h,d2,f.down_res1(h_d1s)),1)); h=f.dense2(h); h=f.dense2_conv2(h)
        d3s=F.interpolate(d2,scale_factor=.5,mode='bilinear',align_corners=True); h_d3s=h; h_d3d=f.skip_down3(torch.cat((h_d3s,h_d2d),1))
        h=h+self.prior_at(priors,h,self.adapt_e3); h=f.dense3_conv1(torch.cat((h,d3s,f.down_res2(h_d2s)),1)); h=f.dense3(h); h=f.dense3_conv2(h)
        d4=F.interpolate(d3s,scale_factor=.5,mode='bilinear',align_corners=True)
        h=h+self.prior_at(priors,h,self.adapt_e4); h=f.dense4_conv1(torch.cat((h,d4,f.down_res3(h_d3s)),1)); h=f.dense4(h)
        h=torch.cat((h,h_d3d),1); h=f.cdown(h); h_skip3=h
        h=h+self.prior_at(priors,h,self.adapt_d1); h=f.updense1_conv(torch.cat((h,d4),1)); h=f.updense1(h); h=f.updense1_duc(h); h_skip1=h
        h=h+self.prior_at(priors,h,self.adapt_d2); h=torch.cat((h,h_d3s,d3s,f.skip_up3(h_skip3)),1); h=f.updense2_conv(h); h=f.updense2(h); h=f.updense2_duc(h); h_skip2=h
        h=h+self.prior_at(priors,h,self.adapt_d3); h=torch.cat((h,h_d2s,d2,f.skip_up1(h_skip1)),1); h=f.updense3_conv(h); h=f.updense3(h); h=f.updense3_duc(h)
        h=torch.cat((h,h_d1s,d1,f.skip_up2(h_skip2)),1); h=f.updense4_conv(h); h=f.updense4(h); h=f.updense4_duc(h)
        h=h+self.prior_at(priors,h,self.adapt_out)
        return safe_depth(f.final(h))

    def forward(self, inp: Dict[str, torch.Tensor], stage: int = None):
        """Stage-aware forward.

        Stage 1 computes only metric alignment and failure estimation. It does not
        execute FDCT, the three experts, router, probabilistic fusion, or refiner.
        Stage 2 additionally executes FDCT/adapters, experts, router, and fusion.
        Stage 3 additionally executes the two shared refinement iterations.
        """
        stage = self.stage if stage is None else int(stage)
        rgb, raw, rel, mask, valid = inp['rgb'], inp['raw'], inp['rel'], inp['mask'], inp['valid']
        boundary, raw_prior, rel_conf = inp['boundary'], inp['raw_prior'], inp['rel_conf']
        raw_valid = (raw > EPS).float() * valid
        sdm = approximate_signed_distance(mask)
        grad_raw = torch.clamp(gradient_mag(raw) / .08, 0, 4)
        grad_rel = torch.clamp(gradient_mag(rel) / .08, 0, 4)

        outside = (1.0 - dilate_binary(mask, ANCHOR_DILATE_KERNEL)).clamp(0, 1)
        anchor = raw_valid * outside * (grad_raw < ANCHOR_GRAD_THR / .08).float()
        if INPUT_REL_ALREADY_METRIC_ALIGNED:
            # Existing cache already contains `rel_aligned`; do not fit another
            # global affine transform. Only the learnable low-frequency field is used.
            rel_global = rel
            global_a = rel.new_ones((rel.shape[0],))
            global_b = rel.new_zeros((rel.shape[0],))
        else:
            rel_global, global_a, global_b = robust_global_align(rel, raw, anchor)
        disc0 = torch.clamp((rel_global - raw) / .75, -1, 1) * raw_valid
        align_x = torch.cat([
            rgb, robust_norm_depth(raw), robust_norm_depth(rel_global), mask,
            boundary, raw_valid, disc0, grad_raw, grad_rel
        ], 1)
        rel_metric, ds, local_bias, align_reg = self.aligner(align_x, rel_global)
        discrepancy = torch.clamp((rel_metric - raw) / .75, -1, 1) * raw_valid

        fail_x = torch.cat([
            rgb, robust_norm_depth(raw), robust_norm_depth(rel_metric), mask,
            boundary, sdm, raw_valid, discrepancy, grad_raw, grad_rel,
            rel_conf, raw_prior
        ], 1)
        fail_feat, fail_logits, source_logb = self.failure_net(fail_x)
        fail_prob = F.softmax(fail_logits, 1)
        raw_logb, rel_logb = source_logb[:, 0:1], source_logb[:, 1:2]
        p_valid = fail_prob[:, 0:1]
        p_fail = 1.0 - p_valid

        # Lightweight placeholders make the loss/metrics interface identical while
        # avoiding all expensive downstream branches in stage 1.
        b, _, h, w = raw.shape
        zeros = raw.new_zeros((b, 1, h, w))
        pi_uniform = raw.new_full((b, 3, h, w), 1.0 / 3.0)
        alpha_stage1 = torch.cat([zeros, torch.ones_like(zeros), zeros], 1)
        common = {
            'rel_input': rel,
            'rel_global': rel_global,
            'rel_metric': rel_metric,
            'fail_logits': fail_logits,
            'fail_prob': fail_prob,
            'raw_logb': raw_logb,
            'rel_logb': rel_logb,
            'anchor': anchor,
            'align_reg': align_reg,
            'global_a': global_a,
            'global_b': global_b,
            'ds': ds,
            'db': local_bias,
        }
        if stage <= 1:
            return {
                **common,
                'final': rel_metric,
                'fused': rel_metric,
                'iter_preds': [],
                'fdct': rel_metric,
                'router_logits': raw.new_zeros((b, 3, h, w)),
                'pi': pi_uniform,
                'expert': rel_metric,
                'expert_candidates': rel_metric.repeat(1, 3, 1, 1),
                'expert_logbs': rel_logb.repeat(1, 3, 1, 1),
                'alpha': alpha_stage1,
                'final_logb': rel_logb,
                'route_entropy': torch.ones_like(zeros),
            }

        rel_bg_resid = inp['rel_bg_resid']
        rel_bg_cov = inp['rel_bg_coverage']
        priors = torch.cat([
            mask, boundary, sdm, raw_valid, rel_conf, raw_prior,
            robust_norm_depth(raw), robust_norm_depth(rel_metric), discrepancy,
            grad_raw, grad_rel, rel_bg_resid, rel_bg_cov
        ], 1)
        fdct = self.forward_adapted_fdct(rgb, raw, priors)

        ctx = torch.cat([fail_x, robust_norm_depth(fdct)], 1)
        shared = self.shared(ctx)
        dm, um = self.missing_expert(shared)
        dd, ud = self.biased_expert(shared)
        dbd, ub = self.boundary_expert(shared)
        router_logits = self.router(torch.cat([shared, fail_prob], 1))
        pi = F.softmax(router_logits, 1)
        deltas = torch.cat([dm, dd, dbd], 1)
        expert_logbs = torch.cat([um, ud, ub], 1)
        mix_delta = (pi * deltas).sum(1, keepdim=True)
        expert = safe_depth(fdct + mix_delta)
        expert_var = (
            pi * (torch.exp(2 * expert_logbs) + deltas.square())
        ).sum(1, keepdim=True) - mix_delta.square()
        expert_logb = .5 * torch.log(expert_var.clamp_min(1e-6)).clamp(-6, 2)

        wr = raw_valid * p_valid.clamp_min(.02) * torch.exp(-raw_logb)
        wm = rel_conf.clamp_min(.05) * torch.exp(-rel_logb)
        we = (.20 + .80 * p_fail) * torch.exp(-expert_logb)
        weights = torch.cat([wr, wm, we], 1)
        alpha = weights / weights.sum(1, keepdim=True).clamp_min(EPS)
        candidates = torch.cat([raw, rel_metric, expert], 1)
        fused = (alpha * candidates).sum(1, keepdim=True)
        source_logbs = torch.cat([raw_logb, rel_logb, expert_logb], 1)
        mix_var = (
            alpha * (torch.exp(2 * source_logbs) + candidates.square())
        ).sum(1, keepdim=True) - fused.square()
        final_logb = .5 * torch.log(mix_var.clamp_min(1e-6)).clamp(-6, 2)
        route_entropy = -(
            pi * torch.log(pi.clamp_min(EPS))
        ).sum(1, keepdim=True) / np.log(3.0)

        cur = safe_depth(fused)
        iter_preds = []
        if stage >= 3:
            for _ in range(2):
                rx = torch.cat([
                    rgb, robust_norm_depth(cur), robust_norm_depth(raw),
                    robust_norm_depth(rel_metric), robust_norm_depth(fdct), mask,
                    boundary, sdm, p_fail, final_logb, route_entropy
                ], 1)
                delta, gate = self.refiner(rx)
                cur = safe_depth(cur + gate * delta)
                iter_preds.append(cur)

        return {
            **common,
            'final': cur,
            'fused': fused,
            'iter_preds': iter_preds,
            'fdct': fdct,
            'router_logits': router_logits,
            'pi': pi,
            'expert': expert,
            'expert_candidates': torch.cat([
                safe_depth(fdct + dm), safe_depth(fdct + dd), safe_depth(fdct + dbd)
            ], 1),
            'expert_logbs': expert_logbs,
            'alpha': alpha,
            'final_logb': final_logb,
            'route_entropy': route_entropy,
        }


# =========================================================
# Input construction
# =========================================================
def build_inputs(batch: Dict[str,torch.Tensor]):
    rgb=force_4d_rgb(batch['rgb'].float()).clamp(0,1)
    raw=safe_depth(force_4d_map(batch['raw_depth'].float()))
    gt=safe_depth(force_4d_map(batch['gt_depth'].float()))
    mask=force_4d_map(batch['mask'].float()).clamp(0,1)
    valid=force_4d_map(batch['valid'].float()).clamp(0,1)
    rel=safe_depth(force_4d_map(batch['rel_aligned'].float()))
    z=torch.zeros_like(mask); o=torch.ones_like(mask)
    def opt(name,default): return force_4d_map(batch[name].float()).clamp(0,1) if name in batch else default
    return {'rgb':rgb,'raw':raw,'gt':gt,'mask':mask,'valid':valid,'rel':rel,'rel_conf':opt('rel_conf',o),'raw_prior':opt('raw_prior',z),'rel_bg_resid':opt('rel_bg_resid',z),'rel_bg_coverage':opt('rel_bg_coverage',z),'boundary':opt('boundary',build_boundary_ring(mask)),'old_base':safe_depth(force_4d_map(batch['base_final'].float())) if 'base_final' in batch else raw}


# Relax cache requirements from the original script.
CachedShardDataset.__getitem__ = (lambda old: (lambda self,idx: _cached_getitem(self,idx)))(CachedShardDataset.__getitem__)
def _cached_getitem(self, idx):
    shard=torch.load(self.shards[idx],map_location='cpu'); out={}
    required=['rgb','raw_depth','gt_depth','mask','valid','rel_aligned']
    for k in required:
        if k not in shard: raise KeyError(f'Cache shard missing {k}: {self.shards[idx]}')
    for k,v in shard.items():
        if torch.is_tensor(v): out[k]=self._squeeze_tensor(v)
    return out


# =========================================================
# True-image micro-batching
# =========================================================
def batch_sample_count(batch: Dict[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise RuntimeError('Cannot determine batch sample count.')


def iter_microbatches(batch: Dict[str, Any], microbatch_size: int):
    n = batch_sample_count(batch)
    microbatch_size = max(1, int(microbatch_size))
    for start in range(0, n, microbatch_size):
        end = min(n, start + microbatch_size)
        part = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == n:
                part[key] = value[start:end]
            else:
                part[key] = value
        yield part, end - start, n


# =========================================================
# Loss / metrics
# =========================================================
def compute_loss(model, batch, stage: int, return_outputs: bool = False):
    inp = build_inputs(batch)
    out = model(inp, stage=stage)
    raw, gt, mask, valid = inp['raw'], inp['gt'], inp['mask'], inp['valid']
    boundary = inp['boundary'] * valid
    labels, raw_err = failure_targets(raw, gt, valid, boundary)
    onehot = F.one_hot(labels[:, 0], num_classes=4).permute(0, 3, 1, 2).float()

    fail_loss = dynamic_focal_loss(out['fail_logits'], labels, valid)
    brier = masked_mean((out['fail_prob'] - onehot).square().sum(1, keepdim=True), valid)

    anchor = out['anchor']
    if anchor.sum().item() > 0:
        align_input = charbonnier(inp['rel'], raw, anchor)
        align_global = charbonnier(out['rel_global'], raw, anchor)
        align_loss = charbonnier(out['rel_metric'], raw, anchor)
    else:
        align_input = raw.new_tensor(0.0)
        align_global = raw.new_tensor(0.0)
        align_loss = raw.new_tensor(0.0)

    # Explicitly supervise the raw/relative uncertainty heads before those heads
    # participate in Stage-2 probabilistic source weighting.
    raw_region = valid * (raw > EPS).float()
    raw_nll = (
        shifted_laplace_nll(raw, gt, out['raw_logb'], raw_region)
        if raw_region.sum().item() > 0 else raw.new_tensor(0.0)
    )
    rel_nll = shifted_laplace_nll(out['rel_metric'], gt, out['rel_logb'], valid)
    input_nll = 0.5 * (raw_nll + rel_nll)

    # Define zero-valued diagnostics for a consistent logging schema.
    depth_mask = raw.new_tensor(0.0)
    depth_all = raw.new_tensor(0.0)
    rmse = masked_rmse(out['final'], gt, valid * mask)
    bd = raw.new_tensor(0.0)
    grad = raw.new_tensor(0.0)
    hard = raw.new_tensor(0.0)
    router_ce = raw.new_tensor(0.0)
    balance = raw.new_tensor(0.0)
    expert_loss = raw.new_tensor(0.0)
    nll = raw.new_tensor(0.0)
    preserve = raw.new_tensor(0.0)
    guard = raw.new_tensor(0.0)
    iter_loss = raw.new_tensor(0.0)

    if stage == 1:
        total = (
            W_FAILURE * fail_loss
            + W_BRIER * brier
            + W_ALIGN * align_loss
            + W_ALIGN_REG * out['align_reg']
            + W_INPUT_NLL * input_nll
        )
    else:
        pred = out['final']
        region_mask = valid * mask
        pred_err = (pred - gt).abs()
        depth_mask = charbonnier(pred, gt, region_mask)
        depth_all = charbonnier(pred, gt, valid)
        rmse = masked_rmse(pred, gt, region_mask)
        bd = masked_mean(pred_err, boundary) if boundary.sum().item() > 0 else pred.new_tensor(0.0)
        grad = gradient_l1(pred, gt, region_mask)
        hard = hard_pixel_rmse(pred, gt, region_mask)

        router_target = (labels - 1).clamp(0, 2)[:, 0]
        failure_region = valid[:, 0] * (labels[:, 0] > 0).float()
        router_ce = (
            masked_mean(F.cross_entropy(out['router_logits'], router_target, reduction='none'), failure_region)
            if failure_region.sum().item() > 0 else pred.new_tensor(0.0)
        )
        mean_pi = (out['pi'] * valid).sum((0, 2, 3)) / valid.sum().clamp_min(EPS)
        balance = ((mean_pi - 1.0 / 3.0) ** 2).mean()

        for k, cls in enumerate([1, 2, 3]):
            reg = valid * (labels == cls).float()
            if reg.sum().item() > 0:
                expert_loss = expert_loss + charbonnier(out['expert_candidates'][:, k:k + 1], gt, reg)

        nll = laplace_nll(pred, gt, out['final_logb'], valid)
        reliable = valid * (raw > EPS).float() * (raw_err <= RELIABLE_RAW_THR).float()
        preserve = (
            masked_mean((pred - raw).abs(), reliable * out['fail_prob'][:, 0:1].detach())
            if reliable.sum().item() > 0 else pred.new_tensor(0.0)
        )
        with torch.no_grad():
            fdct_ref = model.forward_base(inp['rgb'], raw)
            fdct_err = (fdct_ref - gt).abs()
        guard = masked_mean(F.relu(pred_err - fdct_err - FDCT_GUARD_MARGIN), region_mask)

        prev = out['fused'].detach()
        if stage >= 3:
            for ip in out['iter_preds']:
                iter_loss = iter_loss + masked_mean(
                    F.relu((ip - gt).abs() - (prev - gt).abs() + 1e-4), region_mask
                )
                prev = ip.detach()

        total = (
            W_DEPTH_MASK * depth_mask
            + W_DEPTH_ALL * depth_all
            + W_RMSE_MASK * rmse
            + W_BOUNDARY_NEW * bd
            + W_GRAD_NEW * grad
            + W_HARD_NEW * hard
            + W_FAILURE * fail_loss
            + W_BRIER * brier
            + W_ALIGN * align_loss
            + W_ALIGN_REG * out['align_reg']
            + W_INPUT_NLL * input_nll
            + W_EXPERT * expert_loss
            + W_ROUTER * router_ce
            + W_BALANCE * balance
            + W_NLL * nll
            + W_PRESERVE * preserve
            + W_FDCT_GUARD_NEW * guard
            + W_ITER * iter_loss
        )

    with torch.no_grad():
        pred = out['final']
        reg = valid * mask
        pred_label = out['fail_logits'].argmax(1, keepdim=True)
        acc = masked_mean((pred_label == labels).float(), valid)

        cls_metrics = {}
        class_names = {0: 'valid', 1: 'missing', 2: 'biased', 3: 'boundary'}
        recalls = []
        for cls, name in class_names.items():
            precision, recall, f1, support = failure_prf(pred_label, labels, valid, cls)
            cls_metrics[f'precision_{name}'] = float(precision)
            cls_metrics[f'recall_{name}'] = float(recall)
            cls_metrics[f'f1_{name}'] = float(f1)
            cls_metrics[f'support_{name}'] = float(support)
            recalls.append(recall)
        balanced_acc = torch.stack(recalls).mean()

        # Region-wise source weights are much more informative than all-pixel means.
        reliable_region = valid * (raw > EPS).float() * (raw_err <= RELIABLE_RAW_THR).float()
        missing_region = valid * (labels == 1).float()
        biased_region = valid * (labels == 2).float()
        boundary_region = valid * (labels == 3).float()

        def region_weight(channel, region):
            if region.sum().item() <= 0:
                return 0.0
            return float(masked_mean(out['alpha'][:, channel:channel + 1], region))

        stats = {
            'loss_total': float(total),
            'mae_mask': float(masked_mean((pred - gt).abs(), reg)),
            'rmse_mask': float(masked_rmse(pred, gt, reg)),
            'failure': float(fail_loss),
            'brier': float(brier),
            'fail_acc': float(acc),
            'balanced_acc': float(balanced_acc),
            'align_input': float(align_input),
            'align_global': float(align_global),
            'align': float(align_loss),
            'input_nll': float(input_nll),
            'raw_nll': float(raw_nll),
            'rel_nll': float(rel_nll),
            'raw_logb_mean': float(out['raw_logb'].mean()),
            'rel_logb_mean': float(out['rel_logb'].mean()),
            'raw_w': float(out['alpha'][:, 0:1].mean()),
            'rel_w': float(out['alpha'][:, 1:2].mean()),
            'expert_w': float(out['alpha'][:, 2:3].mean()),
            'raw_w_reliable': region_weight(0, reliable_region),
            'rel_w_missing': region_weight(1, missing_region),
            'expert_w_biased': region_weight(2, biased_region),
            'expert_w_boundary': region_weight(2, boundary_region),
            'entropy': float(out['route_entropy'].mean()),
            'depth_mask_loss': float(depth_mask),
            'boundary_loss': float(bd),
            'expert_loss': float(expert_loss),
            'router_loss': float(router_ce),
            'final_nll': float(nll),
        }
        stats.update(cls_metrics)

    if return_outputs:
        return total, stats, inp, out
    return total, stats


@torch.no_grad()
def evaluate(model, loader, stage, desc='Val'):
    model.eval()
    row_names = [
        'Raw Depth',
        'Input relative prior',
        'Metric-aligned prior',
        'Old Base final',
        'FDCT baseline',
        'Failure-aware final',
    ]
    rows = {k: [] for k in row_names}
    aux = []
    for loader_batch in tqdm(loader, desc=desc, leave=False):
        loader_batch = to_device(loader_batch)
        for batch, _, _ in iter_microbatches(loader_batch, VAL_MICROBATCH):
            _, stats, inp, out = compute_loss(model, batch, stage, return_outputs=True)
            raw, gt, mask, valid = inp['raw'], inp['gt'], inp['mask'], inp['valid']
            fdct = model.forward_base(inp['rgb'], raw)
            fdct_hard = fdct * mask + raw * (1 - mask)
            old = inp['old_base'] * mask + raw * (1 - mask)
            rows['Raw Depth'].append(metric_values(raw, raw, gt, mask, valid))
            rows['Input relative prior'].append(metric_values(inp['rel'], raw, gt, mask, valid))
            rows['Metric-aligned prior'].append(metric_values(out['rel_metric'], raw, gt, mask, valid))
            rows['Old Base final'].append(metric_values(old, raw, gt, mask, valid))
            rows['FDCT baseline'].append(metric_values(fdct_hard, raw, gt, mask, valid))
            rows['Failure-aware final'].append(metric_values(out['final'], raw, gt, mask, valid))
            aux.append(stats)
    avg = {k: avg_dicts(v) for k, v in rows.items()}
    avg['_aux'] = avg_dicts(aux)
    return avg


def print_summary(epoch, stage, train_loss, rows):
    print('\n' + '=' * 160)
    print(f'Epoch {epoch} | stage {stage} | train loss {train_loss:.6f}')
    if stage == 1:
        print('[Stage 1 note] Failure-aware final is only the metric-aligned prior; FDCT/experts/router/refiner are intentionally skipped.')
    print(f"{'Variant':<25} | {'MAE_all':>9} | {'RMSE_all':>9} | {'MAE_mask':>9} | {'RMSE_mask':>9} | {'Boundary':>9} | {'BG':>9} | {'Score':>9}")
    names = [
        'Raw Depth', 'Input relative prior', 'Metric-aligned prior',
        'Old Base final', 'FDCT baseline', 'Failure-aware final'
    ]
    for name in names:
        r = rows[name]
        print(
            f"{name:<25} | {fmt(r.get('mae_all')):>9} | {fmt(r.get('rmse_all')):>9} | "
            f"{fmt(r.get('mae_mask')):>9} | {fmt(r.get('rmse_mask')):>9} | "
            f"{fmt(r.get('boundary')):>9} | {fmt(r.get('reliable_bg_disturbance')):>9} | "
            f"{selection_score(r):>9.6f}"
        )
    a = rows['_aux']
    print(
        f"[Failure] acc={a.get('fail_acc', 0):.4f}, balanced_acc={a.get('balanced_acc', 0):.4f}, "
        f"Brier={a.get('brier', 0):.4f}, "
        f"F1(miss/bias/bnd)={a.get('f1_missing', 0):.3f}/{a.get('f1_biased', 0):.3f}/{a.get('f1_boundary', 0):.3f}"
    )
    print(
        f"[Alignment] input={a.get('align_input', 0):.5f}, global={a.get('align_global', 0):.5f}, "
        f"metric={a.get('align', 0):.5f}; input_NLL={a.get('input_nll', 0):.4f}, "
        f"logb(raw/rel)={a.get('raw_logb_mean', 0):.3f}/{a.get('rel_logb_mean', 0):.3f}"
    )
    print(
        f"[Sources all] raw/rel/exp={a.get('raw_w', 0):.3f}/{a.get('rel_w', 0):.3f}/{a.get('expert_w', 0):.3f}; "
        f"[regional] raw@reliable={a.get('raw_w_reliable', 0):.3f}, rel@missing={a.get('rel_w_missing', 0):.3f}, "
        f"exp@biased={a.get('expert_w_biased', 0):.3f}, exp@boundary={a.get('expert_w_boundary', 0):.3f}; "
        f"route_entropy={a.get('entropy', 0):.3f}"
    )


def save_checkpoint(path,model,optimizer,scaler,epoch,stage,rows,history):
    torch.save({
        'epoch': epoch, 'stage': stage, 'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler else None,
        'val': rows.get('Failure-aware final', {}),
        'fdct_val': rows.get('FDCT baseline', {}),
        'all_rows': {k:v for k,v in rows.items() if not k.startswith('_')},
        'aux': rows.get('_aux', {}), 'history': history,
        'config': {
            'INPUT_REL_ALREADY_METRIC_ALIGNED': INPUT_REL_ALREADY_METRIC_ALIGNED,
            'RAW_UNCERT_INIT_B': RAW_UNCERT_INIT_B,
            'REL_UNCERT_INIT_B': REL_UNCERT_INIT_B,
            'W_INPUT_NLL': W_INPUT_NLL,
            'TRAIN_MICROBATCH': TRAIN_MICROBATCH,
            'VAL_MICROBATCH': VAL_MICROBATCH,
        },
    }, str(path))


def write_history(history):
    (OUT_DIR/'failure_aware_train_log.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
    if history:
        keys=sorted(set().union(*[h.keys() for h in history]));
        with open(OUT_DIR/'failure_aware_train_log.csv','w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(history)


def stage_for_epoch(epoch):
    if epoch<=STAGE1_EPOCHS: return 1
    if epoch<=STAGE1_EPOCHS+STAGE2_EPOCHS: return 2
    return 3


# =========================================================
# Main
# =========================================================
def main():
    set_seed(SEED)
    print('='*145); print('FDCT Failure-Aware Probabilistic Depth Completion v3 (calibrated, 8GB-safe)'); print('='*145)
    print(f'DEVICE={DEVICE}, AMP={USE_AMP}\nCACHE_ROOT={CACHE_ROOT}\nFDCT_ROOT={FDCT_ROOT}\nOUT_DIR={OUT_DIR}')
    fdct_mod=load_fdct_module()
    train_shards=load_split_shards(CACHE_ROOT,'train',MAX_TRAIN_SHARDS); val_shards=load_split_shards(CACHE_ROOT,'val',MAX_VAL_SHARDS)
    train_loader=DataLoader(CachedShardDataset(train_shards),batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=(DEVICE=='cuda'),collate_fn=ragged_shard_collate)
    val_loader=DataLoader(CachedShardDataset(val_shards),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=(DEVICE=='cuda'),collate_fn=ragged_shard_collate)
    first=to_device(next(iter(train_loader))); probe=build_inputs(first); print('[Probe]',{k:tuple(v.shape) for k,v in probe.items() if torch.is_tensor(v)})
    model=FailureAwareProbabilisticFDCT(fdct_mod).to(DEVICE); model.load_official_fdct(FDCT_CKPT); model.set_stage(3)
    # Build optimizer with every parameter that may be activated in any stage.
    new_params=[]; adapter_params=[]; head_params=[]
    for n,p in model.named_parameters():
        if n.startswith('fdct_ref.'): continue
        if n.startswith('fdct.'):
            if n.startswith('fdct.final.'): head_params.append(p)
        elif n.startswith('adapt_'): adapter_params.append(p)
        else: new_params.append(p)
    groups=[{'params':new_params,'lr':LR_NEW},{'params':adapter_params,'lr':LR_ADAPTER}]
    if UNFREEZE_FINAL_HEAD_IN_STAGE3: groups.append({'params':head_params,'lr':LR_FDCT_HEAD})
    optimizer=torch.optim.AdamW(groups,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(EPOCHS,1),eta_min=LR_NEW*.05)
    scaler=torch.cuda.amp.GradScaler(enabled=USE_AMP); history=[]; best=float('inf')
    clip_params=[p for g in optimizer.param_groups for p in g['params']]
    for epoch in range(1,EPOCHS+1):
        stage=stage_for_epoch(epoch); model.set_stage(stage); model.train(); stats_all=[]
        pbar=tqdm(train_loader,desc=f'Epoch {epoch}/{EPOCHS} | stage {stage}')
        for step, loader_batch in enumerate(pbar, 1):
            loader_batch = to_device(loader_batch)
            optimizer.zero_grad(set_to_none=True)
            micro_stats = []
            for batch, micro_n, total_n in iter_microbatches(loader_batch, TRAIN_MICROBATCH):
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    total, stats = compute_loss(model, batch, stage)
                    # Preserve the mean-loss semantics of the original concatenated batch.
                    scaled_total = total * (float(micro_n) / float(total_n))
                scaler.scale(scaled_total).backward()
                micro_stats.append(stats)
            scaler.unscale_(optimizer)
            active_clip = [p for p in clip_params if p.requires_grad and p.grad is not None]
            if active_clip:
                torch.nn.utils.clip_grad_norm_(active_clip, CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
            stats = avg_dicts(micro_stats)
            stats_all.append(stats)
            pbar.set_postfix(
                loss=f"{stats.get('loss_total', 0):.4f}",
                rm=f"{stats.get('rmse_mask', 0):.5f}",
                fa=f"{stats.get('fail_acc', 0):.3f}",
                ew=f"{stats.get('expert_w', 0):.2f}",
                mb=TRAIN_MICROBATCH,
            )
            del loader_batch, micro_stats
            if DEVICE == 'cuda' and EMPTY_CACHE_EVERY > 0 and step % EMPTY_CACHE_EVERY == 0:
                torch.cuda.empty_cache()
        scheduler.step(); train_loss=float(np.mean([s['loss_total'] for s in stats_all]))
        rows=evaluate(model,val_loader,stage); print_summary(epoch,stage,train_loss,rows)
        final=rows['Failure-aware final']; score=selection_score(final)
        input_rel = rows['Input relative prior']; metric_rel = rows['Metric-aligned prior']
        h={
            'epoch':epoch, 'stage':stage, 'train_loss':train_loss, 'score':score,
            **{f'final_{k}':v for k,v in final.items()},
            **{f'input_rel_{k}':v for k,v in input_rel.items()},
            **{f'metric_rel_{k}':v for k,v in metric_rel.items()},
            **{f'aux_{k}':v for k,v in rows['_aux'].items()},
        }; history.append(h); write_history(history)
        save_checkpoint(CKPT_DIR/'last.pth',model,optimizer,scaler,epoch,stage,rows,history)
        if stage >= 2 and score < best:
            best = score
            save_checkpoint(CKPT_DIR/'best_score.pth', model, optimizer, scaler, epoch, stage, rows, history)
            print(f'[Best] score={best:.6f}')
    print('Done. Best checkpoint:',CKPT_DIR/'best_score.pth')


if __name__=='__main__':
    main()
