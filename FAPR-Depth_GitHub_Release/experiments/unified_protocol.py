# -*- coding: utf-8 -*-
r"""
FAPR-Depth v6 direct unified-protocol benchmark v4
================================================

This script performs a direct, same-split, same-mask, same-metric comparison of:

    Raw Depth, DFNet, TODE-Trans, TDCNet, SwinDRNet, FDCT, ReMake,
    FAPR-Depth v6 (Ours)

Primary FAPR role
-----------------
The primary v6 model is fixed before evaluation as:

    checkpoints/best_candidate.pth -> Candidate benchmark

The script also reports v6 Safe and Legacy posterior as internal ablations.

Strict protocol
---------------
* All displayed rows are produced by direct inference in this run, or by a
  protocol-hash-matched direct cache created by this script.
* Existing DFNet/ReMake summary CSV files are never merged into the formal table.
* Every prediction is converted to metric depth in metres by its wrapper.
* Every method uses the same ordered cache shards and the same valid/mask tensors.
* Outside the transparent mask, raw sensor depth is preserved:
      D_eval = mask * D_pred + (1-mask) * D_raw
* Metrics are sample-mean: MAE, RMSE, REL, delta thresholds, boundary error,
  reliable-background disturbance, and the same v6 validation Score.

Formal run
----------
    python test_fapr_depth_v6_unified_protocol.py --max-shards 0 --require-all

Quick smoke test
----------------
    python test_fapr_depth_v6_unified_protocol.py --max-shards 10

DFNet/ReMake adapter contract
-----------------------------
The script auto-discovers local wrapper files.  Explicit paths are safer:

    --dfnet-wrapper path/to/dfnet_wrapper.py
    --remake-wrapper path/to/remake_wrapper.py

A wrapper should expose a loader. A standalone inference function is preferred; otherwise v3 uses a bound ``Inferencer.inference`` method:

    def load_remake_model(): ...
    def infer_remake(model, rgb, raw, rel=None, mask=None, use_bgr=False):
        # return HxW metric depth in metres
        ...

Recognised loader/inference names are defined in WRAPPER_SPECS below.
No automatic metric scaling is silently applied; any explicit --*-scale value is
recorded in the protocol manifest.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


# =============================================================================
# DEFAULT LOCAL CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(os.getenv("FAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent)))
TRAIN_SCRIPT = Path(os.getenv("FAPR_TRAIN_SCRIPT", str(PROJECT_ROOT / "train.py")))
CACHE_ROOT = Path(os.getenv("FAPR_CACHE_ROOT", str(PROJECT_ROOT / "data" / "cache")))
FAPR_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "fapr_depth_v6_safe_anchor"
    / "checkpoints"
    / "best_candidate.pth"
)
OUT_DIR = PROJECT_ROOT / "outputs" / "fapr_depth_v6_unified_protocol_test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 6248
LOADER_BATCH_SIZE = 1
NUM_WORKERS = 0
REQUESTED_MICROBATCH = 4

METHOD_ORDER = [
    "Raw Depth",
    "DFNet",
    "TODE-Trans",
    "TDCNet",
    "SwinDRNet",
    "FDCT",
    "ReMake",
    "FAPR-Depth v6 (Ours)",
]

EXTERNAL_METHODS = ["DFNet", "TODE-Trans", "TDCNet", "SwinDRNet", "ReMake"]

# Existing, previously verified wrapper locations.
TODE_WRAPPER = PROJECT_ROOT / "eval_tode_vs_oad_v4b.py"
TDCNET_WRAPPER = PROJECT_ROOT / "eval_tdcnet_vs_oad_v4b_diagnostic.py"
SWINDR_WRAPPER = PROJECT_ROOT / "eval_dreds_swindrnet_vs_oad_v4b_diagnostic.py"

TODE_ROOT = Path(os.getenv("TODE_ROOT", str(PROJECT_ROOT / "third_party" / "TODE-main")))
TDCNET_ROOT = Path(os.getenv("TDCNET_ROOT", str(PROJECT_ROOT / "third_party" / "TDCNet-main")))
SWINDR_ROOT = Path(os.getenv("SWINDR_ROOT", str(PROJECT_ROOT / "third_party" / "SwinDRNet")))

DFNET_WRAPPER_CANDIDATES = [
    PROJECT_ROOT / "eval_dfnet_vs_oad_v4b.py",
    PROJECT_ROOT / "eval_dfnet_vs_oad.py",
    PROJECT_ROOT / "eval_dfnet_unified.py",
    PROJECT_ROOT / "test_dfnet_unified.py",
    PROJECT_ROOT / "dfnet_unified_test.py",
]
REMAKE_WRAPPER_CANDIDATES = [
    PROJECT_ROOT / "eval_remake_vs_oad_v4b.py",
    PROJECT_ROOT / "eval_remake_vs_oad.py",
    PROJECT_ROOT / "eval_remake_unified.py",
    PROJECT_ROOT / "test_remake_unified.py",
    PROJECT_ROOT / "remake_unified_test.py",
]

DFNET_ROOT_CANDIDATES = [Path(os.getenv("DFNET_ROOT", str(PROJECT_ROOT / "third_party" / "DFNet-main")))]
REMAKE_ROOT_CANDIDATES = [Path(os.getenv("REMAKE_ROOT", str(PROJECT_ROOT / "third_party" / "ReMake-main")))]


@dataclass
class WrapperSpec:
    method: str
    wrapper: Optional[Path]
    repo_root: Optional[Path]
    loader_names: Tuple[str, ...]
    infer_names: Tuple[str, ...]
    scale: float = 1.0
    use_bgr: bool = False


WRAPPER_SPECS: Dict[str, Dict[str, Any]] = {
    "DFNet": {
        "wrapper": None,
        "root": None,
        "loaders": (
            "load_official_inferencer",
            "load_dfnet_inferencer",
            "load_dfnet_model",
            "load_transcg_model",
            "load_inferencer",
            "load_model",
        ),
        "infers": (
            "infer_dfnet",
            "run_dfnet_inference",
            "infer_transcg",
            "run_inference",
            "predict_depth",
            "infer",
            "predict",
        ),
    },
    "TODE-Trans": {
        "wrapper": TODE_WRAPPER,
        "root": TODE_ROOT,
        "loaders": ("load_tode_inferencer", "load_tode_model", "load_inferencer"),
        "infers": ("infer_tode", "infer_tode_trans", "run_tode_inference"),
    },
    "TDCNet": {
        "wrapper": TDCNET_WRAPPER,
        "root": TDCNET_ROOT,
        "loaders": (
            "load_tdcnet_inferencer",
            "load_tdc_inferencer",
            "load_tdcnet_model",
            "load_tdc_model",
            "load_inferencer",
        ),
        "infers": (
            "infer_tdcnet",
            "infer_tdc",
            "run_tdcnet_inference",
            "run_tdc_inference",
        ),
    },
    "SwinDRNet": {
        "wrapper": SWINDR_WRAPPER,
        "root": SWINDR_ROOT,
        "loaders": ("load_dreds_model", "load_swindrnet_model"),
        "infers": ("infer_dreds", "infer_swindrnet"),
    },
    "ReMake": {
        "wrapper": None,
        "root": None,
        "loaders": (
            "load_remake_inferencer",
            "load_remake_model",
            "load_inferencer",
            "load_model",
        ),
        "infers": (
            "infer_remake",
            "run_remake_inference",
            "run_inference",
            "predict_depth",
            "infer",
            "predict",
        ),
    },
}


# =============================================================================
# CLI / GENERAL UTILITIES
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct unified-protocol benchmark for FAPR-Depth v6 and baselines."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--train-script", type=Path, default=TRAIN_SCRIPT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=FAPR_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="0 evaluates the complete split; positive values are smoke tests only.",
    )
    parser.add_argument("--microbatch", type=int, default=REQUESTED_MICROBATCH)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--methods",
        type=str,
        default="all",
        help="Comma-separated external methods or all. FAPR/FDCT/Raw are always evaluated.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Fail unless every requested external model is directly evaluated.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse only direct caches whose protocol hash matches this run.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore direct caches.")
    parser.add_argument(
        "--resize-mode",
        choices=("nearest", "bilinear"),
        default="nearest",
        help="Resize external depth outputs to the cache resolution.",
    )
    parser.add_argument("--sanity-samples", type=int, default=64)

    parser.add_argument("--dfnet-wrapper", type=Path, default=None)
    parser.add_argument("--remake-wrapper", type=Path, default=None)
    parser.add_argument("--tode-wrapper", type=Path, default=TODE_WRAPPER)
    parser.add_argument("--tdcnet-wrapper", type=Path, default=TDCNET_WRAPPER)
    parser.add_argument("--swindr-wrapper", type=Path, default=SWINDR_WRAPPER)
    parser.add_argument("--dfnet-root", type=Path, default=None)
    parser.add_argument("--remake-root", type=Path, default=None)
    parser.add_argument("--tode-root", type=Path, default=TODE_ROOT)
    parser.add_argument("--tdcnet-root", type=Path, default=TDCNET_ROOT)
    parser.add_argument("--swindr-root", type=Path, default=SWINDR_ROOT)
    parser.add_argument("--dfnet-scale", type=float, default=1.0)
    parser.add_argument("--remake-scale", type=float, default=1.0)
    parser.add_argument("--tode-scale", type=float, default=1.0)
    parser.add_argument("--tdcnet-scale", type=float, default=1.0)
    parser.add_argument("--swindr-scale", type=float, default=1.0)
    parser.add_argument("--dfnet-bgr", action="store_true")
    parser.add_argument("--remake-bgr", action="store_true")
    parser.add_argument("--tode-bgr", action="store_true")
    parser.add_argument("--tdcnet-bgr", action="store_true")
    parser.add_argument("--swindr-bgr", action="store_true")
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def import_module_from_path(module_name: str, path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Python module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def finite_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def mean_rows(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    out: Dict[str, float] = {}
    for key in keys:
        vals = [finite_float(row.get(key)) for row in rows]
        vals = [x for x in vals if x is not None]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def selection_score(row: Mapping[str, float]) -> float:
    return float(
        row["mae_mask"]
        + row["rmse_mask"]
        + 0.5 * row["boundary"]
        + 0.15 * row["rmse_all"]
    )


def ordered_split_hash(shards: Sequence[Path], split: str) -> str:
    digest = hashlib.sha256()
    digest.update(split.encode("utf-8"))
    for path in shards:
        p = Path(path)
        digest.update(str(p).encode("utf-8"))
        try:
            stat = p.stat()
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
        except OSError:
            pass
    return digest.hexdigest()


def protocol_hash(
    split_hash: str,
    max_shards: int,
    resize_mode: str,
    method: str,
    wrapper: Optional[Path],
    scale: float,
    use_bgr: bool,
) -> str:
    payload = {
        "version": 4,
        "split_hash": split_hash,
        "max_shards": max_shards,
        "aggregation": "sample_mean",
        "fusion": "raw_preserving_transparent_mask",
        "metrics": "v6_metric_values",
        "resize_mode": resize_mode,
        "method": method,
        "wrapper": str(wrapper) if wrapper else None,
        "scale": scale,
        "use_bgr": use_bgr,
    }
    text = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def purge_external_imports() -> None:
    prefixes = (
        "inference",
        "models",
        "utils",
        "datasets",
        "config",
        "configs",
        "networks",
        "trainer",
        "module",
        "model",
        "options",
        "data_loader",
        # ReMake-specific namespace packages. They must not survive when the
        # next external repository is loaded.
        "run_utils",
        "run_tools",
        "relat_depth_models",
    )
    for name in list(sys.modules.keys()):
        if name in prefixes or any(name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[name]


def prepend_path(path: Optional[Path]) -> None:
    if path is None:
        return
    text = str(Path(path))
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def existing_first(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def _repo_marker_score(method: str, path: Path) -> int:
    """Score a directory as the actual import root of an external repository."""
    if not path.is_dir():
        return -1
    score = 0
    if method == "ReMake":
        markers = {
            "configs/inference/remake.yaml": 40,
            "inference.py": 16,
            "main.py": 12,
            "utils/loss_functions.py": 20,
            "utils/tools.py": 20,
            "remake.tar": 8,
        }
    elif method == "DFNet":
        markers = {
            "models": 8,
            "model": 6,
            "utils": 6,
            "configs": 4,
            "config": 4,
            "inference.py": 6,
            "main.py": 4,
        }
    else:
        markers = {}
    for relative, weight in markers.items():
        if (path / Path(relative)).exists():
            score += weight
    return score


def resolve_repo_root(method: str, root: Optional[Path]) -> Optional[Path]:
    """Resolve nested GitHub archive layouts without searching the whole disk."""
    if root is None:
        return None
    root = Path(root)
    if not root.exists():
        return root

    candidates: List[Path] = [root]
    frontier: List[Tuple[Path, int]] = [(root, 0)]
    seen = {str(root.resolve()).lower()}
    while frontier and len(candidates) < 80:
        current, depth = frontier.pop(0)
        if depth >= 2:
            continue
        try:
            children = sorted(
                [p for p in current.iterdir() if p.is_dir() and p.name != "__pycache__"],
                key=lambda p: p.name.lower(),
            )
        except OSError:
            continue
        for child in children:
            try:
                key = str(child.resolve()).lower()
            except OSError:
                key = str(child).lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(child)
            frontier.append((child, depth + 1))

    scored = [(_repo_marker_score(method, p), -len(p.parts), p) for p in candidates]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best = scored[0]
    if best_score > _repo_marker_score(method, root):
        print(f"[Repo root] {method}: resolved nested root {root} -> {best}")
        return best
    return root


def find_callable(module: Any, names: Sequence[str]) -> Tuple[str, Callable[..., Any]]:
    for name in names:
        value = getattr(module, name, None)
        if callable(value):
            return name, value
    raise AttributeError(
        f"No compatible callable in {module.__name__}; tried: {', '.join(names)}"
    )


def wrapper_text_score(path: Path, loaders: Sequence[str], infers: Sequence[str]) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return -1
    score = 0
    for name in loaders:
        if f"def {name}" in text:
            score += 10
    for name in infers:
        if f"def {name}" in text:
            score += 10
    lower = path.name.lower()
    if "eval" in lower:
        score += 2
    if "unified" in lower:
        score += 2
    if "compare" in lower or "summary" in lower:
        score -= 3
    return score


def discover_wrapper(
    project_root: Path,
    explicit: Optional[Path],
    fixed_candidates: Sequence[Path],
    keyword: str,
    loaders: Sequence[str],
    infers: Sequence[str],
) -> Optional[Path]:
    if explicit is not None:
        return explicit
    candidate = existing_first(fixed_candidates)
    if candidate is not None:
        return candidate

    matches: List[Tuple[int, Path]] = []
    try:
        for path in project_root.rglob(f"*{keyword.lower()}*.py"):
            lower_parts = {part.lower() for part in path.parts}
            if "outputs" in lower_parts or "__pycache__" in lower_parts:
                continue
            if path.name == Path(__file__).name:
                continue
            score = wrapper_text_score(path, loaders, infers)
            if score > 0:
                matches.append((score, path))
    except OSError:
        pass
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], str(item[1])))
    return matches[0][1]


def parse_requested_methods(text: str) -> List[str]:
    if text.strip().lower() in {"all", "*"}:
        return list(EXTERNAL_METHODS)
    aliases = {
        "dfnet": "DFNet",
        "tode": "TODE-Trans",
        "tode-trans": "TODE-Trans",
        "tdc": "TDCNet",
        "tdcnet": "TDCNet",
        "swin": "SwinDRNet",
        "swindrnet": "SwinDRNet",
        "dreds": "SwinDRNet",
        "remake": "ReMake",
    }
    out: List[str] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        method = aliases.get(token.lower(), token)
        if method not in EXTERNAL_METHODS:
            raise ValueError(f"Unsupported external method: {token}")
        if method not in out:
            out.append(method)
    return out


# =============================================================================
# METRICS
# =============================================================================
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    denom = mask.sum()
    if float(denom) <= 0:
        return float("nan")
    return float((x * mask).sum().item() / denom.item())


def metric_values(
    train_mod: Any,
    pred: torch.Tensor,
    raw: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
) -> Dict[str, float]:
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
    ratio = torch.maximum(
        pred / gt.clamp_min(float(train_mod.MIN_DEPTH)),
        gt / pred.clamp_min(float(train_mod.MIN_DEPTH)),
    )
    row = {
        "mae_all": masked_mean(abs_err, region_all),
        "rmse_all": math.sqrt(max(masked_mean(sq_err, region_all), 0.0)),
        "mae_mask": masked_mean(abs_err, region_mask),
        "rmse_mask": math.sqrt(max(masked_mean(sq_err, region_mask), 0.0)),
        "rel_mask": masked_mean(
            abs_err / gt.clamp_min(float(train_mod.MIN_DEPTH)), region_mask
        ),
        "delta_105": masked_mean((ratio < 1.05).float(), region_mask),
        "delta_110": masked_mean((ratio < 1.10).float(), region_mask),
        "delta_125": masked_mean((ratio < 1.25).float(), region_mask),
        "boundary": masked_mean(abs_err, boundary),
        "reliable_bg_disturbance": masked_mean(torch.abs(pred - raw), reliable_bg),
    }
    row["score"] = selection_score(row)
    return row


def add_metric_row(
    train_mod: Any,
    store: List[Dict[str, Any]],
    method: str,
    sample_index: int,
    pred: torch.Tensor,
    raw: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    valid: torch.Tensor,
    shard: str,
) -> None:
    row = metric_values(train_mod, pred, raw, gt, mask, valid)
    store.append({"method": method, "sample_index": sample_index, "shard": shard, **row})


# =============================================================================
# DATA / FAPR DIRECT EVALUATION
# =============================================================================
def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True).float()
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }


def slice_batch(batch: Dict[str, Any], start: int, end: int, n: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == n:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def make_loader(train_mod: Any, shards: Sequence[Path], num_workers: int) -> DataLoader:
    return DataLoader(
        train_mod.CachedShardDataset(shards),
        batch_size=LOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=DEVICE == "cuda",
        collate_fn=train_mod.ragged_shard_collate,
        persistent_workers=int(num_workers) > 0,
    )


def load_fapr_model(
    train_mod: Any, checkpoint: Path, device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, Any], str]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"FAPR checkpoint not found: {checkpoint}")
    base_mod = train_mod.load_base_source_module()
    model = train_mod.FailureAwarePosteriorDepth(base_mod).to(device)
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("model_state_dict", payload))
    clean = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"FAPR checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}\n"
            f"missing[:10]={missing[:10]}\nunexpected[:10]={unexpected[:10]}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phase = str(payload.get("phase", "joint"))
    if phase not in {"safe", "proposal", "risk", "joint"}:
        phase = "joint"
    return model, payload, phase


def evaluate_fapr_and_internal(
    train_mod: Any,
    shards: Sequence[Path],
    checkpoint: Path,
    out_dir: Path,
    microbatch: int,
    num_workers: int,
    use_amp: bool,
) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, Any]], Dict[str, Any]]:
    device = torch.device(DEVICE)
    model, payload, phase = load_fapr_model(train_mod, checkpoint, device)
    loader = make_loader(train_mod, shards, num_workers)
    per_sample: List[Dict[str, Any]] = []
    sample_index = 0
    start_time = time.time()

    with torch.inference_mode():
        progress = tqdm(loader, desc="Direct FAPR/FDCT", dynamic_ncols=True)
        for shard_index, cpu_batch in enumerate(progress):
            n = int(train_mod.batch_sample_count(cpu_batch))
            shard = str(shards[shard_index]) if shard_index < len(shards) else ""
            step = max(1, min(int(microbatch), n))
            for start in range(0, n, step):
                end = min(n, start + step)
                part = move_to_device(slice_batch(cpu_batch, start, end, n), device)
                inp = train_mod.build_inputs(part)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(inp, phase=phase, augment_safe=False)

                raw, gt, mask, valid = inp["raw"], inp["gt"], inp["mask"], inp["valid"]
                fdct = out["anchor_depth"] * mask + raw * (1.0 - mask)
                legacy = out["legacy_fused"] * mask + raw * (1.0 - mask)
                safe = out["safe_benchmark"]
                candidate = out["candidate_benchmark"]

                for i in range(int(raw.shape[0])):
                    idx = sample_index + i
                    add_metric_row(
                        train_mod, per_sample, "Raw Depth", idx,
                        raw[i:i+1], raw[i:i+1], gt[i:i+1], mask[i:i+1],
                        valid[i:i+1], shard,
                    )
                    add_metric_row(
                        train_mod, per_sample, "FDCT", idx,
                        fdct[i:i+1], raw[i:i+1], gt[i:i+1], mask[i:i+1],
                        valid[i:i+1], shard,
                    )
                    add_metric_row(
                        train_mod, per_sample, "FAPR legacy posterior", idx,
                        legacy[i:i+1], raw[i:i+1], gt[i:i+1], mask[i:i+1],
                        valid[i:i+1], shard,
                    )
                    add_metric_row(
                        train_mod, per_sample, "FAPR-Depth v6 Safe", idx,
                        safe[i:i+1], raw[i:i+1], gt[i:i+1], mask[i:i+1],
                        valid[i:i+1], shard,
                    )
                    add_metric_row(
                        train_mod, per_sample, "FAPR-Depth v6 (Ours)", idx,
                        candidate[i:i+1], raw[i:i+1], gt[i:i+1], mask[i:i+1],
                        valid[i:i+1], shard,
                    )
                sample_index += int(raw.shape[0])
                del part, inp, out, raw, gt, mask, valid
            progress.set_postfix(samples=sample_index)

    elapsed = time.time() - start_time
    summaries: Dict[str, Dict[str, float]] = {}
    for method in {
        "Raw Depth",
        "FDCT",
        "FAPR legacy posterior",
        "FAPR-Depth v6 Safe",
        "FAPR-Depth v6 (Ours)",
    }:
        summaries[method] = mean_rows(
            [{k: v for k, v in row.items() if k not in {"method", "sample_index", "shard"}}
             for row in per_sample if row["method"] == method]
        )

    meta = {
        "checkpoint": str(checkpoint),
        "checkpoint_phase": phase,
        "checkpoint_epoch": payload.get("refine_epoch"),
        "samples": sample_index,
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_index / max(elapsed, 1.0e-9),
        "primary_output": "best_candidate.pth / Candidate benchmark",
    }
    write_csv(out_dir / "per_sample_fapr_internal.csv", per_sample)
    write_json(out_dir / "fapr_checkpoint_metadata.json", meta)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summaries, per_sample, meta


# =============================================================================
# EXTERNAL WRAPPERS / INFERENCE
# =============================================================================
def build_wrapper_specs(args: argparse.Namespace) -> Dict[str, WrapperSpec]:
    project_root = args.project_root

    df_cfg = WRAPPER_SPECS["DFNet"]
    rm_cfg = WRAPPER_SPECS["ReMake"]
    df_wrapper = discover_wrapper(
        project_root,
        args.dfnet_wrapper,
        DFNET_WRAPPER_CANDIDATES,
        "dfnet",
        df_cfg["loaders"],
        df_cfg["infers"],
    )
    rm_wrapper = discover_wrapper(
        project_root,
        args.remake_wrapper,
        REMAKE_WRAPPER_CANDIDATES,
        "remake",
        rm_cfg["loaders"],
        rm_cfg["infers"],
    )
    df_root = resolve_repo_root(
        "DFNet", args.dfnet_root or existing_first(DFNET_ROOT_CANDIDATES)
    )
    rm_root = resolve_repo_root(
        "ReMake", args.remake_root or existing_first(REMAKE_ROOT_CANDIDATES)
    )

    overrides = {
        "DFNet": (df_wrapper, df_root, args.dfnet_scale, args.dfnet_bgr),
        "TODE-Trans": (
            args.tode_wrapper,
            args.tode_root,
            args.tode_scale,
            args.tode_bgr,
        ),
        "TDCNet": (
            args.tdcnet_wrapper,
            args.tdcnet_root,
            args.tdcnet_scale,
            args.tdcnet_bgr,
        ),
        "SwinDRNet": (
            args.swindr_wrapper,
            args.swindr_root,
            args.swindr_scale,
            args.swindr_bgr,
        ),
        "ReMake": (rm_wrapper, rm_root, args.remake_scale, args.remake_bgr),
    }

    specs: Dict[str, WrapperSpec] = {}
    for method, (wrapper, root, scale, use_bgr) in overrides.items():
        cfg = WRAPPER_SPECS[method]
        specs[method] = WrapperSpec(
            method=method,
            wrapper=wrapper,
            repo_root=root,
            loader_names=tuple(cfg["loaders"]),
            infer_names=tuple(cfg["infers"]),
            scale=float(scale),
            use_bgr=bool(use_bgr),
        )
    return specs


def extract_prediction(value: Any) -> Any:
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        priority = (
            "pred",
            "prediction",
            "depth",
            "pred_depth",
            "completed_depth",
            "output",
            "result",
            "final",
        )
        for key in priority:
            if key in value:
                try:
                    return extract_prediction(value[key])
                except (TypeError, ValueError):
                    pass
        for item in value.values():
            try:
                return extract_prediction(item)
            except (TypeError, ValueError):
                continue
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return extract_prediction(item)
            except (TypeError, ValueError):
                continue
    raise TypeError(f"Cannot extract a depth prediction from type {type(value)!r}")


def resize_prediction(
    prediction: Any,
    shape: Tuple[int, int],
    scale: float,
    resize_mode: str,
) -> np.ndarray:
    pred = extract_prediction(prediction)
    if torch.is_tensor(pred):
        pred = pred.detach().float().cpu().numpy()
    pred = np.asarray(pred, dtype=np.float32)
    pred = np.squeeze(pred)

    if pred.ndim == 3:
        if pred.shape[-1] == 1:
            pred = pred[..., 0]
        elif pred.shape[0] == 1:
            pred = pred[0]
        elif pred.shape[-1] == 3:
            pred = pred.mean(axis=-1)
        elif pred.shape[0] == 3:
            pred = pred.mean(axis=0)
    if pred.ndim != 2:
        raise RuntimeError(f"Cannot interpret prediction shape: {pred.shape}")

    height, width = shape
    if pred.shape != (height, width):
        interpolation = (
            cv2.INTER_NEAREST if resize_mode == "nearest" else cv2.INTER_LINEAR
        )
        pred = cv2.resize(pred, (width, height), interpolation=interpolation)

    pred = pred.astype(np.float32) * float(scale)
    pred[~np.isfinite(pred)] = 0.0
    return pred


def rgb_raw_rel_mask_to_numpy(
    rgb_t: torch.Tensor,
    raw_t: torch.Tensor,
    rel_t: torch.Tensor,
    mask_t: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb_t.detach().float().cpu().numpy()
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    raw = np.squeeze(raw_t.detach().float().cpu().numpy()).astype(np.float32)
    rel = np.squeeze(rel_t.detach().float().cpu().numpy()).astype(np.float32)
    mask = np.squeeze(mask_t.detach().float().cpu().numpy()).astype(np.float32)
    return rgb, raw, rel, mask


def call_external_infer(
    method: str,
    infer_fn: Callable[..., Any],
    model: Any,
    rgb: np.ndarray,
    raw: np.ndarray,
    rel: np.ndarray,
    mask: np.ndarray,
    use_bgr: bool,
) -> Any:
    rgb_arg = rgb[..., ::-1].copy() if use_bgr else rgb
    kw_payload = {
        "model": model,
        "rgb": rgb_arg,
        "raw": raw,
        "depth": raw,
        "rel": rel,
        "relative_depth": rel,
        "mask": mask,
        "use_bgr": use_bgr,
    }

    attempts: List[Callable[[], Any]] = []
    try:
        sig = inspect.signature(infer_fn)
        accepted = {
            name: value
            for name, value in kw_payload.items()
            if name in sig.parameters
        }
        if accepted:
            attempts.append(lambda accepted=accepted: infer_fn(**accepted))
    except (TypeError, ValueError):
        pass

    if method == "ReMake":
        attempts.extend(
            [
                lambda: infer_fn(model, rgb_arg, raw, rel, mask, use_bgr=use_bgr),
                lambda: infer_fn(model, rgb_arg, raw, mask, rel, use_bgr=use_bgr),
                lambda: infer_fn(model, rgb_arg, raw, rel, mask),
                lambda: infer_fn(model, rgb_arg, raw, mask, rel),
                lambda: infer_fn(model, rgb_arg, raw, mask),
                lambda: infer_fn(model, rgb_arg, raw),
            ]
        )
    elif method in {"TODE-Trans", "TDCNet", "DFNet"}:
        attempts.extend(
            [
                lambda: infer_fn(model, rgb_arg, raw, mask, use_bgr=use_bgr),
                lambda: infer_fn(model, rgb_arg, raw, use_bgr=use_bgr),
                lambda: infer_fn(model, rgb_arg, raw, mask),
                lambda: infer_fn(model, rgb_arg, raw),
            ]
        )
    else:
        attempts.extend(
            [
                lambda: infer_fn(model, rgb_arg, raw),
                lambda: infer_fn(model, rgb_arg, raw, mask),
            ]
        )

    attempts.extend(
        [
            lambda: infer_fn(rgb_arg, raw, rel, mask),
            lambda: infer_fn(rgb_arg, raw, mask),
            lambda: infer_fn(rgb_arg, raw),
            lambda: infer_fn(
                model,
                {"rgb": rgb_arg, "raw": raw, "rel": rel, "mask": mask},
            ),
        ]
    )

    errors: List[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(str(exc))
    raise RuntimeError(
        f"No compatible inference signature for {method}. Last errors: {errors[-4:]}"
    )




def _literal_ast_value(node: ast.AST, default: Any = None) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return default


def argparse_defaults_from_source(path: Path) -> Dict[str, Any]:
    """Extract argparse defaults without importing/executing a repository entry point."""
    defaults: Dict[str, Any] = {}
    source = safe_read_text(path)
    if not source:
        return defaults
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return defaults

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "add_argument"
        ):
            continue

        option_strings: List[str] = []
        for arg in node.args:
            value = _literal_ast_value(arg)
            if isinstance(value, str):
                option_strings.append(value)

        kwargs = {
            kw.arg: kw.value
            for kw in node.keywords
            if kw.arg is not None
        }

        dest = _literal_ast_value(kwargs["dest"]) if "dest" in kwargs else None
        if not isinstance(dest, str):
            long_options = [s for s in option_strings if s.startswith("--")]
            selected = long_options[-1] if long_options else (
                option_strings[0] if option_strings else None
            )
            if selected is None:
                continue
            dest = selected.lstrip("-").replace("-", "_")

        if "default" in kwargs:
            value = _literal_ast_value(kwargs["default"])
        else:
            action = _literal_ast_value(kwargs["action"]) if "action" in kwargs else None
            if action == "store_true":
                value = False
            elif action == "store_false":
                value = True
            else:
                value = None
        defaults[dest] = value
    return defaults


def safe_read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return ""
    return ""


def load_remake_inferencer_direct(
    resolved_root: Path,
) -> Tuple[Any, Dict[str, Any]]:
    """Instantiate the repository's real run_utils.Inferencer.

    The existing OAD ReMake wrapper passes a string to Inferencer, but the
    repository implementation expects an argparse-like object with ``cfg``.
    Build that object from main.py defaults and apply explicit inference paths.
    """
    root = Path(resolved_root)
    cfg_path = root / "configs" / "inference" / "remake.yaml"
    checkpoint_path = root / "remake.tar"
    inferencer_path = root / "run_utils" / "inferencer.py"

    missing = [
        str(path)
        for path in (cfg_path, checkpoint_path, inferencer_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "ReMake direct constructor is missing required files: "
            + ", ".join(missing)
        )

    defaults = argparse_defaults_from_source(root / "main.py")
    # Required and common aliases. Extra attributes are harmless and make this
    # robust to small repository revisions.
    defaults.update(
        {
            "cfg": str(cfg_path),
            "config": str(cfg_path),
            "config_path": str(cfg_path),
            "checkpoints": str(checkpoint_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
            "mode": "inference",
            "phase": "inference",
            "local_rank": 0,
            "rank": 0,
            "resume": False,
        }
    )
    args = SimpleNamespace(**defaults)

    module = importlib.import_module("run_utils.inferencer")
    inferencer_cls = getattr(module, "Inferencer")
    loaded = inferencer_cls(args)
    metadata = {
        "constructor": "run_utils.inferencer.Inferencer(args_namespace)",
        "cfg": str(cfg_path),
        "checkpoint": str(checkpoint_path),
        "namespace_keys": sorted(vars(args).keys()),
    }
    print(
        "[ReMake direct loader] "
        f"Inferencer(args), cfg={cfg_path}, checkpoint={checkpoint_path}"
    )
    return loaded, metadata


def install_repository_compatibility_aliases(
    method: str,
    resolved_root: Optional[Path],
) -> Dict[str, str]:
    """Install narrowly-scoped aliases required by original repository wrappers.

    ReMake does not expose ``inference.py`` at repository root. Its actual
    Inferencer is ``run_utils/inferencer.py``, while the existing OAD wrapper
    imports ``Inferencer`` from a top-level module named ``inference``.  Creating
    this alias preserves the wrapper's preprocessing and inference code without
    modifying the external repository.
    """
    aliases: Dict[str, str] = {}
    if method != "ReMake" or resolved_root is None:
        return aliases

    candidate = resolved_root / "run_utils" / "inferencer.py"
    if not candidate.exists():
        return aliases

    # Import with the real package name first so package context is retained,
    # then expose the same module under the legacy top-level name expected by
    # the OAD wrapper.
    try:
        module = importlib.import_module("run_utils.inferencer")
    except Exception:
        module = import_module_from_path("fapr_remake_run_utils_inferencer", candidate)

    if not hasattr(module, "Inferencer"):
        raise AttributeError(
            f"ReMake compatibility module has no Inferencer: {candidate}"
        )
    sys.modules["inference"] = module
    aliases["inference"] = str(candidate)
    print(f"[Compat] ReMake: inference -> {candidate}")
    return aliases


def find_loaded_inference_callable(
    loaded: Any,
    names: Sequence[str] = (
        "inference",
        "infer",
        "infer_dfnet",
        "infer_remake",
        "predict_depth",
        "predict",
        "run_inference",
    ),
) -> Tuple[str, Callable[..., Any]]:
    """Find a bound inference method when a wrapper exposes only a loader.

    The official DFNet wrapper in this project exposes
    ``load_official_inferencer`` but no standalone per-sample inference
    function. The loaded object is the repository's ``Inferencer`` and provides
    ``inference(rgb, depth, ...)``.
    """
    candidates: List[Tuple[str, Any]] = [("loaded", loaded)]
    if isinstance(loaded, Mapping):
        candidates.extend((f"loaded[{key!r}]", value) for key, value in loaded.items())
    elif isinstance(loaded, (tuple, list)):
        candidates.extend((f"loaded[{index}]", value) for index, value in enumerate(loaded))

    for prefix, obj in candidates:
        if obj is None:
            continue
        for name in names:
            value = getattr(obj, name, None)
            if callable(value):
                return f"{prefix}.{name}", value
        if callable(obj):
            return prefix, obj

    raise AttributeError(
        "The wrapper has no recognised inference function and the loaded object "
        f"has no bound method among: {', '.join(names)}"
    )


def load_external_adapter(spec: WrapperSpec) -> Tuple[Any, Callable[..., Any], Dict[str, Any]]:
    if spec.wrapper is None:
        raise FileNotFoundError(
            f"No wrapper discovered for {spec.method}. Pass --{spec.method.lower().replace('-', '')}-wrapper."
        )
    if not spec.wrapper.exists():
        raise FileNotFoundError(f"{spec.method} wrapper not found: {spec.wrapper}")

    resolved_root = resolve_repo_root(spec.method, spec.repo_root)
    spec.repo_root = resolved_root

    # External repositories commonly expose top-level packages named `utils`,
    # `models`, or `datasets`. Purge stale packages and put the actual repository
    # root before the OAD project to avoid namespace shadowing.
    purge_external_imports()
    importlib.invalidate_caches()
    prepend_path(spec.wrapper.parent)
    prepend_path(resolved_root)

    module_name = "fapr_v6_unified_" + spec.method.lower().replace("-", "_")
    old_cwd = Path.cwd()
    compatibility_aliases: Dict[str, str] = {}
    try:
        if resolved_root is not None and resolved_root.exists():
            os.chdir(str(resolved_root))

        # ReMake's true Inferencer lives at run_utils/inferencer.py rather than
        # repository_root/inference.py. Install the exact alias expected by the
        # existing diagnostic wrapper before that wrapper is imported/called.
        compatibility_aliases = install_repository_compatibility_aliases(
            spec.method, resolved_root
        )

        module = import_module_from_path(module_name, spec.wrapper)
        loader_name, loader = find_callable(module, spec.loader_names)

        print(f"[Load] {spec.method}: wrapper={spec.wrapper}")
        if resolved_root:
            print(f"[Load] {spec.method}: root={resolved_root}")
            print(f"[Load] {spec.method}: sys.path[0]={sys.path[0]}")

        direct_loader_meta: Dict[str, Any] = {}
        try:
            loaded = loader()
        except Exception as loader_exc:
            if spec.method != "ReMake" or resolved_root is None:
                raise
            print(
                "[ReMake wrapper loader fallback] "
                f"{loader_name} failed with {loader_exc!r}; "
                "using the repository's real Inferencer(args) constructor."
            )
            loaded, direct_loader_meta = load_remake_inferencer_direct(
                resolved_root
            )

        # Prefer the wrapper's own per-sample adapter because it contains the
        # repository-specific preprocessing and output conversion.  If absent,
        # fall back to a bound method on the loaded official Inferencer.
        try:
            infer_name, infer_fn = find_callable(module, spec.infer_names)
            infer_source = "wrapper"
        except AttributeError:
            infer_name, infer_fn = find_loaded_inference_callable(loaded)
            infer_source = "loaded_object"
            print(
                f"[Adapter fallback] {spec.method}: using bound method "
                f"{infer_name}"
            )
    except Exception as exc:
        os.chdir(str(old_cwd))
        if spec.method == "ReMake":
            raise RuntimeError(
                "ReMake import/load failed after repository-root isolation and "
                "run_utils.inferencer compatibility aliasing. "
                f"resolved_root={resolved_root}; wrapper={spec.wrapper}; "
                f"aliases={compatibility_aliases}; original_error={exc!r}."
            ) from exc
        raise

    # Keep the repository working directory active during this model's direct
    # inference because some original repositories resolve assets lazily.
    model = loaded
    if hasattr(model, "eval"):
        model.eval()
    metadata = {
        "wrapper": str(spec.wrapper),
        "repo_root": str(resolved_root) if resolved_root else None,
        "loader_function": loader_name,
        "infer_function": infer_name,
        "infer_source": infer_source,
        "compatibility_aliases": compatibility_aliases,
        "direct_loader": direct_loader_meta,
        "scale": spec.scale,
        "use_bgr": spec.use_bgr,
        "working_directory": str(Path.cwd()),
    }
    return model, infer_fn, metadata


class PredictionSanity:
    def __init__(self, limit: int):
        self.limit = max(0, int(limit))
        self.rows: List[Dict[str, float]] = []

    def update(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        mask: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        if len(self.rows) >= self.limit:
            return
        region = (mask > 0.5) & (valid > 0.5) & np.isfinite(gt) & (gt > 1.0e-6)
        if not np.any(region):
            return
        p = pred[region]
        g = gt[region]
        finite = np.isfinite(p)
        positive = p > 0
        p_safe = np.where(finite, p, 0.0)
        ratio = p_safe / np.maximum(g, 1.0e-6)
        self.rows.append(
            {
                "finite_ratio": float(finite.mean()),
                "positive_ratio": float(positive.mean()),
                "pred_median": float(np.median(p_safe)),
                "gt_median": float(np.median(g)),
                "median_pred_over_gt": float(np.median(ratio)),
                "median_pred_minus_gt": float(np.median(p_safe - g)),
                "pred_p01": float(np.quantile(p_safe, 0.01)),
                "pred_p99": float(np.quantile(p_safe, 0.99)),
            }
        )

    def summary(self) -> Dict[str, Any]:
        row = mean_rows(self.rows)
        warnings: List[str] = []
        ratio = row.get("median_pred_over_gt")
        finite = row.get("finite_ratio")
        positive = row.get("positive_ratio")
        if ratio is not None and not (0.2 <= ratio <= 5.0):
            warnings.append(
                f"median pred/GT ratio {ratio:.4f} suggests a possible unit/scale mismatch"
            )
        if finite is not None and finite < 0.99:
            warnings.append(f"finite prediction ratio is only {finite:.4f}")
        if positive is not None and positive < 0.90:
            warnings.append(f"positive prediction ratio is only {positive:.4f}")
        return {"samples": len(self.rows), **row, "warnings": warnings}


def evaluate_external_method(
    train_mod: Any,
    shards: Sequence[Path],
    spec: WrapperSpec,
    args: argparse.Namespace,
    split_hash: str,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
    method_dir = args.out_dir / "direct_models" / spec.method.lower().replace("-", "_")
    method_dir.mkdir(parents=True, exist_ok=True)
    p_hash = protocol_hash(
        split_hash,
        args.max_shards,
        args.resize_mode,
        spec.method,
        spec.wrapper,
        spec.scale,
        spec.use_bgr,
    )
    cache_csv = method_dir / "per_sample_metrics.csv"
    cache_meta = method_dir / "metadata.json"

    if args.reuse_cache and not args.force and cache_csv.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if meta.get("protocol_hash") == p_hash:
                rows = read_csv(cache_csv)
                metric_rows = [
                    {
                        key: float(value)
                        for key, value in row.items()
                        if key not in {"method", "sample_index", "shard"}
                        and finite_float(value) is not None
                    }
                    for row in rows
                ]
                summary = mean_rows(metric_rows)
                print(f"[Direct cache] {spec.method}: {cache_csv}")
                return summary, rows, meta
        except Exception:
            pass

    model, infer_fn, adapter_meta = load_external_adapter(spec)
    loader = make_loader(train_mod, shards, args.num_workers)
    per_sample: List[Dict[str, Any]] = []
    sanity = PredictionSanity(args.sanity_samples)
    sample_index = 0
    start_time = time.time()

    with torch.inference_mode():
        progress = tqdm(loader, desc=f"Direct {spec.method}", dynamic_ncols=True)
        for shard_index, cpu_batch in enumerate(progress):
            n = int(train_mod.batch_sample_count(cpu_batch))
            shard = str(shards[shard_index]) if shard_index < len(shards) else ""
            inp = train_mod.build_inputs(cpu_batch)
            rgb_t, raw_t, rel_t = inp["rgb"], inp["raw"], inp["rel"]
            gt_t, mask_t, valid_t = inp["gt"], inp["mask"], inp["valid"]

            for i in range(n):
                rgb, raw, rel, mask = rgb_raw_rel_mask_to_numpy(
                    rgb_t[i], raw_t[i], rel_t[i], mask_t[i]
                )
                prediction = call_external_infer(
                    spec.method,
                    infer_fn,
                    model,
                    rgb,
                    raw,
                    rel,
                    mask,
                    spec.use_bgr,
                )
                pred = resize_prediction(
                    prediction,
                    raw.shape,
                    spec.scale,
                    args.resize_mode,
                )
                fused = raw.copy()
                fused[mask > 0.5] = pred[mask > 0.5]

                gt_np = np.squeeze(gt_t[i].detach().float().cpu().numpy())
                valid_np = np.squeeze(valid_t[i].detach().float().cpu().numpy())
                sanity.update(pred, gt_np, mask, valid_np)

                pred_tensor = torch.from_numpy(fused)[None, None]
                row = metric_values(
                    train_mod,
                    pred_tensor,
                    raw_t[i:i+1],
                    gt_t[i:i+1],
                    mask_t[i:i+1],
                    valid_t[i:i+1],
                )
                per_sample.append(
                    {
                        "method": spec.method,
                        "sample_index": sample_index,
                        "shard": shard,
                        **row,
                    }
                )
                sample_index += 1
            progress.set_postfix(samples=sample_index)

    elapsed = time.time() - start_time
    summary = mean_rows(
        [
            {k: v for k, v in row.items() if k not in {"method", "sample_index", "shard"}}
            for row in per_sample
        ]
    )
    sanity_summary = sanity.summary()
    meta = {
        "method": spec.method,
        "protocol_hash": p_hash,
        "split_hash": split_hash,
        "split": args.split,
        "max_shards": args.max_shards,
        "shards": len(shards),
        "samples": sample_index,
        "elapsed_seconds": elapsed,
        "samples_per_second": sample_index / max(elapsed, 1.0e-9),
        "aggregation": "sample-mean",
        "fusion": "prediction inside transparent mask; raw depth outside mask",
        "prediction_units": "metres after explicit scale",
        "resize_mode": args.resize_mode,
        "adapter": adapter_meta,
        "sanity": sanity_summary,
    }
    write_csv(cache_csv, per_sample)
    write_json(cache_meta, meta)

    del model, infer_fn
    purge_external_imports()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, per_sample, meta


# =============================================================================
# REPORTING
# =============================================================================
def fmt(value: Any, digits: int = 4, percentage: bool = False) -> str:
    x = finite_float(value)
    if x is None:
        return "-"
    if percentage:
        x *= 100.0
        return f"{x:.2f}"
    return f"{x:.{digits}f}"


def print_table(rows: Sequence[Mapping[str, Any]], title: str) -> None:
    print("\n" + "=" * 150)
    print(title)
    print("=" * 150)
    print(
        f"{'Method':30s} | {'RMSE':>9s} | {'REL':>9s} | {'MAE':>9s} | "
        f"{'d1.05(%)':>9s} | {'d1.10(%)':>9s} | {'d1.25(%)':>9s} | {'Score':>9s}"
    )
    print("-" * 150)
    for row in rows:
        print(
            f"{str(row['method']):30s} | "
            f"{fmt(row.get('rmse_mask')):>9s} | "
            f"{fmt(row.get('rel_mask')):>9s} | "
            f"{fmt(row.get('mae_mask')):>9s} | "
            f"{fmt(row.get('delta_105'), percentage=True):>9s} | "
            f"{fmt(row.get('delta_110'), percentage=True):>9s} | "
            f"{fmt(row.get('delta_125'), percentage=True):>9s} | "
            f"{fmt(row.get('score')):>9s}"
        )
    print("=" * 150)


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Method | RMSE | REL | MAE | δ1.05 (%) | δ1.10 (%) | δ1.25 (%) | Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {rmse} | {rel} | {mae} | {d105} | {d110} | {d125} | {score} |".format(
                method=row["method"],
                rmse=fmt(row.get("rmse_mask")),
                rel=fmt(row.get("rel_mask")),
                mae=fmt(row.get("mae_mask")),
                d105=fmt(row.get("delta_105"), percentage=True),
                d110=fmt(row.get("delta_110"), percentage=True),
                d125=fmt(row.get("delta_125"), percentage=True),
                score=fmt(row.get("score")),
            )
        )
    return "\n".join(lines) + "\n"


def latex_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & RMSE & REL & MAE & $\delta_{1.05}$ & $\delta_{1.10}$ & $\delta_{1.25}$ & Score \\",
        r"\midrule",
    ]
    for row in rows:
        name = str(row["method"]).replace("_", r"\_").replace("&", r"\&")
        lines.append(
            f"{name} & {fmt(row.get('rmse_mask'))} & {fmt(row.get('rel_mask'))} & "
            f"{fmt(row.get('mae_mask'))} & {fmt(row.get('delta_105'), percentage=True)} & "
            f"{fmt(row.get('delta_110'), percentage=True)} & "
            f"{fmt(row.get('delta_125'), percentage=True)} & {fmt(row.get('score'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def gap_rows(
    rows: Sequence[Mapping[str, Any]],
    reference_method: str,
) -> List[Dict[str, Any]]:
    reference = next((row for row in rows if row["method"] == reference_method), None)
    if reference is None:
        return []
    out: List[Dict[str, Any]] = []
    metrics = (
        "rmse_mask",
        "rel_mask",
        "mae_mask",
        "delta_105",
        "delta_110",
        "delta_125",
        "score",
    )
    for row in rows:
        item: Dict[str, Any] = {"method": row["method"], "reference": reference_method}
        for metric in metrics:
            a = finite_float(row.get(metric))
            b = finite_float(reference.get(metric))
            if a is None or b is None:
                continue
            item[f"{metric}_minus_reference"] = a - b
            if abs(b) > 1.0e-12:
                item[f"{metric}_relative_percent"] = 100.0 * (a - b) / abs(b)
        out.append(item)
    return out


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    device = torch.device(DEVICE)
    use_amp = device.type == "cuda" and not args.no_amp

    if not args.train_script.exists():
        raise FileNotFoundError(f"v6 training script not found: {args.train_script}")
    train_mod = import_module_from_path("fapr_v6_unified_train_definition", args.train_script)
    train_mod.DEVICE = DEVICE
    train_mod.USE_AMP = use_amp
    train_mod.CACHE_ROOT = args.cache_root

    max_n = None if int(args.max_shards) <= 0 else int(args.max_shards)
    shards = train_mod.load_split_shards(args.cache_root, args.split, max_n)
    if not shards:
        raise RuntimeError(f"No shards found for split={args.split} under {args.cache_root}")
    split_hash = ordered_split_hash(shards, args.split)
    requested = parse_requested_methods(args.methods)
    specs = build_wrapper_specs(args)

    print("=" * 170)
    print("FAPR-Depth v6 DIRECT unified-protocol benchmark")
    print("=" * 170)
    print(f"DEVICE={DEVICE}, AMP={use_amp}")
    print(f"split={args.split}, shards={len(shards)}, max_shards={args.max_shards}")
    print(f"split_hash={split_hash}")
    print(f"checkpoint={args.checkpoint}")
    print("Primary model is fixed: best_candidate.pth -> Candidate benchmark")
    print("No external summary CSV will be merged.")
    if args.max_shards > 0:
        print("[WARNING] This is a subset smoke test, not a formal full-split comparison.")

    summaries, _, fapr_meta = evaluate_fapr_and_internal(
        train_mod,
        shards,
        args.checkpoint,
        args.out_dir,
        args.microbatch,
        args.num_workers,
        use_amp,
    )

    external_metadata: Dict[str, Any] = {}
    failures: Dict[str, str] = {}
    for method in requested:
        spec = specs[method]
        try:
            summary, _, metadata = evaluate_external_method(
                train_mod, shards, spec, args, split_hash
            )
            summaries[method] = summary
            external_metadata[method] = metadata
            for warning in metadata.get("sanity", {}).get("warnings", []):
                print(f"[SANITY WARNING] {method}: {warning}")
        except Exception as exc:
            failures[method] = repr(exc)
            print(f"[DIRECT FAILURE] {method}: {exc!r}")
            print(
                f"  Wrapper contract: load function in {spec.loader_names}; "
                f"inference function in {spec.infer_names}; return metric depth in metres."
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    missing = [method for method in requested if method not in summaries]
    if missing and args.require_all:
        write_json(args.out_dir / "direct_failures.json", failures)
        raise RuntimeError(
            "Formal comparison is incomplete. Missing direct models: "
            + ", ".join(missing)
            + f"\nDetails: {args.out_dir / 'direct_failures.json'}"
        )

    main_rows: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        if method in summaries:
            main_rows.append({"method": method, **summaries[method]})
    internal_order = [
        "Raw Depth",
        "FDCT",
        "FAPR legacy posterior",
        "FAPR-Depth v6 Safe",
        "FAPR-Depth v6 (Ours)",
    ]
    internal_rows = [
        {"method": method, **summaries[method]}
        for method in internal_order
        if method in summaries
    ]

    print_table(
        main_rows,
        f"UNIFIED DIRECT COMPARISON | split={args.split} | "
        f"shards={len(shards)} | samples={fapr_meta['samples']}",
    )

    write_csv(args.out_dir / "unified_main_comparison.csv", main_rows)
    write_json(args.out_dir / "unified_main_comparison.json", main_rows)
    (args.out_dir / "unified_main_comparison.md").write_text(
        markdown_table(main_rows), encoding="utf-8"
    )
    (args.out_dir / "unified_main_comparison.tex").write_text(
        latex_table(main_rows), encoding="utf-8"
    )
    write_csv(args.out_dir / "fapr_v6_internal_ablation.csv", internal_rows)
    write_csv(
        args.out_dir / "gaps_vs_fapr_v6.csv",
        gap_rows(main_rows, "FAPR-Depth v6 (Ours)"),
    )
    write_csv(
        args.out_dir / "gaps_vs_remake.csv",
        gap_rows(main_rows, "ReMake"),
    )

    manifest = {
        "protocol_version": 1,
        "created_unix": time.time(),
        "project_root": str(args.project_root),
        "cache_root": str(args.cache_root),
        "split": args.split,
        "max_shards": args.max_shards,
        "ordered_shard_count": len(shards),
        "ordered_split_hash": split_hash,
        "fapr_checkpoint": str(args.checkpoint),
        "fapr_primary": "best_candidate.pth / Candidate benchmark",
        "aggregation": "sample-mean",
        "metric_region": "valid transparent-mask pixels",
        "boundary_region": "v6 build_boundary_ring(mask) intersect valid",
        "fusion": "prediction inside mask; raw sensor depth outside mask",
        "depth_units": "metres",
        "resize_mode": args.resize_mode,
        "summary_csv_fallback": False,
        "requested_external_methods": requested,
        "completed_external_methods": [
            method for method in requested if method in summaries
        ],
        "missing_external_methods": missing,
        "failures": failures,
        "fapr_metadata": fapr_meta,
        "external_metadata": external_metadata,
    }
    write_json(args.out_dir / "protocol_manifest.json", manifest)
    write_json(args.out_dir / "direct_failures.json", failures)

    report_lines = [
        "FAPR-Depth v6 unified direct-protocol audit",
        "=" * 72,
        f"split={args.split}",
        f"shards={len(shards)}",
        f"split_hash={split_hash}",
        f"FAPR primary={args.checkpoint} / Candidate benchmark",
        "aggregation=sample-mean",
        "fusion=mask * prediction + (1-mask) * raw",
        "external summary fallback=False",
        "",
        "Completed methods: " + ", ".join(row["method"] for row in main_rows),
        "Missing requested external methods: " + (", ".join(missing) if missing else "none"),
        "",
        "Formal status: "
        + (
            "FORMAL COMPLETE"
            if args.max_shards <= 0 and not missing
            else "INCOMPLETE/SMOKE TEST"
        ),
    ]
    if failures:
        report_lines += ["", "Failures:"]
        report_lines += [f"- {method}: {error}" for method, error in failures.items()]
    (args.out_dir / "protocol_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(f"[Saved] {args.out_dir / 'unified_main_comparison.csv'}")
    print(f"[Protocol] {args.out_dir / 'protocol_manifest.json'}")
    if missing:
        print("[Incomplete] Missing direct methods:", ", ".join(missing))
    elif args.max_shards <= 0:
        print("[Formal complete] Every requested method was directly evaluated.")
    else:
        print("[Smoke test complete] Re-run with --max-shards 0 for formal results.")


if __name__ == "__main__":
    main()
