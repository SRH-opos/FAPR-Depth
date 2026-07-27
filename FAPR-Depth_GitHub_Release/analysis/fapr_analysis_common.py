#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared utilities for FAPR-Depth v6 diagnostic analyses.

Place this file in the same directory as the analysis scripts.  The scripts
dynamically import the original v6 training implementation, load the frozen
checkpoint, and evaluate the cached test shards without retraining.

The paper-facing labels deliberately use "Backbone Baseline" and do not expose
the implementation name of the external backbone.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


DEFAULT_PROJECT_ROOT = Path(os.getenv("FAPR_PROJECT_ROOT", str(Path(__file__).resolve().parents[1])))
DEFAULT_TRAIN_SCRIPT = Path(os.getenv("FAPR_TRAIN_SCRIPT", str(DEFAULT_PROJECT_ROOT / "train.py")))
DEFAULT_CACHE_ROOT = Path(os.getenv("FAPR_CACHE_ROOT", str(DEFAULT_PROJECT_ROOT / "data" / "cache")))
DEFAULT_BASE_SOURCE_ROOT = Path(os.getenv("FAPR_BASE_SOURCE_ROOT", str(DEFAULT_PROJECT_ROOT / "third_party" / "FDCT-main")))
DEFAULT_CHECKPOINT = Path(os.getenv("FAPR_CHECKPOINT", str(DEFAULT_PROJECT_ROOT / "weights" / "best_candidate.pth")))
DEFAULT_ANALYSIS_ROOT = Path(os.getenv("FAPR_ANALYSIS_ROOT", str(DEFAULT_PROJECT_ROOT / "outputs" / "analysis")))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def import_module_from_path(name: str, path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Python source not found: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def add_common_args(
    parser: argparse.ArgumentParser,
    analysis_name: str,
    default_phase: str = "joint",
) -> None:
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--base-source-root", type=Path, default=DEFAULT_BASE_SOURCE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_ROOT / analysis_name,
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-shards", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument(
        "--phase",
        type=str,
        default=default_phase,
        choices=["auto", "safe", "proposal", "risk", "joint"],
        help=(
            "Use 'joint' for diagnostics that require both risk heads. "
            "'auto' uses the phase stored in the checkpoint."
        ),
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=6248)


@dataclass
class EvalContext:
    args: argparse.Namespace
    train_mod: Any
    model: torch.nn.Module
    checkpoint_payload: Dict[str, Any]
    checkpoint_phase: str
    phase: str
    shards: List[Path]
    device: torch.device
    use_amp: bool


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    return torch.device(name)


def load_model(train_mod, checkpoint: Path, device: torch.device):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    base_mod = train_mod.load_base_source_module()
    model = train_mod.FailureAwarePosteriorDepth(base_mod).to(device)
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("model_state_dict", payload))
    clean = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"missing={len(missing)}: {missing[:20]}\n"
            f"unexpected={len(unexpected)}: {unexpected[:20]}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def bootstrap(args: argparse.Namespace) -> EvalContext:
    set_seed(int(args.seed))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    use_amp = bool(device.type == "cuda" and not args.no_amp)

    train_mod = import_module_from_path(
        "fapr_v6_training_for_analysis",
        Path(args.train_script),
    )
    # Override the implementation source path before constructing the model.
    train_mod.BASE_SOURCE_ROOT = Path(args.base_source_root)
    train_mod.CACHE_ROOT = Path(args.cache_root)
    train_mod.DEVICE = str(device)

    shards = train_mod.load_split_shards(
        Path(args.cache_root),
        str(args.split),
        None if args.max_shards is None or int(args.max_shards) <= 0 else int(args.max_shards),
    )
    if not shards:
        raise RuntimeError(f"No cache shards found for split={args.split}")

    model, payload = load_model(train_mod, Path(args.checkpoint), device)
    checkpoint_phase = str(payload.get("phase", "joint"))
    phase = checkpoint_phase if args.phase == "auto" else str(args.phase)
    if phase not in {"safe", "proposal", "risk", "joint"}:
        phase = "joint"

    print("=" * 120)
    print("FAPR-Depth v6 paper diagnostic")
    print("=" * 120)
    print(f"device={device}, amp={use_amp}")
    print(f"train_script={args.train_script}")
    print(f"cache_root={args.cache_root}")
    print(f"checkpoint={args.checkpoint}")
    print(
        f"checkpoint_phase={checkpoint_phase}, evaluation_phase={phase}, "
        f"refine_epoch={payload.get('refine_epoch', -1)}"
    )
    print(f"split={args.split}, shards={len(shards)}, microbatch={args.microbatch}")
    print(f"out_dir={args.out_dir}")

    return EvalContext(
        args=args,
        train_mod=train_mod,
        model=model,
        checkpoint_payload=payload,
        checkpoint_phase=checkpoint_phase,
        phase=phase,
        shards=list(shards),
        device=device,
        use_amp=use_amp,
    )


def make_loader(ctx: EvalContext) -> DataLoader:
    return DataLoader(
        ctx.train_mod.CachedShardDataset(ctx.shards),
        batch_size=1,
        shuffle=False,
        num_workers=max(0, int(ctx.args.num_workers)),
        pin_memory=ctx.device.type == "cuda",
        collate_fn=ctx.train_mod.ragged_shard_collate,
        persistent_workers=int(ctx.args.num_workers) > 0,
    )


def batch_sample_count(batch: Mapping[str, Any]) -> int:
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    raise RuntimeError("Cannot determine sample count from batch.")


def slice_batch(batch: Mapping[str, Any], start: int, end: int, total: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == total:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True).float()
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


@torch.inference_mode()
def iter_forward(
    ctx: EvalContext,
    input_transform=None,
) -> Iterator[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]]:
    """
    Yield (inputs, outputs, metadata) for each microbatch.

    input_transform, when supplied, is called as:
        input_transform(inp, shard_index, sample_offset)
    and must return an input dictionary.
    """
    loader = make_loader(ctx)
    microbatch = max(1, int(ctx.args.microbatch))
    autocast_enabled = bool(ctx.use_amp)

    for shard_index, cpu_batch in enumerate(loader):
        total = batch_sample_count(cpu_batch)
        for start in range(0, total, microbatch):
            end = min(total, start + microbatch)
            part = move_batch(slice_batch(cpu_batch, start, end, total), ctx.device)
            inp = ctx.train_mod.build_inputs(part)
            if input_transform is not None:
                inp = input_transform(inp, shard_index, start)

            with torch.autocast(
                device_type=ctx.device.type,
                dtype=torch.float16 if ctx.device.type == "cuda" else torch.bfloat16,
                enabled=autocast_enabled,
            ):
                out = ctx.model(inp, phase=ctx.phase, augment_safe=False)

            meta = {
                "shard_index": shard_index,
                "sample_offset": start,
                "source_shard": str(ctx.shards[shard_index]),
            }
            yield inp, out, meta

            del part, inp, out
            if ctx.device.type == "cuda" and (shard_index + 1) % 100 == 0:
                torch.cuda.empty_cache()


def output_predictions(
    inp: Mapping[str, torch.Tensor],
    out: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Paper-facing output names."""
    mask = inp["mask"]
    raw = inp["raw"]
    return {
        "Raw Depth": raw,
        "Backbone Baseline": out["anchor_depth"] * mask + raw * (1.0 - mask),
        "Posterior Fusion w/o Safety Control": out["legacy_fused"] * mask + raw * (1.0 - mask),
        "Safe Posterior": out["safe_benchmark"],
        "Full Candidate": out["candidate_benchmark"],
        "Risk-Accepted Output": out["benchmark_output"],
    }


def failure_labels_and_regions(
    train_mod,
    inp: Mapping[str, torch.Tensor],
    out: Optional[Mapping[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    labels, raw_error = train_mod.failure_targets(
        inp["raw"], inp["gt"], inp["valid"], inp["boundary"]
    )
    valid_mask = (inp["valid"] > 0.5).float() * (inp["mask"] > 0.5).float()
    boundary_ring = valid_mask * (inp["boundary"] > 0.15).float()
    interior = valid_mask * (inp["boundary"] <= 0.15).float()
    regions: Dict[str, torch.Tensor] = {
        "all_transparent": valid_mask,
        "valid_state": valid_mask * (labels == 0).float(),
        "missing_failure": valid_mask * (labels == 1).float(),
        "biased_failure": valid_mask * (labels == 2).float(),
        "boundary_failure": valid_mask * (labels == 3).float(),
        "any_failure": valid_mask * (labels > 0).float(),
        "boundary_ring": boundary_ring,
        "interior": interior,
        "raw_failure_threshold": valid_mask
        * (
            raw_error
            > (
                float(train_mod.FAIL_ABS_THR)
                + float(train_mod.FAIL_REL_THR) * inp["gt"]
            )
        ).float(),
    }

    if out is not None:
        anchor_error = torch.abs(out["anchor_depth"] - inp["gt"])
        # Hardest 20% of the backbone error, independently for every image.
        hard = torch.zeros_like(valid_mask)
        for bi in range(anchor_error.shape[0]):
            values = anchor_error[bi][valid_mask[bi] > 0.5]
            if values.numel() == 0:
                continue
            threshold = torch.quantile(values.float(), 0.80)
            hard[bi] = valid_mask[bi] * (anchor_error[bi] >= threshold).float()
        regions["hard_backbone_top20"] = hard
    return labels, regions


class PixelMetricAccumulator:
    """Pixel-pooled depth metrics for an arbitrary region."""

    def __init__(self) -> None:
        self.count = 0.0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.rel_sum = 0.0
        self.d105 = 0.0
        self.d110 = 0.0
        self.d125 = 0.0

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        region: torch.Tensor,
        min_depth: float = 0.03,
    ) -> None:
        mask = region > 0.5
        n = int(mask.sum().item())
        if n <= 0:
            return
        p = pred[mask].float()
        y = gt[mask].float()
        err = torch.abs(p - y)
        ratio = torch.maximum(
            p / y.clamp_min(min_depth),
            y / p.clamp_min(min_depth),
        )
        self.count += float(n)
        self.abs_sum += float(err.sum().item())
        self.sq_sum += float((err * err).sum().item())
        self.rel_sum += float((err / y.clamp_min(min_depth)).sum().item())
        self.d105 += float((ratio < 1.05).sum().item())
        self.d110 += float((ratio < 1.10).sum().item())
        self.d125 += float((ratio < 1.25).sum().item())

    def result(self) -> Dict[str, float]:
        if self.count <= 0:
            return {
                "pixels": 0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "rel": float("nan"),
                "delta_105": float("nan"),
                "delta_110": float("nan"),
                "delta_125": float("nan"),
            }
        return {
            "pixels": int(self.count),
            "mae": self.abs_sum / self.count,
            "rmse": math.sqrt(self.sq_sum / self.count),
            "rel": self.rel_sum / self.count,
            "delta_105": self.d105 / self.count,
            "delta_110": self.d110 / self.count,
            "delta_125": self.d125 / self.count,
        }


class MeanAccumulator:
    def __init__(self) -> None:
        self.weight = 0.0
        self.sums: Dict[str, float] = {}

    def update(self, values: Mapping[str, float], weight: float = 1.0) -> None:
        if weight <= 0:
            return
        self.weight += float(weight)
        for key, value in values.items():
            value = float(value)
            if np.isfinite(value):
                self.sums[key] = self.sums.get(key, 0.0) + value * float(weight)

    def result(self) -> Dict[str, float]:
        if self.weight <= 0:
            return {key: float("nan") for key in self.sums}
        return {key: value / self.weight for key, value in self.sums.items()}


class OfficialMetricAccumulator:
    """
    Average the original training script's metric_values output per microbatch.
    With microbatch=1, this matches the paper evaluation convention.
    """

    def __init__(self, train_mod) -> None:
        self.train_mod = train_mod
        self.rows: List[Dict[str, float]] = []

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        raw: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        row = self.train_mod.metric_values(pred, raw, gt, mask, valid)
        self.rows.append(row)

    def result(self) -> Dict[str, float]:
        if not self.rows:
            return {}
        keys = sorted(set().union(*[set(row) for row in self.rows]))
        out = {
            key: float(np.mean([row[key] for row in self.rows if key in row]))
            for key in keys
        }
        out["score"] = float(self.train_mod.selection_score(out))
        return out


class Reservoir:
    """Uniform reservoir sample of numeric rows."""

    def __init__(self, capacity: int, columns: Sequence[str], seed: int = 6248) -> None:
        self.capacity = max(1, int(capacity))
        self.columns = list(columns)
        self.rng = np.random.default_rng(seed)
        self.data = np.empty((self.capacity, len(self.columns)), dtype=np.float64)
        self.size = 0
        self.seen = 0

    def add_arrays(self, **arrays: np.ndarray) -> None:
        values = [np.asarray(arrays[name]).reshape(-1) for name in self.columns]
        if not values:
            return
        n = min(value.size for value in values)
        if n <= 0:
            return
        matrix = np.stack([value[:n] for value in values], axis=1)
        finite = np.all(np.isfinite(matrix), axis=1)
        matrix = matrix[finite]
        for row in matrix:
            self.seen += 1
            if self.size < self.capacity:
                self.data[self.size] = row
                self.size += 1
            else:
                index = int(self.rng.integers(0, self.seen))
                if index < self.capacity:
                    self.data[index] = row

    def frame(self):
        import pandas as pd

        return pd.DataFrame(self.data[: self.size], columns=self.columns)


def tensor_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def chw_rgb(x: torch.Tensor, index: int = 0) -> np.ndarray:
    image = tensor_numpy(x[index]).transpose(1, 2, 0)
    return np.clip(image, 0.0, 1.0)


def map2d(x: torch.Tensor, index: int = 0) -> np.ndarray:
    arr = tensor_numpy(x[index])
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC without sklearn."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[valid], labels[valid]
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = ranks[pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[valid], labels[valid]
    positives = int((labels == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    bins: int = 15,
) -> Tuple[float, List[Dict[str, float]]]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    valid = np.isfinite(probabilities) & np.isfinite(labels)
    probabilities = np.clip(probabilities[valid], 0.0, 1.0)
    labels = labels[valid]
    if probabilities.size == 0:
        return float("nan"), []

    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: List[Dict[str, float]] = []
    ece = 0.0
    for index in range(int(bins)):
        lo, hi = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lo) & (probabilities <= hi)
        else:
            mask = (probabilities >= lo) & (probabilities < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": index,
                    "lower": lo,
                    "upper": hi,
                    "count": 0,
                    "confidence": float("nan"),
                    "accuracy": float("nan"),
                }
            )
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(labels[mask].mean())
        weight = count / probabilities.size
        ece += weight * abs(confidence - accuracy)
        rows.append(
            {
                "bin": index,
                "lower": lo,
                "upper": hi,
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    return float(ece), rows


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    return pearsonr(rankdata(np.asarray(x)), rankdata(np.asarray(y)))


def select_top_records(
    records: List[Dict[str, Any]],
    record: Dict[str, Any],
    key: str,
    k: int,
    largest: bool,
) -> None:
    records.append(record)
    records.sort(key=lambda row: float(row[key]), reverse=largest)
    if len(records) > int(k):
        del records[int(k):]


def save_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    path: Path,
    value_format: str = ".3f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(1.7 * len(col_labels) + 2.5, 0.75 * len(row_labels) + 2.5))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "—" if not np.isfinite(value) else format(value, value_format)
            ax.text(j, i, text, ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_run_manifest(ctx: EvalContext, extra: Optional[Mapping[str, Any]] = None) -> None:
    payload: Dict[str, Any] = {
        "train_script": str(ctx.args.train_script),
        "cache_root": str(ctx.args.cache_root),
        "base_source_root": str(ctx.args.base_source_root),
        "checkpoint": str(ctx.args.checkpoint),
        "checkpoint_phase": ctx.checkpoint_phase,
        "evaluation_phase": ctx.phase,
        "checkpoint_refine_epoch": ctx.checkpoint_payload.get("refine_epoch", -1),
        "split": ctx.args.split,
        "max_shards": ctx.args.max_shards,
        "shards": len(ctx.shards),
        "microbatch": ctx.args.microbatch,
        "device": str(ctx.device),
        "amp": ctx.use_amp,
        "seed": ctx.args.seed,
    }
    if extra:
        payload.update(dict(extra))
    write_json(Path(ctx.args.out_dir) / "run_manifest.json", payload)
