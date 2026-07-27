# -*- coding: utf-8 -*-
r"""
evaluate_fapr_cross_dataset_generalization.py

Zero-shot cross-dataset evaluation for FAPR-Depth v6.

The script reuses the sample/path manifest from the previous TAGE cross-dataset
experiment, but performs NEW FAPR inference and metric computation.

Supported dataset families
--------------------------
- ClearGrasp
- ClearPose

Evaluation protocol
-------------------
- No fine-tuning on ClearGrasp/ClearPose.
- Use the fixed FAPR v6 best-candidate checkpoint.
- Generate a monocular relative-depth prior with the same local
  Depth-Anything backend already used by the ReMake wrapper.
- Robustly align the relative prediction to metric raw depth using only
  reliable background observations.
- Evaluate Raw Depth and FAPR-Depth on the transparent-object mask.
- Preserve existing TAGE fields from the input CSV, so the summarizer can
  report Raw / Previous TAGE / FAPR in one table.

Default input manifest
----------------------
outputs/cross_dataset\
tage_depth_cross_dataset_generalization\all_samples_metrics.csv

Default output
--------------
outputs/cross_dataset\
fapr_depth_cross_dataset_generalization\
    all_samples_metrics.csv
    pred_depths/<dataset>/fapr_depth/*.npy
    run_config.json

Run
---
python \
  cross_dataset/evaluate_fapr_cross_dataset_generalization.py

Quick protocol check
--------------------
python \
  cross_dataset/evaluate_fapr_cross_dataset_generalization.py `
  --max-samples 20
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from PIL import Image
except Exception:
    Image = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# =============================================================================
# Defaults
# =============================================================================
PROJECT_ROOT = Path(os.getenv("FAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent)))

DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "paper_experiments"
    / "tage_depth_cross_dataset_generalization"
    / "all_samples_metrics.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "paper_experiments"
    / "fapr_depth_cross_dataset_generalization"
)
DEFAULT_FAPR_SCRIPT = PROJECT_ROOT / "train_fapr_depth_safe_anchor_v6_8gb.py"
DEFAULT_FAPR_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "fapr_depth_v6_safe_anchor"
    / "checkpoints"
    / "best_candidate.pth"
)
DEFAULT_RELATIVE_WRAPPER = PROJECT_ROOT / "test_remake_unified_transcg.py"

EPS = 1.0e-6
MIN_DEPTH = 1.0e-4
MAX_DEPTH = 10.0


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot ClearGrasp/ClearPose evaluation for FAPR-Depth v6."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fapr-script", type=Path, default=DEFAULT_FAPR_SCRIPT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_FAPR_CKPT)
    parser.add_argument(
        "--relative-wrapper",
        type=Path,
        default=DEFAULT_RELATIVE_WRAPPER,
        help=(
            "Local wrapper exposing load_depthanything_model, "
            "build_rel_transform and preprocess_for_depthanything."
        ),
    )
    parser.add_argument(
        "--prior-mode",
        choices=["depthanything", "csv", "raw_fallback"],
        default="depthanything",
        help=(
            "depthanything: generate and align an external relative prior; "
            "csv: load a precomputed prior path from the manifest; "
            "raw_fallback: disable the external prior by using Raw with zero confidence."
        ),
    )
    parser.add_argument(
        "--relative-column",
        type=str,
        default="rel_aligned",
        help="Manifest column used when --prior-mode csv.",
    )
    parser.add_argument("--input-height", type=int, default=240)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--families",
        type=str,
        default="ClearGrasp,ClearPose",
        help="Comma-separated dataset families.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 means all selected samples.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed rows from an existing output CSV.",
    )
    parser.add_argument(
        "--save-input-prior",
        action="store_true",
        help="Also save the aligned relative prior for diagnostics.",
    )
    parser.add_argument(
        "--strict-relative-backend",
        action="store_true",
        help="Fail instead of falling back to Raw when Depth Anything cannot be loaded.",
    )
    return parser.parse_args()


# =============================================================================
# Basic utilities
# =============================================================================
def import_module_from_path(name: str, path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing Python file: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_torch_load(path: Path, map_location: str = "cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def sanitize_filename(value: Any) -> str:
    text = str(value)
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ', '\t', '\n', '\r']:
        text = text.replace(char, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "sample"


def dataset_family(dataset_name: str) -> str:
    name = str(dataset_name).lower()
    if "cleargrasp" in name:
        return "ClearGrasp"
    if "clearpose" in name:
        return "ClearPose"
    return "Other"


def sample_uid(row: Mapping[str, Any]) -> str:
    if str(row.get("id", "")).strip():
        return sanitize_filename(row["id"])
    return sanitize_filename(
        f"{row.get('dataset', 'dataset')}_{row.get('scene', '')}_{row.get('stem', '')}"
    )


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def safe_depth(array: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(
        np.asarray(array, dtype=np.float32),
        nan=0.0,
        posinf=MAX_DEPTH,
        neginf=0.0,
    )
    return np.clip(value, 0.0, MAX_DEPTH).astype(np.float32)


def squeeze_hw(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    value = np.squeeze(value)
    if value.ndim == 3:
        if value.shape[-1] == 1:
            value = value[..., 0]
        elif value.shape[0] == 1:
            value = value[0]
        elif value.shape[-1] >= 3:
            value = value[..., 0]
        else:
            value = value[0]
    if value.ndim != 2:
        raise RuntimeError(f"Expected HxW array, got shape={value.shape}")
    return value.astype(np.float32)


def load_rgb(path: Any) -> np.ndarray:
    file_path = Path(str(path))
    if not file_path.exists():
        raise FileNotFoundError(f"RGB not found: {file_path}")

    if Image is not None:
        image = Image.open(str(file_path)).convert("RGB")
        return np.asarray(image, dtype=np.float32) / 255.0

    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load RGB: {file_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def load_depth(path: Any) -> np.ndarray:
    file_path = Path(str(path))
    if not file_path.exists():
        raise FileNotFoundError(f"Depth not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        return safe_depth(squeeze_hw(np.load(str(file_path))))
    if suffix == ".npz":
        payload = np.load(str(file_path))
        for key in ["depth", "pred", "output", "final_depth", "arr_0"]:
            if key in payload:
                return safe_depth(squeeze_hw(payload[key]))
        return safe_depth(squeeze_hw(payload[list(payload.keys())[0]]))

    if suffix in [".png", ".tif", ".tiff"]:
        if Image is not None:
            array = np.asarray(Image.open(str(file_path)))
        else:
            array = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise RuntimeError(f"Failed to load depth: {file_path}")
        original_dtype = array.dtype
        array = squeeze_hw(array).astype(np.float32)
        if original_dtype == np.uint16 or float(np.nanmax(array)) > 255.0:
            array = array / 1000.0
        elif float(np.nanmax(array)) > 10.0:
            array = array / 255.0
        return safe_depth(array)

    if suffix == ".exr":
        array = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise RuntimeError(
                f"Failed to load EXR: {file_path}. "
                "Check OpenCV OpenEXR support."
            )
        return safe_depth(squeeze_hw(array))

    raise RuntimeError(f"Unsupported depth format: {file_path}")


def load_mask(path: Any, target_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    file_path = Path(str(path))
    if not file_path.exists():
        raise FileNotFoundError(f"Mask not found: {file_path}")

    if Image is not None:
        array = np.asarray(Image.open(str(file_path)).convert("L"), dtype=np.float32)
    else:
        array = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if array is None:
            raise RuntimeError(f"Failed to load mask: {file_path}")
        array = array.astype(np.float32)

    if float(np.nanmax(array)) > 1.5:
        array /= 255.0
    if target_shape is not None and tuple(array.shape[:2]) != tuple(target_shape):
        array = cv2.resize(
            array,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (array > 0.5).astype(np.float32)


def resize_map(array: np.ndarray, shape: Tuple[int, int], nearest: bool) -> np.ndarray:
    h, w = shape
    value = np.asarray(array, dtype=np.float32)
    if value.shape[:2] == (h, w):
        return value.copy()
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(value, (w, h), interpolation=interpolation).astype(np.float32)


# =============================================================================
# Robust metric alignment for the external relative prior
# =============================================================================
def affine_fit(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    iterations: int = 5,
) -> Tuple[Optional[np.ndarray], float, float, int]:
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(source)
        & np.isfinite(target)
        & (target > MIN_DEPTH)
    )
    x = source[valid].astype(np.float64)
    y = target[valid].astype(np.float64)

    if x.size < 64:
        return None, float("inf"), 0.0, int(x.size)

    # Remove extreme source values before fitting.
    x_lo, x_hi = np.percentile(x, [1.0, 99.0])
    y_lo, y_hi = np.percentile(y, [1.0, 99.0])
    keep = (
        (x >= x_lo)
        & (x <= x_hi)
        & (y >= y_lo)
        & (y <= y_hi)
    )
    x = x[keep]
    y = y[keep]
    if x.size < 64:
        return None, float("inf"), 0.0, int(x.size)

    inlier = np.ones(x.shape[0], dtype=bool)
    scale = 1.0
    offset = 0.0

    for _ in range(max(1, int(iterations))):
        design = np.stack([x[inlier], np.ones(int(inlier.sum()))], axis=1)
        if design.shape[0] < 32:
            break
        solution, *_ = np.linalg.lstsq(design, y[inlier], rcond=None)
        scale = float(solution[0])
        offset = float(solution[1])

        prediction = scale * x + offset
        residual = np.abs(prediction - y)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median))) + EPS
        threshold = max(3.5 * 1.4826 * mad, np.percentile(residual, 75.0))
        new_inlier = residual <= threshold
        if new_inlier.sum() < 32 or np.array_equal(new_inlier, inlier):
            break
        inlier = new_inlier

    if not np.isfinite(scale) or not np.isfinite(offset):
        return None, float("inf"), 0.0, int(x.size)

    fitted = scale * source.astype(np.float64) + offset
    residual = np.abs(scale * x[inlier] + offset - y[inlier])
    fit_mae = float(np.mean(residual)) if residual.size else float("inf")
    coverage = float(inlier.sum() / max(1, valid.sum()))
    return fitted.astype(np.float32), fit_mae, coverage, int(inlier.sum())


def align_relative_to_metric(
    relative: np.ndarray,
    raw: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, Any]:
    """
    Test both direct-depth and inverse-depth hypotheses and choose the lower
    robust background fitting error.
    """
    relative = np.asarray(relative, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    valid = np.asarray(valid, dtype=np.float32)

    # Use reliable raw observations outside the transparent-object mask.
    anchor = (
        (valid > 0.5)
        & (raw > MIN_DEPTH)
        & (mask < 0.5)
        & np.isfinite(relative)
    )

    direct, direct_mae, direct_coverage, direct_n = affine_fit(
        relative,
        raw,
        anchor,
    )

    positive = relative[np.isfinite(relative) & (relative > EPS)]
    inverse_source = np.zeros_like(relative, dtype=np.float32)
    if positive.size:
        stabilizer = max(float(np.percentile(positive, 1.0)) * 0.1, EPS)
    else:
        stabilizer = EPS
    inverse_source = 1.0 / np.maximum(relative, stabilizer)
    inverse, inverse_mae, inverse_coverage, inverse_n = affine_fit(
        inverse_source,
        raw,
        anchor,
    )

    if direct is None and inverse is None:
        return {
            "aligned": raw.copy(),
            "transform": "raw_fallback",
            "fit_mae": float("inf"),
            "coverage": 0.0,
            "anchor_count": 0,
            "confidence": 0.0,
            "residual_map": np.zeros_like(raw, dtype=np.float32),
            "coverage_map": np.zeros_like(raw, dtype=np.float32),
        }

    if inverse_mae < direct_mae:
        aligned = inverse
        transform = "inverse_affine"
        fit_mae = inverse_mae
        coverage = inverse_coverage
        anchor_count = inverse_n
    else:
        aligned = direct
        transform = "direct_affine"
        fit_mae = direct_mae
        coverage = direct_coverage
        anchor_count = direct_n

    aligned = safe_depth(aligned)

    # Confidence is a global source-quality estimate, broadcast as a map.
    anchor_raw = raw[anchor]
    depth_scale = float(np.median(anchor_raw)) if anchor_raw.size else 1.0
    normalized_error = fit_mae / max(depth_scale, 0.05)
    count_factor = min(1.0, anchor_count / 5000.0)
    confidence = float(
        np.clip(
            math.exp(-6.0 * normalized_error)
            * max(0.0, coverage)
            * count_factor,
            0.0,
            1.0,
        )
    )

    residual_map = np.zeros_like(raw, dtype=np.float32)
    residual_map[anchor] = np.abs(aligned[anchor] - raw[anchor])
    coverage_map = anchor.astype(np.float32)

    return {
        "aligned": aligned,
        "transform": transform,
        "fit_mae": fit_mae,
        "coverage": coverage,
        "anchor_count": anchor_count,
        "confidence": confidence,
        "residual_map": residual_map,
        "coverage_map": coverage_map,
    }


# =============================================================================
# Relative-depth backend
# =============================================================================
def load_relative_backend(wrapper_path: Path, device: torch.device) -> Dict[str, Any]:
    # Some local ReMake copies import visualization-only Open3D at module import.
    sys.modules.setdefault("open3d", types.SimpleNamespace())
    wrapper = import_module_from_path("fapr_cross_dataset_relative_wrapper", wrapper_path)

    required = [
        "load_depthanything_model",
        "build_rel_transform",
        "preprocess_for_depthanything",
    ]
    missing = [name for name in required if not callable(getattr(wrapper, name, None))]
    if missing:
        raise AttributeError(
            f"Relative wrapper is missing functions: {missing}"
        )

    model = wrapper.load_depthanything_model()
    transform = wrapper.build_rel_transform()
    if hasattr(model, "eval"):
        model.eval()

    return {
        "wrapper": wrapper,
        "model": model,
        "transform": transform,
        "device": device,
    }


@torch.no_grad()
def infer_relative_depth(
    backend: Dict[str, Any],
    rgb: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    wrapper = backend["wrapper"]
    model = backend["model"]
    transform = backend["transform"]
    device = backend["device"]

    rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    tensor = wrapper.preprocess_for_depthanything(rgb_u8, transform)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    output = model.forward(tensor) if hasattr(model, "forward") else model(tensor)
    if isinstance(output, Mapping):
        for key in ["predicted_depth", "depth", "out", "prediction"]:
            if key in output:
                output = output[key]
                break
    if isinstance(output, (tuple, list)):
        output = output[0]

    if output.ndim == 3:
        output = output.unsqueeze(1)
    elif output.ndim == 2:
        output = output.unsqueeze(0).unsqueeze(0)
    if output.ndim != 4:
        raise RuntimeError(f"Unexpected relative-depth shape: {tuple(output.shape)}")

    output = F.interpolate(
        output.float(),
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )
    relative = output[0, 0].detach().cpu().numpy().astype(np.float32)
    relative[~np.isfinite(relative)] = 0.0
    return relative


# =============================================================================
# FAPR loader and inference
# =============================================================================
def load_fapr_bundle(
    script_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    module = import_module_from_path("fapr_cross_dataset_model", script_path)

    base_root = Path(module.BASE_SOURCE_ROOT)
    if str(base_root) in sys.path:
        sys.path.remove(str(base_root))
    sys.path.insert(0, str(base_root))

    base_module = module.load_base_source_module()
    model = module.FailureAwarePosteriorDepth(base_module).to(device)

    payload = safe_torch_load(checkpoint_path, map_location="cpu")
    state = payload.get("model", payload.get("model_state_dict", payload))
    clean_state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "FAPR checkpoint mismatch:\n"
            f"missing={missing[:20]}\n"
            f"unexpected={unexpected[:20]}"
        )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    checkpoint_phase = str(payload.get("phase", "joint"))
    phase = checkpoint_phase if checkpoint_phase in {"proposal", "risk", "joint"} else "joint"

    return {
        "module": module,
        "model": model,
        "payload": payload,
        "phase": phase,
        "device": device,
    }


@torch.no_grad()
def infer_fapr(
    bundle: Dict[str, Any],
    rgb: np.ndarray,
    raw: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
    prior: Dict[str, Any],
    input_shape: Tuple[int, int],
    use_amp: bool,
) -> np.ndarray:
    module = bundle["module"]
    model = bundle["model"]
    device = bundle["device"]
    phase = bundle["phase"]
    h, w = input_shape

    rgb_model = resize_map(rgb, (h, w), nearest=False)
    raw_model = resize_map(raw, (h, w), nearest=True)
    gt_model = resize_map(gt, (h, w), nearest=True)
    mask_model = resize_map(mask, (h, w), nearest=True)
    valid_model = resize_map(valid, (h, w), nearest=True)
    rel_model = resize_map(prior["aligned"], (h, w), nearest=False)
    residual_model = resize_map(prior["residual_map"], (h, w), nearest=False)
    coverage_model = resize_map(prior["coverage_map"], (h, w), nearest=True)

    raw_prior = ((raw_model <= MIN_DEPTH) & (mask_model > 0.5)).astype(np.float32)
    rel_conf = np.full((h, w), float(prior["confidence"]), dtype=np.float32)

    batch = {
        "rgb": torch.from_numpy(rgb_model.transpose(2, 0, 1)).unsqueeze(0).to(device),
        "raw_depth": torch.from_numpy(raw_model).unsqueeze(0).unsqueeze(0).to(device),
        "gt_depth": torch.from_numpy(gt_model).unsqueeze(0).unsqueeze(0).to(device),
        "mask": torch.from_numpy(mask_model).unsqueeze(0).unsqueeze(0).to(device),
        "valid": torch.from_numpy(valid_model).unsqueeze(0).unsqueeze(0).to(device),
        "rel_aligned": torch.from_numpy(rel_model).unsqueeze(0).unsqueeze(0).to(device),
        "rel_conf": torch.from_numpy(rel_conf).unsqueeze(0).unsqueeze(0).to(device),
        "raw_prior": torch.from_numpy(raw_prior).unsqueeze(0).unsqueeze(0).to(device),
        "rel_bg_resid": torch.from_numpy(residual_model).unsqueeze(0).unsqueeze(0).to(device),
        "rel_bg_coverage": torch.from_numpy(coverage_model).unsqueeze(0).unsqueeze(0).to(device),
    }
    batch = {key: value.float() for key, value in batch.items()}
    inputs = module.build_inputs(batch)

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=bool(use_amp),
    ):
        output = model(
            inputs,
            phase=phase,
            augment_safe=False,
        )

    prediction = output["candidate"][0, 0].detach().float().cpu().numpy()
    prediction = resize_map(prediction, gt.shape, nearest=False)
    prediction = safe_depth(prediction)

    # Keep raw observations outside the transparent mask.
    final = raw.copy()
    final[mask > 0.5] = prediction[mask > 0.5]
    return safe_depth(final)


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(
    prediction: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, float]:
    region = (
        (mask > 0.5)
        & (valid > 0.5)
        & np.isfinite(prediction)
        & np.isfinite(gt)
        & (gt > MIN_DEPTH)
    )
    count = int(region.sum())
    if count <= 0:
        return {
            "num_valid_mask": 0,
            "mae_mask": float("nan"),
            "rmse_mask": float("nan"),
            "rel_mask": float("nan"),
            "delta_105_mask": float("nan"),
            "delta_110_mask": float("nan"),
            "delta_125_mask": float("nan"),
        }

    pred = np.clip(prediction[region], MIN_DEPTH, MAX_DEPTH).astype(np.float64)
    target = np.clip(gt[region], MIN_DEPTH, MAX_DEPTH).astype(np.float64)
    error = np.abs(pred - target)
    ratio = np.maximum(pred / target, target / pred)

    return {
        "num_valid_mask": count,
        "mae_mask": float(np.mean(error)),
        "rmse_mask": float(np.sqrt(np.mean((pred - target) ** 2))),
        "rel_mask": float(np.mean(error / target)),
        "delta_105_mask": float(np.mean(ratio < 1.05)),
        "delta_110_mask": float(np.mean(ratio < 1.10)),
        "delta_125_mask": float(np.mean(ratio < 1.25)),
    }


def add_metric_prefix(
    destination: Dict[str, Any],
    prefix: str,
    metrics: Mapping[str, Any],
) -> None:
    for key, value in metrics.items():
        destination[f"{prefix}_{key}"] = value


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()

    requested_families = {
        value.strip()
        for value in args.families.split(",")
        if value.strip()
    }
    rows = read_csv_rows(args.input_csv)
    rows = [
        row
        for row in rows
        if dataset_family(row.get("dataset", "")) in requested_families
    ]
    if args.max_samples > 0:
        rows = rows[: int(args.max_samples)]
    if not rows:
        raise RuntimeError("No selected ClearGrasp/ClearPose rows were found.")

    output_root = args.output_root
    prediction_root = output_root / "pred_depths"
    output_csv = output_root / "all_samples_metrics.csv"
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    use_amp = (not args.no_amp) and device.type == "cuda"

    fapr = load_fapr_bundle(
        args.fapr_script,
        args.checkpoint,
        device,
    )

    relative_backend = None
    effective_prior_mode = args.prior_mode
    if args.prior_mode == "depthanything":
        try:
            relative_backend = load_relative_backend(
                args.relative_wrapper,
                device,
            )
        except Exception as error:
            if args.strict_relative_backend:
                raise
            print(
                "[WARNING] Could not load relative-depth backend. "
                f"Falling back to raw_fallback.\n{type(error).__name__}: {error}"
            )
            effective_prior_mode = "raw_fallback"

    completed: Dict[str, Dict[str, str]] = {}
    if args.resume and output_csv.exists():
        for previous in read_csv_rows(output_csv):
            if previous.get("status") == "ok":
                completed[sample_uid(previous)] = previous

    result_rows: List[Dict[str, Any]] = []
    if completed:
        result_rows.extend(completed.values())

    progress = tqdm(rows, desc="FAPR cross-dataset generalization", dynamic_ncols=True)
    for source_row in progress:
        uid = sample_uid(source_row)
        if uid in completed:
            continue

        output_row: Dict[str, Any] = dict(source_row)
        output_row["id"] = source_row.get("id", uid)
        output_row["family"] = dataset_family(source_row.get("dataset", ""))
        output_row["fapr_checkpoint"] = str(args.checkpoint)
        output_row["requested_prior_mode"] = args.prior_mode
        output_row["effective_prior_mode"] = effective_prior_mode

        start_time = time.perf_counter()
        try:
            rgb = np.clip(load_rgb(source_row["rgb"]), 0.0, 1.0)
            raw = load_depth(source_row["raw_depth"])
            gt = load_depth(source_row["gt_depth"])

            h, w = gt.shape
            rgb = resize_map(rgb, (h, w), nearest=False)
            raw = resize_map(raw, (h, w), nearest=True)
            mask = load_mask(source_row["mask"], target_shape=(h, w))
            valid = (
                np.isfinite(gt)
                & (gt > MIN_DEPTH)
            ).astype(np.float32)

            if effective_prior_mode == "csv":
                relative_path = str(source_row.get(args.relative_column, "")).strip()
                if not relative_path:
                    raise KeyError(
                        f"Manifest does not contain {args.relative_column!r}."
                    )
                relative = resize_map(
                    load_depth(relative_path),
                    (h, w),
                    nearest=False,
                )
                prior = align_relative_to_metric(
                    relative,
                    raw,
                    mask,
                    valid,
                )
            elif effective_prior_mode == "depthanything":
                relative = infer_relative_depth(
                    relative_backend,
                    rgb,
                    target_shape=(h, w),
                )
                prior = align_relative_to_metric(
                    relative,
                    raw,
                    mask,
                    valid,
                )
            else:
                prior = {
                    "aligned": raw.copy(),
                    "transform": "raw_fallback",
                    "fit_mae": float("nan"),
                    "coverage": 0.0,
                    "anchor_count": 0,
                    "confidence": 0.0,
                    "residual_map": np.zeros_like(raw, dtype=np.float32),
                    "coverage_map": np.zeros_like(raw, dtype=np.float32),
                }

            prediction = infer_fapr(
                fapr,
                rgb,
                raw,
                gt,
                mask,
                valid,
                prior,
                input_shape=(args.input_height, args.input_width),
                use_amp=use_amp,
            )

            dataset_dir = sanitize_filename(source_row.get("dataset", "dataset"))
            prediction_path = (
                prediction_root
                / dataset_dir
                / "fapr_depth"
                / f"{uid}.npy"
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(prediction_path), prediction.astype(np.float32))

            if args.save_input_prior:
                prior_path = (
                    prediction_root
                    / dataset_dir
                    / "relative_prior"
                    / f"{uid}.npy"
                )
                prior_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(prior_path), prior["aligned"].astype(np.float32))
                output_row["fapr_relative_prior_path"] = str(prior_path)

            raw_metrics = compute_metrics(raw, gt, mask, valid)
            fapr_metrics = compute_metrics(prediction, gt, mask, valid)
            add_metric_prefix(output_row, "raw_depth", raw_metrics)
            add_metric_prefix(output_row, "fapr_depth", fapr_metrics)

            output_row.update(
                {
                    "fapr_pred_path": str(prediction_path),
                    "fapr_prior_transform": prior["transform"],
                    "fapr_prior_fit_mae": prior["fit_mae"],
                    "fapr_prior_coverage": prior["coverage"],
                    "fapr_prior_anchor_count": prior["anchor_count"],
                    "fapr_prior_confidence": prior["confidence"],
                    "fapr_runtime_sec": time.perf_counter() - start_time,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as error:
            output_row.update(
                {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "fapr_runtime_sec": time.perf_counter() - start_time,
                }
            )
            print(f"\n[ERROR] {uid}: {output_row['error']}")

        result_rows.append(output_row)

        # Incremental saving protects long cross-dataset runs.
        write_csv_rows(output_csv, result_rows)

    config = {
        "input_csv": str(args.input_csv),
        "output_root": str(output_root),
        "checkpoint": str(args.checkpoint),
        "checkpoint_phase": fapr["phase"],
        "requested_prior_mode": args.prior_mode,
        "effective_prior_mode": effective_prior_mode,
        "relative_wrapper": str(args.relative_wrapper),
        "input_size": [args.input_height, args.input_width],
        "device": str(device),
        "amp": use_amp,
        "families": sorted(requested_families),
        "samples_requested": len(rows),
        "samples_ok": sum(row.get("status") == "ok" for row in result_rows),
        "samples_error": sum(row.get("status") == "error" for row in result_rows),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("FAPR cross-dataset inference completed")
    print("=" * 110)
    print("Metrics CSV :", output_csv)
    print("Predictions :", prediction_root)
    print("Run config  :", output_root / "run_config.json")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
