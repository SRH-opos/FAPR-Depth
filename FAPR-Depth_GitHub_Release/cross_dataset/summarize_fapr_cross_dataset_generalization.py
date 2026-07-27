# -*- coding: utf-8 -*-
r"""
summarize_fapr_cross_dataset_generalization.py

Aggregate FAPR-Depth zero-shot cross-dataset results and generate:

1. A paper-facing dataset summary table:
      Raw Depth | Previous TAGE | FAPR-Depth
2. A metric-improvement table:
      FAPR vs Raw
      FAPR vs Previous TAGE
3. One ClearGrasp qualitative figure.
4. One ClearPose qualitative figure.

The script does NOT rerun model inference.

Default input
-------------
outputs/cross_dataset\
fapr_depth_cross_dataset_generalization\all_samples_metrics.csv

Default FAPR prediction root
----------------------------
outputs/cross_dataset\
fapr_depth_cross_dataset_generalization\pred_depths

Default previous TAGE prediction root
-------------------------------------
outputs/cross_dataset\
tage_depth_cross_dataset_generalization\pred_depths
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image
except Exception:
    Image = None


PROJECT_ROOT = Path(os.getenv("FAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent)))
DEFAULT_EXP_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "paper_experiments"
    / "fapr_depth_cross_dataset_generalization"
)
DEFAULT_METRICS_CSV = DEFAULT_EXP_ROOT / "all_samples_metrics.csv"
DEFAULT_FAPR_PRED_ROOT = DEFAULT_EXP_ROOT / "pred_depths"
DEFAULT_TAGE_PRED_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "paper_experiments"
    / "tage_depth_cross_dataset_generalization"
    / "pred_depths"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_EXP_ROOT / "paper_summary"

EPS = 1.0e-6
MIN_DEPTH = 1.0e-4
MAX_DEPTH = 10.0

DEPTH_CMAP = "viridis"
ERROR_CMAP = "magma"
IMPROVE_CMAP = "coolwarm"

METRICS = [
    ("RMSE", "rmse_mask", "error"),
    ("REL", "rel_mask", "error"),
    ("MAE", "mae_mask", "error"),
    ("δ1.05", "delta_105_mask", "accuracy"),
    ("δ1.10", "delta_110_mask", "accuracy"),
    ("δ1.25", "delta_125_mask", "accuracy"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize FAPR ClearGrasp/ClearPose generalization results."
    )
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--fapr-pred-root", type=Path, default=DEFAULT_FAPR_PRED_ROOT)
    parser.add_argument("--tage-pred-root", type=Path, default=DEFAULT_TAGE_PRED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples-per-family", type=int, default=4)
    parser.add_argument(
        "--selection-reference",
        choices=["tage", "raw"],
        default="tage",
        help="Rank qualitative samples by FAPR MAE gain over TAGE or Raw.",
    )
    parser.add_argument("--crop-to-object", action="store_true")
    parser.add_argument("--no-title", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


# =============================================================================
# CSV and path helpers
# =============================================================================
def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def get_float(
    row: Mapping[str, Any],
    keys: Sequence[str],
    default: float = float("nan"),
) -> float:
    for key in keys:
        value = row.get(key)
        if value not in [None, "", "nan", "None"]:
            try:
                return float(value)
            except Exception:
                pass
    return default


def get_int(
    row: Mapping[str, Any],
    keys: Sequence[str],
    default: int = 0,
) -> int:
    value = get_float(row, keys, float("nan"))
    return int(value) if np.isfinite(value) else default


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


def prediction_path(
    root: Path,
    row: Mapping[str, Any],
    method_key: str,
) -> Optional[Path]:
    dataset_dir = sanitize_filename(row.get("dataset", "dataset"))
    uid = sample_uid(row)
    candidates = [
        root / dataset_dir / method_key / f"{uid}.npy",
    ]

    stem = sanitize_filename(row.get("stem", ""))
    if stem:
        candidates.append(root / dataset_dir / method_key / f"{stem}.npy")

    stem_int = str(row.get("stem_int", "")).strip()
    if stem_int and stem_int not in ["-1", "nan", "None"]:
        try:
            integer = int(float(stem_int))
            candidates.extend(
                [
                    root / dataset_dir / method_key / f"{integer:08d}.npy",
                    root / dataset_dir / method_key / f"{integer}.npy",
                ]
            )
        except Exception:
            pass

    explicit_key = {
        "fapr_depth": "fapr_pred_path",
        "tage_depth": "tage_pred_path",
    }.get(method_key)
    if explicit_key and str(row.get(explicit_key, "")).strip():
        candidates.insert(0, Path(str(row[explicit_key])))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# =============================================================================
# Aggregation
# =============================================================================
def method_specs(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    specs = [
        ("Raw Depth", "raw_depth"),
    ]
    has_tage = any(
        np.isfinite(
            get_float(
                row,
                ["tage_depth_mae_mask", "tage_mae_mask"],
            )
        )
        for row in rows
    )
    if has_tage:
        specs.append(("Previous TAGE", "tage_depth"))
    specs.append(("FAPR-Depth", "fapr_depth"))
    return specs


def metric_keys(method_key: str, suffix: str) -> List[str]:
    aliases = [f"{method_key}_{suffix}"]
    if method_key == "raw_depth":
        aliases.append(f"raw_{suffix}")
    if method_key == "tage_depth":
        aliases.append(f"tage_{suffix}")
    return aliases


def aggregate_method(
    rows: Sequence[Mapping[str, Any]],
    method_name: str,
    method_key: str,
    family: str,
) -> Dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("status", "ok") == "ok"
        and dataset_family(row.get("dataset", "")) == family
    ]

    values: Dict[str, List[Tuple[float, int]]] = {
        suffix: []
        for _, suffix, _ in METRICS
    }

    for row in selected:
        count = get_int(
            row,
            metric_keys(method_key, "num_valid_mask")
            + ["num_valid_mask"],
            0,
        )
        if count <= 0:
            continue
        for _, suffix, _ in METRICS:
            value = get_float(row, metric_keys(method_key, suffix))
            if np.isfinite(value):
                values[suffix].append((value, count))

    output: Dict[str, Any] = {
        "Dataset": family,
        "Method": method_name,
        "Samples": len(selected),
        "ValidPixels": int(
            sum(count for _, count in values["mae_mask"])
        ),
    }

    for display, suffix, kind in METRICS:
        pairs = values[suffix]
        if not pairs:
            output[display] = float("nan")
            continue
        weights = np.asarray([count for _, count in pairs], dtype=np.float64)
        metric_values = np.asarray([value for value, _ in pairs], dtype=np.float64)

        if suffix == "rmse_mask":
            aggregate = math.sqrt(
                float(np.sum(weights * metric_values ** 2) / np.sum(weights))
            )
        else:
            aggregate = float(np.sum(weights * metric_values) / np.sum(weights))
        output[display] = aggregate

    return output


def build_summary(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    specs = method_specs(rows)
    output: List[Dict[str, Any]] = []
    for family in ["ClearGrasp", "ClearPose"]:
        for method_name, method_key in specs:
            output.append(
                aggregate_method(
                    rows,
                    method_name,
                    method_key,
                    family,
                )
            )
    return output


def build_improvement_table(
    summary_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    lookup = {
        (row["Dataset"], row["Method"]): row
        for row in summary_rows
    }
    output: List[Dict[str, Any]] = []

    for family in ["ClearGrasp", "ClearPose"]:
        fapr = lookup.get((family, "FAPR-Depth"))
        if fapr is None:
            continue

        references = [
            ("Raw Depth", "FAPR vs Raw"),
            ("Previous TAGE", "FAPR vs Previous TAGE"),
        ]
        for reference_name, comparison_name in references:
            reference = lookup.get((family, reference_name))
            if reference is None:
                continue

            row: Dict[str, Any] = {
                "Dataset": family,
                "Comparison": comparison_name,
            }
            for display, _, kind in METRICS:
                ref_value = float(reference[display])
                fapr_value = float(fapr[display])
                if not np.isfinite(ref_value) or not np.isfinite(fapr_value):
                    row[display] = float("nan")
                elif kind == "error":
                    row[display] = (
                        100.0 * (ref_value - fapr_value) / ref_value
                        if abs(ref_value) > EPS
                        else float("nan")
                    )
                else:
                    # Accuracy changes are easier to interpret in percentage points.
                    row[display] = 100.0 * (fapr_value - ref_value)
            output.append(row)

    return output


def save_text_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = []
    lines.append(
        f"{'Dataset':<12} | {'Method':<20} | {'RMSE':>9} | {'REL':>9} | "
        f"{'MAE':>9} | {'d1.05':>8} | {'d1.10':>8} | {'d1.25':>8}"
    )
    lines.append("-" * 102)
    for row in rows:
        lines.append(
            f"{row['Dataset']:<12} | {row['Method']:<20} | "
            f"{float(row['RMSE']):>9.6f} | {float(row['REL']):>9.6f} | "
            f"{float(row['MAE']):>9.6f} | "
            f"{100.0 * float(row['δ1.05']):>8.2f} | "
            f"{100.0 * float(row['δ1.10']):>8.2f} | "
            f"{100.0 * float(row['δ1.25']):>8.2f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Image loading
# =============================================================================
def safe_depth(array: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(
        np.asarray(array, dtype=np.float32),
        nan=0.0,
        posinf=MAX_DEPTH,
        neginf=0.0,
    )
    return np.clip(value, 0.0, MAX_DEPTH).astype(np.float32)


def squeeze_hw(array: np.ndarray) -> np.ndarray:
    value = np.squeeze(np.asarray(array))
    if value.ndim == 3:
        if value.shape[-1] == 1:
            value = value[..., 0]
        elif value.shape[0] == 1:
            value = value[0]
        else:
            value = value[..., 0]
    if value.ndim != 2:
        raise RuntimeError(f"Expected HxW array, got {value.shape}")
    return value.astype(np.float32)


def load_rgb(path: Any) -> np.ndarray:
    file_path = Path(str(path))
    if Image is not None:
        return np.asarray(
            Image.open(str(file_path)).convert("RGB"),
            dtype=np.float32,
        ) / 255.0
    image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load RGB: {file_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_depth(path: Any) -> np.ndarray:
    file_path = Path(str(path))
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        return safe_depth(squeeze_hw(np.load(str(file_path))))
    if suffix == ".npz":
        payload = np.load(str(file_path))
        return safe_depth(squeeze_hw(payload[list(payload.keys())[0]]))
    if suffix in [".png", ".tif", ".tiff"]:
        array = np.asarray(Image.open(str(file_path))) if Image is not None else cv2.imread(
            str(file_path), cv2.IMREAD_UNCHANGED
        )
        dtype = array.dtype
        array = squeeze_hw(array).astype(np.float32)
        if dtype == np.uint16 or float(array.max()) > 255.0:
            array /= 1000.0
        elif float(array.max()) > 10.0:
            array /= 255.0
        return safe_depth(array)
    if suffix == ".exr":
        array = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise RuntimeError(f"Failed to load EXR: {file_path}")
        return safe_depth(squeeze_hw(array))
    raise RuntimeError(f"Unsupported depth format: {file_path}")


def load_mask(path: Any, target_shape: Tuple[int, int]) -> np.ndarray:
    file_path = Path(str(path))
    if Image is not None:
        array = np.asarray(Image.open(str(file_path)).convert("L"), dtype=np.float32)
    else:
        array = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if float(array.max()) > 1.5:
        array /= 255.0
    if array.shape != target_shape:
        array = cv2.resize(
            array,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return (array > 0.5).astype(np.float32)


def resize_map(array: np.ndarray, shape: Tuple[int, int], nearest: bool) -> np.ndarray:
    if array.shape[:2] == shape:
        return array.astype(np.float32)
    return cv2.resize(
        np.asarray(array, dtype=np.float32),
        (shape[1], shape[0]),
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
    )


# =============================================================================
# Qualitative selection and plotting
# =============================================================================
def sample_gain(
    row: Mapping[str, Any],
    reference: str,
) -> float:
    fapr_mae = get_float(row, ["fapr_depth_mae_mask"])
    if reference == "tage":
        reference_mae = get_float(row, ["tage_depth_mae_mask", "tage_mae_mask"])
        if not np.isfinite(reference_mae):
            reference_mae = get_float(row, ["raw_depth_mae_mask", "raw_mae_mask"])
    else:
        reference_mae = get_float(row, ["raw_depth_mae_mask", "raw_mae_mask"])

    if not np.isfinite(reference_mae) or not np.isfinite(fapr_mae):
        return -float("inf")
    return reference_mae - fapr_mae


def select_diverse_samples(
    rows: Sequence[Mapping[str, Any]],
    family: str,
    k: int,
    reference: str,
) -> List[Dict[str, Any]]:
    candidates = [
        dict(row)
        for row in rows
        if row.get("status", "ok") == "ok"
        and dataset_family(row.get("dataset", "")) == family
        and np.isfinite(sample_gain(row, reference))
    ]

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(str(row.get("dataset", family)), []).append(row)

    for key in groups:
        groups[key].sort(
            key=lambda item: sample_gain(item, reference),
            reverse=True,
        )

    selected: List[Dict[str, Any]] = []
    used = set()

    # First pass: one sample from each concrete dataset subset.
    for key in sorted(groups):
        if len(selected) >= k:
            break
        for row in groups[key]:
            uid = sample_uid(row)
            if uid not in used:
                selected.append(row)
                used.add(uid)
                break

    # Fill with the strongest remaining examples.
    all_sorted = sorted(
        candidates,
        key=lambda item: sample_gain(item, reference),
        reverse=True,
    )
    for row in all_sorted:
        if len(selected) >= k:
            break
        uid = sample_uid(row)
        if uid not in used:
            selected.append(row)
            used.add(uid)

    return selected[:k]


def crop_to_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    arrays: Sequence[np.ndarray],
    margin: int = 16,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        return rgb, mask, [np.asarray(array) for array in arrays]
    h, w = mask.shape
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + margin + 1)
    rgb_crop = rgb[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    array_crops = [
        array[y0:y1, x0:x1] if array.ndim == 2 else array[y0:y1, x0:x1, :]
        for array in arrays
    ]
    return rgb_crop, mask_crop, array_crops


def normalize_depths(
    depths: Sequence[np.ndarray],
    valid: np.ndarray,
) -> Tuple[List[np.ndarray], float, float]:
    values = []
    for depth in depths:
        region = (valid > 0.5) & np.isfinite(depth) & (depth > MIN_DEPTH)
        if np.any(region):
            values.append(depth[region])
    if not values:
        return [np.zeros_like(depths[0]) for _ in depths], 0.0, 1.0
    merged = np.concatenate(values)
    lo, hi = np.percentile(merged, [1.0, 99.0])
    if hi <= lo + EPS:
        hi = lo + 1.0
    output = [
        np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        for depth in depths
    ]
    return output, float(lo), float(hi)


def load_qualitative_sample(
    row: Mapping[str, Any],
    fapr_root: Path,
    tage_root: Path,
) -> Optional[Dict[str, Any]]:
    try:
        rgb = load_rgb(row["rgb"])
        raw = load_depth(row["raw_depth"])
        gt = load_depth(row["gt_depth"])
        h, w = gt.shape
        rgb = resize_map(rgb, (h, w), nearest=False)
        raw = resize_map(raw, (h, w), nearest=True)
        mask = load_mask(row["mask"], (h, w))
        valid = (np.isfinite(gt) & (gt > MIN_DEPTH)).astype(np.float32)

        fapr_path = prediction_path(fapr_root, row, "fapr_depth")
        if fapr_path is None:
            return None
        fapr = resize_map(load_depth(fapr_path), (h, w), nearest=False)

        tage_path = prediction_path(tage_root, row, "tage_depth")
        tage = (
            resize_map(load_depth(tage_path), (h, w), nearest=False)
            if tage_path is not None
            else None
        )

        return {
            "rgb": rgb,
            "mask": mask,
            "gt": gt,
            "raw": raw,
            "tage": tage,
            "fapr": fapr,
            "valid": valid,
        }
    except Exception as error:
        print(f"[WARNING] Could not load qualitative sample {sample_uid(row)}: {error}")
        return None


def save_family_figure(
    selected_rows: Sequence[Mapping[str, Any]],
    family: str,
    fapr_root: Path,
    tage_root: Path,
    output_root: Path,
    crop_object: bool,
    dpi: int,
    show_title: bool,
) -> None:
    loaded = []
    for row in selected_rows:
        sample = load_qualitative_sample(row, fapr_root, tage_root)
        if sample is not None:
            loaded.append((row, sample))
    if not loaded:
        return

    include_tage = any(sample["tage"] is not None for _, sample in loaded)
    titles = ["RGB", "Mask", "GT Depth", "Raw"]
    if include_tage:
        titles.append("Previous TAGE")
    titles += ["FAPR-Depth", "Raw Error"]
    if include_tage:
        titles.append("TAGE Error")
    titles += ["FAPR Error"]
    if include_tage:
        titles.append("FAPR Gain over TAGE")
    else:
        titles.append("FAPR Gain over Raw")

    figure, axes = plt.subplots(
        len(loaded),
        len(titles),
        figsize=(1.85 * len(titles), 1.75 * len(loaded)),
        squeeze=False,
        dpi=220,
    )

    for row_index, (row, sample) in enumerate(loaded):
        rgb = sample["rgb"]
        mask = sample["mask"]
        gt = sample["gt"]
        raw = sample["raw"]
        tage = sample["tage"]
        fapr = sample["fapr"]
        valid = sample["valid"]

        depth_inputs = [gt, raw, fapr] + ([tage] if tage is not None else [])
        depth_vis, _, _ = normalize_depths(depth_inputs, valid)
        gt_vis = depth_vis[0]
        raw_vis = depth_vis[1]
        fapr_vis = depth_vis[2]
        tage_vis = depth_vis[3] if tage is not None else None

        region = (mask > 0.5) & (valid > 0.5)
        raw_error = np.abs(raw - gt)
        fapr_error = np.abs(fapr - gt)
        tage_error = np.abs(tage - gt) if tage is not None else None

        error_values = [raw_error[region], fapr_error[region]]
        if tage_error is not None:
            error_values.append(tage_error[region])
        merged_errors = np.concatenate(
            [value[np.isfinite(value)] for value in error_values]
        )
        error_max = (
            float(np.percentile(merged_errors, 98.0))
            if merged_errors.size
            else 0.05
        )
        error_max = max(error_max, 1.0e-5)

        reference_error = tage_error if tage_error is not None else raw_error
        gain = reference_error - fapr_error
        gain_values = gain[region]
        gain_max = (
            float(np.percentile(np.abs(gain_values[np.isfinite(gain_values)]), 98.0))
            if np.any(np.isfinite(gain_values))
            else 0.02
        )
        gain_max = max(gain_max, 1.0e-5)

        panels: List[Tuple[np.ndarray, Optional[str], float, float]] = [
            (rgb, None, 0.0, 1.0),
            (mask, "gray", 0.0, 1.0),
            (gt_vis, DEPTH_CMAP, 0.0, 1.0),
            (raw_vis, DEPTH_CMAP, 0.0, 1.0),
        ]
        if tage is not None:
            panels.append((tage_vis, DEPTH_CMAP, 0.0, 1.0))
        panels += [
            (fapr_vis, DEPTH_CMAP, 0.0, 1.0),
            (np.where(region, raw_error, np.nan), ERROR_CMAP, 0.0, error_max),
        ]
        if tage_error is not None:
            panels.append(
                (np.where(region, tage_error, np.nan), ERROR_CMAP, 0.0, error_max)
            )
        panels += [
            (np.where(region, fapr_error, np.nan), ERROR_CMAP, 0.0, error_max),
            (np.where(region, gain, np.nan), IMPROVE_CMAP, -gain_max, gain_max),
        ]

        if crop_object:
            rgb_crop, mask_crop, cropped = crop_to_mask(
                rgb,
                mask,
                [panel[0] for panel in panels[1:]],
            )
            panel_metadata = panels[1:]
            panels = [(rgb_crop, None, 0.0, 1.0)] + [
                (image, meta[1], meta[2], meta[3])
                for image, meta in zip(cropped, panel_metadata)
            ]

        for column_index, (image, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[row_index, column_index]
            if cmap is None:
                axis.imshow(np.clip(image, 0.0, 1.0))
            else:
                color_map = plt.get_cmap(cmap).copy()
                color_map.set_bad("black")
                axis.imshow(image, cmap=color_map, vmin=vmin, vmax=vmax)
                axis.set_facecolor("black")
            axis.axis("off")
            if row_index == 0:
                axis.set_title(titles[column_index], fontsize=10.5, fontweight="bold")

        fapr_mae = get_float(row, ["fapr_depth_mae_mask"])
        reference_mae = (
            get_float(row, ["tage_depth_mae_mask", "tage_mae_mask"])
            if tage is not None
            else get_float(row, ["raw_depth_mae_mask", "raw_mae_mask"])
        )
        axes[row_index, 0].set_ylabel(
            f"{str(row.get('dataset', family)).replace('ClearGrasp-', 'CG-').replace('ClearPose-', 'CP-')}\n"
            f"{row.get('scene', '')} / {row.get('stem', '')}\n"
            f"MAE {reference_mae:.3f}→{fapr_mae:.3f}",
            fontsize=7.8,
            rotation=90,
            labelpad=18,
        )

    if show_title:
        figure.suptitle(
            f"{family} Zero-Shot Generalization",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )

    plt.subplots_adjust(
        left=0.045,
        right=0.998,
        top=0.90 if show_title else 0.97,
        bottom=0.02,
        wspace=0.012,
        hspace=0.05,
    )
    prefix = family.lower()
    figure.savefig(
        output_root / f"{prefix}_qualitative.png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    figure.savefig(
        output_root / f"{prefix}_qualitative.pdf",
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = [
        row
        for row in read_csv_rows(args.metrics_csv)
        if row.get("status", "ok") == "ok"
        and dataset_family(row.get("dataset", "")) in {"ClearGrasp", "ClearPose"}
    ]
    if not rows:
        raise RuntimeError("No completed ClearGrasp/ClearPose rows found.")

    summary = build_summary(rows)
    improvements = build_improvement_table(summary)

    write_csv_rows(
        args.output_root / "cross_dataset_summary_table.csv",
        summary,
    )
    write_csv_rows(
        args.output_root / "cross_dataset_improvement_table.csv",
        improvements,
    )
    save_text_table(
        args.output_root / "cross_dataset_summary_table.txt",
        summary,
    )

    selected_manifest: List[Dict[str, Any]] = []
    for family in ["ClearGrasp", "ClearPose"]:
        selected = select_diverse_samples(
            rows,
            family,
            k=args.samples_per_family,
            reference=args.selection_reference,
        )
        selected_manifest.extend(selected)
        write_csv_rows(
            args.output_root / f"{family.lower()}_selected_samples.csv",
            selected,
        )
        save_family_figure(
            selected,
            family,
            args.fapr_pred_root,
            args.tage_pred_root,
            args.output_root,
            crop_object=args.crop_to_object,
            dpi=args.dpi,
            show_title=not args.no_title,
        )

    metadata = {
        "metrics_csv": str(args.metrics_csv),
        "fapr_pred_root": str(args.fapr_pred_root),
        "tage_pred_root": str(args.tage_pred_root),
        "selection_reference": args.selection_reference,
        "samples_per_family": args.samples_per_family,
        "summary_rows": summary,
        "improvement_rows": improvements,
    }
    (args.output_root / "summary_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDataset summary")
    for row in summary:
        print(row)
    print("\nMetric improvements")
    for row in improvements:
        print(row)
    print("\nSaved to:", args.output_root)


if __name__ == "__main__":
    main()
