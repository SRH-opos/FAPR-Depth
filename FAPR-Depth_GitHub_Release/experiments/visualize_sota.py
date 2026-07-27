# -*- coding: utf-8 -*-
r"""
visualize_fapr_depth_top4_comparison_paper_style.py

Purpose
-------
Generate a paper-style qualitative comparison figure for FAPR-Depth and the
same strong external baselines used by the previous TAGE qualitative script.

Default method order:
    1) Raw Depth
    2) TODE-Trans
    3) TDCNet
    4) ReMake
    5) FAPR-Depth (Ours)

Layout:
    - left side: Scene label / RGB / GT Depth;
    - right side: per-method Depth and Error Map;
    - each scene occupies two rows;
    - one shared error colorbar with a fixed 0--0.05 m range.

Paper-facing naming rule
------------------------
Only FAPR-Depth is shown for our method. The internal completion backbone is
not displayed or named in the figure.

Expected local assets
---------------------
    - TODE wrapper:   path/to/tode_wrapper.py
    - TDCNet wrapper: path/to/tdcnet_wrapper.py
    - ReMake wrapper: path/to/remake_wrapper.py
    - FAPR train/model definition:
          train.py
    - FAPR main checkpoint:
          weights/best_candidate.pth
    - cached test split:
          data/cache/test

Run
---
& E:/Anaconda/envs/yolov8/python.exe `
  F:/TransCG_OAD/visualize_fapr_depth_top4_comparison_paper_style.py
"""

from pathlib import Path
import os
import sys
import json
import time
import types
import importlib.util
from functools import lru_cache
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(os.getenv("FAPR_PROJECT_ROOT", str(Path(__file__).resolve().parent)))
CACHE_ROOT = Path(os.getenv("FAPR_CACHE_ROOT", str(PROJECT_ROOT / "data" / "cache")))
SPLIT = "test"

# Selected sample indices on the chosen split (0-based flat sample index).
# These follow your previous qualitative figure examples.
FIGURE_SAMPLE_INDICES = [211, 1626, 95, 1365, 1564]

# Output directory.
OUT_DIR = PROJECT_ROOT / "outputs" / "fapr_depth_top4_visual_comparison_paper_style" / SPLIT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_COMBINED_PNG = OUT_DIR / f"fapr_depth_top4_visual_comparison_paper_style_{SPLIT}.png"
SAVE_COMBINED_PDF = OUT_DIR / f"fapr_depth_top4_visual_comparison_paper_style_{SPLIT}.pdf"
SAVE_META_JSON = OUT_DIR / f"fapr_depth_top4_visual_comparison_paper_style_{SPLIT}.json"

# Current qualitative comparison methods:
# Raw Depth + selected strong comparison methods.
# We intentionally do NOT show the internal completion-backbone row in the figure.
METHOD_SPECS = [
    ("Raw", "raw_depth"),
    ("TODE-Trans", "tode_trans"),
    ("TDCNet", "tdcnet"),
    ("ReMake", "remake"),
    ("FAPR-Depth (Ours)", "fapr_depth"),
]

# External wrapper / model scripts.
TODE_WRAPPER_SCRIPT = PROJECT_ROOT / "eval_tode_vs_oad_v4b.py"
TDCNET_WRAPPER_SCRIPT = PROJECT_ROOT / "eval_tdcnet_vs_oad_v4b_diagnostic.py"
REMAKE_WRAPPER_SCRIPT = PROJECT_ROOT / "test_remake_unified_transcg.py"
FAPR_TRAIN_SCRIPT = Path(os.getenv("FAPR_TRAIN_SCRIPT", str(PROJECT_ROOT / "train.py")))
FAPR_CANDIDATE_CKPT = (
    PROJECT_ROOT
    / "outputs"
    / "fapr_depth_v6_safe_anchor"
    / "checkpoints"
    / "best_candidate.pth"
)

# Error-map visualization range in meters.
ERROR_VMAX = 0.05

# Paper visualization colormaps.
# Use a different colormap for depth and error to avoid visual ambiguity.
DEPTH_CMAP_NAME = "viridis"   # for GT/method depth maps
ERROR_CMAP_NAME = "turbo"     # for error maps

# Device.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# Generic helpers
# =========================================================
def import_module_from_path(name: str, path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepend_path(path: Path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def find_function(module, names: List[str]):
    for n in names:
        fn = getattr(module, n, None)
        if callable(fn):
            return fn
    raise AttributeError(
        f"Could not find any function {names} in module {getattr(module, '__file__', module)}"
    )


def purge_external_imports():
    """Remove repo-local module names that often collide across TODE/TDCNet/ReMake/TAGE."""
    prefixes = [
        "inference", "models", "utils", "datasets", "config", "configs",
        "networks", "trainer", "module", "relat_depth_models",
    ]
    for name in list(sys.modules.keys()):
        if name in prefixes or any(name.startswith(p + ".") for p in prefixes):
            del sys.modules[name]


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def resize_to_hw(arr: np.ndarray, hw: Tuple[int, int], interp=cv2.INTER_NEAREST) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    h, w = hw
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 3:
            # If a 3-channel depth accidentally appears, average it.
            arr = arr.mean(axis=-1)
    if arr.shape != (h, w):
        arr = cv2.resize(arr, (w, h), interpolation=interp)
    arr[~np.isfinite(arr)] = 0.0
    return arr.astype(np.float32)


def make_fuse_depth(pred_depth: np.ndarray, raw_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fused = np.asarray(raw_depth, dtype=np.float32).copy()
    pred_depth = np.asarray(pred_depth, dtype=np.float32)
    mask = np.asarray(mask) > 0
    fused[mask] = pred_depth[mask]
    return fused.astype(np.float32)


def add_panel_label(ax, text: str, fontsize=11, rotation=0):
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        rotation=rotation,
    )


def normalize_depth_group(depth_list: List[np.ndarray], valid: np.ndarray) -> List[np.ndarray]:
    vals = []
    valid_mask = np.asarray(valid) > 0
    for d in depth_list:
        d = np.asarray(d, dtype=np.float32)
        m = valid_mask & np.isfinite(d) & (d > 0)
        if np.any(m):
            vals.append(d[m])
    if not vals:
        return [np.zeros_like(depth_list[0], dtype=np.float32) for _ in depth_list]

    vv = np.concatenate(vals, axis=0)
    lo = float(np.percentile(vv, 1))
    hi = float(np.percentile(vv, 99))
    if hi <= lo + 1e-6:
        hi = lo + 1.0

    outs = []
    for d in depth_list:
        x = (np.asarray(d, dtype=np.float32) - lo) / (hi - lo)
        x = np.clip(x, 0.0, 1.0)
        outs.append(x.astype(np.float32))
    return outs


def normalize_error_for_paper(err: np.ndarray, mask: np.ndarray, valid: np.ndarray, vmax=0.15) -> np.ndarray:
    err = np.asarray(err, dtype=np.float32)
    region = (np.asarray(mask) > 0) & (np.asarray(valid) > 0)
    show = np.full_like(err, np.nan, dtype=np.float32)
    if vmax <= 0:
        vmax = 1e-6
    show[region] = np.clip(err[region] / float(vmax), 0.0, 1.0)
    return show


def masked_depth_visual(depth: np.ndarray, mask: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalize a depth map to [0,1] and show only the transparent-object region on a black background."""
    depth = np.asarray(depth, dtype=np.float32)
    region = (np.asarray(mask) > 0) & (np.asarray(valid) > 0) & np.isfinite(depth) & (depth > 0)
    out = np.full_like(depth, np.nan, dtype=np.float32)
    denom = max(hi - lo, 1e-6)
    if np.any(region):
        out[region] = np.clip((depth[region] - lo) / denom, 0.0, 1.0)
    return out


def raw_depth_visual(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Show raw depth in grayscale with invalid pixels as black, similar to prior paper figures."""
    depth = np.asarray(depth, dtype=np.float32)
    region = (np.asarray(valid) > 0) & np.isfinite(depth) & (depth > 0)
    out = np.full_like(depth, np.nan, dtype=np.float32)
    if np.any(region):
        vals = depth[region]
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))
        denom = max(hi - lo, 1e-6)
        out[region] = np.clip((depth[region] - lo) / denom, 0.0, 1.0)
    return out


def normalize_error_object_only(err: np.ndarray, mask: np.ndarray, valid: np.ndarray, vmax=0.05) -> np.ndarray:
    """Error map on black background, object region only, with paper-style fixed range."""
    err = np.asarray(err, dtype=np.float32)
    region = (np.asarray(mask) > 0) & (np.asarray(valid) > 0) & np.isfinite(err)
    out = np.full_like(err, np.nan, dtype=np.float32)
    vmax = max(float(vmax), 1e-6)
    if np.any(region):
        out[region] = np.clip(err[region] / vmax, 0.0, 1.0)
    return out


def scene_depth_range(depths: list, mask: np.ndarray, valid: np.ndarray) -> tuple:
    region_base = (np.asarray(mask) > 0) & (np.asarray(valid) > 0)
    vals = []
    for d in depths:
        d = np.asarray(d, dtype=np.float32)
        m = region_base & np.isfinite(d) & (d > 0)
        if np.any(m):
            vals.append(d[m])
    if not vals:
        return 0.0, 1.0
    vals = np.concatenate(vals, axis=0)
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if hi <= lo + 1e-6:
        hi = lo + 1.0
    return lo, hi


# =========================================================
# Cache loading and sample extraction
# =========================================================
def load_split_manifest(cache_root: Path, split: str) -> Dict[str, Any]:
    split_dir = cache_root / split
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def load_shard(path_str: str) -> Dict[str, Any]:
    return safe_torch_load(Path(path_str), map_location="cpu")


def squeeze_leading_one(x: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(x) and x.ndim >= 1 and x.shape[0] == 1:
        # cached shards often store [1, N, ...]
        # remove only the artificial leading singleton.
        x = x.squeeze(0)
    return x


def shard_num_samples(shard: Dict[str, Any]) -> int:
    ref = shard.get("rgb", None)
    if ref is None or not torch.is_tensor(ref):
        raise RuntimeError("Cache shard missing tensor key 'rgb'.")
    ref = squeeze_leading_one(ref)
    if ref.ndim < 1:
        raise RuntimeError(f"Unexpected rgb shape in shard: {tuple(ref.shape)}")
    return int(ref.shape[0])


def build_sample_locator(cache_root: Path, split: str):
    manifest = load_split_manifest(cache_root, split)
    split_dir = cache_root / split
    shards = [split_dir / s["file"] for s in manifest["shards"]]
    locator = []
    running = 0
    print("[Cache] building flat-sample locator ...")
    for shard_path in tqdm(shards, desc="Index shards"):
        shard = load_shard(str(shard_path))
        n = shard_num_samples(shard)
        locator.append({
            "path": shard_path,
            "start": running,
            "end": running + n,
            "count": n,
        })
        running += n
    print(f"[Cache] split={split}, samples={running}, shards={len(locator)}")
    return locator, running


def locate_sample(locator: List[Dict[str, Any]], flat_idx: int) -> Tuple[Path, int]:
    for item in locator:
        if item["start"] <= flat_idx < item["end"]:
            return item["path"], int(flat_idx - item["start"])
    raise IndexError(f"Sample index out of range: {flat_idx}")


def extract_tensor_sample(t: torch.Tensor, local_idx: int):
    t = squeeze_leading_one(t)
    if t.ndim == 0:
        return t
    if t.shape[0] <= local_idx:
        raise IndexError(f"local_idx={local_idx} out of tensor shape={tuple(t.shape)}")
    return t[local_idx]


def load_cache_sample(locator: List[Dict[str, Any]], flat_idx: int) -> Dict[str, Any]:
    shard_path, local_idx = locate_sample(locator, flat_idx)
    shard = load_shard(str(shard_path))
    out = {"idx": flat_idx, "scene_label": f"Scene {flat_idx}"}
    for k, v in shard.items():
        if torch.is_tensor(v):
            out[k] = extract_tensor_sample(v, local_idx).float().contiguous()
    return out


# =========================================================
# TODE-Trans loader / inference
# =========================================================
def load_tode_trans():
    purge_external_imports()
    if not TODE_WRAPPER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing TODE wrapper script: {TODE_WRAPPER_SCRIPT}")

    # TODE repo path used in your previous scripts.
    tode_root = Path(os.getenv("TODE_ROOT", str(PROJECT_ROOT / "third_party" / "TODE-main")))
    prepend_path(tode_root)

    mod = import_module_from_path("fapr_visual_tode_wrapper", TODE_WRAPPER_SCRIPT)
    load_fn = find_function(mod, ["load_tode_inferencer", "load_tode_model", "load_inferencer"])
    infer_fn = find_function(mod, ["infer_tode", "infer_tode_trans", "run_tode_inference"])

    print("[Load] TODE-Trans ...")
    model = load_fn()
    return model, infer_fn


def infer_tode_adapter(fn, model, rgb_uint8: np.ndarray, raw_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    try:
        pred = fn(model, rgb_uint8, raw_depth, use_bgr=False)
    except TypeError:
        try:
            pred = fn(model, rgb_uint8, raw_depth, mask, use_bgr=False)
        except TypeError:
            try:
                pred = fn(model, rgb_uint8, raw_depth)
            except TypeError:
                pred = fn(model, rgb_uint8, raw_depth, mask)
    return resize_to_hw(pred, raw_depth.shape, interp=cv2.INTER_NEAREST)


def load_tdcnet():
    purge_external_imports()
    if not TDCNET_WRAPPER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing TDCNet wrapper script: {TDCNET_WRAPPER_SCRIPT}")

    tdc_root = Path(os.getenv("TDCNET_ROOT", str(PROJECT_ROOT / "third_party" / "TDCNet-main")))
    prepend_path(tdc_root)

    mod = import_module_from_path("fapr_visual_tdcnet_wrapper", TDCNET_WRAPPER_SCRIPT)
    load_fn = find_function(
        mod,
        ["load_tdcnet_inferencer", "load_tdc_inferencer", "load_tdcnet_model", "load_tdc_model", "load_inferencer"],
    )
    infer_fn = find_function(
        mod,
        ["infer_tdcnet", "infer_tdc", "run_tdcnet_inference", "run_tdc_inference"],
    )

    print("[Load] TDCNet ...")
    model = load_fn()
    return model, infer_fn


def infer_tdcnet_adapter(fn, model, rgb_uint8: np.ndarray, raw_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    try:
        pred = fn(model, rgb_uint8, raw_depth, use_bgr=False)
    except TypeError:
        try:
            pred = fn(model, rgb_uint8, raw_depth, mask, use_bgr=False)
        except TypeError:
            try:
                pred = fn(model, rgb_uint8, raw_depth)
            except TypeError:
                pred = fn(model, rgb_uint8, raw_depth, mask)
    return resize_to_hw(pred, raw_depth.shape, interp=cv2.INTER_NEAREST)


# =========================================================
# ReMake loader / inference
# =========================================================
def load_remake_bundle():
    purge_external_imports()
    if not REMAKE_WRAPPER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing ReMake wrapper script: {REMAKE_WRAPPER_SCRIPT}")

    mod = import_module_from_path("fapr_visual_remake_wrapper", REMAKE_WRAPPER_SCRIPT)

    # In some environments ReMake's visualization dependency open3d may be missing.
    try:
        import open3d  # noqa: F401
    except Exception:
        sys.modules.setdefault("open3d", types.SimpleNamespace())

    # TODE/TDCNet may have already imported their own `models` / `utils` packages.
    # ReMake imports `models.remake`, and that file imports `utils.visualization`.
    # In some local ReMake copies, visualization is not needed for inference and may be
    # missing or shadowed by another repository's `utils`.  We therefore:
    #   1) purge colliding modules;
    #   2) put ReMake root at the front of sys.path;
    #   3) register a small dummy `utils.visualization` module with the two functions
    #      imported by ReMake but not used during evaluation.
    purge_external_imports()

    remake_root = Path(getattr(mod, "REMAKE_ROOT", os.getenv("REMAKE_ROOT", str(PROJECT_ROOT / "third_party" / "ReMake-main"))))
    if str(remake_root) in sys.path:
        sys.path.remove(str(remake_root))
    sys.path.insert(0, str(remake_root))

    # Create a namespace-like utils package that can still resolve ReMake's
    # real utils.transform while also providing a dummy utils.visualization.
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = [str(remake_root / "utils")]
    sys.modules["utils"] = utils_pkg

    vis_mod = types.ModuleType("utils.visualization")
    def _unused_visualization_stub(*args, **kwargs):
        return None
    vis_mod.plot_realat_depth = _unused_visualization_stub
    vis_mod.run_gradcam_on_encoder_img = _unused_visualization_stub
    sys.modules["utils.visualization"] = vis_mod

    print("[Load] ReMake ...")
    model = mod.load_remake_model()
    rel_model = mod.load_depthanything_model()
    rel_transform = mod.build_rel_transform()

    bundle = {
        "module": mod,
        "model": model,
        "rel_model": rel_model,
        "rel_transform": rel_transform,
    }
    return bundle


@torch.no_grad()
def infer_remake(bundle: Dict[str, Any], rgb_float: np.ndarray, raw_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mod = bundle["module"]
    model = bundle["model"]
    rel_model = bundle["rel_model"]
    rel_transform = bundle["rel_transform"]

    # Cache RGB is assumed to be RGB float in [0,1]. Convert to uint8 RGB.
    rgb_uint8 = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)

    model_h, model_w = mod.MODEL_SIZE
    eval_h, eval_w = mod.EVAL_SIZE

    rgb_m = cv2.resize(rgb_uint8, (model_w, model_h), interpolation=cv2.INTER_AREA)
    raw_m = cv2.resize(raw_depth.astype(np.float32), (model_w, model_h), interpolation=cv2.INTER_NEAREST)
    mask_m = cv2.resize(mask.astype(np.float32), (model_w, model_h), interpolation=cv2.INTER_NEAREST)

    rgb_t = torch.from_numpy((rgb_m.astype(np.float32) / 255.0).transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    raw_t = torch.from_numpy(raw_m).float().unsqueeze(0).to(DEVICE)
    mask_t = torch.from_numpy((mask_m > 0).astype(np.float32)).float().unsqueeze(0).to(DEVICE)

    rel_in = mod.preprocess_for_depthanything(rgb_m.astype(np.uint8), rel_transform).unsqueeze(0).to(DEVICE)
    rel_depth = rel_model.forward(rel_in).unsqueeze(1)
    rel_depth = F.interpolate(rel_depth, size=(model_h, model_w), mode="bilinear", align_corners=False)

    pred_model = model(rgb_t, rel_depth, raw_t, mask_t).view(1, 1, model_h, model_w)
    pred_eval = F.interpolate(pred_model, size=(eval_h, eval_w), mode="bilinear", align_corners=False)

    pred_np = pred_eval[0, 0].detach().cpu().numpy().astype(np.float32)
    return resize_to_hw(pred_np, raw_depth.shape, interp=cv2.INTER_NEAREST)


# =========================================================
# FAPR-Depth loader / inference
# =========================================================
def load_fapr_model():
    if not FAPR_TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing FAPR train/model script: {FAPR_TRAIN_SCRIPT}")
    if not FAPR_CANDIDATE_CKPT.exists():
        raise FileNotFoundError(f"Missing FAPR checkpoint: {FAPR_CANDIDATE_CKPT}")

    # ReMake/TDCNet/TODE repositories use generic top-level package names.
    # Clear them before importing the FAPR model definition and place the
    # completion source directory at the front of sys.path.
    purge_external_imports()
    mod = import_module_from_path("fapr_v6_model_for_visual", FAPR_TRAIN_SCRIPT)
    base_root = Path(mod.BASE_SOURCE_ROOT)
    base_root_text = str(base_root)
    if base_root_text in sys.path:
        sys.path.remove(base_root_text)
    sys.path.insert(0, base_root_text)
    base_mod = mod.load_base_source_module()

    print("[Load] FAPR-Depth v6 Candidate ...")
    model = mod.FailureAwarePosteriorDepth(base_mod).to(DEVICE)
    payload = safe_torch_load(FAPR_CANDIDATE_CKPT, map_location="cpu")
    state = payload.get("model", payload.get("model_state_dict", payload))
    clean = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "FAPR checkpoint/model mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}\n"
            f"missing[:10]={missing[:10]}\n"
            f"unexpected[:10]={unexpected[:10]}"
        )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    checkpoint_phase = str(payload.get("phase", "joint"))
    if checkpoint_phase not in {"proposal", "risk", "joint"}:
        # Candidate output requires the proposal branch. This fallback is only
        # used for an unusual checkpoint without a compatible recorded phase.
        evaluation_phase = "joint"
    else:
        evaluation_phase = checkpoint_phase

    print(
        f"[Load] FAPR checkpoint phase={checkpoint_phase}, "
        f"evaluation phase={evaluation_phase}, "
        f"refine_epoch={payload.get('refine_epoch', -1)}"
    )
    return {
        "module": mod,
        "model": model,
        "payload": payload,
        "phase": evaluation_phase,
    }


@torch.no_grad()
def infer_fapr_model(bundle: Dict[str, Any], sample: Dict[str, Any]) -> np.ndarray:
    mod = bundle["module"]
    model = bundle["model"]
    phase = bundle["phase"]

    batch = {
        key: value.to(DEVICE, non_blocking=True).float()
        for key, value in sample.items()
        if torch.is_tensor(value)
    }
    inp = mod.build_inputs(batch)

    with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
        out = model(
            inp,
            phase=phase,
            augment_safe=False,
        )

    # Candidate benchmark is the accepted paper/deployment output and already
    # preserves raw observations outside the transparent mask.
    pred = out["candidate_benchmark"][0, 0]
    return pred.detach().float().cpu().numpy().astype(np.float32)


# =========================================================
# Figure generation
# =========================================================
def save_combined_depth_error_figure(rows: List[Dict[str, Any]], save_png: Path, save_pdf: Path):
    """
    Paper-style qualitative figure closer to the second reference figure:
        - depth maps use a dedicated depth colormap (viridis by default)
        - error maps keep a separate error colormap (turbo)
        - RGB at left
        - GT depth on left-bottom
        - method depth maps shown mainly on the transparent-object region with a black background
        - error maps shown on the transparent-object region with a black background
        - a single vertical color bar on the right for error magnitude
    """
    method_specs = METHOD_SPECS
    n_scenes = len(rows)
    n_methods = len(method_specs)

    # No GridSpec column for colorbar. We reserve right-side whitespace and place
    # a shorter standalone colorbar there, so it will not collide with the last title.
    n_cols = 4 + n_methods
    n_grid_rows = 2 * n_scenes

    depth_cmap = plt.get_cmap(DEPTH_CMAP_NAME).copy()
    depth_cmap.set_bad(color="black")
    error_cmap = plt.get_cmap(ERROR_CMAP_NAME).copy()
    error_cmap.set_bad(color="black")
    raw_cmap = plt.get_cmap("gray").copy()
    raw_cmap.set_bad(color="black")

    width_ratios = [0.78, 0.34, 1.58, 0.34] + [1.58] * n_methods
    total_width = sum(width_ratios)

    # Add a little extra canvas width for the standalone colorbar on the right.
    fig = plt.figure(figsize=(total_width * 1.12 + 0.65, 1.60 * n_grid_rows), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(
        n_grid_rows,
        n_cols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[1.0] * n_grid_rows,
        wspace=0.035,
        hspace=0.025,
    )

    # Model headers are attached to the top-row Axes instead of being placed
    # with fig.text(...).  This makes each title use the true column center
    # after GridSpec / subplots_adjust / bbox_inches="tight", so the headers
    # stay aligned with the corresponding image columns.

    err_mappable = None

    for i, row in enumerate(rows):
        r0 = 2 * i
        r1 = r0 + 1

        rgb = row["rgb"]
        gt = row["gt_depth"]
        mask = row["mask"]
        valid = row["valid"]

        method_depths = [row[key] for _, key in method_specs]
        lo, hi = scene_depth_range(method_depths + [gt], mask, valid)
        gt_show = masked_depth_visual(gt, mask, valid, lo, hi)

        ax_scene = fig.add_subplot(gs[r0:r1 + 1, 0])
        add_panel_label(ax_scene, row.get("scene_label", f"Scene {row.get('idx', i)}"), fontsize=12, rotation=0)

        ax_left_top_label = fig.add_subplot(gs[r0, 1])
        add_panel_label(ax_left_top_label, "RGB", fontsize=11, rotation=90)

        ax_left_bottom_label = fig.add_subplot(gs[r1, 1])
        add_panel_label(ax_left_bottom_label, "GT Depth", fontsize=11, rotation=90)

        ax_rgb = fig.add_subplot(gs[r0, 2])
        ax_rgb.imshow(rgb)
        ax_rgb.axis("off")

        ax_gt = fig.add_subplot(gs[r1, 2])
        ax_gt.imshow(gt_show, cmap=depth_cmap, vmin=0, vmax=1)
        ax_gt.set_facecolor("black")
        ax_gt.axis("off")

        ax_right_top_label = fig.add_subplot(gs[r0, 3])
        add_panel_label(ax_right_top_label, "Depth", fontsize=11, rotation=90)

        ax_right_bottom_label = fig.add_subplot(gs[r1, 3])
        add_panel_label(ax_right_bottom_label, "Error Map", fontsize=11, rotation=90)

        for j, (title, key) in enumerate(method_specs):
            col = 4 + j
            pred = row[key]

            if key == "raw_depth":
                depth_show = raw_depth_visual(pred, valid)
                depth_map_cmap = raw_cmap
                ax_face = "white"
            else:
                depth_show = masked_depth_visual(pred, mask, valid, lo, hi)
                depth_map_cmap = depth_cmap
                ax_face = "black"

            ax_depth = fig.add_subplot(gs[r0, col])
            ax_depth.imshow(depth_show, cmap=depth_map_cmap, vmin=0, vmax=1)
            ax_depth.set_facecolor(ax_face)
            ax_depth.axis("off")

            # Add model names only above the first scene.
            # Because the title belongs to ax_depth itself, it is exactly
            # centered over the corresponding method column.
            if i == 0:
                display_title = "Ours" if title == "FAPR-Depth (Ours)" else title
                ax_depth.set_title(
                    display_title,
                    fontsize=14,
                    fontweight="bold",
                    pad=8,
                    loc="center",
                )

            err = np.abs(pred - gt)
            err_show = normalize_error_object_only(err, mask=mask, valid=valid, vmax=0.05)

            ax_err = fig.add_subplot(gs[r1, col])
            im = ax_err.imshow(err_show, cmap=error_cmap, vmin=0, vmax=1)
            ax_err.set_facecolor("black")
            ax_err.axis("off")
            err_mappable = im


    # Add one shared vertical error colorbar on the right.
    # Shorter than the previous version to match common paper figures and reduce clutter.
    if err_mappable is not None:
        cax = fig.add_axes([0.940, 0.33, 0.016, 0.36])
        cb = fig.colorbar(err_mappable, cax=cax)
        cb.set_ticks([0.0, 1.0])
        cb.set_ticklabels(["0", "0.05 m"])
        # Conventional orientation: low error at the bottom and high error at the top.
        cb.ax.tick_params(labelsize=11, length=0)
        cb.outline.set_linewidth(0.6)
        cb.set_label(
            "Absolute error (m)",
            fontsize=11,
            rotation=90,
            labelpad=10,
        )

    # Keep enough room on the right for the colorbar.
    plt.subplots_adjust(
        left=0.020,
        right=0.915,
        top=0.955,
        bottom=0.012,
        wspace=0.035,
        hspace=0.025,
    )
    fig.savefig(save_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(save_pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    plt.close("all")


# =========================================================
# Main
# =========================================================
@torch.no_grad()
def main():
    print("=" * 120)
    print("FAPR-Depth top4 qualitative comparison")
    print("=" * 120)
    print("DEVICE:      ", DEVICE)
    print("CACHE_ROOT:  ", CACHE_ROOT)
    print("SPLIT:       ", SPLIT)
    print("Samples:     ", FIGURE_SAMPLE_INDICES)
    print("OUT_DIR:     ", OUT_DIR)
    print("Method order:", [x[0] for x in METHOD_SPECS])
    print("Depth cmap:  ", DEPTH_CMAP_NAME)
    print("Error cmap:  ", ERROR_CMAP_NAME)

    locator, total_samples = build_sample_locator(CACHE_ROOT, SPLIT)
    for idx in FIGURE_SAMPLE_INDICES:
        if idx < 0 or idx >= total_samples:
            raise IndexError(f"FIGURE_SAMPLE_INDICES contains {idx}, but split only has {total_samples} samples")

    # Load all comparison models.
    tode_model, tode_infer = load_tode_trans()
    tdc_model, tdc_infer = load_tdcnet()
    remake_bundle = load_remake_bundle()
    fapr_bundle = load_fapr_model()

    rows = []
    meta = {
        "split": SPLIT,
        "figure_sample_indices": FIGURE_SAMPLE_INDICES,
        "method_order": [m[0] for m in METHOD_SPECS],
        "display_name_notes": {
            "Ours": "FAPR-Depth v6 best_candidate.pth / Candidate benchmark",
        },
        "outputs": {
            "png": str(SAVE_COMBINED_PNG),
            "pdf": str(SAVE_COMBINED_PDF),
        },
    }

    for flat_idx in tqdm(FIGURE_SAMPLE_INDICES, desc="Generating qualitative rows"):
        sample = load_cache_sample(locator, flat_idx)

        rgb_t = sample["rgb"]
        raw_t = sample["raw_depth"]
        gt_t = sample["gt_depth"]
        mask_t = sample["mask"]
        valid_t = sample["valid"]

        # Convert cache tensors to numpy display format.
        rgb_np = to_numpy(rgb_t)
        if rgb_np.ndim == 3 and rgb_np.shape[0] == 3:
            rgb_np = np.transpose(rgb_np, (1, 2, 0))
        rgb_np = np.clip(rgb_np, 0.0, 1.0).astype(np.float32)

        raw_np = resize_to_hw(to_numpy(raw_t), (240, 320), interp=cv2.INTER_NEAREST)
        gt_np = resize_to_hw(to_numpy(gt_t), (240, 320), interp=cv2.INTER_NEAREST)
        mask_np = resize_to_hw(to_numpy(mask_t), (240, 320), interp=cv2.INTER_NEAREST)
        valid_np = resize_to_hw(to_numpy(valid_t), (240, 320), interp=cv2.INTER_NEAREST)

        # TODE-Trans.
        tode_pred = infer_tode_adapter(tode_infer, tode_model, (rgb_np * 255).astype(np.uint8), raw_np, mask_np)
        tode_fuse = make_fuse_depth(tode_pred, raw_np, mask_np)

        # TDCNet.
        tdc_pred = infer_tdcnet_adapter(tdc_infer, tdc_model, (rgb_np * 255).astype(np.uint8), raw_np, mask_np)
        tdc_fuse = make_fuse_depth(tdc_pred, raw_np, mask_np)

        # ReMake.
        remake_pred = infer_remake(remake_bundle, rgb_np, raw_np, mask_np)
        remake_fuse = make_fuse_depth(remake_pred, raw_np, mask_np)

        # FAPR-Depth v6 Candidate benchmark.
        fapr_ours = infer_fapr_model(fapr_bundle, sample)
        fapr_ours = resize_to_hw(
            fapr_ours,
            raw_np.shape,
            interp=cv2.INTER_NEAREST,
        )

        row = {
            "idx": flat_idx,
            "scene_label": f"Scene {flat_idx}",
            "rgb": rgb_np,
            "raw_depth": raw_np,
            "gt_depth": gt_np,
            "mask": mask_np,
            "valid": valid_np,
            "tode_trans": tode_fuse,
            "tdcnet": tdc_fuse,
            "remake": remake_fuse,
            "fapr_depth": fapr_ours,
        }
        rows.append(row)

    save_combined_depth_error_figure(rows, SAVE_COMBINED_PNG, SAVE_COMBINED_PDF)
    SAVE_META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved:")
    print("  ", SAVE_COMBINED_PNG)
    print("  ", SAVE_COMBINED_PDF)
    print("  ", SAVE_META_JSON)
    print("\nDone.")


if __name__ == "__main__":
    main()
