#!/usr/bin/env python3
"""Run FAPR-Depth inference on one cached `.pt` shard.

The public research scripts use precomputed cache shards. A shard should contain
at least: rgb, raw_depth, mask, valid, rel_aligned. If gt_depth is absent, raw
depth is used only to satisfy the common input builder; no accuracy metrics are
computed by this script.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def import_train_module(repo_root: Path):
    train_path = repo_root / "train.py"
    spec = importlib.util.spec_from_file_location("fapr_train_public", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> Dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("model_state_dict", payload))
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}")
    return payload


def normalize_shard(shard: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    required = ("rgb", "raw_depth", "mask", "valid", "rel_aligned")
    missing = [key for key in required if key not in shard]
    if missing:
        raise KeyError(f"Missing required cache fields: {missing}")
    result = {k: v for k, v in shard.items() if torch.is_tensor(v)}
    if "gt_depth" not in result:
        result["gt_depth"] = result["raw_depth"].clone()
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-shard", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--phase", choices=["safe", "proposal", "risk", "joint"], default="joint")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    os.environ["FAPR_PROJECT_ROOT"] = str(repo_root)
    os.environ["FAPR_BASE_SOURCE_ROOT"] = str(args.base_source_root.resolve())
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    train_mod = import_train_module(repo_root)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    base_mod = train_mod.load_base_source_module()
    model = train_mod.FailureAwarePosteriorDepth(base_mod).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    shard = torch.load(args.cache_shard, map_location="cpu", weights_only=False)
    batch = normalize_shard(shard)
    batch = {k: v.to(device).float() for k, v in batch.items()}
    inp = train_mod.build_inputs(batch)

    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        out = model(inp, phase=args.phase, augment_safe=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = {
        "benchmark_output": out["benchmark_output"].detach().cpu().numpy(),
        "candidate_benchmark": out["candidate_benchmark"].detach().cpu().numpy(),
        "safe_benchmark": out["safe_benchmark"].detach().cpu().numpy(),
        "failure_probability": out["fail_prob"].detach().cpu().numpy(),
        "source_allocation": out["alpha"].detach().cpu().numpy(),
        "expert_routing": out["pi"].detach().cpu().numpy(),
        "uncertainty": out["final_logb"].detach().cpu().numpy(),
        "safe_gate": out["safe_gate"].detach().cpu().numpy(),
        "acceptance": out["acceptance"].detach().cpu().numpy(),
    }
    np.savez_compressed(args.output_dir / "fapr_outputs.npz", **selected)
    print(f"Saved outputs to: {args.output_dir / 'fapr_outputs.npz'}")


if __name__ == "__main__":
    main()
