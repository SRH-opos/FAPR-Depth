#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Profile FAPR-Depth v6 efficiency and export a paper-ready table.

Measured cumulative output paths
--------------------------------
1. Backbone Baseline
2. Posterior Fusion + Backbone
3. Safe Posterior
4. Full Candidate
5. Risk-Accepted Output

Reported fields
---------------
- active/cumulative parameter count;
- incremental parameter count;
- profiled GFLOPs for supported operators;
- mean / median / p90 latency;
- throughput;
- total peak CUDA allocation;
- activation-memory increase over the loaded-model baseline.

Notes
-----
- Latency is measured with batch size 1 on one cached test sample.
- CUDA timings use torch.cuda.Event.
- Profiled FLOPs come from torch.profiler and may undercount unsupported custom
  operators. The output table labels them accordingly.
- The full model remains loaded for every stage, so total peak VRAM reflects
  deployment with the complete checkpoint. Activation delta is the cleaner
  stage-to-stage comparison.

Outputs
-------
fapr_efficiency_table.csv
fapr_efficiency_table.md
fapr_efficiency_details.json
fapr_parameter_groups.csv
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch

from fapr_analysis_common import (
    add_common_args,
    batch_sample_count,
    bootstrap,
    make_loader,
    move_batch,
    slice_batch,
    write_csv,
    write_json,
    write_run_manifest,
)


STAGE_ORDER = [
    "Backbone Baseline",
    "Posterior Fusion + Backbone",
    "Safe Posterior",
    "Full Candidate",
    "Risk-Accepted Output",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile FAPR v6 parameters, FLOPs, latency and CUDA memory."
    )
    add_common_args(parser, "09_efficiency", default_phase="joint")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="Skip torch.profiler FLOP estimation.",
    )
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=1,
        help="Forward repeats inside the FLOP profiler.",
    )
    return parser.parse_args()


def unique_parameters(
    modules_or_parameters: Iterable[Any],
) -> List[torch.nn.Parameter]:
    parameters: List[torch.nn.Parameter] = []
    seen: Set[int] = set()

    for item in modules_or_parameters:
        if isinstance(item, torch.nn.Module):
            iterator = item.parameters()
        elif isinstance(item, torch.nn.Parameter):
            iterator = [item]
        else:
            try:
                iterator = iter(item)
            except TypeError:
                continue

        for parameter in iterator:
            if not isinstance(parameter, torch.nn.Parameter):
                continue
            identifier = id(parameter)
            if identifier in seen:
                continue
            seen.add(identifier)
            parameters.append(parameter)

    return parameters


def parameter_count(parameters: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(parameter.numel() for parameter in parameters))


def adapter_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    names = [
        "adapt_first",
        "adapt_e1",
        "adapt_e2",
        "adapt_e3",
        "adapt_e4",
        "adapt_d1",
        "adapt_d2",
        "adapt_d3",
        "adapt_out",
    ]
    return [getattr(model, name) for name in names]


def build_parameter_groups(model: torch.nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    posterior_modules = [
        model.base_stream,
        *adapter_modules(model),
        model.aligner,
        model.failure_net,
        model.shared,
        model.missing_expert,
        model.biased_expert,
        model.boundary_expert,
        model.router,
    ]

    groups = {
        "Backbone reference": unique_parameters([model.base_reference]),
        "Adapted completion and posterior": unique_parameters(posterior_modules),
        "Safe Anchor head": unique_parameters([model.safe_anchor]),
        "Proposal and refinement-risk head": unique_parameters([model.risk_refiner]),
        "Safe direct gate": unique_parameters(
            list(model.safe_anchor.gate_parameters())
        ),
        "Safe counterfactual risk": unique_parameters(
            list(model.safe_anchor.risk_parameters())
        ),
        "Detail proposal": unique_parameters(
            list(model.risk_refiner.proposal_parameters())
        ),
        "Proposal risk estimator": unique_parameters(
            list(model.risk_refiner.risk_parameters())
        ),
        "Full model": unique_parameters([model]),
    }
    return groups


def stage_parameter_sets(
    groups: Mapping[str, Sequence[torch.nn.Parameter]],
) -> Dict[str, List[torch.nn.Parameter]]:
    backbone = unique_parameters(groups["Backbone reference"])
    posterior = unique_parameters(
        [
            groups["Backbone reference"],
            groups["Adapted completion and posterior"],
        ]
    )
    safe = unique_parameters(
        [
            posterior,
            groups["Safe Anchor head"],
        ]
    )
    full = unique_parameters(
        [
            safe,
            groups["Proposal and refinement-risk head"],
        ]
    )

    return {
        "Backbone Baseline": backbone,
        "Posterior Fusion + Backbone": posterior,
        "Safe Posterior": safe,
        "Full Candidate": full,
        "Risk-Accepted Output": full,
    }


def get_one_input(ctx) -> Dict[str, torch.Tensor]:
    loader = make_loader(ctx)
    cpu_batch = next(iter(loader))
    total = batch_sample_count(cpu_batch)
    cpu_part = slice_batch(cpu_batch, 0, 1, total)
    gpu_part = move_batch(cpu_part, ctx.device)
    return ctx.train_mod.build_inputs(gpu_part)


def make_stage_functions(
    ctx,
    inp: Dict[str, torch.Tensor],
) -> Dict[str, Callable[[], Any]]:
    model = ctx.model

    def backbone():
        return model.forward_reference(
            inp["rgb"],
            inp["raw"],
        )

    def posterior_plus_backbone():
        posterior = model.forward_posterior(inp)
        anchor = model.forward_reference(
            inp["rgb"],
            inp["raw"],
        )
        return posterior["fused"], anchor

    def safe():
        return model(
            inp,
            phase="safe",
            augment_safe=False,
        )["safe_benchmark"]

    def candidate():
        return model(
            inp,
            phase="proposal",
            augment_safe=False,
        )["candidate_benchmark"]

    def joint():
        return model(
            inp,
            phase="joint",
            augment_safe=False,
        )["benchmark_output"]

    return {
        "Backbone Baseline": backbone,
        "Posterior Fusion + Backbone": posterior_plus_backbone,
        "Safe Posterior": safe,
        "Full Candidate": candidate,
        "Risk-Accepted Output": joint,
    }


def execute_stage(
    ctx,
    function: Callable[[], Any],
) -> Any:
    with torch.inference_mode():
        with torch.autocast(
            device_type=ctx.device.type,
            dtype=(
                torch.float16
                if ctx.device.type == "cuda"
                else torch.bfloat16
            ),
            enabled=ctx.use_amp,
        ):
            return function()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup_stage(
    ctx,
    function: Callable[[], Any],
    warmup: int,
) -> None:
    for _ in range(max(0, int(warmup))):
        output = execute_stage(ctx, function)
        del output
    synchronize(ctx.device)


def measure_latency(
    ctx,
    function: Callable[[], Any],
    repeats: int,
) -> Dict[str, float]:
    repeats = max(1, int(repeats))
    latencies: List[float] = []

    if ctx.device.type == "cuda":
        starts = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(repeats)
        ]
        ends = [
            torch.cuda.Event(enable_timing=True)
            for _ in range(repeats)
        ]

        for index in range(repeats):
            starts[index].record()
            output = execute_stage(ctx, function)
            ends[index].record()
            del output

        torch.cuda.synchronize(ctx.device)
        latencies = [
            float(start.elapsed_time(end))
            for start, end in zip(starts, ends)
        ]
    else:
        for _ in range(repeats):
            start = time.perf_counter()
            output = execute_stage(ctx, function)
            elapsed = (time.perf_counter() - start) * 1000.0
            del output
            latencies.append(float(elapsed))

    values = np.asarray(latencies, dtype=np.float64)
    return {
        "latency_mean_ms": float(np.mean(values)),
        "latency_median_ms": float(np.median(values)),
        "latency_p90_ms": float(np.percentile(values, 90)),
        "latency_std_ms": float(np.std(values)),
        "throughput_fps": float(1000.0 / np.mean(values)),
    }


def measure_memory(
    ctx,
    function: Callable[[], Any],
) -> Dict[str, float]:
    if ctx.device.type != "cuda":
        return {
            "baseline_allocated_mb": float("nan"),
            "peak_allocated_mb": float("nan"),
            "activation_delta_mb": float("nan"),
            "peak_reserved_mb": float("nan"),
        }

    torch.cuda.empty_cache()
    synchronize(ctx.device)
    baseline = float(
        torch.cuda.memory_allocated(ctx.device)
        / (1024.0 ** 2)
    )
    torch.cuda.reset_peak_memory_stats(ctx.device)

    output = execute_stage(ctx, function)
    synchronize(ctx.device)

    peak_allocated = float(
        torch.cuda.max_memory_allocated(ctx.device)
        / (1024.0 ** 2)
    )
    peak_reserved = float(
        torch.cuda.max_memory_reserved(ctx.device)
        / (1024.0 ** 2)
    )
    del output
    synchronize(ctx.device)

    return {
        "baseline_allocated_mb": baseline,
        "peak_allocated_mb": peak_allocated,
        "activation_delta_mb": max(0.0, peak_allocated - baseline),
        "peak_reserved_mb": peak_reserved,
    }


def profile_flops(
    ctx,
    function: Callable[[], Any],
    repeats: int,
) -> Tuple[float, str]:
    try:
        from torch.profiler import (
            ProfilerActivity,
            profile,
        )

        activities = [ProfilerActivity.CPU]
        if ctx.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
            profile_memory=False,
        ) as profiler:
            for _ in range(max(1, int(repeats))):
                output = execute_stage(ctx, function)
                del output
            synchronize(ctx.device)

        total_flops = 0.0
        for event in profiler.key_averages():
            value = getattr(event, "flops", 0)
            if value:
                total_flops += float(value)

        total_flops /= max(1, int(repeats))
        if total_flops <= 0:
            return float("nan"), "torch.profiler returned no FLOP counts"
        return total_flops / 1.0e9, ""
    except Exception as error:
        return float("nan"), f"{type(error).__name__}: {error}"


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "Stage",
        "Params (M)",
        "ΔParams (M)",
        "GFLOPs*",
        "Latency (ms)",
        "P90 (ms)",
        "FPS",
        "Peak VRAM (MB)",
        "Activation Δ (MB)",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]

    for row in rows:
        def fmt(value, digits=3):
            if value is None or not np.isfinite(float(value)):
                return "—"
            return f"{float(value):.{digits}f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["stage"]),
                    fmt(row["active_params_m"], 3),
                    fmt(row["incremental_params_m"], 3),
                    fmt(row["profiled_gflops"], 2),
                    fmt(row["latency_mean_ms"], 2),
                    fmt(row["latency_p90_ms"], 2),
                    fmt(row["throughput_fps"], 2),
                    fmt(row["peak_allocated_mb"], 1),
                    fmt(row["activation_delta_mb"], 1),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "*GFLOPs are torch.profiler supported-operator estimates and may undercount custom operators.*",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ctx = bootstrap(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp = get_one_input(ctx)
    stage_functions = make_stage_functions(ctx, inp)

    groups = build_parameter_groups(ctx.model)
    stages = stage_parameter_sets(groups)

    group_rows = []
    for name, parameters in groups.items():
        group_rows.append(
            {
                "parameter_group": name,
                "parameters": parameter_count(parameters),
                "parameters_million": parameter_count(parameters) / 1.0e6,
            }
        )

    # Explicit v6-trainable heads.
    v6_trainable_parameters = unique_parameters(
        [
            groups["Safe direct gate"],
            groups["Safe counterfactual risk"],
            groups["Detail proposal"],
            groups["Proposal risk estimator"],
        ]
    )
    group_rows.append(
        {
            "parameter_group": "All v6 trainable heads (unique)",
            "parameters": parameter_count(v6_trainable_parameters),
            "parameters_million": parameter_count(v6_trainable_parameters) / 1.0e6,
        }
    )

    rows: List[Dict[str, Any]] = []
    previous_params = 0

    for stage in STAGE_ORDER:
        function = stage_functions[stage]
        active_params = parameter_count(stages[stage])

        print("\n" + "=" * 100)
        print("Profiling:", stage)
        print("=" * 100)

        warmup_stage(
            ctx,
            function,
            args.warmup,
        )
        latency = measure_latency(
            ctx,
            function,
            args.repeats,
        )
        memory = measure_memory(
            ctx,
            function,
        )

        if args.skip_flops:
            gflops = float("nan")
            flops_error = "Skipped by --skip-flops"
        else:
            gflops, flops_error = profile_flops(
                ctx,
                function,
                args.profile_repeats,
            )

        row = {
            "stage": stage,
            "input_height": int(inp["rgb"].shape[-2]),
            "input_width": int(inp["rgb"].shape[-1]),
            "batch_size": int(inp["rgb"].shape[0]),
            "amp": bool(ctx.use_amp),
            "active_parameters": int(active_params),
            "active_params_m": active_params / 1.0e6,
            "incremental_parameters": int(active_params - previous_params),
            "incremental_params_m": (active_params - previous_params) / 1.0e6,
            "profiled_gflops": gflops,
            "flops_note": flops_error,
            **latency,
            **memory,
        }
        rows.append(row)
        previous_params = active_params

        print(json.dumps(row, indent=2, ensure_ascii=False))

    write_csv(out_dir / "fapr_efficiency_table.csv", rows)
    write_csv(out_dir / "fapr_parameter_groups.csv", group_rows)
    write_json(
        out_dir / "fapr_efficiency_details.json",
        {
            "protocol": {
                "device": str(ctx.device),
                "amp": ctx.use_amp,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "input_shape": list(inp["rgb"].shape),
                "checkpoint": str(args.checkpoint),
            },
            "efficiency_rows": rows,
            "parameter_groups": group_rows,
        },
    )
    write_run_manifest(
        ctx,
        {
            "analysis": "efficiency",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "input_shape": list(inp["rgb"].shape),
        },
    )

    markdown = markdown_table(rows)
    (out_dir / "fapr_efficiency_table.md").write_text(
        markdown,
        encoding="utf-8",
    )

    print("\n" + markdown)
    print("\nParameter groups")
    for row in group_rows:
        print(row)
    print("\nSaved to:", out_dir)


if __name__ == "__main__":
    main()
