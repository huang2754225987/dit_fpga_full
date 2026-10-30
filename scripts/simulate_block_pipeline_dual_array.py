#!/usr/bin/env python3
"""Explore a row-sliced 32x32 + 16x32 systolic-array schedule.

This is an analytical scheduling model, not RTL or implementation evidence.
The existing 32x32 array is unchanged. A hypothetical 16x32 INT8 sidecar uses
half as many MAC lanes and can process one 16-row sub-tile in the time that
the 32x32 array processes one 32-row tile. The main array can process the same
16-row sub-tile in half that time. Durations scale linearly with row count;
this optimistic assumption must be replaced by measured RTL tile latencies
before the sidecar is committed to hardware.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import simulate_block_pipeline as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "pipeline_sim_dual_array"
OUT.mkdir(parents=True, exist_ok=True)

MAIN = "ARRAY32"
SIDE = "ARRAY16x32"
RESOURCES = {MAIN, SIDE, "LN_A", "LN_B", "LN_C", "R1", "R2", "SFT"}


def split_row_tasks():
    original = base.build_tasks()
    tasks: dict[str, tuple[str, int, list[str]]] = {}
    for name, (resource, duration, deps) in original.items():
        for half in range(2):
            split_name = f"{name}.h{half}"
            if resource == "ARRAY":
                # A 16-row sub-tile: half the 32x32 array's time on the main
                # array, or a full 32-row-tile time on a 16x32 sidecar.
                split_resource = "ARRAY_FLEX"
                split_duration = duration
            else:
                split_resource = resource
                split_duration = math.ceil(duration / 2)

            split_deps: list[str] = []
            for dep in deps:
                # Same row tile means data dependency is local to this half.
                parent_name = name.rsplit(".h", 1)[0]
                suffix = parent_name[parent_name.rfind(".b"):] if ".b" in parent_name else ""
                branch_barrier = parent_name.startswith("sqk.")
                if not branch_barrier and suffix and dep.endswith(suffix):
                    split_deps.append(f"{dep}.h{half}")
                else:
                    # Cross-row barriers (the QKV K/V-complete fence before
                    # attention) must wait for both halves of every row tile.
                    split_deps.extend((f"{dep}.h0", f"{dep}.h1"))
            tasks[split_name] = (split_resource, split_duration, split_deps)
    return tasks


def simulate():
    tasks = split_row_tasks()
    free = {resource: 0 for resource in RESOURCES}
    busy = {resource: 0 for resource in RESOURCES}
    start: dict[str, int] = {}
    finish: dict[str, int] = {}

    def fast_duration(name: str) -> int:
        resource, duration, _ = tasks[name]
        return math.ceil(duration / 2) if resource == "ARRAY_FLEX" else duration

    def dep_finish(name: str, cache: dict[str, int]) -> int:
        if name not in cache:
            _, _, deps = tasks[name]
            cache[name] = max((dep_finish(dep, cache) for dep in deps), default=0) + fast_duration(name)
        return cache[name]

    cache: dict[str, int] = {}
    order = sorted(tasks, key=lambda name: (dep_finish(name, cache) - fast_duration(name), -dep_finish(name, cache)))

    for name in order:
        resource, duration, deps = tasks[name]
        dep_ready = max((finish[dep] for dep in deps), default=0)
        if resource == "ARRAY_FLEX":
            candidates = []
            for array, factor in ((MAIN, 0.5), (SIDE, 1.0)):
                run = math.ceil(duration * factor)
                task_start = max(dep_ready, free[array])
                candidates.append((task_start + run, task_start, array, run))
            _, task_start, chosen, run = min(candidates)
            start[name], finish[name] = task_start, task_start + run
            free[chosen] = finish[name]
            busy[chosen] += run
        else:
            task_start = max(dep_ready, free[resource])
            start[name], finish[name] = task_start, task_start + duration
            free[resource] = finish[name]
            busy[resource] += duration

    for name, (_, _, deps) in tasks.items():
        if any(finish[dep] > start[name] for dep in deps):
            raise AssertionError(f"dependency violation in {name}")

    makespan = max(finish.values())
    # MAC work is unchanged from the base model; peak capacity counts both arrays.
    one_array_ideal = math.ceil(base.BLOCK_MACS / (32 * 32))
    total_mac_peak = (32 * 32 + 16 * 32) * makespan
    report = {
        "model": "optimistic 16-row split schedule: existing 32x32 plus hypothetical 16x32 sidecar",
        "status": "analytical estimate only; no sidecar RTL, synthesis, or timing closure",
        "single_array_baseline_cycles": base.simulate()[0],
        "dual_array_estimated_cycles": makespan,
        "estimated_speedup_vs_single_array": base.simulate()[0] / makespan,
        "single_array_mac_lower_bound_cycles": one_array_ideal,
        "sidecar_gain_vs_single_array_ideal": round(one_array_ideal / makespan, 4),
        "resource_busy_cycles": dict(sorted(busy.items())),
        "resource_busy_fraction": {key: round(value / makespan, 4) for key, value in sorted(busy.items())},
        "array_mac_utilization_combined": round(base.BLOCK_MACS / total_mac_peak, 4),
        "dsp48_estimate": {"main_32x32": 1024, "sidecar_16x32": 512, "combined": 1536, "xczu7ev_capacity": 1728, "capacity_fraction": round(1536 / 1728, 4)},
        "row_split_assumption": "16-row work takes 0.5x base 32-row cycles on 32x32 and 1.0x on 16x32; streaming unit durations halve",
        "tasks": len(tasks),
    }
    fixed_step_cycles = 30528 + 294912 + 590336 + 10362 + 12800
    no_skip_step = 14 * makespan + fixed_step_cycles
    skip_step = int(14 * (1 - 0.40857) * makespan + fixed_step_cycles)
    report["step_estimates_at_200mhz"] = {
        "no_skip_ms": round(no_skip_step / 200_000, 1),
        "skip_trace_ms": round(skip_step / 200_000, 1),
        "image_50_steps_skip_trace_s": round(skip_step * 50 / 200_000_000, 2),
        "includes": "modeled block schedule plus existing fixed front-end/final/Euler estimate",
        "excludes": "RTL DMA, DDR wait/traffic, full-system timing closure, and image-level RTL execution",
    }
    (OUT / "dual_array_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    return report


if __name__ == "__main__":
    print(json.dumps(simulate(), indent=2))
