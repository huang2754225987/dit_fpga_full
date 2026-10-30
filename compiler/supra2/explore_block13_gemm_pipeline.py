"""Capacity-checked, non-executable block-13 GEMM schedule exploration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compile_block13_tile_program import MEASURED
from explore_instruction_overlap import Task, schedule


# M, N, K come from the connected non-PROGRAMMED_TILES block controller.
SHAPES = {
    "QKV": (256, 1728, 576), "SPROJ": (256, 576, 576),
    "CQ": (256, 576, 576), "CKV": (32, 1152, 576),
    "CPROJ": (256, 576, 576), "MLP0": (256, 2304, 576),
    "MLP2": (256, 576, 2304),
}


def capacities(name: str) -> tuple[int, int]:
    _, n, k = SHAPES[name]
    # Entire INT8 matrix + one FP32 scale and bias per output column.
    weight = n * k + 2 * n * 4
    # One M tile of INT8 activations plus one FP32 token scale per row.
    activation = 32 * k + 32 * 4
    return weight, activation


def graph(name: str, mode: str, row_cycles: int) -> list[Task]:
    m, n, k = SHAPES[name]
    if mode not in ("serial", "overlap") or row_cycles <= 0:
        raise ValueError("invalid schedule options")
    # Existing phase counter is a proxy for one weight load, not calibrated DMA.
    weight_wait = MEASURED[name][1]
    prep_cycles = 32 * row_cycles * k // 576
    # One M tile traverses N/32 output tiles, each with a complete K reduction.
    # One WAIT_ACC cycle is illustrative; 32-row drain is kept serial.
    gemm_cycles = (n // 32) * (1 + k + 32 + 32 + 3 + 1 + 32)
    tasks = [Task("weight", "dma", weight_wait, buffer="W0")]
    for row in range(m // 32):
        prep_deps = []
        if row >= 2:
            prep_deps.append(f"gemm{row-2}")
        if mode == "serial":
            prep_deps.append("weight")
            if row:
                prep_deps.append(f"gemm{row-1}")
        tasks.append(Task(f"prep{row}", "prep", prep_cycles,
                          tuple(prep_deps), f"A{row % 2}"))
        deps = ["weight", f"prep{row}"]
        if row:
            deps.append(f"gemm{row-1}")
        tasks.append(Task(f"gemm{row}", "array", gemm_cycles,
                          tuple(deps), f"A{row % 2},W0"))
    return tasks


def validate(rows: list[dict], name: str) -> None:
    by_name = {row["task"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("duplicate task")
    for row in rows:
        if any(by_name[d]["end"] > row["start"] for d in row["dependencies"]):
            raise ValueError("dependency violation")
    for resource in ("dma", "prep", "array"):
        spans = sorted((r["start"], r["end"]) for r in rows if r["resource"] == resource)
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError("resource overlap")
    for row in range(SHAPES[name][0] // 32):
        prep, gemm = by_name[f"prep{row}"], by_name[f"gemm{row}"]
        if prep["end"] > gemm["start"] or by_name["weight"]["end"] > gemm["start"]:
            raise ValueError("input is not ready")
        if row >= 2 and by_name[f"gemm{row-2}"]["end"] > prep["start"]:
            raise ValueError("activation buffer overwritten before last consumer")


def explore(weight_bytes: int, activation_bytes: int, row_cycles: int) -> dict:
    if weight_bytes <= 0 or activation_bytes <= 0 or row_cycles <= 0:
        raise ValueError("capacities and frontend cycles must be positive")
    stages = []
    for name, (m, n, k) in SHAPES.items():
        w, a = capacities(name)
        stage = {"stage": name, "shape_m_n_k": [m, n, k],
                 "weight_bytes_per_buffer": w, "activation_bytes_per_buffer": a,
                 "capacity_ok": w <= weight_bytes and a <= activation_bytes}
        if not stage["capacity_ok"]:
            stage["reason"] = "full-stage weight or M-tile activation exceeds a buffer"
            stages.append(stage)
            continue
        for mode in ("serial", "overlap"):
            rows = schedule(graph(name, mode, row_cycles))
            validate(rows, name)
            stage[mode] = {"cycles": max(r["end"] for r in rows), "tasks": rows}
        stages.append(stage)
    return {
        "status": "non_executable_uncalibrated_per_stage_exploration",
        "limitations": ["Inputs assumed available at cycle zero; independent stages, not a block schedule",
                        "One weight load per isolated stage; does not model cross-stage prefetch",
                        "No attention, inter-stage dependencies, port conflicts or measured frontend timing",
                        "No ISA output; two weight buffers are capacity assumptions only"],
        "configuration": {"weight_bytes_per_buffer": weight_bytes,
                          "activation_bytes_per_buffer": activation_bytes,
                          "weight_buffer_count": 2, "activation_buffer_count": 2,
                          "frontend_cycles_per_row_k576_assumed": row_cycles},
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-buffer-bytes", type=int, default=1280)
    parser.add_argument("--activation-buffer-bytes", type=int, default=73856)
    parser.add_argument("--frontend-row-cycles", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = explore(args.weight_buffer_bytes, args.activation_buffer_bytes,
                     args.frontend_row_cycles)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
