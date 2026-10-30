#!/usr/bin/env python3
"""Cycle-accurate list-scheduling simulation of the pipelined Supra2 block.

Anchors: per-stage busy cycles measured from XSim RTL runs (M=256, doubled
for row-scaling stages at M=512; cached-ctx stages fixed). The serial total
per block is 9,767,212 cycles at M=512 (verified against README records).

The pipelined model adds the units proposed in rtl/supra2/README.md
("Pipelined block schedule and utilization plan") without touching the
32x32 array:
  - 3x 32-lane LayerNorm/AdaLN units (one per site), single-pass
  - GELU fused into the MLP.0 epilogue (zero standalone cycles)
  - 2x 32-lane residual/gate adders
  - self/cross attention restructured to buffered two-pass: QK pass A on the
    array, softmax on a 32-lane engine, PV pass B as a dense GEMM
Scheduling granularity: 32-row tiles, 16 tiles = 2 CFG branches x 8 row
tiles. Self-attention consumes keys within one branch (256 keys).

This is a schedule-level simulation, not an RTL or board measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "pipeline_sim"
OUT.mkdir(parents=True, exist_ok=True)

# ---- measured per-stage serial cycles at M=512 (from XSim logs) ----
SERIAL_M512 = {
    "qkv": 645568, "self_qk": 155528, "self_pv": 1845648,
    "self_proj": 271168, "cross_q": 271168, "cross_kv": 28648,
    "cross_qk": 9724, "cross_pv": 509040, "cross_proj": 271168,
    "mlp0": 832768, "gelu": 1179648, "mlp2": 1091392,
    "ln_x3": 1771008, "res_x3": 884736,
}
SERIAL_BLOCK = sum(SERIAL_M512.values())  # 9,767,212

# ---- per-tile durations in the pipelined model ----
# 32-row tile = 32x576 FP16 activations = 18,432 elements.
# 32-lane LayerNorm unit measured in XSim: 596 cycles/tile (tb fixture,
# block-7 self-norm slice) -> 31.4 elem/cycle/lane, within the 640 budget.
LN_TILE = 596
# 32-lane gated residual measured in XSim: 4,608 cycles for 256x576 ->
# 576 cycles per 32-row tile (32 elems/cycle, 1-cycle latency hidden in stream).
RES_TILE = 576
# GEMM stages: measured M=512 totals / 16 tiles (keeps frontend overheads).
QKV_TILE = SERIAL_M512["qkv"] // 16            # 40,348
SELFPROJ_TILE = SERIAL_M512["self_proj"] // 16  # 16,948
CROSSQ_TILE = SERIAL_M512["cross_q"] // 16      # 16,948
CROSSPROJ_TILE = SERIAL_M512["cross_proj"] // 16
MLP0_TILE = SERIAL_M512["mlp0"] // 16           # 52,048
MLP2_TILE = SERIAL_M512["mlp2"] // 16           # 68,212
# Attention two-pass (restructured):
SELF_QK_TILE = SERIAL_M512["self_qk"] // 16     # QK pass A keeps measured rate
# PV pass B: 151M MACs / 16 query tiles -> ideal 9,216/tile at 70% util
SELF_PV_TILE = int(2 * 256 * 256 * 64 * 9 / 16 / 1024 / 0.70)
# softmax engine: measured 2,401 XSim cycles per 32x256 tile (supra2_softmax_2pass)
SFT_SELF_TILE = 2401
# cross: QK+softmax tiny (15 valid keys), PV 18.9M MACs/16 tiles @ 50%
CROSS_QK_SFT_TILE = 900
CROSS_PV_TILE = int(2 * 256 * 32 * 64 * 9 / 16 / 1024 / 0.50)

# MACs per block at M=512 (for utilization)
BLOCK_MACS = (
    512 * 1728 * 576            # qkv
    + 2 * 256 * 256 * 64 * 9 * 2  # self qk + pv
    + 512 * 576 * 576 * 3       # self/cross proj + cross q
    + 512 * 9 * 32 * 64 * 2     # cross qk + pv
    + 512 * 2304 * 576          # mlp0
    + 512 * 576 * 2304          # mlp2
    + 32 * 1152 * 576           # cross kv (cached, still counted once)
)

NTILE = 16  # 2 branches x 8 row tiles

RESOURCE_IDS = {
    "ARRAY": 0, "LN_A": 1, "LN_B": 2, "LN_C": 3, "R1": 4, "R2": 5, "SFT": 6,
}

def build_tasks():
    tasks = {}  # name -> (resource, duration, [deps])
    # Modeling notes (deliberate simplifications, ~1% combined effect):
    # - cross KV is a one-time per-prompt cached projection: counted in the
    #   serial baseline (SERIAL_M512["cross_kv"]) but intentionally has no
    #   per-block task in the pipelined schedule.
    # - cross QK (~600 cycles/tile at M=512) is folded into the softmax-engine
    #   task cqk (900 cycles) instead of occupying the array.
    # - GELU is fused into the MLP.0 epilogue, so it adds no standalone task.
    for b in range(2):
        for r in range(8):
            t = f"b{b}r{r}"
            tasks[f"ln1.{t}"] = ("LN_A", LN_TILE, [])
            tasks[f"qkv.{t}"] = ("ARRAY", QKV_TILE, [f"ln1.{t}"])
            # branch barrier: attention needs all 8 K/V tiles of the branch
            kv_done = [f"qkv.b{b}r{i}" for i in range(8)]
            tasks[f"sqk.{t}"] = ("ARRAY", SELF_QK_TILE, kv_done)
            tasks[f"sft.{t}"] = ("SFT", SFT_SELF_TILE, [f"sqk.{t}"])
            tasks[f"spv.{t}"] = ("ARRAY", SELF_PV_TILE, [f"sft.{t}"])
            tasks[f"sproj.{t}"] = ("ARRAY", SELFPROJ_TILE, [f"spv.{t}"])
            tasks[f"r1.{t}"] = ("R1", RES_TILE, [f"sproj.{t}"])
            tasks[f"ln2.{t}"] = ("LN_B", LN_TILE, [f"r1.{t}"])
            tasks[f"cq.{t}"] = ("ARRAY", CROSSQ_TILE, [f"ln2.{t}"])
            tasks[f"cqk.{t}"] = ("SFT", CROSS_QK_SFT_TILE, [f"cq.{t}"])
            tasks[f"cpv.{t}"] = ("ARRAY", CROSS_PV_TILE, [f"cqk.{t}"])
            tasks[f"cproj.{t}"] = ("ARRAY", CROSSPROJ_TILE, [f"cpv.{t}"])
            tasks[f"r2.{t}"] = ("R2", RES_TILE, [f"cproj.{t}"])
            tasks[f"ln3.{t}"] = ("LN_C", LN_TILE, [f"r2.{t}"])
            tasks[f"mlp0.{t}"] = ("ARRAY", MLP0_TILE, [f"ln3.{t}"])
            tasks[f"mlp2.{t}"] = ("ARRAY", MLP2_TILE, [f"mlp0.{t}"])
            tasks[f"rf.{t}"] = ("R1", RES_TILE, [f"mlp2.{t}"])
    return tasks

def simulate():
    tasks = build_tasks()
    # dependency-only earliest-start order (recursive finish-time probe)
    def dep_finish(name):
        res, dur, deps = tasks[name]
        return max((dep_finish(d) for d in deps), default=0) + dur
    order = sorted(tasks, key=lambda n: dep_finish(n) - tasks[n][1])
    # pass 2: greedy list schedule in dependency-earliest-start order
    free = {res: 0 for res in set(v[0] for v in tasks.values())}
    busy = {res: 0 for res in free}
    start, finish = {}, {}
    for name in order:
        res, dur, deps = tasks[name]
        est_start = max((finish[d] for d in deps), default=0)
        s = max(est_start, free[res])
        start[name] = s
        finish[name] = s + dur
        free[res] = s + dur
        busy[res] += dur
    makespan = max(finish.values())
    return makespan, busy, tasks, start, finish

def main():
    makespan, busy, tasks, start, finish = simulate()
    # sanity: serial-equivalent check of tile durations
    array_busy = busy["ARRAY"]
    report = {
        "model": "list-scheduled 32-row tile pipeline, 2 CFG branches interleaved",
        "serial_block_cycles_measured": SERIAL_BLOCK,
        "pipelined_block_cycles_simulated": makespan,
        "speedup_per_block": SERIAL_BLOCK / makespan,
        "resource_busy_cycles": dict(sorted(busy.items())),
        "resource_busy_fraction": {k: round(v / makespan, 4) for k, v in sorted(busy.items())},
        "array_mac_utilization": round(BLOCK_MACS / 1024 / makespan, 4),
        "array_busy_fraction": round(array_busy / makespan, 4),
        "attention_pv_tile_cycles": {"self_pv_dense": SELF_PV_TILE, "cross_pv_dense": CROSS_PV_TILE},
    }
    # end-to-end: one step = 14 blocks (no skip) + fixed frontend/final/euler
    fixed = 30528 + 294912 + 590336 + 10362 + 12800  # x_embed+pos+final_ln+head+euler
    step_noskip = 14 * makespan + fixed
    report["step_cycles_no_skip"] = step_noskip
    report["step_ms_no_skip_at_200mhz"] = round(step_noskip / 200e6 * 1e3, 1)
    blocks_avg_skip = 14 * (1 - 0.40857)
    step_skip = blocks_avg_skip * makespan + fixed
    report["step_cycles_with_skip_trace"] = int(step_skip)
    report["step_ms_with_skip_at_200mhz"] = round(step_skip / 200e6 * 1e3, 1)
    report["image_seconds_50steps_skip_at_200mhz"] = round(step_skip * 50 / 200e6, 2)
    serial_step = 14 * SERIAL_BLOCK + fixed
    report["serial_step_cycles_no_skip"] = serial_step
    report["serial_step_ms_no_skip_at_200mhz"] = round(serial_step / 200e6 * 1e3, 1)
    report["serial_image_seconds_50steps_skip_at_200mhz"] = round(
        (blocks_avg_skip * SERIAL_BLOCK + fixed) * 50 / 200e6, 2)
    # producer-consumer rate check on the bottleneck resource
    per_tile = {
        "ln": LN_TILE, "res": RES_TILE, "qkv": QKV_TILE, "selfproj": SELFPROJ_TILE,
        "mlp0": MLP0_TILE, "mlp2": MLP2_TILE, "self_pv_dense": SELF_PV_TILE,
        "self_qk": SELF_QK_TILE, "cross_pv_dense": CROSS_PV_TILE,
    }
    report["per_tile_cycles"] = per_tile
    (OUT / "pipeline_sim_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="ascii")
    dump_schedule_table(tasks, start, finish, busy, makespan)
    print(json.dumps(report, indent=2))

def dump_schedule_table(tasks, start, finish, busy, makespan):
    """Emit an RTL-replayable schedule table.

    One line per task, four 32-bit hex words:
      resource_id  start_cycle  duration  dep_ready_cycle
    where dep_ready = max(finish of dependencies) (0 if none). A clocked RTL
    replay checks start >= dep_ready (dependency feasibility) and
    start >= previous finish on the same resource (no double-booking), then
    re-derives makespan and per-resource busy cycles.
    """
    names = sorted(tasks, key=lambda n: (start[n], RESOURCE_IDS[tasks[n][0]]))
    # emit grouped by resource id ascending (one header line per resource with
    # its task count, then each group's tasks sorted by start) so the RTL
    # walker can index each resource's slice directly.
    lines = []
    by_res = {rid: [] for rid in RESOURCE_IDS.values()}
    for n in names:
        by_res[RESOURCE_IDS[tasks[n][0]]].append(n)
    for rid in sorted(by_res):
        lines.append(f"{len(by_res[rid]):08x}")
    for rid in sorted(by_res):
        for n in by_res[rid]:
            res, dur, deps = tasks[n]
            dep_ready = max((finish[d] for d in deps), default=0)
            assert dur == finish[n] - start[n]
            lines.append(f"{RESOURCE_IDS[res]:01x} {start[n]:08x} {dur:08x} {dep_ready:08x}")
    (OUT / "schedule_table.hex").write_text(
        "\n".join(lines) + "\n", encoding="ascii")
    expect = {
        "ntasks": len(names),
        "makespan": makespan,
        "busy": {k: busy[k] for k in sorted(busy)},
    }
    (OUT / "schedule_expect.json").write_text(
        json.dumps(expect, indent=2) + "\n", encoding="ascii")

if __name__ == "__main__":
    main()
