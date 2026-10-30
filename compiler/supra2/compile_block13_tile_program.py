"""Compile a bank-aware 32x32 tile schedule for the connected block-13 RTL.

The output is a scheduling prototype: command descriptors and a cycle plan
for the proposed tile engine. The current full datapath still executes its
operators through the verified stage FSM; the generated schedule is replayed
separately by the RTL schedule checker.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path


OUT_DEFAULT = Path(__file__).resolve().parents[2] / "build/supra2_tile_program"
ROWS, FEAT, TILE, HEADS, HEAD_DIM, CONTEXT_ROWS = 256, 576, 32, 9, 64, 32
RESOURCE_NAMES = ("ARRAY", "LN", "ATTENTION", "GELU", "RESIDUAL", "DMA", "DDR")
RESOURCE_ID = {name: i for i, name in enumerate(RESOURCE_NAMES)}

STAGES = {
    "LN0": 0, "QKV": 1, "ATTN_SELF": 2, "SPROJ": 3, "RES0": 4,
    "LN1": 5, "CQ": 6, "CKV": 7, "ATTN_CROSS": 8, "CPROJ": 9,
    "RES1": 10, "LN2": 11, "MLP0": 12, "GELU": 13, "MLP2": 14,
    "RES2": 15,
}

# Measured from the passing compiler-driven block-13 AXI512 + board MIG +
# DDR4 run (rtl tile program). Each QLinear total includes its serialized
# weight-load wait; each attention stage is distributed over row/head tiles.
MEASURED = {
    "QKV": (531165, 71749), "ATTN_SELF": (2163744, 0),
    "SPROJ": (290073, 25633), "CQ": (288366, 23926),
    "CKV": (96672, 51431), "ATTN_CROSS": (409824, 0),
    "CPROJ": (290246, 25806), "MLP0": (652342, 95438),
    "GELU": (18448, 0), "MLP2": (808007, 91983),
    "LN0": (14152, 0), "LN1": (14152, 0), "LN2": (14152, 0),
    "RES0": (4624, 0), "RES1": (4624, 0), "RES2": (4624, 0),
}
INPUT_CYCLES, OUTPUT_CYCLES = 178076, 43190

OPCODE = {"DMA_WEIGHT": 1, "GEMM_TILE": 2, "ATTN_TILE": 3,
          "LN_TILE": 4, "RES_TILE": 5, "GELU_TILE": 6,
          "DMA_INPUT": 7, "DMA_OUTPUT": 8, "END": 15}


@dataclass
class Task:
    name: str
    opcode: str
    stage: str
    row: int
    n_tiles: int
    k_tiles: int
    duration: int
    resources: tuple[int, ...]
    reads: tuple[int, ...] = ()
    writes: tuple[int, ...] = ()
    deps: list[str] = field(default_factory=list)
    start: int = 0
    finish: int = 0
    dep_ready: int = 0
    src_bank: int = 0
    dst_bank: int = 0


def memory_plan() -> dict:
    # Each ID is an independently arbitrated synchronous bank. Sizes include
    # 64-byte AXI512 alignment. The budget is an explicit scratchpad target,
    # not a claim about Vivado's eventual BRAM/URAM inference.
    entries = [
        ("x_tile_0", 0, 36864, "input/residual tile ping"),
        ("x_tile_1", 1, 36864, "input/residual tile pong"),
        ("ln_tile_0", 2, 36864, "LayerNorm tile ping"),
        ("ln_tile_1", 3, 36864, "LayerNorm tile pong"),
        ("self_q", 4, 294912, "256x576 persistent self-attention Q, FP16"),
        ("cross_q_0", 5, 36864, "32x576 cross-query ping"),
        ("cross_q_1", 6, 36864, "32x576 cross-query pong"),
        ("self_kv", 7, 589824, "256x576 K and V, FP16"),
        ("score_0", 8, 16384, "32x256 FP16 score tile"),
        ("score_1", 9, 16384, "32x256 FP16 score tile ping-pong"),
        ("attn_out_0", 10, 36864, "attention output tile ping"),
        ("attn_out_1", 11, 36864, "attention output tile pong"),
        ("context_input", 12, 36864, "32x576 projected context input"),
        ("context_kv", 13, 73728, "32x576 K and V, FP16"),
        ("mlp_0", 14, 147456, "32x2304 FP16 MLP tile ping, URAM"),
        ("mlp_1", 15, 147456, "32x2304 FP16 MLP tile pong, URAM"),
        ("weight_0", 16, 1280, "1 KiB INT8 tile + FP32 scale/bias"),
        ("weight_1", 17, 1280, "prefetch weight tile ping-pong"),
        ("adaln", 18, 9216, "8x576 FP16 AdaLN parameters"),
        ("proj_0", 19, 36864, "32x576 residual update ping"),
        ("proj_1", 20, 36864, "32x576 residual update pong"),
    ]
    offset = 0
    banks = []
    lifetimes = {
        "x_tile_0": "input through final output; residual ping-pong",
        "x_tile_1": "input through final output; residual ping-pong",
        "ln_tile_0": "one normalization tile until its consumer GEMM completes",
        "ln_tile_1": "one normalization tile until its consumer GEMM completes",
        "self_q": "QKV completion through all self-attention query/head tiles",
        "cross_q_0": "CQ tile production through its cross-attention consumers",
        "cross_q_1": "CQ tile production through its cross-attention consumers",
        "self_kv": "QKV completion through all self-attention query/head tiles",
        "score_0": "one attention query/head score tile",
        "score_1": "one attention query/head score tile; alternating bank",
        "attn_out_0": "attention output through the matching projection tile",
        "attn_out_1": "attention output through the matching projection tile",
        "context_input": "input context through CKV projection",
        "context_kv": "CKV projection through all cross-attention consumers",
        "mlp_0": "MLP0 output through GELU or MLP2 consumer",
        "mlp_1": "MLP0 output through GELU or MLP2 consumer; alternating bank",
        "weight_0": "one QLinear stage's current matrix tile and scale/bias",
        "weight_1": "next QLinear stage's prefetched matrix tile and scale/bias",
        "adaln": "all three normalization/residual sites in the block",
        "proj_0": "projection/update tile until residual consumes it",
        "proj_1": "projection/update tile until residual consumes it; alternating bank",
    }
    styles = {"self_q": "ultra", "self_kv": "ultra", "mlp_0": "ultra",
              "mlp_1": "ultra", "score_0": "distributed", "score_1": "distributed",
              "weight_0": "lutram", "weight_1": "lutram"}
    for name, bank_id, size, purpose in entries:
        offset = (offset + 63) & ~63
        banks.append({"name": name, "bank_id": bank_id, "offset": offset,
                      "bytes": size, "purpose": purpose,
                      "target_ram_style": styles.get(name, "block"),
                      "lifetime": lifetimes[name],
                      "read_ports": 1, "write_ports": 1})
        offset += size
    return {"alignment_bytes": 64, "capacity_bytes": 1_835_008,
            "allocated_bytes": offset,
            "mapping_status": "compiler allocation target; Vivado inference not yet measured",
            "allocated_bytes_by_style": {
                style: sum(b["bytes"] for b in banks if b["target_ram_style"] == style)
                for style in sorted({b["target_ram_style"] for b in banks})},
            "banks": banks}


def encode(task: Task, descriptor_index: int, dependency_index: int,
           completion_token: int) -> int:
    """128-bit issue word with two sources, two destinations and token IDs."""
    fields8 = (OPCODE[task.opcode], STAGES.get(task.stage, 0xFF), task.row,
               task.n_tiles, task.k_tiles,
               task.reads[0] if task.reads else 0xFF,
               task.reads[1] if len(task.reads) > 1 else 0xFF,
               task.writes[0] if task.writes else 0xFF,
               task.writes[1] if len(task.writes) > 1 else 0xFF)
    if any(not 0 <= x <= 255 for x in fields8):
        raise ValueError(f"instruction byte field overflow for {task.name}: {fields8}")
    if not all(0 <= x <= 0xFFFF for x in (descriptor_index, dependency_index, completion_token)):
        raise ValueError(f"instruction token/descriptor overflow for {task.name}")
    fields = (*fields8, descriptor_index, dependency_index, completion_token, 0)
    word = 0
    for value, width in zip(fields, (8,)*9 + (16, 16, 16, 8)):
        word = (word << width) | value
    return word


def _duration(stage: str, pieces: int, index: int) -> int:
    total, wait = MEASURED[stage]
    compute = total - wait
    q, r = divmod(compute, pieces)
    return q + (index < r)


def build_tasks() -> list[Task]:
    tasks: list[Task] = []
    by_name: dict[str, Task] = {}
    bank_ids = {b["name"]: b["bank_id"] for b in memory_plan()["banks"]}

    def add(name: str, opcode: str, stage: str, row: int, nt: int, kt: int,
            dur: int, primary: str, reads=(), writes=(), deps=(),
            src=0, dst=0) -> Task:
        locks = {RESOURCE_ID[primary]}
        locks.update(7 + int(x) for x in reads)
        locks.update(7 + int(x) for x in writes)
        task = Task(name, opcode, stage, row, nt, kt, max(1, int(dur)),
                    tuple(sorted(locks)), tuple(reads), tuple(writes), list(deps),
                    src_bank=src, dst_bank=dst)
        tasks.append(task)
        by_name[name] = task
        return task

    x_banks = (bank_ids["x_tile_0"], bank_ids["x_tile_1"])
    ln_banks = (bank_ids["ln_tile_0"], bank_ids["ln_tile_1"])
    cross_q_banks = (bank_ids["cross_q_0"], bank_ids["cross_q_1"])
    proj_banks = (bank_ids["proj_0"], bank_ids["proj_1"])
    wt_banks = (bank_ids["weight_0"], bank_ids["weight_1"])
    input_task = add("dma.input", "DMA_INPUT", "LN0", 0, 0, 0,
                     INPUT_CYCLES, "DMA", writes=x_banks, src=0, dst=x_banks[0])

    # One DDR stage descriptor load, followed by a 32-row x output-channel
    # macro-tile command. N/K dimensions are 32x32 array tiles.
    qlinear = {
        "QKV": (1728, 576, "ln_tile", "self_kv"),
        "SPROJ": (576, 576, "attn_out", "x_tile"),
        "CQ": (576, 576, "ln_tile", "q_tile"),
        "CKV": (1152, 576, "context_kv", "context_kv"),
        "CPROJ": (576, 576, "attn_out", "x_tile"),
        "MLP0": (2304, 576, "ln_tile", "mlp"),
        "MLP2": (576, 2304, "mlp", "x_tile"),
    }
    load_task = {}
    for i, stage in enumerate(qlinear):
        _, wait = MEASURED[stage]
        if not wait:
            wait = 1
        wb = wt_banks[i & 1]
        load_task[stage] = add(f"dma.weight.{stage}", "DMA_WEIGHT", stage,
                               0, 0, 0, wait, "DMA", writes=(wb,),
                               deps=(input_task.name,), src=0, dst=wb)

    ln_tasks, gemm_tasks, attn_tasks, residual_tasks, gelu_tasks = {}, {}, {}, {}, {}
    for stage in ("LN0", "LN1", "LN2"):
        for row in range(ROWS // TILE):
            prev = (input_task.name if stage == "LN0" else
                    f"{'RES0' if stage == 'LN1' else 'RES1'}.r{row}")
            src = x_banks[row & 1] if stage in ("LN0", "LN2") else x_banks[1 - (row & 1)]
            dst = ln_banks[row & 1]
            t = add(f"{stage}.r{row}", "LN_TILE", stage, row, FEAT // TILE, 0,
                    _duration(stage, 8, row), "LN", reads=(src, bank_ids["adaln"]),
                    writes=(dst,), deps=(prev,), src=src, dst=dst)
            ln_tasks[(stage, row)] = t

    # QKV tasks establish the persistent K/V cache. Self-attention waits for
    # all row tiles because every query tile consumes all 256 K/V rows.
    for stage, (n, k, src_name, dst_name) in qlinear.items():
        rows = 1 if stage == "CKV" else 8
        for row in range(rows):
            wb = wt_banks[list(qlinear).index(stage) & 1]
            if stage == "QKV":
                deps = (ln_tasks[("LN0", row)].name, load_task[stage].name)
                reads = (ln_banks[row & 1], wb)
                writes = (bank_ids["self_q"], bank_ids["self_kv"])
            elif stage == "SPROJ":
                deps = (f"ATTN_SELF.r{row}.all_heads", load_task[stage].name)
                reads = (bank_ids[f"attn_out_{row & 1}"], wb)
                writes = (proj_banks[row & 1],)
            elif stage == "CQ":
                deps = (ln_tasks[("LN1", row)].name, load_task[stage].name)
                reads = (ln_banks[row & 1], wb)
                writes = (cross_q_banks[row & 1],)
            elif stage == "CKV":
                deps = (input_task.name, load_task[stage].name)
                reads = (bank_ids["context_input"], wb)
                writes = (bank_ids["context_kv"],)
            elif stage == "CPROJ":
                deps = (f"ATTN_CROSS.r{row}.all_heads", load_task[stage].name)
                reads = (bank_ids[f"attn_out_{row & 1}"], wb)
                writes = (proj_banks[row & 1],)
            elif stage == "MLP0":
                deps = (ln_tasks[("LN2", row)].name, load_task[stage].name)
                reads = (ln_banks[row & 1], wb)
                writes = (bank_ids[f"mlp_{row & 1}"],)
            else:  # MLP2
                deps = (f"GELU.r{row}", load_task[stage].name)
                reads = (bank_ids[f"mlp_{1 - (row & 1)}"], wb)
                writes = (proj_banks[row & 1],)
            dur = (_duration(stage, rows, row) if stage != "CKV"
                   else MEASURED[stage][0] - MEASURED[stage][1])
            t = add(f"{stage}.r{row}", "GEMM_TILE", stage, row,
                    (n + TILE - 1) // TILE, (k + TILE - 1) // TILE,
                    dur, "ARRAY", reads=reads, writes=writes, deps=deps,
                    src=reads[0], dst=writes[0])
            gemm_tasks[(stage, row)] = t

    # The current RTL exposes one physical QLinear weight store.  The abstract
    # bank plan has two ping-pong banks for a future prefetch implementation,
    # but commands must not overwrite the active matrix before its GEMMs retire.
    # Keep the loads ordered by consumers so static issue order is executable
    # on today's datapath as well as on a future dual-bank implementation.
    weight_use_rows = {
        "QKV": [f"QKV.r{row}" for row in range(8)],
        "SPROJ": [f"SPROJ.r{row}" for row in range(8)],
        "CQ": [f"CQ.r{row}" for row in range(8)],
        "CKV": ["CKV.r0"],
        "CPROJ": [f"CPROJ.r{row}" for row in range(8)],
        "MLP0": [f"MLP0.r{row}" for row in range(8)],
        "MLP2": [f"MLP2.r{row}" for row in range(8)],
    }
    weight_order = tuple(weight_use_rows)
    for previous, current in zip(weight_order, weight_order[1:]):
        load_task[current].deps = weight_use_rows[previous]

    # The QKV->K/V global barrier is explicit; each row/head issue then carries
    # an eight-key-tile loop for self attention or one 32-key context tile.
    for row in range(8):
        for head in range(HEADS):
            name = f"ATTN_SELF.r{row}.h{head}"
            t = add(name, "ATTN_TILE", "ATTN_SELF", row, head, 8,
                    _duration("ATTN_SELF", 8 * HEADS, row * HEADS + head),
                    "ARRAY", reads=(bank_ids["self_q"], bank_ids["self_kv"]),
                    writes=(bank_ids[f"score_{(row + head) & 1}"],
                            bank_ids[f"attn_out_{row & 1}"]),
                    deps=tuple(gemm_tasks[("QKV", r)].name for r in range(8)),
                    src=bank_ids["self_kv"], dst=bank_ids[f"attn_out_{row & 1}"])
            attn_tasks[("ATTN_SELF", row, head)] = t
        # Stage projection waits for all nine heads for this query tile.
        # The alias task is zero-cost in the schedule, represented by the
        # maximum finish of its nine producers in the dependency list below.

    # Fill row/head fan-in dependencies after all self-attention tasks exist.
    for row in range(8):
        proj = gemm_tasks[("SPROJ", row)]
        proj.deps = [attn_tasks[("ATTN_SELF", row, h)].name for h in range(HEADS)] + [load_task["SPROJ"].name]

    for row in range(8):
        for stage, prior in (("RES0", "SPROJ"), ("RES1", "CPROJ"), ("RES2", "MLP2")):
            if stage == "RES0":
                prev_x, next_x = x_banks[row & 1], x_banks[1 - (row & 1)]
            elif stage == "RES1":
                prev_x, next_x = x_banks[1 - (row & 1)], x_banks[row & 1]
            else:
                prev_x, next_x = x_banks[row & 1], x_banks[1 - (row & 1)]
            update = proj_banks[row & 1]
            t = add(f"{stage}.r{row}", "RES_TILE", stage, row, 0, 0,
                    _duration(stage, 8, row), "RESIDUAL",
                    reads=(prev_x, update),
                    writes=(next_x,), deps=(gemm_tasks[(prior, row)].name,),
                    src=prev_x, dst=next_x)
            residual_tasks[(stage, row)] = t
        gelu = add(f"GELU.r{row}", "GELU_TILE", "GELU", row,
                   2304 // TILE, 0, _duration("GELU", 8, row), "GELU",
            reads=(bank_ids[f"mlp_{row & 1}"],),
                   writes=(bank_ids[f"mlp_{1 - (row & 1)}"],),
                   deps=(gemm_tasks[("MLP0", row)].name,),
                   src=bank_ids[f"mlp_{row & 1}"], dst=bank_ids[f"mlp_{1 - (row & 1)}"])
        gelu_tasks[row] = gelu

    # Cross-attention Q tile and cached context K/V are independent producers.
    for row in range(8):
        for head in range(HEADS):
            name = f"ATTN_CROSS.r{row}.h{head}"
            t = add(name, "ATTN_TILE", "ATTN_CROSS", row, head, 1,
                    _duration("ATTN_CROSS", 8 * HEADS, row * HEADS + head),
                    "ARRAY", reads=(cross_q_banks[row & 1], bank_ids["context_kv"]),
                    writes=(bank_ids[f"score_{(row + head) & 1}"],
                            bank_ids[f"attn_out_{row & 1}"]),
                    deps=(gemm_tasks[("CQ", row)].name, gemm_tasks[("CKV", 0)].name),
                    src=bank_ids["context_kv"], dst=bank_ids[f"attn_out_{row & 1}"])
            attn_tasks[("ATTN_CROSS", row, head)] = t
        cproj = gemm_tasks[("CPROJ", row)]
        cproj.deps = [attn_tasks[("ATTN_CROSS", row, h)].name for h in range(HEADS)] + [load_task["CPROJ"].name]

    # Whole activation input/output traffic and every stage weight load are
    # explicit program commands; output starts after all RES2 tiles complete.
    output = add("dma.output", "DMA_OUTPUT", "RES2", 0, 0, 0,
                 OUTPUT_CYCLES, "DMA", reads=x_banks,
                 deps=tuple(residual_tasks[("RES2", r)].name for r in range(8)),
                 src=x_banks[0], dst=0)
    return tasks


def schedule(tasks: list[Task]) -> tuple[list[Task], int]:
    by_name = {t.name: t for t in tasks}
    pending = set(by_name)
    order = []
    finish = {}
    resource_free: dict[int, int] = {}
    while pending:
        ready = [by_name[n] for n in pending if all(d in finish for d in by_name[n].deps)]
        if not ready:
            raise ValueError(f"dependency cycle or missing task: {sorted(pending)[:8]}")
        ready.sort(key=lambda t: (max((finish[d] for d in t.deps), default=0), -t.duration, t.name))
        task = ready[0]
        task.dep_ready = max((finish[d] for d in task.deps), default=0)
        task.start = max([task.dep_ready] + [resource_free.get(r, 0) for r in task.resources])
        task.finish = task.start + task.duration
        for resource in task.resources:
            resource_free[resource] = task.finish
        finish[task.name] = task.finish
        order.append(task)
        pending.remove(task.name)
    return order, max(finish.values(), default=0)


def _instruction_plan(order: list[Task], output_dir: Path) -> None:
    issue = sorted(order, key=lambda t: (t.start, t.resources[0], t.name))
    task_ids = {task.name: i + 1 for i, task in enumerate(issue)}
    dependency_tokens: list[int] = []
    dependency_ranges: dict[str, tuple[int, int]] = {}
    for task in issue:
        first = len(dependency_tokens)
        dependency_tokens.extend(task_ids[d] for d in task.deps)
        dependency_ranges[task.name] = (first, len(task.deps))
    words = [encode(t, STAGES.get(t.stage, 0xFFFF),
                    dependency_ranges[t.name][0], task_ids[t.name])
             | dependency_ranges[t.name][1]
             for t in issue]
    words.append(OPCODE["END"] << 120)
    (output_dir / "block13_tile_program.hex").write_text(
        "".join(f"{w:032x}\n" for w in words), encoding="ascii")
    (output_dir / "block13_tile_program.bin").write_bytes(
        b"".join(w.to_bytes(16, "little") for w in words))
    (output_dir / "block13_tile_program.json").write_text(json.dumps({
        "isa": "supra2-tile-v2-prototype", "word_bits": 128,
        "encoding": "opcode[127:120], stage[119:112], row[111:104], N/head[103:96], K/keys[95:88], src0[87:80], src1[79:72], dst0[71:64], dst1[63:56], descriptor[55:40], dependency_list[39:24], completion_token[23:8], flags[7:0]",
        "issue_count": len(words), "dependency_word_count": len(dependency_tokens),
        "tile": "32 rows x 32 columns; QLinear K reduction encoded as k_tiles",
        "commands": [{"pc": i, "word": f"{words[i]:032x}", "op": t.opcode,
                      "task": t.name, "stage": t.stage, "row_tile": t.row,
                      "n_tiles_or_head": t.n_tiles, "k_tiles_or_keys": t.k_tiles,
                      "scheduled_start": t.start, "duration": t.duration,
                      "wait_for": t.deps,
                      "dependency_list_start": dependency_ranges[t.name][0],
                      "dependency_count": dependency_ranges[t.name][1],
                      "dependency_token_ids": [task_ids[d] for d in t.deps],
                      "completion_token_id": task_ids[t.name],
                      "src_banks": list(t.reads), "dst_banks": list(t.writes),
                      "resource_ids": list(t.resources)}
                     for i, t in enumerate(issue)] +
                    [{"pc": len(issue), "word": f"{words[-1]:032x}", "op": "END"}],
    }, indent=2) + "\n", encoding="ascii")
    (output_dir / "block13_tile_dependencies.hex").write_text(
        "".join(f"{token:04x}\n" for token in dependency_tokens), encoding="ascii")
    (output_dir / "block13_tile_dependencies.bin").write_bytes(
        b"".join(token.to_bytes(2, "little") for token in dependency_tokens))


def _schedule_table(order: list[Task], makespan: int, output_dir: Path) -> int:
    by_resource = {i: [] for i in range(7 + len(memory_plan()["banks"]))}
    for task in order:
        for resource in task.resources:
            by_resource[resource].append(task)
    lines = [f"{len(by_resource[r]):08x}" for r in sorted(by_resource)]
    for resource in sorted(by_resource):
        for task in sorted(by_resource[resource], key=lambda t: (t.start, t.name)):
            lines.append(f"{resource:08x} {task.start:08x} {task.duration:08x} {task.dep_ready:08x}")
    total_events = sum(map(len, by_resource.values()))
    serial_task_cycles = sum(task.duration for task in order)
    (output_dir / "block13_tile_schedule.hex").write_text("\n".join(lines) + "\n", encoding="ascii")
    resource_busy = {RESOURCE_NAMES[i]: sum(t.duration for t in by_resource[i])
                     for i in range(len(RESOURCE_NAMES))}
    (output_dir / "block13_tile_schedule.json").write_text(json.dumps({
        "task_count": len(order), "rtl_replay_event_count": total_events,
        "makespan_cycles": makespan, "resource_busy_cycles": resource_busy,
        "serialized_task_duration_cycles": serial_task_cycles,
        "estimated_overlap_savings_cycles": serial_task_cycles - makespan,
        "estimated_overlap_savings_percent": (100.0 * (serial_task_cycles - makespan) /
                                                serial_task_cycles if serial_task_cycles else 0.0),
        "schedule_parallelism": (serial_task_cycles / makespan if makespan else 0.0),
        "scope": "compiler schedule estimate anchored to measured block13 phase counters; RTL replay checks dependencies/resource exclusivity, no datapath is invoked by the replay",
    }, indent=2) + "\n", encoding="ascii")
    return total_events


def compile_plan(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mem = memory_plan()
    if mem["allocated_bytes"] > mem["capacity_bytes"]:
        raise ValueError("planned scratch banks exceed the configured budget")
    tasks = build_tasks()
    order, makespan = schedule(tasks)
    if any(t.start < t.dep_ready or t.finish - t.start != t.duration for t in order):
        raise AssertionError("invalid dependency schedule")
    _instruction_plan(order, output_dir)
    replay_events = _schedule_table(order, makespan, output_dir)
    serial_task_cycles = sum(task.duration for task in order)
    (output_dir / "block13_memory_plan.json").write_text(json.dumps(mem, indent=2) + "\n", encoding="ascii")
    result = {"instructions": len(order) + 1, "schedule_tasks": len(order),
              "rtl_replay_events": replay_events, "makespan_cycles": makespan,
              "serialized_task_duration_cycles": serial_task_cycles,
              "estimated_overlap_savings_cycles": serial_task_cycles - makespan,
              "estimated_overlap_savings_percent": (100.0 * (serial_task_cycles - makespan) /
                                                       serial_task_cycles if serial_task_cycles else 0.0),
              "rtl_issue_model": "current RTL waits for each command completion; overlap savings are a static schedule estimate",
              "allocated_bytes": mem["allocated_bytes"],
              "scratch_capacity_bytes": mem["capacity_bytes"]}
    (output_dir / "block13_compile_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(compile_plan(args.output_dir), indent=2))


if __name__ == "__main__":
    resource_free: dict[int, int] = {}
    main()
