"""Explore a small dependency-safe overlap graph; emits no executable ISA."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    name: str
    resource: str
    duration: int
    deps: tuple[str, ...] = ()
    buffer: str = ""


def build_tasks(overlap: bool) -> list[Task]:
    """Two hypothetical stages, each with three 32-row macro tiles.

    Stage weights are retained until all stage consumers finish. Activations
    use ping/pong slots; M2 cannot overwrite M0 until GEMM M0 has completed.
    Durations and capacities are illustrative, not measured RTL parameters.
    """
    tasks = [Task("load_s0", "dma", 100, buffer="W0"),
             Task("load_s1", "dma", 100,
                  () if overlap else ("gemm_s0_m2",), "W1")]
    for stage in range(2):
        for row in range(3):
            name = f"s{stage}_m{row}"
            prep_deps = []
            if row >= 2:
                prep_deps.append(f"gemm_s{stage}_m{row - 2}")
            # Same activation slots are shared across stages, so a later
            # producer must wait for the preceding stage's last slot user.
            if stage:
                prep_deps.append(f"gemm_s0_m{2 if row % 2 == 0 else 1}")
                # This example represents stage outputs feeding the next stage.
                prep_deps.append(f"gemm_s0_m{row}")
            tasks.append(Task(f"prep_{name}", "prep", 40,
                              tuple(dict.fromkeys(prep_deps)), f"A{row % 2}"))
            deps = [f"prep_{name}", f"load_s{stage}"]
            if row:
                deps.append(f"gemm_s{stage}_m{row - 1}")
            tasks.append(Task(f"gemm_{name}", "array", 150, tuple(deps), f"A{row % 2},W{stage}"))
    return tasks


def schedule(tasks: list[Task]) -> list[dict]:
    by_name = {t.name: t for t in tasks}
    if len(by_name) != len(tasks):
        raise ValueError("duplicate task name")
    pending, done, free, result = set(by_name), {}, {}, []
    while pending:
        ready = [by_name[n] for n in pending if all(d in done for d in by_name[n].deps)]
        if not ready:
            raise ValueError("missing dependency or dependency cycle")
        def start(t: Task) -> int:
            return max([free.get(t.resource, 0)] + [done[d] for d in t.deps])
        task = min(ready, key=lambda t: (start(t), t.name))
        begin = start(task)
        end = begin + task.duration
        free[task.resource] = done[task.name] = end
        result.append({"task": task.name, "resource": task.resource,
                       "buffer": task.buffer, "start": begin, "end": end,
                       "dependencies": list(task.deps)})
        pending.remove(task.name)
    return result


def check(rows: list[dict]) -> None:
    by_name = {r["task"]: r for r in rows}
    for r in rows:
        assert all(by_name[d]["end"] <= r["start"] for d in r["dependencies"])
    for resource in ("dma", "prep", "array"):
        events = sorted((r["start"], r["end"]) for r in rows if r["resource"] == resource)
        assert all(a[1] <= b[0] for a, b in zip(events, events[1:]))
    for stage in range(2):
        for row in range(3):
            prep = by_name[f"prep_s{stage}_m{row}"]
            gemm = by_name[f"gemm_s{stage}_m{row}"]
            assert prep["end"] <= gemm["start"]
            if row >= 2:
                assert by_name[f"gemm_s{stage}_m{row - 2}"]["end"] <= prep["start"]
    assert by_name["gemm_s0_m2"]["end"] <= by_name["prep_s1_m0"]["start"]
    assert by_name["gemm_s0_m1"]["end"] <= by_name["prep_s1_m1"]["start"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    serial, overlap = schedule(build_tasks(False)), schedule(build_tasks(True))
    check(serial)
    check(overlap)
    serial_end = max(r["end"] for r in serial)
    overlap_end = max(r["end"] for r in overlap)
    assert overlap_end <= serial_end
    report = {"status": "illustrative_not_rtl_or_block13_prediction",
              "timing_unit": "hypothetical_cycles",
              "note": "Only stage-weight prefetch differs; activation preparation overlaps in both cases.",
              "serial_weight_schedule_cycles": serial_end,
              "prefetch_schedule_cycles": overlap_end,
              "serial_weight_schedule": serial, "prefetch_schedule": overlap}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
