"""Pass 9 - LoopPassScheduler: list scheduling + latency annotation.

Python reimplementation of ``src/pass/LoopPassScheduler.{h,cpp}``.

Two phases:

  1. ``annotate_inst_latency`` - attach a cycle latency to every
     instruction based on its type and the Arch model:
       * Load   : step * (ceil(step_size/factor) + dram_latency)
       * Store  : step * (step_size + dram_latency) + reduce/sync cost
       * Matmult: batch * M * ceil(K / (multipliers/inner_tile)) + pe_latency
       * Vector/Move/Kernel: 1
     Same-core +5, cross-core +26 (AXI bus) on top.

  2. ``sched_parallel_blocks`` - list (table) scheduling: maintain a
     ``ready`` queue (no pending predecessors), an ``active`` list, and a
     per-core resource map (``DDR-i``, ``PE-i``, ``VEC-i`` each capacity 1).
     Each cycle: issue ready instructions whose resource is free (set
     start cycle), move to active; complete active instructions whose
     start+latency <= cycle (set end cycle, release resource, unlock
     successors). Deadlock-guarded.

After this pass every instruction carries ``start_cycle`` /
``end_cycle`` / ``latency`` attributes that LoopPassEventGen reads.
"""
from __future__ import annotations

import math
from typing import List, Dict, Set

from bir.instruction import (LoadInst, StoreInst, MatmultInst, MoveInst,
                             VectorInst)


class LoopSchedulerPass:
    """Pass: loop_scheduler - list scheduling with latency annotation."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, module) -> int:
        arch = module.get_arch_model() if hasattr(module, "get_arch_model") else None
        total = 0
        for fn in module:
            for bb in fn:
                self._annotate_latency(bb, arch)
                total += self._schedule(bb, arch)
        print(f"[LoopScheduler] scheduled, total_cycles={total}")
        return 0

    # ------------------------------------------------------------------
    # latency annotation
    # ------------------------------------------------------------------
    def _annotate_latency(self, bb, arch):
        dram_lat = getattr(arch, "dram_access_latency", 20) if arch else 20
        pe_lat = getattr(arch, "pe_latency", 0) if arch else 0
        multipliers = getattr(arch, "multipliers", 1024) if arch else 1024
        sync = getattr(arch, "sync_delay", 1) if arch else 1
        inner_tile = 32   # inner-product tile size for the systolic array

        for inst in bb:
            if isinstance(inst, LoadInst):
                step = self._attr(inst, "step", 1)
                step_size = self._attr(inst, "step_size", 1)
                lat = step * (self._ceil_div(step_size, 1) + dram_lat)
                inst.set_attr("latency", lat)
            elif isinstance(inst, StoreInst):
                step = self._attr(inst, "step", 1)
                step_size = self._attr(inst, "step_size", 1)
                lat = step * (step_size + dram_lat) + self._store_extra(inst) + sync
                inst.set_attr("latency", lat)
            elif isinstance(inst, MatmultInst):
                batch = self._attr(inst, "batch", 1)
                M = self._attr(inst, "M", 1)
                K = self._attr(inst, "K", 1)
                out_per_cycle = max(multipliers // inner_tile, 1)
                cycle_per_fmap = self._ceil_div(K, out_per_cycle)
                lat = batch * M * cycle_per_fmap + pe_lat
                inst.set_attr("latency", lat)
                inst.set_attr("ker_round", 1)
                inst.set_attr("output_num", M)
            elif isinstance(inst, (MoveInst, VectorInst)):
                inst.set_attr("latency", 1)
            else:
                inst.set_attr("latency", 1)

    def _store_extra(self, inst) -> int:
        """Extra cycles for reduce ops folded into a store."""
        post = getattr(inst, "post_ops", []) or []
        if any(getattr(p, "type", "") == "rmsnorm" for p in post):
            return 45
        if any(getattr(p, "type", "") == "softmax" for p in post):
            return 20
        return 0

    # ------------------------------------------------------------------
    # list scheduling
    # ------------------------------------------------------------------
    def _schedule(self, bb, arch) -> int:
        instrs = list(bb)
        if not instrs:
            return 0
        # ready = instructions with no predecessors
        ready: List = [i for i in instrs if not i.pred]
        active: List = []
        done: Set[int] = set()
        # resource map: resource_name -> in-use cycle until
        rmap: Dict[str, int] = {}
        cycle = 0
        max_iter = 80000

        while (ready or active) and max_iter > 0:
            max_iter -= 1
            # issue ready instructions whose resource is free
            issued_now: List = []
            for inst in ready:
                res = getattr(inst, "dep_resource", "") or ""
                # resource free if not in rmap, or rmap[res] <= cycle
                if res and res in rmap and rmap[res] > cycle:
                    continue
                inst.set_attr("start_cycle", cycle)
                issued_now.append(inst)
                if res:
                    rmap[res] = cycle + max(self._attr(inst, "latency", 1), 1)
                active.append(inst)
            for inst in issued_now:
                ready.remove(inst)

            # advance cycle to next completion
            if active:
                latencies = [self._attr(i, "latency", 1) for i in active]
                starts = [self._attr(i, "start_cycle", cycle) for i in active]
                next_complete = min(s + l for s, l in zip(starts, latencies))
                cycle = max(cycle + 1, next_complete)
            else:
                cycle += 1

            # complete active instructions whose start+latency <= cycle
            still_active = []
            for inst in active:
                st = self._attr(inst, "start_cycle", 0)
                lat = self._attr(inst, "latency", 1)
                if st + lat <= cycle:
                    inst.set_attr("end_cycle", st + lat)
                    done.add(id(inst))
                    # unlock successors
                    for succ in inst.succ:
                        if all(id(p) in done for p in succ.pred) and succ not in ready and id(succ) not in done:
                            ready.append(succ)
                else:
                    still_active.append(inst)
            active = still_active

        return cycle

    # ------------------------------------------------------------------
    def _attr(self, inst, name, default):
        v = inst.get_attr(name) if hasattr(inst, "get_attr") else None
        if v is None:
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def _ceil_div(self, a, b):
        b = b if b else 1
        return (a + b - 1) // b


__all__ = ["LoopSchedulerPass"]
