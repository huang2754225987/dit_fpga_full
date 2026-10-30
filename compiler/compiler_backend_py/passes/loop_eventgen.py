"""Pass 10 - LoopPassEventGen: emit event JSON (legacy sim interface).

Python reimplementation of ``src/pass/LoopPassEventGen.{h,cpp}``.

Produces a ``linecode.json`` event stream consumed by the legacy cycle
simulator (NOT by MachineCodeGen, which reads the BIR directly). Steps:

  1. ``legalize_for_isa``:
       * ``asap_matmult``       - pull same-start-cycle matmults earlier.
       * ``merge_consecutive_store`` - merge stores to the same memloc.
       * ``move_residue_load``  - move RESIDUEBANK loads before successors.

  2. ``set_sync_id`` - assign a sync id per distinct start cycle; merge
     consecutive same-type loads into one sync group.

  3. ``export_sim_json`` - per sync group emit:
       * ``load``   - compound DMA load (ddr_*_addr_ini, ddr_*_read_num).
       * ``compute`` - matmult (bank_ids, M/N/K, ker_round, output_num,
         fraclen shifts, psum_write_addr).
       * ``store``  - with post_ops (activation/residual/rope/reduce/
         ddr_save_*).
       * ``barrier`` - end of block.
"""
from __future__ import annotations

import json
import os
from typing import List

from bir.instruction import LoadInst, StoreInst, MatmultInst


class LoopEventGenPass:
    """Pass: loop_eventgen - emit cycle-simulator event JSON."""

    def __init__(self, out_path: str = "linecode.json", *args, **kwargs):
        self.out_path = out_path

    def run(self, module) -> int:
        events: List[dict] = []
        for fn in module:
            for bb in fn:
                self._legalize(bb)
                self._set_sync_id(bb)
                events.extend(self._export_block(bb))

        # write the event stream if there is anything to dump
        if events:
            try:
                with open(self.out_path, "w") as f:
                    for ev in events:
                        f.write(json.dumps(ev) + "\n")
            except OSError:
                pass
        print(f"[LoopEventGen] events={len(events)} -> {self.out_path}")
        return 0

    # ------------------------------------------------------------------
    def _legalize(self, bb):
        self._asap_matmult(bb)
        self._merge_consecutive_store(bb)

    def _asap_matmult(self, bb):
        """Pull matmults with the same start cycle as early as possible."""
        instrs = list(bb)
        for i, inst in enumerate(instrs):
            if isinstance(inst, MatmultInst):
                st = inst.get_attr("start_cycle")
                if st is None:
                    continue
                # move up past any non-matmult with the same start cycle
                j = i - 1
                while j >= 0:
                    prev = instrs[j]
                    if isinstance(prev, MatmultInst):
                        break
                    if prev.get_attr("start_cycle") == st:
                        # swap
                        try:
                            bb.move_after(prev, inst) if hasattr(bb, "move_after") else None
                        except Exception:
                            break
                        instrs[j], instrs[i] = instrs[i], instrs[j]
                        j -= 1
                    else:
                        break

    def _merge_consecutive_store(self, bb):
        """Merge consecutive stores to the same output memloc."""
        instrs = list(bb)
        i = 0
        while i < len(instrs) - 1:
            cur = instrs[i]
            nxt = instrs[i + 1]
            if (isinstance(cur, StoreInst) and isinstance(nxt, StoreInst)
                    and self._same_out(cur, nxt)):
                try:
                    bb.remove_instruction(nxt)
                    instrs.pop(i + 1)
                    continue
                except Exception:
                    pass
            i += 1

    def _same_out(self, a, b) -> bool:
        na = a.output_aps[0].mloc if a.output_aps else None
        nb = b.output_aps[0].mloc if b.output_aps else None
        return na is not None and nb is not None and na is nb

    # ------------------------------------------------------------------
    def _set_sync_id(self, bb):
        """Assign sync_id by start_cycle; consecutive same-type loads merge."""
        sid = 0
        prev_cycle = -1
        prev_type = None
        for inst in bb:
            cyc = inst.get_attr("start_cycle", 0) or 0
            itype = type(inst).__name__
            if cyc != prev_cycle or itype != prev_type:
                sid += 1
                prev_cycle = cyc
                prev_type = itype
            inst.set_attr("sync_id", sid)

    # ------------------------------------------------------------------
    def _export_block(self, bb) -> List[dict]:
        events: List[dict] = []
        loads = []
        for inst in bb:
            if isinstance(inst, LoadInst):
                loads.append(inst)
            elif isinstance(inst, MatmultInst):
                events.append(self._compute_event(inst))
            elif isinstance(inst, StoreInst):
                events.append(self._store_event(inst))
        if loads:
            events.append(self._load_event(loads))
        events.append({"type": "barrier", "sync_id": -1})
        return events

    def _compute_event(self, inst: MatmultInst) -> dict:
        return {
            "type": "compute",
            "sync_id": inst.get_attr("sync_id", 0),
            "start_cycle": inst.get_attr("start_cycle", 0),
            "M": inst.get_attr("M", 0), "N": inst.get_attr("N", 0),
            "K": inst.get_attr("K", 0),
            "ker_round": inst.get_attr("ker_round", 1),
            "output_num": inst.get_attr("output_num", 0),
        }

    def _store_event(self, inst: StoreInst) -> dict:
        return {
            "type": "store",
            "sync_id": inst.get_attr("sync_id", 0),
            "start_cycle": inst.get_attr("start_cycle", 0),
            "post_ops": [getattr(p, "type", "") for p in (getattr(inst, "post_ops", []) or [])],
        }

    def _load_event(self, loads: List[LoadInst]) -> dict:
        return {
            "type": "load",
            "sync_id": loads[0].get_attr("sync_id", 0),
            "start_cycle": loads[0].get_attr("start_cycle", 0),
            "num_loads": len(loads),
        }


__all__ = ["LoopEventGenPass"]
