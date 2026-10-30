"""Pass 8 - ADA: allocation-aware dependency analysis (RAW/WAR/WAW).

Python reimplementation of ``src/pass/ADA.{h,cpp}``.

Builds the instruction dependency graph so the scheduler can issue
instructions in a legal order. Three dependency kinds:

  * RAW (true)    - this instruction reads a memory location that an
                    earlier instruction wrote. (producer -> consumer)
  * WAR (anti)    - this instruction writes a location an earlier one reads.
  * WAW (output)  - this instruction writes a location an earlier one wrote.

Each instruction also gets a ``dep_resource`` string naming the hardware
resource it occupies (``DDR-{bank}`` for loads/stores, ``PE-{pid}`` for
matmults, ``VEC-{pid}`` for vector/move/kernel). Two instructions with the
same resource that touch overlapping memory get a resource dependency edge
(serializing the resource).

DRAM locations are skipped for true/anti/output (they flow through DMA and
are synchronized by barriers).
"""
from __future__ import annotations

from bir.memory import MemoryType
from bir.instruction import LoadInst, StoreInst, MatmultInst, MoveInst, VectorInst

_DRAM = {MemoryType.DRAM}


class ADAPass:
    """Pass: ada - build the instruction dependency graph."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, module) -> int:
        n_edges = 0
        for fn in module:
            for bb in fn:
                n_edges += self._process_block(bb)
        print(f"[ADA] dependency edges added: {n_edges}")
        return 0

    # ------------------------------------------------------------------
    def _process_block(self, bb) -> int:
        instrs = list(bb)
        n = 0
        # last_def[name] = instruction that last wrote memloc 'name'
        # last_use[name] = instruction that last read  memloc 'name'
        last_def: dict = {}
        last_use: dict = {}

        for inst in instrs:
            in_names = self._input_names(inst)
            out_names = self._output_names(inst)

            # RAW: each input that was previously defined -> edge
            for nm in in_names:
                prev = last_def.get(nm)
                if prev is not None and prev is not inst:
                    inst.add_dependency(prev)
                    n += 1

            # WAR: each output that was previously read -> edge
            for nm in out_names:
                prev = last_use.get(nm)
                if prev is not None and prev is not inst:
                    inst.add_dependency(prev)
                    n += 1

            # WAW: each output that was previously written -> edge
            for nm in out_names:
                prev = last_def.get(nm)
                if prev is not None and prev is not inst:
                    inst.add_dependency(prev)
                    n += 1

            # update tables (skip DRAM)
            for ap in inst.input_aps:
                if ap.mloc is not None and ap.mloc.mem_type not in _DRAM:
                    last_use[self._key(ap)] = inst
            for ap in inst.output_aps:
                if ap.mloc is not None and ap.mloc.mem_type not in _DRAM:
                    last_def[self._key(ap)] = inst
                    last_use[self._key(ap)] = inst

            # assign hardware resource + resource dependencies
            self._assign_resource(inst, instrs)

        # re-number local_index
        for i, inst in enumerate(instrs):
            inst.local_index = i
        return n

    # ------------------------------------------------------------------
    def _input_names(self, inst):
        names = []
        for ap in inst.input_aps:
            if ap.mloc is not None and ap.mloc.mem_type not in _DRAM:
                names.append(self._key(ap))
        return names

    def _output_names(self, inst):
        names = []
        for ap in inst.output_aps:
            if ap.mloc is not None and ap.mloc.mem_type not in _DRAM:
                names.append(self._key(ap))
        return names

    def _key(self, ap):
        """Dependency key: memloc name + bank id (per-tensor)."""
        bank = getattr(ap.mloc, "bank_id", 0)
        return f"{ap.mloc.to_string()}-{bank}"

    # ------------------------------------------------------------------
    def _assign_resource(self, inst, all_instrs):
        """Set dep_resource and add resource-dependency edges.

        Resource naming (mirrors C++ assign_resource):
          Load/Store        -> DDR-{bank_id}
          Matmult           -> PE-{partition_id}
          Move/Vector/Kernel-> VEC-{partition_id}
        """
        pid = getattr(inst, "partition_id", 0)
        if isinstance(inst, (LoadInst, StoreInst)):
            bank = 0
            if inst.input_aps:
                bank = getattr(inst.input_aps[0].mloc, "bank_id", 0) or 0
            inst.dep_resource = f"DDR-{bank}"
        elif isinstance(inst, MatmultInst):
            inst.dep_resource = f"PE-{pid}"
        elif isinstance(inst, (MoveInst, VectorInst)):
            inst.dep_resource = f"VEC-{pid}"
        else:
            inst.dep_resource = f"VEC-{pid}"

        # resource dependency: serialize same-resource instructions that
        # touch overlapping on-chip memory.
        if inst.dep_resource and inst.dep_resource.startswith(("PE-", "VEC-")):
            for prev in all_instrs:
                if prev is inst:
                    break
                if getattr(prev, "dep_resource", "") == inst.dep_resource:
                    if self._overlaps_any(prev, inst):
                        inst.add_dependency(prev)


    def _overlaps_any(self, a, b):
        """True if any output AP of a overlaps any input/output AP of b."""
        for ap_a in a.output_aps:
            if ap_a.mloc is None or ap_a.mloc.mem_type in _DRAM:
                continue
            for ap_b in (b.input_aps + b.output_aps):
                if ap_b.mloc is None or ap_b.mloc.mem_type in _DRAM:
                    continue
                if ap_a.mloc is ap_b.mloc:
                    return True
                try:
                    if ap_a.overlap(ap_b):
                        return True
                except Exception:
                    continue
        return False


__all__ = ["ADAPass"]
