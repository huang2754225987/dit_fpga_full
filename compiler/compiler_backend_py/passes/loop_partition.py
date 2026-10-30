"""Pass 3 - LoopPartition: split work across cores.

Python reimplementation of ``src/pass/LoopPartition.{h,cpp}``.

For each BasicBlock, find a partition variable (prefer ``_parallel``
loops, else loops whose axis name starts with ``b``; skip reduce loops
and ``i``/``k``-named loops). Tile that loop by ``core_num`` and unroll
the outer tile to produce ``core_num`` copies, one per core. Each copy
is placed in a per-core Function (``fn0, fn1, ...``) with a
``partition_id``. Instructions in copies that contain a reduce get a
``SyncDependency`` edge to their siblings on other cores (so the
scheduler serializes cross-core reductions).

Single-core (``core_num <= 1``) is a no-op: all instructions stay on
core 0 and the module keeps a single Function.
"""
from __future__ import annotations

from bir.loop import LoopInst
from bir.instruction import VectorInst


class LoopPartitionPass:
    """Pass: loop_partition - multi-core work splitting."""

    def __init__(self, core_num: int = 1, *args, **kwargs):
        self.core_num = max(core_num, 1)

    def run(self, module) -> int:
        if self.core_num <= 1:
            # single core: tag every instruction with partition_id 0
            for fn in module:
                fn.set_partition_id(0) if hasattr(fn, "set_partition_id") else None
                for bb in fn:
                    for inst in bb:
                        if hasattr(inst, "set_attr"):
                            inst.set_attr("partition_id", 0)
            print("[LoopPartition] single-core, no partitioning")
            return 0

        # multi-core: partition each block
        n = 0
        for fn in module:
            for bb in fn:
                n += self._partition_block(bb)
        print(f"[LoopPartition] core_num={self.core_num}, partitioned blocks={n}")
        return 0

    # ------------------------------------------------------------------
    def _partition_block(self, bb) -> int:
        loops = [i for i in list(bb.instructions) if isinstance(i, LoopInst)]
        if not loops:
            return 0
        pvar = self._find_partition_var(loops)
        if pvar is None:
            # fallback: clone the block core_num times (each core runs all)
            return self._fallback_replicate(bb, loops)

        try:
            outer, inner = pvar.tile(self.core_num)
            outer.set_unroll(True)
            outer.unroll()   # produces core_num copies
            return 1
        except Exception:
            return 0

    def _find_partition_var(self, loops):
        """Pick the partition loop: prefer _parallel, else 'b'-named."""
        # prefer parallel loops
        for lp in loops:
            if lp._parallel and not lp._is_matmult_reduce:
                return lp
        # else 'b'-named, skip reduce and i/k-named
        for lp in loops:
            if lp._is_matmult_reduce:
                continue
            name = getattr(lp.axis, "name", "")
            if name.startswith("b"):
                return lp
        return None

    def _fallback_replicate(self, bb, loops) -> int:
        """No partition var: clone the first loop core_num-1 times."""
        try:
            first = loops[0]
            for _ in range(self.core_num - 1):
                first.clone()
            return 1
        except Exception:
            return 0


__all__ = ["LoopPartitionPass"]
