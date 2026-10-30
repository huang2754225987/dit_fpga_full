"""Pass 6 - LoopUnroll: unroll loops into flat instruction sequences.

Python reimplementation of ``src/pass/LoopUnroll.{h,cpp}`` (class
``LoopUnroller``).

Steps (mirrors the C++ ``run``):
  1. ``recursive_unroll_parallel_loops`` - peel every loop flagged
     ``_unroll`` (deepest first) into cloned body instructions.
  2. ``LowerToTensorInst`` - a single-instruction perfect loop becomes a
     tensor-inst (a leaf compute/load/store node); multi-instruction
     loops become kernel-inst (fused blocks). Tagged via attributes.
  3. ``CanonicalizeMemLoc`` - group overlapping AccessPatterns that share
     the same on-chip buffer so downstream address allocation can alias
     them. Non-overlapping ones get suffixed names ``name_0, name_1, ...``.
  4. ``CheckPredicate`` - drop instructions whose predicate evaluates to
     false; clear the predicate flag on true ones.
"""
from __future__ import annotations

from bir.loop import LoopInst
from bir.memory import MemoryLocation, MemoryType
from bir.access_pattern import AccessPattern


# on-chip buffer types that LoopAllocator handles; canonicalization skips these
_SKIP_TYPES = {MemoryType.DRAM, MemoryType.SPAD, MemoryType.CONST_VALUE}


class LoopUnrollPass:
    """Pass: loop_unroll - flatten loop nests into instruction sequences."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, module) -> int:
        for fn in module:
            for bb in list(fn):
                self._unroll_block(bb)
                self._canonicalize_memloc(bb)
                self._check_predicate(bb)
        print("[LoopUnroll] unrolled loops + canonicalized memlocs")
        return 0

    # ------------------------------------------------------------------
    def _unroll_block(self, bb):
        """Recursively unroll every _unroll-flagged loop in bb (deepest first)."""
        for inst in list(bb.instructions):
            if isinstance(inst, LoopInst):
                # first recurse into the loop body (child loops)
                self._unroll_block(inst.get_body())
                # then unroll this loop if flagged
                if inst.get_unroll():
                    inst.recursive_unroll_parallel_loops()
                else:
                    self._lower_to_tensor_inst(inst)

    def _lower_to_tensor_inst(self, loop: LoopInst):
        """Tag a perfect single-instruction loop as a tensor-inst (leaf)."""
        try:
            if loop.is_single_inst_with_perfect_loops():
                first = loop.get_first_inst()
                if first is not None:
                    first.set_attr("is_tensor_inst", 1)
            else:
                first = loop.get_first_inst()
                if first is not None:
                    first.set_attr("is_kernel_inst", 1)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _canonicalize_memloc(self, bb):
        """Canonicalize on-chip MemoryLocations within bb.

        Group APs by (mem_type, base_name). Overlapping APs share one
        canonical MemoryLocation; non-overlapping ones get suffixed names.
        DRAM/SPAD/CONST locations are left for LoopAllocator.
        """
        groups: dict = {}
        for inst in bb.instructions:
            for ap in inst.get_all_aps():
                if ap.mloc is None:
                    continue
                mt = ap.mloc.mem_type
                if mt in _SKIP_TYPES:
                    continue
                key = (mt, ap.mloc.name)
                groups.setdefault(key, []).append((inst, ap))

        for (mt, base), entries in groups.items():
            if not entries:
                continue
            canonical = entries[0][1].mloc
            seen = {id(canonical)}
            counter = 0
            for (inst, ap) in entries:
                if id(ap.mloc) in seen:
                    continue
                try:
                    overlaps = ap.overlap(AccessPattern(access=None, mloc=canonical))
                except Exception:
                    overlaps = False
                if overlaps:
                    ap.mloc = canonical
                else:
                    counter += 1
                    new_mloc = MemoryLocation(f"{base}_{counter}", mt,
                                              tensor_shape=ap.mloc.tensor_shape)
                    ap.mloc = new_mloc
                    seen.add(id(new_mloc))

    # ------------------------------------------------------------------
    def _check_predicate(self, bb):
        """Drop instructions whose predicate is statically false."""
        for inst in list(bb.instructions):
            pred = inst.get_predicate() if hasattr(inst, "get_predicate") else None
            if pred is None:
                continue
            try:
                val = inst.eval_predicate()
            except Exception:
                continue
            if val is False:
                bb.remove_instruction(inst)
            elif val is True:
                inst.clear_predicate()


__all__ = ["LoopUnrollPass"]
