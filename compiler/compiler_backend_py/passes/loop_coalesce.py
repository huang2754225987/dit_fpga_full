"""Pass 4 - LoopMemoryCoalesce: coalesce small DMA loads.

Python reimplementation of ``src/pass/LoopMemoryCoalesce.{h,cpp}``.

Walks single-instruction perfect load loops that are flagged ``_unroll``.
For those whose loaded buffer name contains ``gamma``, ``beta``, or
``residue`` (small per-layer constants / residual vectors), clear the
unroll flag so the DMA engine merges the per-iteration loads into a
single larger transfer instead of ``core_num`` small ones.

This is a name-string-match policy, exactly like the C++ reference
(``disable_loop_unroll`` checks for "beta"/"gamma"/"residue" in the
mloc name).
"""
from __future__ import annotations

from bir.loop import LoopInst
from bir.instruction import LoadInst

_COALESCE_KEYS = ("beta", "gamma", "residue")


class LoopCoalescePass:
    """Pass: loop_coalesce - merge small DMA loads."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, module) -> int:
        n = 0
        for fn in module:
            for bb in fn:
                n += self._coalesce_block(bb)
        print(f"[LoopCoalesce] coalesced loops={n}")
        return 0

    # ------------------------------------------------------------------
    def _coalesce_block(self, bb) -> int:
        n = 0
        for inst in list(bb.instructions):
            if not isinstance(inst, LoopInst):
                continue
            # recurse into the loop body first
            n += self._coalesce_block(inst.get_body())
            # then check this loop
            if not inst.get_unroll():
                continue
            if not self._is_single_load_loop(inst):
                continue
            first = self._first_inst(inst)
            if first is None:
                continue
            mloc_name = self._loaded_name(first)
            if mloc_name and any(k in mloc_name.lower() for k in _COALESCE_KEYS):
                inst.set_unroll(False)
                n += 1
        return n

    def _is_single_load_loop(self, loop: LoopInst) -> bool:
        try:
            return loop.is_single_inst_with_perfect_loops()
        except Exception:
            return False

    def _first_inst(self, loop: LoopInst):
        try:
            return loop.get_first_inst()
        except Exception:
            return None

    def _loaded_name(self, inst) -> str:
        """The output mloc name of a LoadInst (where the data lands)."""
        for ap in getattr(inst, "output_aps", []) or []:
            if ap.mloc is not None:
                return ap.mloc.name or ""
        return ""


__all__ = ["LoopCoalescePass"]
