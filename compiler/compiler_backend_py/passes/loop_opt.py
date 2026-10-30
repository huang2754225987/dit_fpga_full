"""Pass 5 - LoopOpt: fuse loops + reduce PSUM pressure.

Python reimplementation of ``src/pass/LoopOpt.{h,cpp}``.

Two optimizations:

  1. ``reduce_parallel_psum_groups`` - if the required PSUM buffer depth
     (j_unroll_size * i_free_tripcount) exceeds the hardware
     ``psum_buffer_depth``, tile the ``j`` loop by ``reduce_factor`` so
     each parallel group fits. Loops without a ``j`` child get
     ``replicate``'d into ``reduce_factor`` parallel groups instead.

  2. ``plain_fuse`` - fuse adjacent unrolled outer loops that share the
     same iteration axis. When a GEMM loop is present, fuse tail-first
     (loops after the GEMM), then the GEMM with its successor, then
     head-first (loops before the GEMM). Fusion requires identical axes
     and no loop-carried dependency (checked by ``LoopInst.fuse``).
"""
from __future__ import annotations

from bir.loop import LoopInst
from bir.instruction import MatmultInst


class LoopOptPass:
    """Pass: loop_opt - loop fusion + PSUM pressure reduction."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, module) -> int:
        arch = module.get_arch_model() if hasattr(module, "get_arch_model") else None
        psum_depth = getattr(arch, "psum_buffer_depth", 1024) if arch else 1024
        n_fused = 0
        n_reduced = 0
        for fn in module:
            for bb in fn:
                n_reduced += self._reduce_psum(bb, psum_depth)
                n_fused += self._plain_fuse(bb)
        print(f"[LoopOpt] psum-reduced loops={n_reduced}, fused pairs={n_fused}")
        return 0

    # ------------------------------------------------------------------
    def _reduce_psum(self, bb, psum_depth: int) -> int:
        """Tile j-loops (or replicate) so parallel PSUM groups fit."""
        n = 0
        for inst in list(bb.instructions):
            if not isinstance(inst, LoopInst):
                continue
            try:
                required = self._required_psum_depth(inst)
            except Exception:
                required = 0
            if required <= psum_depth:
                continue
            reduce_factor = (required + psum_depth - 1) // psum_depth
            if reduce_factor <= 1:
                continue
            jloop = self._find_child_by_key(inst, "j")
            if jloop is not None:
                try:
                    outer, inner = jloop.tile(reduce_factor)
                    outer.set_unroll(True)
                    n += 1
                except Exception:
                    pass
            else:
                try:
                    inst.replicate("r", reduce_factor)
                    n += 1
                except Exception:
                    pass
        return n

    def _required_psum_depth(self, loop: LoopInst) -> int:
        j = self._find_child_by_key(loop, "j", check_unroll=True)
        i = self._find_child_by_key(loop, "i", check_unroll=False)
        j_tc = j.axis.tripcount() if j else 1
        i_tc = i.axis.tripcount() if i else 1
        return j_tc * i_tc

    def _find_child_by_key(self, loop: LoopInst, key: str,
                           check_unroll: bool = False):
        """Find a descendant loop whose axis name starts with `key`."""
        try:
            for (inst, _depth) in loop.visit():
                if isinstance(inst, LoopInst):
                    if inst.axis.name.startswith(key):
                        if not check_unroll or inst.get_unroll():
                            return inst
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    def _plain_fuse(self, bb) -> int:
        """Fuse adjacent unrolled outer loops."""
        loops = [i for i in list(bb.instructions) if isinstance(i, LoopInst)]
        if not loops:
            return 0
        n = 0
        gemm_idx = -1
        for idx, lp in enumerate(loops):
            if self._has_matmult(lp):
                gemm_idx = idx
                break

        if gemm_idx >= 0:
            # tail-first (after GEMM)
            for i in range(len(loops) - 1, gemm_idx + 1, -1):
                if self._fuse_pair(loops[i - 1], loops[i]):
                    n += 1
            # GEMM with successor
            if gemm_idx + 1 < len(loops):
                if self._fuse_pair(loops[gemm_idx], loops[gemm_idx + 1]):
                    n += 1
            # head-first (before GEMM)
            for i in range(gemm_idx - 1, -1, -1):
                if self._fuse_pair(loops[i], loops[i + 1]):
                    n += 1
        else:
            for i in range(len(loops) - 1, 0, -1):
                if self._fuse_pair(loops[i - 1], loops[i]):
                    n += 1
        return n

    def _fuse_pair(self, l0: LoopInst, l1: LoopInst) -> bool:
        """Fuse two loops if both unrolled and axes match. Returns success."""
        if l0 is l1:
            return False
        if not (l0.get_unroll() and l1.get_unroll()):
            return False
        try:
            return bool(l0.fuse(l1))
        except Exception:
            return False

    def _has_matmult(self, loop: LoopInst) -> bool:
        try:
            for (inst, _d) in loop.visit():
                if isinstance(inst, MatmultInst):
                    return True
        except Exception:
            pass
        return False


__all__ = ["LoopOptPass"]
