"""BIR loop iteration space (LoopAxis).

Python reimplementation of ``bir/LoopAxis.h``.

A ``LoopAxis`` describes the iteration space of one loop: a half-open range
``[lb, ub)`` stepped by ``stride``. It owns the ``VarExpr`` that names the
induction variable and carries flags describing whether the loop is parallel
or on-chip. The axis is what gets split during tiling.
"""

from __future__ import annotations

import math
from typing import Tuple


class LoopAxis:
    """The iteration space of a single loop nest level.

    Attributes:
        lb, ub, stride: half-open range ``[lb, ub)`` with step ``stride``.
        name: name of the induction variable (display + keying).
        var: the ``VarExpr`` bound to this axis (created lazily on first
            access of :py:meth:`get_var`; set by :py:meth:`set_var`).
        is_parallel: this axis drives parallel cores.
        is_on_chip: this axis drives on-chip buffers (vs. DRAM).
    """

    def __init__(self, name: str, lb: int = 0, ub: int = 0, stride: int = 1):
        self.name = name
        self.lb = lb
        self.ub = ub
        self.stride = stride
        self.is_parallel = False
        self.is_on_chip = False
        # cached/computed
        self._tripcount = -1
        self.last_ub = -1          # residue size, set when tiling non-uniformly
        self._loopinst = None     # back-ref to owning LoopInst, if any
        self._var = None           # owned VarExpr

    # -- identity -------------------------------------------------------
    def __repr__(self):
        return (f"LoopAxis({self.name!r}, lb={self.lb}, ub={self.ub}, "
                f"stride={self.stride})")

    def __eq__(self, other):
        if not isinstance(other, LoopAxis):
            return NotImplemented
        # Axis identity is by object (loop-carried var substitution relies on
        # the axis pointer, exactly like C++).
        return self is other

    def __hash__(self):
        return id(self)

    # -- getters / setters ---------------------------------------------
    def set_lb(self, lb: int) -> None:
        self.lb = lb
        self._tripcount = -1

    def set_ub(self, ub: int) -> None:
        self.ub = ub
        self._tripcount = -1

    def set_stride(self, stride: int) -> None:
        self.stride = stride
        self._tripcount = -1

    def set_name(self, name: str) -> None:
        self.name = name

    def get_lb(self) -> int:
        return self.lb

    def get_ub(self) -> int:
        return self.ub

    def get_stride(self) -> int:
        return self.stride

    def get_name(self) -> str:
        return self.name

    # -- tripcount ------------------------------------------------------
    def tripcount(self) -> int:
        """Number of iterations ``[lb, ub)`` stepped by ``stride``."""
        if self._tripcount == -1:
            if self.stride == 0:
                self._tripcount = 0
            else:
                self._tripcount = (self.ub - self.lb) // self.stride
        return self._tripcount

    get_tripcount = tripcount

    # -- var / loopinst back-refs ---------------------------------------
    def set_var(self, var) -> None:
        self._var = var
        if var is not None:
            var.axis = self

    def get_var(self):
        if self._var is None:
            # lazy import to avoid a circular dependency at module load time
            from .expr import VarExpr
            self._var = VarExpr(self.name, axis=self)
        return self._var

    def set_loop_inst(self, loopinst) -> None:
        self._loopinst = loopinst

    def get_loop_inst(self):
        return self._loopinst

    # -- tiling ---------------------------------------------------------
    def tile(self, tile_size: int) -> Tuple["LoopAxis", "LoopAxis"]:
        """Split this axis into (outer, inner).

        ``outer`` iterates over ``ceil(tripcount/tile_size)`` tiles,
        ``inner`` iterates over ``tile_size`` elements per tile. The residue
        (when tripcount is not a multiple of tile_size) is recorded in
        ``inner.last_ub`` so the caller can install a bounds predicate.

        This mutates *self* in place (it becomes ``outer``) and returns the
        freshly-created inner axis, matching ``LoopAxis::tile`` in C++.
        """
        tc = self.tripcount()
        tile_cnt = math.ceil(tc / tile_size) if tile_size > 0 else 0
        residue = tc % tile_size
        # self becomes the outer axis
        self.ub = tile_cnt
        self._tripcount = -1
        self.tripcount()
        inner = LoopAxis(self.name + "_1", 0, tile_size, 1)
        if residue != 0:
            inner.last_ub = residue
        self.name = self.name + "_0"
        return self, inner

    # -- debugging ------------------------------------------------------
    def to_string(self, level: int = 0) -> str:
        indent = "  " * level
        s = f"{indent}for {self.name} : [{self.lb},{self.ub})"
        if self.stride != 1:
            s += f" step {self.stride}"
        if self.is_parallel:
            s += "  // parallel"
        if self.is_on_chip:
            s += " // onchip"
        return s

    def __str__(self) -> str:
        return self.to_string()

    # -- cloning --------------------------------------------------------
    def clone(self) -> "LoopAxis":
        axis = LoopAxis(self.name, self.lb, self.ub, self.stride)
        axis.is_parallel = self.is_parallel
        axis.is_on_chip = self.is_on_chip
        axis._tripcount = -1
        axis.last_ub = self.last_ub
        return axis
