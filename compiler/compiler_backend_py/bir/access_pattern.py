"""BIR memory access patterns.

Python reimplementation of ``bir/AccessPattern.h`` / ``.cpp``.

An ``AccessPattern`` is the (Expr per tensor dim, MemoryLocation) pair that
describes which slice of which buffer an instruction touches. The symbolic
form (one affine Expr per dimension, outer-to-inner) is what the tiling and
unrolling passes rewrite.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from .expr import Expr, VarExpr, RangeExpr

if TYPE_CHECKING:
    from .memory import MemoryLocation
    from .loop_axis import LoopAxis


class AccessPattern:
    """A symbolic memory access: one Expr per tensor dimension + a memory loc.

    Attributes:
        access: list of Expr, outer dimension first. Each Expr is either a
            scalar (a point access) or a RangeExpr (a span access).
        mloc: the MemoryLocation this access references.
    """

    def __init__(self, access: Optional[List[Expr]] = None,
                 mloc: Optional["MemoryLocation"] = None):
        self.access: List[Expr] = list(access) if access is not None else []
        self.mloc = mloc

    # -- accessors -----------------------------------------------------
    def set_mem_loc(self, mloc: "MemoryLocation") -> None:
        self.mloc = mloc

    def get_mem_loc(self) -> Optional["MemoryLocation"]:
        return self.mloc

    def get_access(self, idx: int) -> Expr:
        return self.access[idx]

    def get_last_access(self) -> Expr:
        return self.access[-1]

    def add_access(self, e: Expr) -> None:
        self.access.append(e)

    def set_access(self, idx: int, e: Expr) -> None:
        self.access[idx] = e

    def get_dims(self) -> int:
        return len(self.access)

    # -- queries --------------------------------------------------------
    def find_var_dim(self, var: VarExpr) -> int:
        """Return the index of the first dim whose Expr references ``var``.

        ``-1`` if not found. Mirrors ``AccessPattern::FindVarDim``.
        """
        for i, e in enumerate(self.access):
            vars_in_e = e.collect_vars()
            for v in vars_in_e:
                if v == var:
                    return i
        return -1

    def substitute(self, var: VarExpr, expr: Expr) -> "AccessPattern":
        """Return a copy with ``var`` replaced by ``expr`` in every dim."""
        new_access = [e.substitute(var, expr) for e in self.access]
        return AccessPattern(new_access, self.mloc)

    def overlap(self, other: "AccessPattern") -> bool:
        """Do these two APs touch the same memory?

        Conservative: same memory location, and every dimension's range
        intersects. Non-RangeExpr (point) accesses are treated as touching
        only when structurally equal.
        """
        if self.mloc is None or other.mloc is None:
            return False
        # C++ uses getLoc() (resolved parent loc); we just compare the loc.
        if self.mloc != other.mloc:
            return False
        if len(self.access) != len(other.access):
            return False
        for a, b in zip(self.access, other.access):
            if isinstance(a, RangeExpr) and isinstance(b, RangeExpr):
                if not a.overlap(b):
                    return False
            else:
                if a != b:
                    return False
        return True

    # -- cloning -------------------------------------------------------
    def clone(self) -> "AccessPattern":
        new_access = [e.clone() for e in self.access]
        return AccessPattern(new_access, self.mloc)

    # -- debugging -----------------------------------------------------
    def to_string(self) -> str:
        if self.mloc is not None:
            head = self.mloc.to_string()
        else:
            head = "<no-mloc>"
        parts = []
        for e in self.access:
            if isinstance(e, RangeExpr):
                parts.append(e.to_string())
            else:
                parts.append(f"[{e.to_string()}]")
        return head + "".join(parts)

    def __repr__(self):
        return f"AccessPattern({self.to_string()})"

    def __str__(self):
        return self.to_string()


class PhysicalAccessPattern(AccessPattern):
    """A symbolic AP plus the resolved physical placement.

    Once a symbolic AP is lowered against a particular buffer allocation, we
    attach the (bank_id, offset, step, step_size, stride) tuple so the
    codegen pass can emit concrete DMA / load addresses.
    """

    def __init__(self, bank_id: int = 0, offset: int = 0, step: int = 0,
                 step_size: int = 0, stride: int = 0,
                 access: Optional[List[Expr]] = None,
                 mloc: Optional["MemoryLocation"] = None,
                 ap: Optional[AccessPattern] = None):
        if ap is not None:
            access = [e.clone() for e in ap.access]
            mloc = ap.mloc
        super().__init__(access=access, mloc=mloc)
        self.bank_id = bank_id
        self.offset = offset
        self.step = step
        self.step_size = step_size
        self.stride = stride

    def clone(self) -> "PhysicalAccessPattern":
        return PhysicalAccessPattern(
            bank_id=self.bank_id, offset=self.offset, step=self.step,
            step_size=self.step_size, stride=self.stride,
            access=[e.clone() for e in self.access], mloc=self.mloc)

    def to_string(self) -> str:
        base = super().to_string()
        return (f"{base}[#{self.bank_id}:{self.offset}+:{self.step_size}"
                f"x{self.step}s{self.stride}]")

    def __repr__(self):
        return f"PhysicalAccessPattern({self.to_string()})"
