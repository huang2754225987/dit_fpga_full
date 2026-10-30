"""BIR LoopInst: the composite loop instruction.

Python reimplementation of ``bir/Loop.h`` / ``Loop.cpp``.

A ``LoopInst`` is an ``Instruction`` whose body is a ``LoopBody`` (a
BasicBlock). It supports the canonical loop transformations the BIR backend
relies on:

  * :py:meth:`tile`     - split one loop into outer * factor + inner
  * :py:meth:`unroll`   - peel the loop into a flat list of cloned insts
  * :py:meth:`fuse`     - merge two loops with identical axes
  * :py:meth:`interchange` - swap two loops in the same nest
  * :py:meth:`replicate` - wrap a loop in an outer [0, factor) loop
  * :py:meth:`clone`    - deep copy
  * :py:meth:`visit`    - pre-order traversal

The tile/unroll algorithms follow the C++ reference closely: tile rewrites
every access pattern's Expr using ``Expr.substitute`` (old_var ->
outer*factor + inner), and unroll deep-copies the body and substitutes the
var with a ConstExpr for each iteration.
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Any

from .instruction import (Instruction, InstructionType, NoOpInst,
                          VectorInst)
from .loop_axis import LoopAxis
from .expr import Expr, VarExpr, ConstExpr, ArithExpr, ExprOp
from .access_pattern import AccessPattern


class LoopBody:
    """A loop's body: just a BasicBlock plus a back-pointer to the LoopInst."""

    def __init__(self, loop: Optional["LoopInst"] = None):
        # lazy import to avoid a circular dependency at module load time
        from .module import BasicBlock
        self.loop = loop
        self.body = BasicBlock("body", None)
        # stash the owner so BasicBlock consumers can walk up to the parent
        # loop without scanning the whole function.
        self.body._loop_owner = loop

    def get_body(self):
        return self.body

    def get_parent(self) -> Optional["LoopInst"]:
        return self.loop


class LoopInst(Instruction):
    """A loop instruction (opcode == Loop)."""

    def __init__(self, name: str, parent=None, lb: int = 0, ub: int = 0,
                 stride: int = 1):
        super().__init__("loop-" + name, InstructionType.Loop, parent)
        # axis + var + body
        self.axis = LoopAxis(name, lb, ub, stride)
        self.axis.set_loop_inst(self)
        self.body = LoopBody(self)
        self.body.loop = self
        self.axis.set_var(VarExpr(self.axis.name, axis=self.axis))
        self.var = self.axis.get_var()
        # transformation flags
        self._unroll = False
        self._spatial = False
        self._parallel = False
        self._is_matmult_reduce = False
        self._free = False
        self._splitable = False

    # -- accessors -----------------------------------------------------
    def set_unroll(self, unroll: bool) -> None:
        self._unroll = unroll

    def get_unroll(self) -> bool:
        return self._unroll

    def set_spatial(self, spatial: bool) -> None:
        self._spatial = spatial

    def get_spatial(self) -> bool:
        return self._spatial

    def set_axis(self, axis: LoopAxis) -> None:
        self.axis = axis
        self.axis.set_loop_inst(self)

    def get_axis(self) -> LoopAxis:
        return self.axis

    def get_var(self) -> VarExpr:
        return self.axis.get_var()

    def get_body(self):
        return self.body.get_body()

    def get_parent_block(self):
        return self.parent

    # ------------------------------------------------------------------
    # traversal
    # ------------------------------------------------------------------
    def visit(self, depth: int = 0) -> List[Tuple[Instruction, int]]:
        """Pre-order traversal: this loop, then its body (recursing into
        nested loops). Returns (instruction, depth) pairs."""
        result: List[Tuple[Instruction, int]] = [(self, depth)]
        for inst in self.get_body():
            if isinstance(inst, LoopInst):
                result.extend(inst.visit(depth + 1))
            else:
                result.append((inst, depth + 1))
        return result

    def get_child_loops(self) -> List["LoopInst"]:
        """All loops nested directly or indirectly under this one."""
        children: List[LoopInst] = []
        for inst in self.get_body():
            if isinstance(inst, LoopInst):
                children.append(inst)
                children.extend(inst.get_child_loops())
        return children

    def get_parent_loops(self) -> List["LoopInst"]:
        """Walk up the parent-loop chain (outermost first)."""
        parents: List[LoopInst] = []
        cur = self
        while cur.parent is not None and _block_owner_is_loop(cur.parent):
            owner = _block_owner_is_loop(cur.parent)
            parents.append(owner)
            cur = owner
        parents.reverse()
        return parents

    def get_loopnest(self) -> List["LoopInst"]:
        """This loop plus all nested loops (outermost first)."""
        nest = [self]
        nest.extend(self.get_child_loops())
        return nest

    def get_first_inst(self) -> Instruction:
        """First non-loop instruction in the (recursively) first child."""
        for inst in self.get_body():
            if isinstance(inst, LoopInst):
                return inst.get_first_inst()
            return inst
        # empty body - return a fresh NoOp as a sentinel
        return NoOpInst("empty")

    def is_single_inst_with_perfect_loops(self) -> bool:
        """Body is a chain of single-instruction perfect loops, ending in one
        non-loop instruction."""
        if len(self.get_body()) != 1:
            return False
        only = self.get_body()[0]
        if isinstance(only, LoopInst):
            return only.is_single_inst_with_perfect_loops()
        return True

    # ------------------------------------------------------------------
    # tile
    # ------------------------------------------------------------------
    def tile(self, factor: int, inner_size: int = 0) -> Tuple["LoopInst", "LoopInst"]:
        """Split this loop into an outer (factor iterations) + inner loop.

        Rewrites every access pattern in the body so that the old loop var is
        replaced by ``outer * factor + inner``. Returns ``(outer_loop, inner_loop)``
        where ``outer_loop`` is ``self`` (mutated in place).
        """
        # snapshot the body instructions (we'll move them into the inner loop)
        body = self.get_body()
        insts = list(body.instructions)

        # update the axis: this loop becomes the outer, ub := factor
        axis = self.axis
        ub = axis.ub
        axis.ub = factor
        axis._tripcount = -1
        lb_inner = 0
        ub_inner = ub // factor
        if inner_size > 0:
            ub_inner = inner_size
        need_shape_predicate = (inner_size > 0 and ub != factor * inner_size)

        name = axis.name
        axis.set_name(name + "o")
        old_var = self.get_var()
        # fresh var bound to the (renamed) outer axis
        self.var = VarExpr(axis.name, axis=axis)
        axis.set_var(self.var)

        # build the inner loop inside this loop's body
        inner = LoopInst(name + "i", body, lb_inner, ub_inner, 1)
        body.add_instruction(inner)
        # placeholder so we can moveBefore it
        ph = NoOpInst("placeholder", inner.get_body())
        inner.get_body().add_instruction(ph)
        for inst in insts:
            inner.get_body().move_before(inst, ph)
        inner.get_body().remove_instruction(ph)

        # rewrite every access pattern + predicate
        c0 = ConstExpr(inner.get_axis().tripcount())
        cu = ConstExpr(ub)
        # outer_var * c0 + inner_var
        replacement: Expr = self.get_var() * c0 + inner.get_var()
        for inst, _depth in self.visit():
            if isinstance(inst, LoopInst):
                continue
            aps = list(inst.input_aps) + list(inst.output_aps)
            for ap in aps:
                ap.access = [e.substitute(old_var, replacement).simplify()
                             for e in ap.access]
            if inst.has_predicate():
                inst.set_predicate(inst.get_predicate().substitute(old_var, replacement).simplify())
            if need_shape_predicate:
                inst.add_predicate(self.get_var() * c0 + inner.get_var() < cu)
        return self, inner

    # ------------------------------------------------------------------
    # unroll
    # ------------------------------------------------------------------
    def unroll(self) -> List[Instruction]:
        """Replace this loop with its body replicated per iteration.

        For each ``i`` in ``[lb, ub)`` step ``stride``: clone every body
        instruction, substitute ``var`` with ``ConstExpr(i)``, and append the
        clones to the parent block just before this loop. Finally remove this
        loop. Returns the list of unrolled instructions.
        """
        unrolled: List[Instruction] = []
        var = self.get_var()
        bb = self.parent
        if bb is None:
            return unrolled
        axis = self.axis
        i = axis.lb
        while i < axis.ub:
            c = ConstExpr(i)
            for inst in list(self.get_body().instructions):
                ci = inst.clone()
                ci.recursive_replace_expr(var, c)
                bb.move_before(ci, self)
                unrolled.append(ci)
            i += axis.stride
        bb.remove_instruction(self)
        return unrolled

    def recursive_unroll_parallel_loops(self) -> None:
        """Unroll every nested loop flagged ``_unroll`` (deepest first)."""
        loops = self.get_child_loops()
        loops.insert(0, self)
        loops.reverse()  # deepest first
        for loop in loops:
            if loop.get_unroll():
                loop.unroll()

    # ------------------------------------------------------------------
    # fuse
    # ------------------------------------------------------------------
    def fuse(self, other: "LoopInst") -> bool:
        """Merge ``other`` into ``self`` (same axes required).

        Returns True on success. The two loops must live in the same block
        and have identical (lb, ub, stride). A conservative loop-carried
        dependency check guards the merge.
        """
        l0 = self
        l1 = other
        if l0.parent is not l1.parent:
            return False
        a0, a1 = l0.axis, l1.axis
        if not (a0.lb == a1.lb and a0.ub == a1.ub and a0.stride == a1.stride):
            return False
        if _has_loop_carried_dependency(l0, l1):
            return False
        # move l1's body into l0's body, replacing l1.var with l0.var
        nop = NoOpInst("ph", l0.get_body())
        l0.get_body().add_instruction(nop)
        for inst in list(l1.get_body().instructions):
            inst.recursive_replace_expr(l1.get_var(), l0.get_var())
            l0.get_body().move_before(inst, nop)
        l0.get_body().remove_instruction(nop)
        l0.parent.remove_instruction(l1)
        return True

    # ------------------------------------------------------------------
    # interchange
    # ------------------------------------------------------------------
    def interchange(self, other: "LoopInst") -> bool:
        """Swap two loops in the same nest.

        Returns True on success. The two loops must be directly nested
        (outer/inner) with only perfect single-instruction loops between them.
        """
        # requirement 1: same nest (one is an ancestor of the other)
        valid = False
        outer = inner = None
        children = self.get_child_loops()
        if other in children:
            outer, inner = self, other
            valid = True
        if not valid:
            children = other.get_child_loops()
            if self in children:
                outer, inner = other, self
                valid = True
        if not valid:
            return False

        # requirement 2: perfect loops between outer and inner
        t = outer
        while t is not inner:
            visited_loop = 0
            for inst in t.get_body():
                if isinstance(inst, LoopInst) and visited_loop == 0:
                    t = inst
                    visited_loop += 1
                else:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            return False

        # swap the two loops' bodies (inner becomes outer, outer becomes inner)
        inner_body_insts = list(inner.get_body().instructions)
        outer_body_insts = list(outer.get_body().instructions)
        # detach inner from outer's body and move outer's other insts out
        # The simplest faithful swap: exchange the body instruction lists.
        inner.get_body().instructions = outer_body_insts
        outer.get_body().instructions = inner_body_insts
        # fix up parent back-refs
        for inst in inner.get_body().instructions:
            inst.parent = inner.get_body()
        for inst in outer.get_body().instructions:
            inst.parent = outer.get_body()
        # swap axis names so dumps still read sensibly
        outer.axis, inner.axis = inner.axis, outer.axis
        outer.var, inner.var = inner.var, outer.var
        return True

    # ------------------------------------------------------------------
    # replicate
    # ------------------------------------------------------------------
    def replicate(self, name: str, factor: int) -> "LoopInst":
        """Wrap ``self`` in an outer ``[0, factor)`` loop named ``name``.

        The new outer loop is inserted into this loop's parent block
        immediately after this loop, and this loop is moved into the new
        loop's body. Returns the new outer loop.
        """
        bb = self.parent
        outer = LoopInst(name, bb, 0, factor, 1)
        bb.move_after(outer, self)
        nop = NoOpInst("ph", outer.get_body())
        outer.get_body().add_instruction(nop)
        outer.get_body().move_before(self, nop)
        outer.get_body().remove_instruction(nop)
        return outer

    # ------------------------------------------------------------------
    # clone
    # ------------------------------------------------------------------
    def clone(self, create_var: bool = False) -> "LoopInst":
        """Deep-copy this loop into the same parent block.

        When ``create_var`` is False (the default) the clone shares this
        loop's axis var (matches the C++ default). When True a fresh axis+var
        is allocated.
        """
        bb = self.parent
        axis = self.axis
        clone = LoopInst(axis.name, bb, axis.lb, axis.ub, axis.stride)
        clone._unroll = self._unroll
        clone._spatial = self._spatial
        clone._free = self._free
        clone._parallel = self._parallel
        clone._is_matmult_reduce = self._is_matmult_reduce
        clone._splitable = self._splitable
        if not create_var:
            clone.axis.set_var(self.get_var())
            clone.var = self.get_var()
        # move clone right after self
        if bb is not None:
            bb.move_after(clone, self)
        # seed body with a NoOp pivot, clone every body inst, then drop pivot
        ph = NoOpInst("ph0", clone.get_body())
        clone.get_body().add_instruction(ph)
        for inst in list(self.get_body().instructions):
            if isinstance(inst, LoopInst):
                ci = inst.clone(create_var)
            else:
                ci = inst.clone()
            clone.get_body().move_before(ci, ph)
        clone.get_body().remove_instruction(ph)
        # copy APs / mlocs / predicate that live on the loop instruction itself
        clone.input_aps = [ap.clone() for ap in self.input_aps]
        clone.output_aps = [ap.clone() for ap in self.output_aps]
        clone.input_mlocs = list(self.input_mlocs)
        clone.output_mlocs = list(self.output_mlocs)
        if self.has_predicate():
            clone.predicate = self.predicate.clone()
        clone.attributes = dict(self.attributes)
        return clone

    def _create_empty_clone(self) -> "LoopInst":
        return LoopInst(self.axis.name, self.parent,
                        self.axis.lb, self.axis.ub, self.axis.stride)

    # ------------------------------------------------------------------
    # debugging
    # ------------------------------------------------------------------
    def to_string(self) -> str:
        lines = []
        for inst, depth in self.visit():
            indent = "  " * depth
            if isinstance(inst, LoopInst):
                s = indent + inst.axis.to_string()
                if inst._unroll:
                    s += " // unroll"
                if inst._spatial:
                    s += " // spatial"
                if inst._free:
                    s += " // free"
                lines.append(s)
            else:
                lines.append(indent + str(inst))
        return "\n".join(lines)

    def __repr__(self):
        return f"LoopInst({self.axis.name!r}, [{self.axis.lb},{self.axis.ub}))"

    def __str__(self):
        return self.to_string()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _block_owner_is_loop(block) -> Optional[LoopInst]:
    """If ``block`` is the body of a LoopInst, return that LoopInst.

    BasicBlock.parent is normally a Function, but for a loop body the parent
    is None (we set ``LoopBody.body.parent = None``). We instead find the
    owning loop by scanning the parent function for a loop whose body equals
    this block - but that's expensive. The cheaper path: LoopBody stores
    ``loop``; for a BasicBlock that is a loop body we stash the owner on
    ``block._loop_owner`` when we build it.
    """
    owner = getattr(block, "_loop_owner", None)
    if isinstance(owner, LoopInst):
        return owner
    return None


def _has_loop_carried_dependency(l0: LoopInst, l1: LoopInst) -> bool:
    """Conservative loop-carried dependency check between two loops.

    Mirrors the C++ ``has_loop_carried_dependency``: collect l0's producers
    (output AP 0 of every non-loop inst) and l1's consumers (input APs of
    every non-loop inst); if a producer and consumer share a memory location
    and the loop var appears in only one of them, there's a carried dep.
    """
    l0_insts = [i for i, _ in l0.visit() if not isinstance(i, LoopInst)]
    producers: List[AccessPattern] = []
    for inst in l0_insts:
        if inst.output_aps:
            producers.append(inst.output_aps[0])
    l1_insts = [i for i, _ in l1.visit() if not isinstance(i, LoopInst)]
    consumers: List[AccessPattern] = []
    for inst in l1_insts:
        consumers.extend(inst.input_aps)
    for p in producers:
        for c in consumers:
            if p.mloc is None or c.mloc is None:
                continue
            if p.mloc != c.mloc:
                continue
            d0 = p.find_var_dim(l0.get_var())
            d1 = c.find_var_dim(l1.get_var())
            if d0 != d1 and (d0 == -1 or d1 == -1):
                return True
    return False


def loop_reorder(before: List[LoopInst], after: List[LoopInst]) -> None:
    """Reorder a nest of loops via pairwise interchange.

    ``before`` is the current ordering; ``after`` is the target ordering.
    Each element of ``after`` must appear in ``before``.
    """
    if len(before) != len(after) or len(before) == 0:
        return
    loops = list(before)
    pos_map = {loops[i]: i for i in range(len(loops))}
    for a in after:
        if a not in pos_map:
            return
    for i in range(len(before)):
        if loops[i] is not after[i]:
            j = pos_map[after[i]]
            pos_map[after[i]] = i
            pos_map[loops[i]] = j
            loops[i].interchange(after[i])
            loops[i], loops[j] = loops[j], loops[i]
