"""BIR instruction hierarchy.

Python reimplementation of ``bir/Instruction.h`` / ``.cpp``.

The instruction hierarchy models the operations the NPU can run: atomic
``Load``/``Store``/``Matmult``/``Move``/``Memset``/``Vector``/``Transpose``
ops, the composite ``Loop`` (see loop.py) and ``Kernel`` (fused block) ops,
and the sentinel ``NoOp``. Each instruction carries its input/output
access patterns and memory locations, a dependency graph (pred/succ lists),
an optional guard predicate, and a free-form attribute dict.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from .expr import Expr, VarExpr
from .access_pattern import AccessPattern

if TYPE_CHECKING:
    from .memory import MemoryLocation
    from .module import BasicBlock


class InstructionType(Enum):
    """The opcode carried by every Instruction."""
    Load = "Load"
    Store = "Store"
    Matmult = "Matmult"
    Move = "Move"
    Memset = "Memset"
    Vector = "Vector"
    Transpose = "Transpose"
    Loop = "Loop"
    NoOp = "NoOp"
    Kernel = "Kernel"

    def __str__(self) -> str:
        return self.value


class VectorType(Enum):
    """Vector-op sub-opcode for ``VectorInst``."""
    EltwiseAdd = "EltwiseAdd"
    EltwiseSub = "EltwiseSub"
    EltwiseMul = "EltwiseMul"
    EltwiseDiv = "EltwiseDiv"
    EltwiseExp = "EltwiseExp"
    EltwiseGELU = "EltwiseGELU"
    EltwiseSiLU = "EltwiseSiLU"
    ReduceMax = "ReduceMax"
    ReduceAdd = "ReduceAdd"
    ReduceMean = "ReduceMean"
    ReduceMeanSqrt = "ReduceMeanSqrt"
    ComplexMul = "ComplexMul"
    RMSNorm = "RMSNorm"
    Softmax = "Softmax"
    RoPE = "RoPE"

    def __str__(self) -> str:
        return self.value


# Set of reduce-family vector ops (used by VectorInst.is_reduce).
_REDUCE_OPS = {VectorType.ReduceMax, VectorType.ReduceAdd,
               VectorType.ReduceMean, VectorType.ReduceMeanSqrt}


class Instruction:
    """Base class for all BIR instructions."""

    def __init__(self, name: str, opcode: InstructionType,
                 parent: Optional["BasicBlock"] = None):
        self.name = name
        self.opcode = opcode
        self.parent = parent  # owning BasicBlock (or None for a free inst)

        # dependency lists (raw instruction pointers, like C++ pred/succ)
        self.pred: List["Instruction"] = []
        self.succ: List["Instruction"] = []
        self.pred_edges: List[Any] = []
        self.succ_edges: List[Any] = []

        # access patterns + memory locations
        self.input_aps: List[AccessPattern] = []
        self.output_aps: List[AccessPattern] = []
        self.input_mlocs: List["MemoryLocation"] = []
        self.output_mlocs: List["MemoryLocation"] = []

        # predicate (guard Expr) - None means always-execute
        self.predicate: Optional[Expr] = None

        # back-reference to the DFG node, once lowered
        self.dfg_node = None

        # nesting info
        self.parent_loops: List[Any] = []
        self.local_index: int = -1
        self.parent_inst: Optional["Instruction"] = None
        self.child_insts: List["Instruction"] = []

        # free-form attributes (C++ uses unordered_map<string,int>; we use a
        # plain dict so callers can stash arbitrary values).
        self.attributes: Dict[str, Any] = {}
        self.dep_resource: str = ""

        # iteration map (populated by LoopInst.unroll)
        self.unrolled_var_iter_map: Dict[VarExpr, int] = {}

    # -- opcode helpers -------------------------------------------------
    def get_opcode(self) -> InstructionType:
        return self.opcode

    def get_parent(self) -> Optional["BasicBlock"]:
        return self.parent

    def set_parent(self, parent: "BasicBlock") -> None:
        self.parent = parent

    # -- dependency edges ----------------------------------------------
    def add_dependency(self, other: "Instruction") -> None:
        """Make ``self`` depend on ``other`` (other must run first)."""
        if other not in self.pred:
            self.pred.append(other)
        if self not in other.succ:
            other.succ.append(self)

    def remove_dependency(self, other: "Instruction") -> None:
        """Drop the self-depends-on-other edge (both directions)."""
        if other in self.pred:
            self.pred.remove(other)
        if self in other.succ:
            other.succ.remove(self)

    # -- access patterns -----------------------------------------------
    def add_input_ap(self, ap: AccessPattern) -> None:
        self.input_aps.append(ap)

    def get_input_ap(self, index: int) -> AccessPattern:
        return self.input_aps[index]

    def add_output_ap(self, ap: AccessPattern) -> None:
        self.output_aps.append(ap)

    def get_output_ap(self, index: int) -> AccessPattern:
        return self.output_aps[index]

    def get_all_aps(self) -> List[AccessPattern]:
        return list(self.input_aps) + list(self.output_aps)

    # -- memory locations ---------------------------------------------
    def add_input_mem_loc(self, mloc: "MemoryLocation") -> None:
        self.input_mlocs.append(mloc)

    def get_input_mem_loc(self, index: int) -> "MemoryLocation":
        return self.input_mlocs[index]

    def add_output_mem_loc(self, mloc: "MemoryLocation") -> None:
        self.output_mlocs.append(mloc)

    def get_output_mem_loc(self, index: int) -> "MemoryLocation":
        return self.output_mlocs[index]

    def get_all_mem_locs(self) -> List["MemoryLocation"]:
        return list(self.input_mlocs) + list(self.output_mlocs)

    # -- attributes ----------------------------------------------------
    def set_attr(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def get_attr(self, key: str, default: Any = None) -> Any:
        return self.attributes.get(key, default)

    def has_attr(self, key: str) -> bool:
        return key in self.attributes

    # -- predicate (guard Expr) ----------------------------------------
    def has_predicate(self) -> bool:
        return self.predicate is not None

    def set_predicate(self, e: Optional[Expr]) -> None:
        self.predicate = e

    def get_predicate(self) -> Optional[Expr]:
        return self.predicate

    def clear_predicate(self) -> None:
        self.predicate = None

    def add_predicate(self, e: Expr) -> None:
        """Conjoin ``e`` onto the existing predicate (or set it)."""
        if self.predicate is not None:
            self.predicate = self.predicate & e
        else:
            self.predicate = e

    def eval_predicate(self) -> bool:
        if not self.has_predicate():
            return True
        return self.predicate.evaluate_bool({})

    # -- expr rewriting (used by tile / unroll / fuse) -----------------
    def recursive_replace_expr(self, e0: Expr, e1: Expr) -> None:
        """Replace every occurrence of ``e0`` with ``e1`` in this instruction.

        For a plain instruction this rewrites every access-pattern Expr (and
        the predicate). For a LoopInst this recurses into the body - that
        override lives in loop.py.
        """
        for ap in self.get_all_aps():
            ap.access = [e.substitute(e0, e1).simplify() for e in ap.access]
        if self.has_predicate():
            self.predicate = self.predicate.substitute(e0, e1).simplify()

    # -- cloning -------------------------------------------------------
    def clone(self) -> "Instruction":
        """Deep-copy this instruction into the same parent block.

        Mirrors ``Instruction::clone``: it creates a sibling of the same
        opcode and copies the APs/mlocs/predicate. LoopInst overrides this.
        """
        new = self._create_empty_clone()
        new.pred = list(self.pred)
        new.succ = list(self.succ)
        new.attributes = dict(self.attributes)
        new.input_aps = [ap.clone() for ap in self.input_aps]
        new.output_aps = [ap.clone() for ap in self.output_aps]
        new.input_mlocs = list(self.input_mlocs)
        new.output_mlocs = list(self.output_mlocs)
        if self.has_predicate():
            new.predicate = self.predicate.clone()
        new.dfg_node = self.dfg_node
        new.unrolled_var_iter_map = dict(self.unrolled_var_iter_map)
        new.dep_resource = self.dep_resource
        return new

    def _create_empty_clone(self) -> "Instruction":
        """Construct a same-opcode sibling with no APs/mlocs yet.

        Subclasses override this to invoke their own constructor.
        """
        # We can't ``addInstruction`` into a BasicBlock here (that would
        # create a circular dependency on the module layer); instead we
        # construct in place and let the caller attach to a block if needed.
        return Instruction(self.name + "-clone", self.opcode, self.parent)

    # -- debugging -----------------------------------------------------
    def to_string(self) -> str:
        out_parts = []
        for ap in self.output_aps:
            out_parts.append(ap.to_string())
        out_str = ",".join(out_parts)
        in_parts = []
        for ap in self.input_aps:
            in_parts.append(ap.to_string())
        in_str = ",".join(in_parts)
        s = f"{out_str} <- {self.opcode.value}({in_str})"
        if self.has_predicate():
            s += f" if {self.predicate.to_string()}"
        if self.local_index != -1:
            pred_ids = [str(p.local_index) for p in self.pred]
            succ_ids = [str(s2.local_index) for s2 in self.succ]
            s += f"  //id={self.local_index},pred=[{' '.join(pred_ids)}]," \
                 f"succ=[{' '.join(succ_ids)}]"
        return s

    def __repr__(self):
        return f"{self.opcode.value}({self.name!r})"

    def __str__(self):
        return self.to_string()


# ---------------------------------------------------------------------------
# Concrete instruction subclasses
# ---------------------------------------------------------------------------

class NoOpInst(Instruction):
    """A no-op placeholder used as a move/insert pivot."""
    def __init__(self, name: str, parent: Optional["BasicBlock"] = None):
        super().__init__(name, InstructionType.NoOp, parent)

    def _create_empty_clone(self) -> "NoOpInst":
        return NoOpInst(self.name + "-clone", self.parent)

    def to_string(self) -> str:
        return f"noop  // {self.name}"


class LoadInst(Instruction):
    """DRAM -> on-chip load."""
    def __init__(self, name: str, parent: Optional["BasicBlock"] = None):
        super().__init__(name, InstructionType.Load, parent)

    def _create_empty_clone(self) -> "LoadInst":
        return LoadInst(self.name + "-clone", self.parent)


class StoreInst(Instruction):
    """on-chip -> DRAM store.

    Carries ``pattern`` (the post-store reduction pattern, default 3 in C++)
    and ``post_ops`` (DFG nodes queued to run after the store).
    """
    def __init__(self, name: str, parent: Optional["BasicBlock"] = None):
        super().__init__(name, InstructionType.Store, parent)
        self.pattern: int = 3
        self.post_ops: List[Any] = []

    def set_pattern(self, p: int) -> None:
        self.pattern = p

    def get_pattern(self) -> int:
        return self.pattern

    def _create_empty_clone(self) -> "StoreInst":
        st = StoreInst(self.name + "-clone", self.parent)
        st.pattern = self.pattern
        st.post_ops = list(self.post_ops)
        return st


class MatmultInst(Instruction):
    """Matrix multiply: a x b -> psum. ``stride`` is the output stride."""
    def __init__(self, name: str, parent: Optional["BasicBlock"] = None):
        super().__init__(name, InstructionType.Matmult, parent)
        self.stride: int = 1

    def set_stride(self, s: int) -> None:
        self.stride = s

    def get_stride(self) -> int:
        return self.stride

    def _create_empty_clone(self) -> "MatmultInst":
        mm = MatmultInst(self.name + "-clone", self.parent)
        mm.stride = self.stride
        return mm


class MoveInst(Instruction):
    """on-chip -> on-chip data movement."""
    def __init__(self, name: str, parent: Optional["BasicBlock"] = None):
        super().__init__(name, InstructionType.Move, parent)

    def _create_empty_clone(self) -> "MoveInst":
        return MoveInst(self.name + "-clone", self.parent)


class VectorInst(Instruction):
    """Elementwise / reduce / fused vector op.

    Attributes:
        op: the VectorType sub-opcode.
        imm: an immediate operand (used by some activations / scales).
        inout_left_shift: (input_shift, output_shift) fraclen adjustment.
    """
    def __init__(self, name: str, op: VectorType,
                 parent: Optional["BasicBlock"] = None):
        super().__init__(name, InstructionType.Vector, parent)
        self.op = op
        self.imm: float = 0
        self.inout_left_shift = (0, 0)

    def is_reduce(self) -> bool:
        return self.op in _REDUCE_OPS

    def _create_empty_clone(self) -> "VectorInst":
        v = VectorInst(self.name + "-clone", self.op, self.parent)
        v.imm = self.imm
        v.inout_left_shift = self.inout_left_shift
        return v

    def to_string(self) -> str:
        base = super().to_string()
        # Inject the vector sub-op after the opcode, e.g. "Vector-Softmax(...)"
        return base.replace(f"{self.opcode.value}(", f"{self.opcode.value}-{self.op.value}(", 1)


class TransposeInst(Instruction):
    """Data transpose. ``dims`` lists the swapped dimension indices."""
    def __init__(self, name: str, parent: Optional["BasicBlock"] = None):
        super().__init__(name, InstructionType.Transpose, parent)
        self.dims: List[int] = []

    def _create_empty_clone(self) -> "TransposeInst":
        t = TransposeInst(self.name + "-clone", self.parent)
        t.dims = list(self.dims)
        return t


class KernelInst(Instruction):
    """A fused block of instructions (used to lower a perfect loop nest).

    The kernel owns a ``LoopBody`` (a BasicBlock) and exposes the contained
    instructions as its body. ``has_reduce_inst`` reports whether the kernel
    contains a reduce-family vector op.
    """
    def __init__(self, name: str, parent: Optional["BasicBlock"] = None):
        super().__init__(name, InstructionType.Kernel, parent)
        self.body: Any = None  # a LoopBody (set by initialize_from)

    def initialize_from(self, loop) -> None:
        """Adopt ``loop``'s body into this kernel.

        Mirrors ``KernelInst::initialize_from``: creates a fresh LoopBody,
        seeds it with a NoOp pivot, moves the loop after the pivot, then
        removes the pivot - leaving the loop as the kernel's sole child.
        """
        from .loop import LoopBody
        self.body = LoopBody(None)
        bb = self.body.body
        from .module import BasicBlock
        noop = NoOpInst("noop", bb)
        bb.add_instruction(noop)
        # move the loop after the noop within this kernel's body
        if loop.parent is not None:
            loop.parent.remove_instruction(loop)
        loop.parent = bb
        bb.instructions.append(loop)
        bb.remove_instruction(noop)

    def has_reduce_inst(self) -> bool:
        """True if this kernel's body contains a reduce vector op."""
        from .loop import LoopInst
        if self.body is None:
            return False
        loop = None
        for inst in self.body.body:
            if isinstance(inst, LoopInst):
                loop = inst
                break
        if loop is None:
            return False
        for inst, _depth in loop.visit():
            if isinstance(inst, VectorInst) and inst.is_reduce():
                return True
        return False

    def _create_empty_clone(self) -> "KernelInst":
        return KernelInst(self.name + "-clone", self.parent)
