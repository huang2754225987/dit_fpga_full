"""BIR containers: BasicBlock, Function, Module.

Python reimplementation of ``bir/BasicBlock``, ``bir/Function``, ``bir/Module``.

These are the hierarchical containers that own instructions. A Module owns
Functions; a Function owns BasicBlocks; a BasicBlock owns an ordered list of
Instructions. The C++ side uses a templated ``NamedObjectContainer`` mixin
(``insertElement``/``removeElement``/``getElementByName``); we use plain
Python lists with helper methods.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Iterator, TYPE_CHECKING

from .instruction import Instruction, NoOpInst

if TYPE_CHECKING:
    from .arch import Arch


class BasicBlock:
    """An ordered list of instructions.

    Attributes:
        name: block name (unique within a Function).
        instructions: ordered list of Instructions (insertion order == run order).
        complex_instructions: instructions marked as composite (kept for
            parity with C++, used by some passes).
    """

    def __init__(self, name: str, parent: Optional["Function"] = None):
        self.name = name
        self.parent = parent
        self.instructions: List[Instruction] = []
        self.complex_instructions: List[Instruction] = []

    # -- mutation ------------------------------------------------------
    def add_instruction(self, inst: Instruction) -> Instruction:
        """Append ``inst`` to this block and fix up the parent back-ref."""
        inst.parent = self
        self.instructions.append(inst)
        return inst

    def remove_instruction(self, inst: Instruction) -> None:
        if inst in self.instructions:
            self.instructions.remove(inst)
        if inst.parent is self:
            inst.parent = None

    def detach(self, inst: Instruction) -> None:
        """Remove ``inst`` from whatever block currently owns it (if any)."""
        if inst.parent is not None and inst.parent is not self:
            inst.parent.remove_instruction(inst)
        elif inst.parent is None and inst in self.instructions:
            self.instructions.remove(inst)

    def insert_before(self, inst: Instruction, pivot: Instruction) -> None:
        """Insert ``inst`` immediately before ``pivot`` (cross-block safe)."""
        self.detach(inst)
        idx = self.instructions.index(pivot)
        inst.parent = self
        self.instructions.insert(idx, inst)

    def insert_after(self, inst: Instruction, pivot: Instruction) -> None:
        """Insert ``inst`` immediately after ``pivot`` (cross-block safe)."""
        self.detach(inst)
        idx = self.instructions.index(pivot)
        inst.parent = self
        self.instructions.insert(idx + 1, inst)

    def move_before(self, inst: Instruction, pivot: Instruction) -> None:
        """Move ``inst`` so it sits immediately before ``pivot``.

        Handles cross-block moves: the instruction is first detached from its
        current parent (if any), then inserted into this block.
        """
        self.detach(inst)
        idx = self.instructions.index(pivot)
        inst.parent = self
        self.instructions.insert(idx, inst)

    def move_after(self, inst: Instruction, pivot: Instruction) -> None:
        """Move ``inst`` so it sits immediately after ``pivot`` (cross-block)."""
        self.detach(inst)
        idx = self.instructions.index(pivot)
        inst.parent = self
        self.instructions.insert(idx + 1, inst)

    def get_instruction_by_name(self, name: str) -> Optional[Instruction]:
        for inst in self.instructions:
            if inst.name == name:
                return inst
        return None

    def add_complex_instruction(self, inst: Instruction) -> None:
        self.complex_instructions.append(inst)

    # -- iteration / size ---------------------------------------------
    def __iter__(self) -> Iterator[Instruction]:
        return iter(self.instructions)

    def __len__(self) -> int:
        return len(self.instructions)

    def size(self) -> int:
        return len(self.instructions)

    # -- cloning -------------------------------------------------------
    def clone(self, clone_insts: bool = True) -> "BasicBlock":
        """Return a sibling block in the same function.

        When ``clone_insts`` is True, deep-copy every instruction (using a
        NoOp pivot so instructions land in the right order, matching C++).
        """
        f = self.parent
        nb = f.add_basic_block(self.name + "-clone") if f is not None else BasicBlock(self.name + "-clone")
        nb.complex_instructions = list(self.complex_instructions)
        if clone_insts:
            pivot = NoOpInst("noop", nb)
            nb.add_instruction(pivot)
            for inst in self.instructions:
                ci = inst.clone()
                nb.move_before(ci, pivot)
            nb.remove_instruction(pivot)
        return nb

    # -- debugging -----------------------------------------------------
    def to_string(self) -> str:
        return "\n".join(str(i) for i in self.instructions)

    def __repr__(self):
        return f"BasicBlock({self.name!r}, {len(self.instructions)} insts)"

    def __str__(self):
        return self.to_string()


class Function:
    """A named collection of BasicBlocks plus the function's mem locs.

    Attributes:
        name: function name.
        basic_blocks: ordered list of BasicBlocks.
        mem_locs: memory locations owned by this function.
        partition_id: index used when this function is mapped to a core.
    """

    def __init__(self, name: str, parent: Optional["Module"] = None):
        self.name = name
        self.parent = parent
        self.basic_blocks: List[BasicBlock] = []
        self.mem_locs: Dict[str, Any] = {}
        self.partition_id: int = 0

    # -- mutation ------------------------------------------------------
    def add_basic_block(self, name: str) -> BasicBlock:
        bb = BasicBlock(name, self)
        self.basic_blocks.append(bb)
        return bb

    def remove_basic_block(self, bb: BasicBlock) -> None:
        if bb in self.basic_blocks:
            self.basic_blocks.remove(bb)

    def get_basic_block_by_name(self, name: str) -> Optional[BasicBlock]:
        for bb in self.basic_blocks:
            if bb.name == name:
                return bb
        return None

    def add_memory_location(self, mloc) -> None:
        self.mem_locs[mloc.name] = mloc

    def memory_locations(self) -> List[Any]:
        return list(self.mem_locs.values())

    # -- accessors -----------------------------------------------------
    def get_partition_id(self) -> int:
        return self.partition_id

    def set_partition_id(self, pid: int) -> None:
        self.partition_id = pid

    # -- iteration -----------------------------------------------------
    def __iter__(self) -> Iterator[BasicBlock]:
        return iter(self.basic_blocks)

    def __len__(self) -> int:
        return len(self.basic_blocks)

    # -- debugging -----------------------------------------------------
    def __repr__(self):
        return f"Function({self.name!r}, {len(self.basic_blocks)} blocks)"

    def __str__(self):
        out = [f"function {self.name}:"]
        for bb in self.basic_blocks:
            out.append(f"  {bb.name}:")
            for inst in bb:
                out.append(f"    {inst}")
        return "\n".join(out)


class Module:
    """The top-level container: a list of Functions + the arch model.

    Attributes:
        functions: ordered list of Functions.
        arch: the target ``Arch`` (hardware parameters).
        mem_locs: module-global memory locations (shared across functions).
        dfg: the data-flow graph, once built (None until lowered).
    """

    def __init__(self, arch: Optional["Arch"] = None):
        self.functions: List[Function] = []
        self.arch = arch
        self.dfg: Any = None
        self.mem_locs: Dict[str, Any] = {}

        # tiling / parse knobs (parity with C++ Module)
        self.constrained_tiling: bool = False
        self.tiling_factor_base: Dict[str, int] = {}
        self.parse_layer_count: int = -1
        self.parse_start_layer: int = 1

    # -- mutation ------------------------------------------------------
    def add_function(self, name: str) -> Function:
        f = Function(name, self)
        self.functions.append(f)
        return f

    def remove_function(self, f: Function) -> None:
        if f in self.functions:
            self.functions.remove(f)

    def get_function_by_name(self, name: str) -> Optional[Function]:
        for f in self.functions:
            if f.name == name:
                return f
        return None

    def get_arch_model(self) -> Optional["Arch"]:
        if self.arch is None:
            # lazy-default so a bare Module() still has an arch model
            from .arch import Arch
            self.arch = Arch()
        return self.arch

    def set_tiling_size_base(self, name: str, tz: int) -> None:
        self.constrained_tiling = True
        if name not in self.tiling_factor_base:
            self.tiling_factor_base[name] = tz
        else:
            from math import gcd
            old = self.tiling_factor_base[name]
            g = gcd(old, tz)
            self.tiling_factor_base[name] = g

    def parse_predicate(self, layer_index: int) -> bool:
        if self.parse_layer_count == -1:
            return True
        return (layer_index <= self.parse_start_layer + self.parse_layer_count - 1
                and layer_index >= self.parse_start_layer)

    # -- iteration -----------------------------------------------------
    def __iter__(self) -> Iterator[Function]:
        return iter(self.functions)

    def __len__(self) -> int:
        return len(self.functions)

    # -- debugging -----------------------------------------------------
    def __repr__(self):
        return f"Module({len(self.functions)} functions)"

    def __str__(self):
        return "\n\n".join(str(f) for f in self.functions)
