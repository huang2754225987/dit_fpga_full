"""BIR memory locations.

Python reimplementation of ``bir/MemoryLocation.h`` / ``.cpp``.

A ``MemoryLocation`` is a typed, addressed region of a particular on-chip or
off-chip memory (DRAM, FB, WB, PSUM, ...). It carries the symbolic tensor
shape, the data type / quantisation info, and the physical placement
(offset, step, stride, bank id). Memory locations are referenced by
``AccessPattern``s to describe which slice of which buffer an instruction
reads or writes.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any


class MemoryType(Enum):
    """Which physical memory a location lives in."""
    DRAM = "DRAM"
    FB = "FB"            # feature buffer (on-chip SRAM0)
    WB = "WB"            # weight buffer
    BB = "BB"            # bias buffer
    PSUM = "PSUM"        # partial-sum accumulator
    ROWBANK = "ROWBANK"
    CONST_VALUE = "CONST_VALUE"
    VREG = "VREG"        # virtual register (DFG-level)
    SPAD = "SPAD"
    RESIDUEBANK = "RESIDUEBANK"
    WEIGHTMASK = "WEIGHTMASK"

    def __str__(self) -> str:
        return self.value


class DataType(Enum):
    """Element data type / quantisation width."""
    FIX32 = "FIX32"
    FIX16 = "FIX16"
    FIX8 = "FIX8"
    FP16 = "FP16"
    FP8 = "FP8"

    def __str__(self) -> str:
        return self.value


# bits-per-element lookup (matches C++ DataTypeToBits)
_DTYPE_BITS = {
    DataType.FIX32: 32,
    DataType.FIX16: 16,
    DataType.FIX8: 8,
    DataType.FP16: 16,
    DataType.FP8: 8,
}


def data_type_to_bits(dt: DataType) -> int:
    return _DTYPE_BITS.get(dt, 8)


class MemoryLocation:
    """A symbolic + physical memory region.

    The symbolic side is the ``tensor_shape`` (list of ints, outer-to-inner)
    plus the ``data_type`` / ``fraclen`` / ``psum_fraclen`` quantisation info.
    The physical side is the ``(offset, step, step_size, stride, bank_id)``
    tuple describing where in the buffer the data is laid out.
    """

    def __init__(self, name: str, mem_type: MemoryType,
                 data_type: DataType = DataType.FP16, fraclen: int = 0,
                 tensor_shape: Optional[List[int]] = None):
        self.name = name
        self.mem_type = mem_type
        self.tensor_shape: List[int] = list(tensor_shape) if tensor_shape else []
        self.data_type = data_type
        self.fraclen = fraclen
        self.psum_fraclen = 0

        # physical placement
        self.offset = 0
        self.step = 0
        self.step_size = 0
        self.stride = 0
        self.bank_id = 0
        self.extra_offset = 0
        self.layout = ""
        self.expected_layout = ""

        # flags / roles
        self.is_static = False
        self.is_residue = False
        self.is_output = False
        self.is_input = False
        self.is_scalar = False
        self.is_kv_cache = False
        self.transposed = False
        self.sparse = False
        self.read_pattern_name = ""

        # kv cache metadata
        self.kv_cache_info: Dict[str, Any] = {}

        # hierarchy
        self.parent: Optional[MemoryLocation] = None
        self.parent_mloc: Optional[MemoryLocation] = None
        self.concat_locations: List[MemoryLocation] = []

        # readers / writers (instructions that touch this location)
        self.readers: List[Any] = []
        self.writers: List[Any] = []

        self.residue_at = 0

    # -- identity / equality -------------------------------------------
    def __eq__(self, other):
        if not isinstance(other, MemoryLocation):
            return NotImplemented
        # Two locations are "the same" when they point at the same underlying
        # buffer cell: same name + same type. This matches the C++ ``equal``
        # shortcut used by AccessPattern::overlap.
        return self.name == other.name and self.mem_type == other.mem_type

    def __hash__(self):
        return hash((self.name, self.mem_type))

    def __repr__(self):
        return f"MemoryLocation({self.name!r}, {self.mem_type})"

    # -- convenience predicates ----------------------------------------
    def is_fwb(self) -> bool:
        """Feature / weight / bias buffer (any on-chip state buffer)."""
        return self.mem_type in (MemoryType.FB, MemoryType.WB, MemoryType.BB)

    def is_allocated(self) -> bool:
        return self.step != 0

    def is_state_buffer(self) -> bool:
        return self.mem_type not in (MemoryType.DRAM, MemoryType.PSUM)

    # -- accessors -----------------------------------------------------
    def get_data_width(self) -> int:
        return data_type_to_bits(self.data_type)

    def get_bytes(self) -> int:
        return self.get_data_width() // 8

    # -- mutation ------------------------------------------------------
    def set_tensor_shape(self, shape: List[int]) -> None:
        self.tensor_shape = list(shape)

    def set_parent(self, parent: "MemoryLocation") -> None:
        self.parent = parent

    def add_reader(self, inst) -> None:
        if inst not in self.readers:
            self.readers.append(inst)

    def add_writer(self, inst) -> None:
        if inst not in self.writers:
            self.writers.append(inst)

    def remove_reader(self, inst) -> None:
        if inst in self.readers:
            self.readers.remove(inst)

    def remove_writer(self, inst) -> None:
        if inst in self.writers:
            self.writers.remove(inst)

    # -- debugging -----------------------------------------------------
    def to_string(self) -> str:
        """Compact identifier used by dumps and dependency checks.

        Matches the C++ ``toString`` convention ``<name>_<type>_<bank>``.
        """
        return f"{self.name}_{self.mem_type.value}_{self.bank_id}"

    def __str__(self) -> str:
        return self.to_string()
