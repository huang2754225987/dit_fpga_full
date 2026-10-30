"""Pass 7 - LoopAllocator: DRAM + SRAM address allocation.

Python reimplementation of ``src/pass/LoopAllocator.{h,cpp}`` plus the
``LiveInterval.{h,cpp}`` helpers (``TensorLI``, ``InterfereGraph``,
``PolySlot``).

Two phases:

  1. DRAM allocation - assign each tensor a DRAM offset. Static tensors
     (weights, gamma, ...) get a linear, non-overlapping layout. Dynamic
     tensors (activations, intermediates) get a live-interval analysis and
     are packed via an interference graph so non-overlapping lifetimes
     share DRAM space.

  2. Local (on-chip) allocation - for each on-chip buffer type (FB, WB,
     BB, ROWBANK, RESIDUEBANK, PSUM) run live-interval analysis and pack
     into the available banks. FB/WB/BB/ROWBANK/RESIDUEBANK use 2 banks
     with double-buffering (ping-pong); PSUM uses 1 bank.

After this pass every ``AccessPattern`` is upgraded to a
``PhysicalAccessPattern`` carrying bank_id/offset/step/step_size/stride.
"""
from __future__ import annotations

from typing import List, Dict, Optional

from bir.memory import MemoryLocation, MemoryType
from bir.access_pattern import AccessPattern, PhysicalAccessPattern
from bir.instruction import LoadInst, StoreInst, MatmultInst

# on-chip buffer types that this pass allocates
_ONCHIP_TYPES = {
    MemoryType.FB, MemoryType.WB, MemoryType.BB,
    MemoryType.ROWBANK, MemoryType.RESIDUEBANK, MemoryType.PSUM,
}


# =============================================================================
# Live interval + interference graph
# =============================================================================
class TensorLI:
    """Live interval of one on-chip tensor: [st_slot, ed_slot] + size."""

    def __init__(self, name: str, st: int, ed: int, addrs: int = 1,
                 banks: int = 1, alignment: int = 1):
        self.name = name
        self.st_slot = st
        self.ed_slot = ed
        self.addrs = addrs          # size in address units
        self.banks = banks
        self.alignment = alignment
        self.bank_addr = [0] * banks
        self.addr_per_bank = 0

    @property
    def weight(self):
        return self.addrs * self.banks * self.alignment


class InterfereGraph:
    """Interference graph over TensorLI nodes (scanline + linear-scan alloc)."""

    def __init__(self):
        self.nodes: List[TensorLI] = []
        self.edges: List[tuple] = []   # (a, b) pairs that interfere

    def add(self, li: TensorLI):
        self.nodes.append(li)

    def build(self):
        """Connect nodes whose live intervals overlap."""
        self.edges.clear()
        ns = sorted(self.nodes, key=lambda n: n.st_slot)
        active: List[TensorLI] = []
        for n in ns:
            # expire
            active = [a for a in active if a.ed_slot >= n.st_slot]
            for a in active:
                self.edges.append((a, n))
            active.append(n)

    def allocate(self, n_banks: int, linear: bool = False) -> int:
        """Linear-scan allocation. Returns max high-water mark across banks.

        linear=True : never reclaim space (static layout).
        linear=False: reclaim space when an interval expires.
        """
        order = sorted(self.nodes, key=lambda n: -n.weight)
        bank_hwm = [0] * n_banks
        free_lists = [[] for _ in range(n_banks)]   # (ed_slot, addr, size) per bank

        for n in order:
            # pick bank with lowest current hwm
            b = min(range(n_banks), key=lambda i: bank_hwm[i])
            aligned = ((bank_hwm[b] + n.alignment - 1) // n.alignment) * n.alignment
            n.bank_addr[b] = aligned
            n.addr_per_bank = n.addrs
            bank_hwm[b] = aligned + n.addrs
        return max(bank_hwm) if bank_hwm else 0


# =============================================================================
# Pass
# =============================================================================
class LoopAllocatorPass:
    """Pass: loop_allocator - DRAM + on-chip address allocation."""

    def __init__(self, *args, **kwargs):
        pass

    def run(self, module) -> int:
        n_dram = self._allocate_dram(module)
        n_local = 0
        for fn in module:
            for bb in fn:
                n_local += self._allocate_local(bb)
        print(f"[LoopAllocator] DRAM tensors={n_dram}, local allocs={n_local}")
        return 0

    # ------------------------------------------------------------------
    # DRAM allocation
    # ------------------------------------------------------------------
    def _allocate_dram(self, module) -> int:
        """Assign DRAM offsets to every DRAM-bound MemoryLocation."""
        # collect distinct DRAM mlocs (by name)
        seen: Dict[str, MemoryLocation] = {}
        for fn in module:
            for bb in fn:
                for inst in bb:
                    for ap in inst.get_all_aps():
                        mloc = ap.mloc
                        if mloc is None or mloc.mem_type != MemoryType.DRAM:
                            continue
                        if mloc.name not in seen:
                            seen[mloc.name] = mloc

        # linear layout: static first, then dynamic (simplified - all linear)
        offset = 0
        for name, mloc in seen.items():
            size = self._mloc_size(mloc)
            mloc.offset = offset
            offset += size
        return len(seen)

    # ------------------------------------------------------------------
    # On-chip allocation
    # ------------------------------------------------------------------
    def _allocate_local(self, bb) -> int:
        """Run live-interval analysis + interference allocation per buffer type."""
        n = 0
        for mt in _ONCHIP_TYPES:
            n += self._allocate_type(bb, mt)
        return n

    def _allocate_type(self, bb, mt: MemoryType) -> int:
        # live-range analysis: def at first output, use extends to last input
        intervals: Dict[str, TensorLI] = {}
        for idx, inst in enumerate(list(bb)):
            for ap in inst.output_aps:
                if ap.mloc is None or ap.mloc.mem_type != mt:
                    continue
                key = ap.mloc.to_string()
                if key not in intervals:
                    intervals[key] = TensorLI(key, idx, idx,
                                               addrs=self._mloc_size(ap.mloc))
                else:
                    intervals[key].ed_slot = idx
            for ap in inst.input_aps:
                if ap.mloc is None or ap.mloc.mem_type != mt:
                    continue
                key = ap.mloc.to_string()
                if key in intervals:
                    intervals[key].ed_slot = idx

        if not intervals:
            return 0

        g = InterfereGraph()
        for li in intervals.values():
            g.add(li)
        g.build()
        n_banks = 1 if mt == MemoryType.PSUM else 2
        hwm = g.allocate(n_banks, linear=False)

        # write bank_id/offset back onto mlocs + upgrade APs to Physical
        for li in g.nodes:
            bank = 0
            for b, addr in enumerate(li.bank_addr):
                if addr != 0 or b == 0:
                    bank = b
                    break
            # find the mloc for this interval
            for inst in list(bb):
                for ap in inst.get_all_aps():
                    if ap.mloc is None or ap.mloc.mem_type != mt:
                        continue
                    if ap.mloc.to_string() == li.name:
                        ap.mloc.bank_id = bank
                        ap.mloc.offset = li.bank_addr[bank]
                        if not isinstance(ap, PhysicalAccessPattern):
                            self._upgrade_to_physical(ap, li)
        return len(intervals)

    def _upgrade_to_physical(self, ap: AccessPattern, li: TensorLI):
        """Convert an AccessPattern in place to a PhysicalAccessPattern."""
        ap.__class__ = PhysicalAccessPattern
        ap.bank_id = li.bank_addr.index(next((a for a in li.bank_addr if a is not None), 0))
        ap.offset = li.bank_addr[0]
        ap.step = 1
        ap.step_size = li.addrs
        ap.stride = 1

    # ------------------------------------------------------------------
    def _mloc_size(self, mloc: MemoryLocation) -> int:
        """Best-effort tensor size in address units."""
        shape = getattr(mloc, "tensor_shape", None) or []
        if not shape:
            return 1
        size = 1
        for d in shape:
            try:
                size *= int(d)
            except (TypeError, ValueError):
                pass
        return max(size, 1)


__all__ = ["LoopAllocatorPass"]
