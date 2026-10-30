"""Target hardware parameters (the ``Arch`` model).

Python reimplementation of ``src/Arch.h``.

The defaults match the C++ side (32x32 systolic array, FP16, 200 MHz). A few
model-config defaults (``heads``/``kv_heads``/``layers``/...) are exposed so
LLM-shaped workloads can be configured without recompiling.
"""

from __future__ import annotations

from typing import Optional


class Arch:
    """Static description of the NPU target.

    Everything is a plain int so passes can read/write knobs directly.
    """

    def __init__(self):
        # ---- core / compute ----
        self.core_num: int = 1
        self.compute_engine: int = 1
        self.psum_bank: int = 1
        self.input_buffer_bank: int = 2
        self.dram_bank: int = self.core_num

        # ---- precision ----
        self.precision: int = 16          # FP16
        self.data_bytes: int = 2          # bytes per element (FP16)

        # ---- systolic array ----
        self.dsp: int = 1024             # 32x32 array = 1024 MACs
        # multipliers = dsp * (2 / data_bytes); for FP16 that's 1024
        self.multipliers: int = self.dsp * (2 // self.data_bytes)
        self.sparse_weight_factor: int = 1
        self.dfce: bool = False

        # ---- on-chip buffers ----
        self.fmap_buffer_bw_bytes: int = 64
        self.fmap_buffer_depth: int = 1024          # SRAM0 rows -> 64KB
        self.psum_buffer_depth: int = self.fmap_buffer_depth
        self.w_buffer_bw_bytes: int = self.multipliers * self.data_bytes
        self.w_buffer_depth: int = 32               # 32x32 array row count
        self.b_buffer_bw_bytes: int = 64
        self.b_buffer_depth: int = 1

        # ---- DRAM ----
        self.DRAM_bw_bytes: int = 64
        self.dram_addr_cnt: int = (1 << 34)
        self.dram_fmap_addr_offset: int = 0
        self.dram_w_addr_offset: int = 1638400
        self.dram_b_addr_offset: int = 6144000
        self.dram_inst_addr_offset: int = 7372800

        # ---- latency ----
        self.dram_access_latency: int = 20
        self.pe_latency: int = 0
        self.sync_delay: int = 1
        self.axi_cross_ddr_latency: int = 26

        # ---- clock ----
        self.operation_frequency: int = 200_000_000   # 200 MHz

        # accumulation width
        self.data_width_during_accumulation: int = 42

        # ---- LLM model config (Qwen3-4B defaults in C++) ----
        self.heads: int = 32
        self.kv_heads: int = 8
        self.layers: int = 36
        self.hidden_size: int = 2560
        self.head_dim: int = 80
        self.intermediate_size: int = 9728
        self.context_length: int = 1
        self.max_total_onchip_length: int = 256

    # -- setters -------------------------------------------------------
    def set_dsp(self, dsp: int) -> None:
        self.dsp = dsp
        self.multipliers = dsp * (2 // self.data_bytes)
        self.w_buffer_bw_bytes = self.multipliers * self.data_bytes

    # -- derived getters (parity with Arch.h) -------------------------
    def get_elements_per_dram_addr(self) -> int:
        return self.DRAM_bw_bytes // self.data_bytes

    def get_simd_parallelism(self) -> int:
        return self.multipliers

    def get_fmap_buffer_bytes(self) -> int:
        """Total feature-buffer size in bytes (64B/row * depth)."""
        return self.fmap_buffer_bw_bytes * self.fmap_buffer_depth

    def get_w_buffer_bytes(self) -> int:
        return self.w_buffer_bw_bytes * self.w_buffer_depth

    def get_fb_depth(self) -> int:
        return self.fmap_buffer_depth

    def get_wb_depth(self) -> int:
        return self.w_buffer_depth

    def get_bb_depth(self) -> int:
        return self.b_buffer_depth

    def get_psum_depth(self) -> int:
        return self.psum_buffer_depth

    def get_fb_bank(self) -> int:
        return self.input_buffer_bank

    def get_wb_bank(self) -> int:
        return self.input_buffer_bank

    def get_bb_bank(self) -> int:
        return self.input_buffer_bank

    def get_psum_bank(self) -> int:
        return self.psum_bank

    def get_dram_bank(self) -> int:
        return self.dram_bank

    def get_compute_engine_cnt(self) -> int:
        return self.compute_engine

    def get_wb_width(self) -> int:
        return self.w_buffer_bw_bytes // self.data_bytes

    def get_dram_latency(self) -> int:
        return self.dram_access_latency

    def get_pe_latency(self) -> int:
        return self.pe_latency

    def get_freq(self) -> int:
        return self.operation_frequency

    # -- debugging -----------------------------------------------------
    def __repr__(self):
        return (f"Arch(core={self.core_num}, dsp={self.dsp}, "
                f"data_bytes={self.data_bytes}, "
                f"fmap_bytes={self.get_fmap_buffer_bytes()})")

    def __str__(self):
        return self.__repr__()
