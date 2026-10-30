"""FPGA compiler contracts for the 256-token Supra2 projection workload.

These tests emit instruction objects only. They do not instantiate the PyTorch
model, run Transformer inference, or claim RTL/FPGA execution.
"""

import sys
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.compiler_backend_py import isa, static_linear
from compiler.supra2.compile_projection_plan import choose_token_batch, projection_specs
from compiler.supra2.export_fp16_weights import (
    pack_fp16_bias,
    pack_fp16_weight_tiles,
)
from compiler.supra2.prepare_projection_smoke import build_projection_smoke_program
from compiler.supra2.verify_projection_smoke import read_dirty_range


class Supra2ProjectionContractTests(unittest.TestCase):
    def test_fake_ddr_dump_reconstructs_little_endian_bytes(self):
        expected = bytes(range(16))
        word = int.from_bytes(expected, byteorder="little")
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            dump = Path(directory) / "dirty.hex"
            dump.write_text(f"00080000 {word:032x} ffff\n", encoding="ascii")
            self.assertEqual(read_dirty_range(dump, 0x80000, 16), expected)

    def test_fake_ddr_dump_rejects_missing_output_bytes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
            dump = Path(directory) / "dirty.hex"
            dump.write_text("00080000 00000000000000000000000000000000 0001\n",
                            encoding="ascii")
            with self.assertRaises(ValueError):
                read_dirty_range(dump, 0x80000, 16)

    def test_fp16_weight_tile_is_k_major_and_zero_padded(self):
        import torch

        weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        packed = pack_fp16_weight_tiles(weight)
        self.assertEqual(len(packed), 32 * 32 * 2)
        # Physical WRF gather order is K row, then the 32 output lanes.
        self.assertEqual(struct.unpack("<4e", packed[:8]), (1.0, 4.0, 0.0, 0.0))
        self.assertEqual(struct.unpack("<4e", packed[32 * 2:32 * 2 + 8]),
                         (2.0, 5.0, 0.0, 0.0))
        self.assertEqual(struct.unpack("<2e", packed[2 * 32 * 2:2 * 32 * 2 + 4]),
                         (3.0, 6.0))
        self.assertEqual(packed[-2:], b"\0\0")

    def test_fp16_bias_pads_to_aligned_32_lane_vectors(self):
        import torch

        packed = pack_fp16_bias(torch.tensor([1.0, -2.0, 0.5]), 32)
        self.assertEqual(len(packed), 64)
        self.assertEqual(struct.unpack("<3e", packed[:6]), (1.0, -2.0, 0.5))
        self.assertEqual(packed[-2:], b"\0\0")

    def test_model_projection_inventory_is_complete(self):
        specs = projection_specs(depth=14)
        self.assertEqual(len(specs), 118)
        self.assertEqual(len({name for name, _, _ in specs}), 118)
        self.assertEqual(sum(input_kind == "latent_tokens"
                             for _, _, input_kind in specs), 6 * 14 + 2)
        self.assertEqual(sum(input_kind == "text_context"
                             for _, _, input_kind in specs), 15)

    def test_256_token_batches_fit_output_page(self):
        cases = (
            (576, 32, 1152, 8),
            (1728, 16, 3456, 16),
            (1536, 16, 3072, 16),
            (2304, 8, 4608, 32),
        )
        for output_width, expected_rows, expected_stride, expected_batches in cases:
            with self.subTest(output_width=output_width):
                rows, stride = choose_token_batch(256, output_width)
                self.assertEqual((rows, stride), (expected_rows, expected_stride))
                self.assertEqual((256 + rows - 1) // rows, expected_batches)
                self.assertLessEqual(rows * stride, 64 * 1024)

    def test_real_fp16_batch_emits_valid_gemm_program(self):
        # Representative 576 -> 2304 MLP expansion tile at M=8.
        region = static_linear.LinearRegion(
            weight=0x100000,
            scale_zero=None,
            rows=2304,
            cols=576,
            nbits=16,
            tile_bytes=2048,
            groups=0,
        )
        program = static_linear.emit_fp16_batch(
            region,
            activation_base=0,
            output_base=0x4000,
            tokens=8,
            activation_page=1,
            output_page=0,
            activation_row_stride_bytes=576 * 2,
            output_row_stride_bytes=4608,
        )

        gemms = [instruction for instruction in program
                 if instruction.opcode == isa.UOpcode.GEMM]
        self.assertGreaterEqual(len(gemms), region.n_tiles)
        self.assertLess(len(gemms), region.n_tiles * region.k_tiles)
        self.assertTrue(all((instruction.vd & 0x3) == int(isa.Precision.FP16)
                            for instruction in gemms))
        self.assertTrue(all(0 <= instruction.encode() <= 0xFFFFFFFF
                            for instruction in program))
        self.assertEqual(program[0].opcode, isa.UOpcode.NOP)  # CFG_M setup

    def test_fp16_static_dma_supports_exact_2k_tile_transfer(self):
        page, dma = isa.static_dma(0x12000, l1_bank=2, fp16=True,
                                   transfer_bytes=2048)
        self.assertEqual(page.opcode, isa.UOpcode.NOP)
        self.assertEqual(dma.opcode, isa.UOpcode.DMA_RD)
        self.assertEqual(dma.vs1, 0x5)
        self.assertEqual(dma.vs2, 5)  # RTL static-length code for 2048 bytes
        with self.assertRaises(ValueError):
            isa.static_dma(0x12000, l1_bank=2, fp16=True,
                           transfer_bytes=1024)

    def test_linear_dma_write_encodes_aligned_burst_lengths(self):
        instr = isa.dma_wr_linear_burst(0x2000, 4096, output_window=3, page=1)
        self.assertEqual(instr.opcode, isa.UOpcode.DMA_WR)
        self.assertEqual(instr.vd, 3)
        self.assertEqual(instr.vs1, 0xD)
        self.assertEqual(instr.vs2, 15)
        with self.assertRaises(ValueError):
            isa.dma_wr_linear_burst(0x2001, 4096)

    def test_x_embed_smoke_program_covers_dma_gemm_and_writeback(self):
        program = build_projection_smoke_program()
        gemms = [instruction for instruction in program
                 if instruction.opcode == isa.UOpcode.GEMM]
        dma_reads = [instruction for instruction in program
                     if instruction.opcode == isa.UOpcode.DMA_RD]
        dma_writes = [instruction for instruction in program
                      if instruction.opcode == isa.UOpcode.DMA_WR]
        self.assertEqual(len(gemms), 8 * 18)
        self.assertEqual(len(dma_reads), 1 + 8 * 19)  # bias + per batch: activation and weights
        self.assertEqual(sum(instruction.vs1 == 0xA for instruction in dma_reads), 8)
        fp16_weight_reads = [instruction for instruction in dma_reads
                             if instruction.vs1 == 0x5]
        self.assertEqual(len(fp16_weight_reads), 1 + 8 * 18)
        self.assertTrue(all(instruction.vs2 == 5
                            for instruction in fp16_weight_reads[1:]))
        self.assertEqual(len(dma_writes), 8 * 9)
        self.assertEqual({instruction.vd for instruction in dma_writes}, set(range(8)))
        write_positions = [index for index, instruction in enumerate(program)
                           if instruction.opcode == isa.UOpcode.DMA_WR]
        for batch in range(8):
            positions = write_positions[batch * 9:(batch + 1) * 9]
            self.assertEqual(positions, list(range(positions[0], positions[0] + 9)))
            self.assertEqual(program[positions[-1] + 1].opcode, isa.UOpcode.BARRIER)
        self.assertEqual(program[-1].opcode, isa.UOpcode.END)
        self.assertTrue(all((instruction.vd & 0x3) == int(isa.Precision.FP16)
                            for instruction in gemms))


if __name__ == "__main__":
    unittest.main()
