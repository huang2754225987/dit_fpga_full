import unittest

from compiler.supra2.flash_attention import (
    AttentionHead,
    compile_head,
    compile_self_attention,
)
from compiler.compiler_backend_py.isa import UOpcode


def make_head(index=0):
    return AttentionHead(
        q_base=0,
        k_base=4096,
        v_base=8192,
        score_base=12288,
        alpha_base=0,
        output_base=0,
        # Heads execute sequentially and therefore reuse the 32 query slots.
        state_slot_base=0,
        key_tile_loaders=tuple(() for _ in range(8)),
    )


class Supra2FlashAttentionCompilerTests(unittest.TestCase):
    def test_emits_tiled_qk_online_softmax_pv_program(self):
        program = compile_head(make_head())
        opcodes = [instruction.opcode for instruction in program]
        self.assertEqual(opcodes.count(UOpcode.GEMM), 32)
        softmax = [instruction for instruction in program
                   if instruction.opcode == UOpcode.SOFTMAX]
        self.assertEqual(len(softmax), 256)
        self.assertTrue(all(instruction.imm & 0x40 for instruction in softmax))
        self.assertEqual(sum(bool(instruction.imm & 0x80)
                             for instruction in softmax), 32)
        self.assertEqual(opcodes[-1], UOpcode.END)

    def test_emits_all_nine_heads_and_rejects_missing_load_tiles(self):
        program = compile_self_attention([make_head(i) for i in range(9)])
        self.assertEqual(sum(instruction.opcode == UOpcode.END
                             for instruction in program), 72)
        self.assertEqual(sum(instruction.opcode == UOpcode.GEMM
                             for instruction in program), 2304)
        self.assertEqual(sum(instruction.opcode == UOpcode.SOFTMAX
                             for instruction in program), 18432)
        with self.assertRaisesRegex(ValueError, "8 K/V tile loaders"):
            compile_head(AttentionHead(
                0, 4096, 8192, 12288, 0, 0, 0, tuple(() for _ in range(7))))


if __name__ == "__main__":
    unittest.main()
