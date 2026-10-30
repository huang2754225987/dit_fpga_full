import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_qlinear_epilogue import (
    DEFAULT_EXPORT,
    prepare,
)
from compiler.supra2.prepare_int8_golden_tile import prepare as prepare_int8_tile


class Supra2QLinearEpilogueFixtureTests(unittest.TestCase):
    def test_integrated_qlinear_tile_fixture_preserves_int8_and_fp16_bits(self):
        if not (DEFAULT_EXPORT / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 golden export has not been extracted")
        with tempfile.TemporaryDirectory(
                prefix="supra2_qlinear_", dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
            output = Path(temp)
            manifest = prepare_int8_tile(
                DEFAULT_EXPORT, output, "blocks.0.self_attn.qkv", 64, 512)
            self.assertEqual(manifest["k"], 576)
            for filename, count in (
                ("activation_int8.hex", 32 * 576),
                ("weight_int8.hex", 32 * 576),
                ("accumulator_int32.hex", 32 * 32),
                ("token_scale_fp32.hex", 32),
                ("weight_scale_fp32.hex", 32),
                ("bias_fp32.hex", 32),
                ("output_fp16.hex", 32 * 32),
            ):
                self.assertEqual(len((output / filename).read_text().splitlines()), count)
            actual_output = np.fromiter(
                (int(line, 16) for line in
                 (output / "output_fp16.hex").read_text().splitlines()),
                dtype=np.uint16)
            with np.load(DEFAULT_EXPORT / "golden_unit_layers.npz",
                         allow_pickle=False) as unit:
                expected = unit[
                    "blocks.0.self_attn.qkv.y_out"][64:96, 512:544]
                np.testing.assert_array_equal(
                    actual_output, expected.view(np.uint16).reshape(-1))

    def test_golden_fixture_contains_one_32_by_32_tile(self):
        if not (DEFAULT_EXPORT / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 golden export has not been extracted")
        with tempfile.TemporaryDirectory(
                prefix="supra2_epilogue_", dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
            output = Path(temp)
            manifest = prepare(DEFAULT_EXPORT, output)
            self.assertEqual(manifest["lanes"], 32)
            self.assertEqual(len((output / "acc_int32.hex").read_text().splitlines()),
                             32 * 32)
            self.assertEqual(len((output / "token_scale_fp32.hex").read_text().splitlines()),
                             32)
            self.assertEqual(len((output / "weight_scale_fp32.hex").read_text().splitlines()),
                             32)
            self.assertEqual(len((output / "bias_fp32.hex").read_text().splitlines()),
                             32)
            self.assertEqual(len((output / "expected_fp16.hex").read_text().splitlines()),
                             32 * 32)
            actual_bits = np.fromiter(
                (int(line, 16) for line in
                 (output / "expected_fp16.hex").read_text().splitlines()),
                dtype=np.uint16)
            with np.load(DEFAULT_EXPORT / "golden_unit_layers.npz",
                         allow_pickle=False) as unit:
                expected_bits = unit["blocks.0.self_attn.qkv.y_out"][:32, :32].view(
                    np.uint16).reshape(-1)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_fp32_chain_reference_matches_package_fp16_within_storage_error(self):
        if not (DEFAULT_EXPORT / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 golden export has not been extracted")
        with np.load(DEFAULT_EXPORT / "golden_unit_layers.npz",
                     allow_pickle=False) as unit, \
                np.load(DEFAULT_EXPORT / "weights_quant_int8.npz",
                        allow_pickle=False) as weights:
            for layer in ("blocks.0.self_attn.qkv", "blocks.0.mlp.2"):
                with self.subTest(layer=layer):
                    acc = unit[layer + ".acc_int32"].astype(np.float32)
                    scale_t = unit[layer + ".s_t"].astype(np.float32)
                    scale_w = weights[layer + ".ws"].astype(np.float32)
                    bias = weights[layer + ".bias"].astype(np.float32)
                    result = (acc * scale_t[:, None]) * scale_w[None, :] + bias[None, :]
                    result_fp16 = result.astype(np.float16)
                    expected = unit[layer + ".y_out"].astype(np.float32)
                    relative_max = (np.max(np.abs(result_fp16.astype(np.float32) - expected)) /
                                    np.max(np.abs(expected)))
                    self.assertLess(relative_max, 0.0045)


if __name__ == "__main__":
    unittest.main()
