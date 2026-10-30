import unittest

import numpy as np

from compiler.supra2.activation_quant import (
    block_hadamard,
    quantize_activations,
    quantize_symmetric_int8,
)


class Supra2ActivationQuantTests(unittest.TestCase):
    def test_block_hadamard_matches_explicit_normalized_matrix(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(3, 16)).astype(np.float32)
        h4 = np.array(
            [[1, 1, 1, 1], [1, -1, 1, -1],
             [1, 1, -1, -1], [1, -1, -1, 1]], dtype=np.float32) / 2
        expected = x @ np.kron(np.eye(4, dtype=np.float32), h4)
        np.testing.assert_allclose(block_hadamard(x, 4), expected,
                                   rtol=0, atol=3e-7)

    def test_quantizer_uses_per_row_absmax_and_round_to_even(self):
        x = np.array([[0, 0, 0], [0.5, 1.0, -1.0]], dtype=np.float32)
        q, scale = quantize_symmetric_int8(x)
        np.testing.assert_array_equal(q, [[0, 0, 0], [64, 127, -127]])
        np.testing.assert_allclose(scale, [0, 1 / 127], rtol=0, atol=1e-10)

    def test_archive_rotated_tensor_matches_fp16_storage_tolerance(self):
        from pathlib import Path
        import json

        root = Path(__file__).resolve().parents[1]
        export = root / "build" / "supra2_hw_export" / "export"
        if not (export / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 golden export has not been extracted")
        manifest = json.loads((export / "manifest.json").read_text(
            encoding="utf-8"))
        h_blocks = {item["name"]: item["h_block"]
                    for item in manifest["layer_table"]}
        with np.load(export / "golden_unit_layers.npz", allow_pickle=False) as unit:
            for name in ("blocks.0.self_attn.qkv", "blocks.0.mlp.2"):
                with self.subTest(layer=name):
                    rotated, _, _ = quantize_activations(
                        unit[name + ".x_raw"], h_blocks[name])
                    error = np.max(np.abs(
                        rotated - unit[name + ".x_rot"].astype(np.float32)))
                    # x_raw/x_rot are stored as FP16, while the original
                    # quantization was performed before that storage rounding.
                    self.assertLess(error, 0.0095)

    def test_zero_rows_have_zero_quantized_values_and_finite_scale(self):
        q, scale = quantize_symmetric_int8(np.zeros((2, 8), np.float32))
        np.testing.assert_array_equal(q, np.zeros((2, 8), np.int8))
        np.testing.assert_array_equal(scale, np.zeros(2, np.float32))


if __name__ == "__main__":
    unittest.main()
