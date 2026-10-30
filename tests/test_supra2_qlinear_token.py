import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_qlinear_token import DEFAULT_EXPORT, prepare


class Supra2QLinearTokenFixtureTests(unittest.TestCase):
    def test_fixture_connects_fp16_quantizer_int8_gemm_and_epilogue(self):
        if not (DEFAULT_EXPORT / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 golden export has not been extracted")
        cases = (
            ("blocks.0.self_attn.qkv", 0, 0),
            ("blocks.0.self_attn.qkv", 64, 512),
            ("blocks.0.mlp.2", 24, 0),
            ("blocks.0.mlp.2", 128, 320),
        )
        for layer, row, col in cases:
            with self.subTest(layer=layer, row=row, column_start=col):
                with tempfile.TemporaryDirectory(
                        prefix="supra2_qtoken_",
                        dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
                    output = Path(temp)
                    manifest = prepare(DEFAULT_EXPORT, output, layer, row, col)
                    k = manifest["k"]
                    self.assertEqual(manifest["n"], 32)
                    q = np.fromiter(
                        (int(v, 16) for v in
                         (output / "expected_quantized_int8.hex").read_text().splitlines()),
                        dtype=np.uint8).view(np.int8)
                    weights = np.fromiter(
                        (int(v, 16) for v in
                         (output / "weight_int8.hex").read_text().splitlines()),
                        dtype=np.uint8).view(np.int8).reshape(32, k)
                    expected_acc = np.fromiter(
                        (int(v, 16) for v in
                         (output / "expected_accumulator_int32.hex").read_text().splitlines()),
                        dtype=np.uint32).view(np.int32)
                    np.testing.assert_array_equal(
                        q.astype(np.int64) @ weights.astype(np.int64).T,
                        expected_acc.astype(np.int64))

                    with np.load(DEFAULT_EXPORT / "golden_unit_layers.npz",
                                 allow_pickle=False) as unit:
                        package_y = unit[layer + ".y_out"][row, col:col+32]
                        package_absmax = np.max(
                            np.abs(unit[layer + ".y_out"].astype(np.float32)))
                    compute_y = np.fromiter(
                        (int(v, 16) for v in
                         (output / "expected_compute_fp16.hex").read_text().splitlines()),
                        dtype=np.uint16).view(np.float16).astype(np.float32)
                    error = np.max(np.abs(compute_y - package_y.astype(np.float32)))
                    self.assertLess(error / package_absmax, 0.0045)


if __name__ == "__main__":
    unittest.main()
