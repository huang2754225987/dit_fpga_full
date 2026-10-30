import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_hadamard import (
    DEFAULT_EXPORT, prepare, prepare_synthetic_partial_k,
)


class Supra2HadamardFixtureTests(unittest.TestCase):
    def test_partial_k_fixture_pads_to_one_rtl_lane_beat(self):
        with tempfile.TemporaryDirectory(
                prefix="supra2_had_partial_",
                dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
            output = Path(temp)
            manifest = prepare_synthetic_partial_k(output)
            self.assertEqual((manifest["k"], manifest["h_block"]), (16, 16))
            self.assertEqual(len((output / "input_raw_fp16.hex").read_text().splitlines()), 16)
            self.assertEqual(len((output / "expected_rotated_fp16.hex").read_text().splitlines()), 16)

    def test_archive_rows_export_fp16_hadamard_fixtures(self):
        if not (DEFAULT_EXPORT / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 golden export has not been extracted")
        for layer, row, expected_h in (
                ("blocks.0.self_attn.qkv", 0, 64),
                ("blocks.0.mlp.2", 24, 256)):
            with self.subTest(layer=layer, row=row):
                with tempfile.TemporaryDirectory(
                        prefix="supra2_hadamard_",
                        dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
                    output = Path(temp)
                    manifest = prepare(DEFAULT_EXPORT, output, layer, row)
                    self.assertEqual(manifest["h_block"], expected_h)
                    raw_bits = np.fromiter(
                        (int(v, 16) for v in
                         (output / "input_raw_fp16.hex").read_text().splitlines()),
                        dtype=np.uint16)
                    expected_bits = np.fromiter(
                        (int(v, 16) for v in
                         (output / "expected_rotated_fp16.hex").read_text().splitlines()),
                        dtype=np.uint16)
                    package_bits = np.fromiter(
                        (int(v, 16) for v in
                         (output / "package_rotated_fp16.hex").read_text().splitlines()),
                        dtype=np.uint16)
                    with np.load(DEFAULT_EXPORT / "golden_unit_layers.npz",
                                 allow_pickle=False) as unit:
                        np.testing.assert_array_equal(
                            raw_bits, unit[layer + ".x_raw"][row].view(np.uint16))
                        np.testing.assert_array_equal(
                            package_bits, unit[layer + ".x_rot"][row].view(np.uint16))
                    self.assertEqual(raw_bits.size, expected_bits.size)
                    self.assertEqual(expected_bits.size % 32, 0)


if __name__ == "__main__":
    unittest.main()
