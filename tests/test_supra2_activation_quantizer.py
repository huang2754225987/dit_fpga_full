import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_activation_quantizer import (
    DEFAULT_EXPORT,
    prepare,
)


class Supra2ActivationQuantizerFixtureTests(unittest.TestCase):
    def test_fp16_token_fixture_matches_reference_rounding_and_scale(self):
        if not (DEFAULT_EXPORT / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 golden export has not been extracted")
        for layer, row in (("blocks.0.self_attn.qkv", 0),
                           ("blocks.0.mlp.2", 24)):
            with self.subTest(layer=layer):
                with tempfile.TemporaryDirectory(
                        prefix="supra2_quantizer_",
                        dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
                    output = Path(temp)
                    manifest = prepare(DEFAULT_EXPORT, output, layer, row)
                    with np.load(DEFAULT_EXPORT / "golden_unit_layers.npz",
                                 allow_pickle=False) as unit:
                        source = unit[layer + ".x_rot"][row]
                    input_bits = np.fromiter(
                        (int(line, 16) for line in
                         (output / "input_fp16.hex").read_text().splitlines()),
                        dtype=np.uint16)
                    output_q = np.fromiter(
                        (int(line, 16) for line in
                         (output / "expected_int8.hex").read_text().splitlines()),
                        dtype=np.uint8).view(np.int8)
                    output_scale = int(
                        (output / "expected_scale_fp32.hex").read_text().strip(), 16)
                    np.testing.assert_array_equal(input_bits, source.view(np.uint16))
                    values = source.astype(np.float32)
                    scale = np.float32(np.max(np.abs(values)) / np.float32(127))
                    expected_q = np.clip(
                        np.rint(values / scale), -128, 127).astype(np.int8)
                    np.testing.assert_array_equal(output_q, expected_q)
                    self.assertEqual(output_scale,
                                     np.asarray([scale]).view(np.uint32)[0])
                    self.assertEqual(manifest["k"], source.size)


if __name__ == "__main__":
    unittest.main()
