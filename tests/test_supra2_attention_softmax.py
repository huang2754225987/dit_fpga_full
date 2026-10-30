import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_attention_softmax import DEFAULT_EXPORT, prepare


class Supra2AttentionSoftmaxFixtureTests(unittest.TestCase):
    def test_full_block_score_and_softmax_fixture(self):
        if not (DEFAULT_EXPORT / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 unit golden export has not been extracted")
        build = DEFAULT_EXPORT.parents[2] / "build"
        with tempfile.TemporaryDirectory(prefix="supra2_attn_softmax_", dir=build) as temp:
            output = Path(temp)
            manifest = prepare(DEFAULT_EXPORT, output)
            self.assertEqual((manifest["queries"], manifest["heads"],
                              manifest["keys"]), (256, 9, 256))
            expected_path = output / "expected_probs_fp16.hex"
            self.assertEqual(len(expected_path.read_text().splitlines()), 256*9*256)
            expected_bits = np.fromiter(
                (int(v, 16) for v in expected_path.read_text().splitlines()),
                dtype=np.uint16,
            )
            probs = expected_bits.view(np.float16).reshape(256, 9, 256)
            np.testing.assert_allclose(probs.astype(np.float32).sum(axis=2),
                                       np.ones((256, 9)), rtol=0, atol=0.001)


if __name__ == "__main__":
    unittest.main()
