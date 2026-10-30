from __future__ import annotations

import unittest

import numpy as np

from compiler.supra2.prepare_position_add import add_position_embedding


class Supra2PositionAddTests(unittest.TestCase):
    def test_fp16_add_broadcasts_single_position_table(self):
        tokens = np.zeros((256, 576), dtype=np.float16)
        tokens[:, 0] = np.arange(256, dtype=np.float16)
        position = np.full((1, 256, 576), np.float16(0.5), dtype=np.float16)
        result = add_position_embedding(tokens, position)
        self.assertEqual(result.shape, (256, 576))
        np.testing.assert_array_equal(result[:, 0], tokens[:, 0] + np.float16(0.5))
        np.testing.assert_array_equal(result[:, 1], np.full(256, 0.5, dtype=np.float16))

    def test_rejects_unexpected_geometry(self):
        with self.assertRaises(ValueError):
            add_position_embedding(np.zeros((256, 575)), np.zeros((256, 576)))


if __name__ == "__main__":
    unittest.main()
