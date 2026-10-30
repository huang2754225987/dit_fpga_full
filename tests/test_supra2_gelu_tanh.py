import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_gelu_tanh import generate_lut


class Supra2GeluTanhTests(unittest.TestCase):
    def test_rom_shape_symmetry_and_limits(self):
        build_dir = Path(__file__).resolve().parents[1] / "build"
        with tempfile.TemporaryDirectory(prefix="supra2_gelu_", dir=build_dir) as temp:
            path = Path(temp) / "lut.hex"
            generate_lut(path)
            values = np.fromiter((int(v, 16) for v in path.read_text().splitlines()),
                                 dtype=np.uint16).view(np.float16)
            self.assertEqual(values.size, 4096)
            self.assertAlmostEqual(float(values[2048]), 0.0, places=5)
            self.assertLess(abs(float(values[0])), 1e-5)
            self.assertAlmostEqual(float(values[2304]), 0.8413, places=3)
            self.assertGreater(float(values[-1]), 7.99)


if __name__ == "__main__":
    unittest.main()
