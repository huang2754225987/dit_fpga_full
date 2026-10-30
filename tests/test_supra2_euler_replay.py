import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_euler_replay import DEFAULT_EXPORT, prepare


class Supra2EulerReplayFixtureTests(unittest.TestCase):
    def test_archive_50_step_replay_fixture(self):
        if not (DEFAULT_EXPORT / "golden_steps.npz").is_file():
            self.skipTest("Supra2 step export has not been extracted")
        with tempfile.TemporaryDirectory(
                prefix="supra2_euler_replay_",
                dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
            output = Path(temp)
            manifest = prepare(DEFAULT_EXPORT, output)
            self.assertEqual((manifest["steps"], manifest["elements"],
                              manifest["lanes"]), (50, 4096, 32))
            self.assertLess(manifest["software_reference_rel_l2_vs_z_final"], 0.01)
            self.assertEqual(len((output / "velocity_steps_fp16.hex")
                                 .read_text().splitlines()), 50 * 4096)
            self.assertEqual(len((output / "reference_final_fp16.hex")
                                 .read_text().splitlines()), 4096)
            with np.load(DEFAULT_EXPORT / "golden_steps.npz",
                         allow_pickle=False) as steps:
                z0_bits = np.fromiter(
                    (int(v, 16) for v in
                     (output / "z0_fp16.hex").read_text().splitlines()),
                    dtype=np.uint16)
                np.testing.assert_array_equal(z0_bits, steps["z0"].view(np.uint16).flat)


if __name__ == "__main__":
    unittest.main()
