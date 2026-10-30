from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_layernorm_modulate import DEFAULT_EXPORT, prepare

TEST_TMP_ROOT = DEFAULT_EXPORT.parents[2] / "build"


class Supra2LayerNormFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (DEFAULT_EXPORT / "golden_block.npz").is_file():
            raise unittest.SkipTest("Supra2 hardware export is not available")

    def test_missing_block_activation_requires_explicit_input(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as temp:
            output = Path(temp) / "fixture"
            with self.assertRaises(KeyError):
                prepare(DEFAULT_EXPORT, output, block=1)

    def test_runtime_input_and_shared_timestep_fallback(self) -> None:
        with np.load(DEFAULT_EXPORT / "golden_block.npz", allow_pickle=False) as blocks:
            x = np.asarray(blocks["b0.x"], dtype=np.float16)
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as temp:
            root = Path(temp)
            input_file = root / "input.hex"
            input_file.write_text(
                "".join(f"{int(v):04x}\n" for v in x.view(np.uint16).flat),
                encoding="ascii",
            )
            manifest = prepare(DEFAULT_EXPORT, root / "fixture", block=1,
                               input_file=input_file)
            self.assertEqual(manifest["block"], 1)
            self.assertIn("RTL-produced activation", manifest["activation_source"])
            self.assertEqual(manifest["timestep_source"],
                             "archive b0.c (shared timestep vector; b0.c equals b7.c)")
            output = np.fromiter(
                (int(v, 16) for v in (root / "fixture" / "x_fp16.hex").read_text().splitlines()),
                dtype=np.uint16,
            ).view(np.float16).reshape(256, 576)
            np.testing.assert_array_equal(output, x)

    def test_explicit_timestep_file_is_used(self) -> None:
        with np.load(DEFAULT_EXPORT / "golden_block.npz", allow_pickle=False) as blocks:
            x = np.asarray(blocks["b0.x"], dtype=np.float16)
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as temp:
            root = Path(temp)
            input_file = root / "input.hex"
            c_file = root / "timestep.hex"
            input_file.write_text("".join(f"{int(v):04x}\n" for v in x.view(np.uint16).flat),
                                  encoding="ascii")
            c = np.zeros(576, dtype=np.float16)
            c_file.write_text("".join(f"{int(v):04x}\n" for v in c.view(np.uint16)),
                              encoding="ascii")
            manifest = prepare(DEFAULT_EXPORT, root / "fixture", block=1,
                               input_file=input_file, c_file=c_file)
            self.assertIn("explicit FP16 timestep vector", manifest["timestep_source"])

    def test_mlp_branch_selects_mlp_adaln_vectors(self) -> None:
        with np.load(DEFAULT_EXPORT / "golden_block.npz", allow_pickle=False) as blocks:
            x = np.asarray(blocks["b0.x"], dtype=np.float16)
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as temp:
            root = Path(temp)
            input_file = root / "input.hex"
            input_file.write_text(
                "".join(f"{int(v):04x}\n" for v in x.view(np.uint16).flat),
                encoding="ascii",
            )
            output = root / "fixture"
            manifest = prepare(DEFAULT_EXPORT, output, block=1,
                               input_file=input_file, branch="mlp")
            self.assertEqual(manifest["branch"], "mlp")
            for selected, branch_specific in (("scale_fp16.hex", "scale_mlp_fp16.hex"),
                                               ("shift_fp16.hex", "shift_mlp_fp16.hex")):
                selected_values = np.fromiter(
                    (int(v, 16) for v in (output / selected).read_text().splitlines()),
                    dtype=np.uint16,
                )
                branch_values = np.fromiter(
                    (int(v, 16) for v in (output / branch_specific).read_text().splitlines()),
                    dtype=np.uint16,
                )
                np.testing.assert_array_equal(selected_values, branch_values)


if __name__ == "__main__":
    unittest.main()
