import tempfile
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.prepare_qlinear_projection import DEFAULT_EXPORT, prepare
from compiler.supra2.prepare_x_embed_qlinear_projection import (
    prepare as prepare_x_embed,
)
from compiler.supra2.merge_qlinear_batches import merge as merge_qlinear_batches


class Supra2QLinearProjectionFixtureTests(unittest.TestCase):
    def test_x_embed_fixture_uses_archive_latent_and_partial_k(self):
        if not (DEFAULT_EXPORT / "golden_steps.npz").is_file():
            self.skipTest("Supra2 step export has not been extracted")
        with tempfile.TemporaryDirectory(
                prefix="supra2_x_embed_",
                dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
            output = Path(temp)
            manifest = prepare_x_embed(DEFAULT_EXPORT, output)
            self.assertEqual((manifest["m"], manifest["n"], manifest["k"],
                              manifest["h_block"]), (256, 576, 16, 16))
            q = np.fromiter(
                (int(v, 16) for v in
                 (output / "expected_quantized_int8.hex").read_text().splitlines()),
                dtype=np.uint8).view(np.int8).reshape(256, 16)
            w = np.fromiter(
                (int(v, 16) for v in
                 (output / "weight_int8.hex").read_text().splitlines()),
                dtype=np.uint8).view(np.int8).reshape(576, 16)
            golden = np.fromiter(
                (int(v, 16) for v in
                 (output / "accumulator_int32.hex").read_text().splitlines()),
                dtype=np.uint32).view(np.int32).reshape(256, 576)
            np.testing.assert_array_equal(q.astype(np.int64) @ w.astype(np.int64).T,
                                          golden.astype(np.int64))

    def test_full_qkv_projection_archive_contract(self):
        if not (DEFAULT_EXPORT / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 golden export has not been extracted")
        with tempfile.TemporaryDirectory(
                prefix="supra2_qkv_projection_",
                dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
            output = Path(temp)
            manifest = prepare(DEFAULT_EXPORT, output)
            self.assertEqual((manifest["m"], manifest["n"], manifest["k"]),
                             (256, 1728, 576))
            with np.load(DEFAULT_EXPORT / "golden_unit_layers.npz",
                         allow_pickle=False) as unit, \
                    np.load(DEFAULT_EXPORT / "weights_quant_int8.npz",
                             allow_pickle=False) as weights:
                x = unit[manifest["layer"] + ".x_int8"].astype(np.int64)
                w = weights[manifest["layer"] + ".qw"].astype(np.int64)
                acc = unit[manifest["layer"] + ".acc_int32"].astype(np.int64)
                np.testing.assert_array_equal(x @ w.T, acc)
            self.assertEqual(len((output / "accumulator_int32.hex").read_text().splitlines()),
                             256 * 1728)
            self.assertEqual(len((output / "reference_output_fp16.hex").read_text().splitlines()),
                             256 * 1728)

    def test_qkv_row_slice_matches_corresponding_full_fixture_rows(self):
        if not (DEFAULT_EXPORT / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 golden export has not been extracted")
        with tempfile.TemporaryDirectory(
                prefix="supra2_qkv_row_slice_",
                dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
            root = Path(temp)
            full_dir, slice_dir = root / "full", root / "slice"
            full_manifest = prepare(
                DEFAULT_EXPORT, full_dir, fp16_hadamard_frontend=True)
            slice_manifest = prepare(
                DEFAULT_EXPORT, slice_dir, fp16_hadamard_frontend=True,
                row_start=64, row_count=32)
            self.assertEqual((slice_manifest["row_start"],
                              slice_manifest["row_count"], slice_manifest["m"]),
                             (64, 32, 32))
            for filename in (
                    "raw_input_fp16.hex", "rotated_fp16.hex",
                    "expected_quantized_int8.hex", "accumulator_int32.hex",
                    "token_scale_fp32.hex", "output_fp16.hex",
                    "reference_output_fp16.hex"):
                with self.subTest(filename=filename):
                    full = (full_dir / filename).read_text().splitlines()
                    sliced = (slice_dir / filename).read_text().splitlines()
                    if filename == "token_scale_fp32.hex":
                        width = 1
                    elif filename in ("accumulator_int32.hex", "output_fp16.hex",
                                      "reference_output_fp16.hex"):
                        width = 1728
                    else:
                        width = 576
                    first = 64 * width
                    self.assertEqual(sliced, full[first:first + 32 * width])
            self.assertEqual(full_manifest["m"], 256)

    def test_qlinear_batch_merge_checks_exact_full_tensor(self):
        with tempfile.TemporaryDirectory(
                prefix="supra2_merge_batches_",
                dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
            root = Path(temp)
            batch_root = root / "batches"
            expected = np.arange(16, dtype=np.float16).reshape(4, 4)
            for index in range(2):
                batch_dir = batch_root / f"batch_{index:02d}"
                batch_dir.mkdir(parents=True)
                batch = expected[index * 2:(index + 1) * 2]
                (batch_dir / "rtl_output_fp16.hex").write_text(
                    "".join(f"{int(value):04x}\n" for value in batch.view(np.uint16).flat),
                    encoding="ascii")
            expected_file = root / "expected.hex"
            expected_file.write_text(
                "".join(f"{int(value):04x}\n" for value in expected.view(np.uint16).flat),
                encoding="ascii")
            output_file = root / "merged.hex"
            metrics = merge_qlinear_batches(
                batch_root, 2, 2, 4, expected_file, output_file, cycles=123)
            self.assertEqual(metrics["verification"], "PASS")
            self.assertEqual(metrics["rtl_cycles_sum"], 123)
            actual = np.fromiter(
                (int(value, 16) for value in output_file.read_text().splitlines()),
                dtype=np.uint16)
            np.testing.assert_array_equal(actual, expected.view(np.uint16).flat)

    def test_full_qkv_fp16_frontend_stays_within_output_gate(self):
        if not (DEFAULT_EXPORT / "golden_unit_layers.npz").is_file():
            self.skipTest("Supra2 golden export has not been extracted")
        with tempfile.TemporaryDirectory(
                prefix="supra2_qkv_frontend_",
                dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
            output = Path(temp)
            manifest = prepare(DEFAULT_EXPORT, output,
                               fp16_hadamard_frontend=True)
            self.assertEqual(manifest["activation_source"],
                         "FP16 x_raw -> FP32-stage Hadamard -> FP16 -> dynamic INT8")
            with np.load(DEFAULT_EXPORT / "golden_unit_layers.npz",
                         allow_pickle=False) as unit:
                package_y = unit[manifest["layer"] + ".y_out"].astype(np.float32)
                package_absmax = np.max(np.abs(package_y))
            reference_y = np.fromiter(
                (int(v, 16) for v in
                 (output / "reference_output_fp16.hex").read_text().splitlines()),
                dtype=np.uint16).view(np.float16).astype(np.float32).reshape(256, 1728)
            self.assertLess(np.max(np.abs(reference_y - package_y)) / package_absmax,
                            0.0045)
            self.assertEqual(len((output / "raw_input_fp16.hex").read_text().splitlines()),
                             256 * 576)
            self.assertEqual(len((output / "expected_quantized_int8.hex").read_text().splitlines()),
                             256 * 576)

    def test_rtl_activation_uses_its_own_projection_golden(self):
        if not (DEFAULT_EXPORT / "weights_quant_int8.npz").is_file():
            self.skipTest("Supra2 quantized weight export has not been extracted")
        with tempfile.TemporaryDirectory(
                prefix="supra2_rtl_activation_",
                dir=DEFAULT_EXPORT.parents[2] / "build") as temp:
            root = Path(temp)
            raw_file, output = root / "rtl_input.hex", root / "projection"
            raw = np.random.default_rng(19).normal(0, 0.1, (256, 2304)).astype(np.float16)
            raw_file.write_text("".join(f"{int(v):04x}\n" for v in raw.view(np.uint16).flat),
                                encoding="ascii")
            manifest = prepare(DEFAULT_EXPORT, output, layer="blocks.0.mlp.2",
                               fp16_hadamard_frontend=True, raw_input_file=raw_file)
            self.assertIn("RTL-produced raw activation", manifest["status"])
            expected = np.fromiter(
                (int(v, 16) for v in (output / "output_fp16.hex").read_text().splitlines()),
                dtype=np.uint16)
            reference = np.fromiter(
                (int(v, 16) for v in (output / "reference_output_fp16.hex").read_text().splitlines()),
                dtype=np.uint16)
            np.testing.assert_array_equal(expected, reference)


if __name__ == "__main__":
    unittest.main()
