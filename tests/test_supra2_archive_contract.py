import json
import unittest
from pathlib import Path

import numpy as np

from compiler.supra2.export_quantized_weights import verify as verify_weight_image


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "build" / "supra2_hw_export" / "export"


class Supra2ArchiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = (
            EXPORT / "manifest.json",
            EXPORT / "weights_quant_int8.npz",
            EXPORT / "weights_passthrough_bf16.npz",
            EXPORT / "golden_unit_layers.npz",
            EXPORT / "golden_block.npz",
            EXPORT / "golden_steps.npz",
            EXPORT / "skip_trace.json",
        )
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Extract supra2-img-hw-export.tar.gz first; missing " +
                ", ".join(missing))
        cls.manifest = json.loads((EXPORT / "manifest.json").read_text(
            encoding="utf-8"))
        cls.quant = np.load(EXPORT / "weights_quant_int8.npz",
                            allow_pickle=False)
        cls.passthrough = np.load(EXPORT / "weights_passthrough_bf16.npz",
                                  allow_pickle=False)
        cls.unit = np.load(EXPORT / "golden_unit_layers.npz",
                           allow_pickle=False)

    def test_weight_inventory_matches_quantization_contract(self):
        quant_layers = self.manifest["layer_table"]
        self.assertEqual(len(quant_layers), 100)
        self.assertEqual(len({item["name"] for item in quant_layers}), 100)
        self.assertEqual(len(self.quant.files), 300)
        self.assertEqual(len(self.passthrough.files), 37)
        for layer in quant_layers:
            name = layer["name"]
            out_features, in_features = layer["out"], layer["d_in"]
            self.assertEqual(self.quant[name + ".qw"].shape,
                             (out_features, in_features))
            self.assertEqual(self.quant[name + ".qw"].dtype, np.int8)
            self.assertEqual(self.quant[name + ".ws"].shape, (out_features,))
            self.assertEqual(self.quant[name + ".bias"].shape, (out_features,))

    def test_golden_integer_accumulators_are_exact(self):
        for name in ("blocks.0.self_attn.qkv", "blocks.0.mlp.2"):
            with self.subTest(layer=name):
                activation = self.unit[name + ".x_int8"].astype(np.int64)
                weight = self.quant[name + ".qw"].astype(np.int64)
                expected = self.unit[name + ".acc_int32"]
                np.testing.assert_array_equal(activation @ weight.T, expected)

    def test_fpga_weight_image_round_trips_all_archive_tensors(self):
        result = verify_weight_image(
            EXPORT, ROOT / "build" / "supra2_quant_export")
        self.assertEqual(result["quantized_layers_verified"], 100)
        self.assertEqual(result["passthrough_tensors_verified"], 37)
        self.assertEqual(result["status"], "pass")

    def test_dequantized_golden_output_is_within_fp16_storage_error(self):
        for name in ("blocks.0.self_attn.qkv", "blocks.0.mlp.2"):
            with self.subTest(layer=name):
                acc = self.unit[name + ".acc_int32"].astype(np.float64)
                scale = self.unit[name + ".s_t"].astype(np.float64)
                weight_scale = self.quant[name + ".ws"].astype(np.float64)
                bias = self.quant[name + ".bias"].astype(np.float64)
                output = acc * scale[:, None] * weight_scale[None, :] + bias[None, :]
                expected = self.unit[name + ".y_out"].astype(np.float32)
                relative_max = (np.max(np.abs(output - expected)) /
                                np.max(np.abs(expected)))
                self.assertLess(relative_max, 0.0045)

    def test_end_to_end_golden_contract_shapes_and_skip_rate(self):
        with np.load(EXPORT / "golden_block.npz", allow_pickle=False) as blocks:
            self.assertEqual(blocks["b0.x"].shape, (256, 576))
            self.assertEqual(blocks["b0.ctx"].shape, (128, 576))
            self.assertEqual(blocks["b0.y"].shape, (256, 576))
        with np.load(EXPORT / "golden_steps.npz", allow_pickle=False) as steps:
            self.assertEqual(steps["z0"].shape, (4, 32, 32))
            self.assertEqual(steps["ctx_all"].shape, (2, 128, 768))
            self.assertEqual(steps["v_per_step"].shape, (50, 4, 32, 32))
            self.assertEqual(steps["z_final"].shape, (4, 32, 32))
        trace = json.loads((EXPORT / "skip_trace.json").read_text(
            encoding="utf-8"))
        self.assertEqual(len(trace), 50)
        self.assertTrue(all(len(row) == 14 for row in trace))
        actual_ratio = sum(sum(row) for row in trace) / (50 * 14)
        self.assertAlmostEqual(actual_ratio, 0.4085714285714286)


if __name__ == "__main__":
    unittest.main()
