from __future__ import annotations

import unittest
import json

from compiler.supra2.prepare_dit_ddr_image import (
    DEFAULT_EXPORT,
    DEFAULT_OUTPUT,
    DEFAULT_WEIGHT_IMAGE,
    build,
    verify,
)


class Supra2DitDdrImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output_dir = DEFAULT_OUTPUT
        cls.manifest = build(DEFAULT_EXPORT, DEFAULT_WEIGHT_IMAGE, cls.output_dir)

    def test_full_payload_and_descriptor_counts(self) -> None:
        result = verify(self.output_dir)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["quantized_layers"], 100)
        self.assertEqual(result["passthrough_tensors"], 37)
        self.assertEqual(len(self.manifest["block_stage_layer_ids"]), 14)
        self.assertTrue(all(len(row) == 7 for row in self.manifest["block_stage_layer_ids"]))
        self.assertEqual(self.manifest["skip_trace_shape"], [50, 14])
        self.assertEqual(self.manifest["axi_512bit_beats"] * 64,
                         self.manifest["payload_bytes"])

    def test_dma_offsets_obey_beat_and_tile_alignment(self) -> None:
        for record in self.manifest["components"]:
            self.assertEqual(record["offset"] % record["alignment"], 0, record["name"])
        for layer in self.manifest["quantized_layers"]:
            self.assertEqual(layer["weight_offset"] % 1024, 0, layer["name"])
            self.assertEqual(layer["scale_offset"] % 64, 0, layer["name"])
            self.assertEqual(layer["bias_offset"] % 64, 0, layer["name"])

    def test_block_stage_ids_cover_manifest_layers(self) -> None:
        names = [record["name"] for record in self.manifest["quantized_layers"]]
        for block, row in enumerate(self.manifest["block_stage_layer_ids"]):
            self.assertEqual(
                [names[index] for index in row],
                [
                    f"blocks.{block}.self_attn.qkv",
                    f"blocks.{block}.self_attn.proj",
                    f"blocks.{block}.cross_attn.q",
                    f"blocks.{block}.cross_attn.kv",
                    f"blocks.{block}.cross_attn.proj",
                    f"blocks.{block}.mlp.0",
                    f"blocks.{block}.mlp.2",
                ],
            )

    def test_block_stage_dma_descriptors_resolve_layer_metadata(self) -> None:
        layers = self.manifest["quantized_layers"]
        descriptors = self.manifest["block_stage_descriptors"]
        self.assertEqual(len(descriptors), 14)
        self.assertTrue(all(len(row) == 7 for row in descriptors))
        for block, row in enumerate(descriptors):
            for stage, descriptor in enumerate(row):
                layer_id = self.manifest["block_stage_layer_ids"][block][stage]
                self.assertEqual(descriptor["layer_id"], layer_id)
                for field in ("weight_offset", "scale_offset", "bias_offset",
                              "weight_bytes", "scale_bytes", "bias_bytes",
                              "weight_beats", "scale_beats", "bias_beats",
                              "out", "d_in", "n_tiles", "k_tiles"):
                    self.assertEqual(descriptor[field], layers[layer_id][field])
                self.assertEqual(descriptor["weight_beats"],
                                 descriptor["weight_bytes"] // 64)
                self.assertEqual(descriptor["scale_beats"],
                                 descriptor["scale_bytes"] // 64)
                self.assertEqual(descriptor["bias_beats"],
                                 descriptor["bias_bytes"] // 64)
                self.assertEqual(descriptor["k_tile_stride_bytes"], 1024)
                self.assertEqual(descriptor["n_tile_stride_bytes"],
                                 descriptor["k_tiles"] * 1024)
        package = (self.output_dir / "supra2_dit_ddr_layout.sv").read_text(
            encoding="ascii")
        for table in ("BLOCK_STAGE_WEIGHT_OFFSET", "BLOCK_STAGE_SCALE_OFFSET",
                      "BLOCK_STAGE_BIAS_OFFSET", "BLOCK_STAGE_OUT", "BLOCK_STAGE_IN",
                      "BLOCK_STAGE_N_TILES", "BLOCK_STAGE_K_TILES",
                      "BLOCK_STAGE_N_TILE_STRIDE_BYTES",
                      "BLOCK_STAGE_K_TILE_STRIDE_BYTES", "BLOCK_STAGE_SCALE_BYTES",
                      "BLOCK_STAGE_BIAS_BYTES", "BLOCK_STAGE_WEIGHT_BEATS",
                      "BLOCK_STAGE_SCALE_BEATS", "BLOCK_STAGE_BIAS_BEATS"):
            self.assertIn(f"localparam integer {table} [0:13][0:6]", package)

    def test_every_weight_tile_maps_to_sixteen_page_safe_axi_beats(self) -> None:
        for row in self.manifest["block_stage_descriptors"]:
            for descriptor in row:
                for n_tile in range(descriptor["n_tiles"]):
                    for k_tile in range(descriptor["k_tiles"]):
                        offset = (descriptor["weight_offset"] +
                                  n_tile * descriptor["n_tile_stride_bytes"] +
                                  k_tile * descriptor["k_tile_stride_bytes"])
                        self.assertEqual(offset % 1024, 0)
                        self.assertLessEqual((offset & 0xFFF) + 1024, 4096)
                        self.assertLessEqual(offset + 1024,
                                             descriptor["weight_offset"] +
                                             descriptor["weight_bytes"])

    def test_block_adaln_and_runtime_support_descriptors(self) -> None:
        tensors = self.manifest["passthrough_tensors"]
        tensor_ids = self.manifest["passthrough_ids"]
        adaln = self.manifest["block_adaln_descriptors"]
        self.assertEqual(len(adaln), 14)
        for block, descriptor in enumerate(adaln):
            for kind in ("weight", "bias"):
                name = f"blocks.{block}.adaln.1.{kind}"
                tensor_id = tensor_ids[name]
                self.assertEqual(descriptor[f"{kind}_id"], tensor_id)
                self.assertEqual(descriptor[f"{kind}_offset"], tensors[tensor_id]["offset"])
                self.assertEqual(descriptor[f"{kind}_bytes"], tensors[tensor_id]["bytes"])
        self.assertEqual(tensors[tensor_ids["pos_embed"]]["name"], "pos_embed")
        self.assertEqual(tensors[tensor_ids["final.linear.weight"]]["name"],
                         "final.linear.weight")
        package = (self.output_dir / "supra2_dit_ddr_layout.sv").read_text(
            encoding="ascii")
        self.assertIn("localparam integer CFG_ROWS = 512", package)
        self.assertIn("localparam integer FLOW_STEPS = 50", package)
        self.assertIn("localparam integer BLOCK_ADALN_WEIGHT_OFFSET [0:13]", package)

    def test_compiled_timestep_adaln_tables_are_in_ddr_image(self) -> None:
        runtime = {item["name"]: item for item in self.manifest["runtime_inputs"]}
        self.assertEqual(runtime["block_adaln_params_fp16"]["shape"], [50, 14, 8, 576])
        self.assertEqual(runtime["final_adaln_params_fp16"]["shape"], [50, 2, 576])
        self.assertEqual(runtime["timestep_embedding_fp16"]["shape"], [50, 576])
        package = (self.output_dir / "supra2_dit_ddr_layout.sv").read_text(
            encoding="ascii")
        self.assertIn("BLOCK_ADALN_RUNTIME_OFFSET", package)
        self.assertIn("FINAL_ADALN_RUNTIME_OFFSET", package)

    def test_skip_masks_preserve_the_package_schedule(self) -> None:
        source_path = DEFAULT_EXPORT / "skip_trace.json"
        skip_trace = json.loads(source_path.read_text(encoding="utf-8"))
        masks = self.manifest["block_skip_masks"]
        self.assertEqual(len(masks), 50)
        self.assertEqual(self.manifest["block_skip_encoding"],
                         "bit 1 means skip; bit index is block id")
        for step, row in enumerate(skip_trace):
            expected = sum(int(skip) << block for block, skip in enumerate(row))
            self.assertEqual(masks[step], expected, f"step {step}")
        self.assertEqual(self.manifest["block_skip_count"], 286)
        package = (self.output_dir / "supra2_dit_ddr_layout.sv").read_text(
            encoding="ascii")
        self.assertIn("localparam logic [13:0] BLOCK_SKIP_MASK [0:49]", package)


if __name__ == "__main__":
    unittest.main()
