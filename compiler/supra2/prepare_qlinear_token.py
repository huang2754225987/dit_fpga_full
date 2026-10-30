#!/usr/bin/env python3
"""Build one FP16-x_rot -> INT8 GEMM -> FP16 QLinear token fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .prepare_hadamard import _fp16_hadamard


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
LANES = 32


def _write_hex(path: Path, values: np.ndarray, width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text("".join(f"{int(v) & mask:0{width}x}\n"
                             for v in values.flat), encoding="ascii")


def prepare(export_dir: Path, output_dir: Path,
            layer: str = "blocks.0.self_attn.qkv", row: int = 0,
            column_start: int = 0) -> dict:
    with np.load(export_dir / "golden_unit_layers.npz", allow_pickle=False) as unit, \
            np.load(export_dir / "weights_quant_int8.npz", allow_pickle=False) as weights:
        raw_all = unit[f"{layer}.x_raw"]
        if not 0 <= row < raw_all.shape[0]:
            raise ValueError("row is outside the supplied unit golden")
        raw = np.ascontiguousarray(raw_all[row], dtype=np.float16)
        package_rotated = np.ascontiguousarray(
            unit[f"{layer}.x_rot"][row], dtype=np.float16)
        layer_manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8"))
        layer_entry = next(item for item in layer_manifest["layer_table"]
                           if item["name"] == layer)
        h_block = int(layer_entry["h_block"])
        rotated = _fp16_hadamard(raw, h_block)
        weight = weights[f"{layer}.qw"]
        output_features, k = weight.shape
        if not 0 <= column_start <= output_features - LANES:
            raise ValueError("column_start must select 32 output channels")
        cols = slice(column_start, column_start + LANES)
        weight_tile = np.ascontiguousarray(weight[cols])
        weight_scale = np.ascontiguousarray(weights[f"{layer}.ws"][cols],
                                            dtype=np.float32)
        bias = np.ascontiguousarray(weights[f"{layer}.bias"][cols],
                                    dtype=np.float32)
        package_output = np.ascontiguousarray(
            unit[f"{layer}.y_out"][row, cols], dtype=np.float16)
        package_absmax = np.float32(
            np.max(np.abs(unit[f"{layer}.y_out"].astype(np.float32))))

    x = rotated.astype(np.float32)
    absmax = np.max(np.abs(x))
    reference_scale = np.float32(absmax / np.float32(127.0))
    quantized = np.clip(np.rint(
        (x.astype(np.float64) * 127.0) / float(absmax)),
        -128, 127).astype(np.int8)
    # Match the RTL scale path: FP16 multiply by the rounded FP16 value of 1/127.
    hardware_scale = np.float32(
        np.float16(absmax).astype(np.float16) * np.float16(1.0 / 127.0))
    accumulator = quantized.astype(np.int64) @ weight_tile.astype(np.int64).T
    output_fp32 = (accumulator.astype(np.float32) * hardware_scale) * weight_scale + bias
    expected_output = output_fp32.astype(np.float16)
    compute_absmax = np.float32(np.max(np.abs(output_fp32)))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "input_fp16.hex", raw.view(np.uint16), 4)
    _write_hex(output_dir / "expected_quantized_int8.hex",
               quantized.view(np.uint8), 2)
    _write_hex(output_dir / "weight_int8.hex", weight_tile, 2)
    _write_hex(output_dir / "expected_accumulator_int32.hex", accumulator, 8)
    _write_hex(output_dir / "weight_scale_fp32.hex", weight_scale.view(np.uint32), 8)
    _write_hex(output_dir / "bias_fp32.hex", bias.view(np.uint32), 8)
    _write_hex(output_dir / "expected_compute_fp16.hex",
               expected_output.view(np.uint16), 4)
    _write_hex(output_dir / "package_output_fp16.hex",
               package_output.view(np.uint16), 4)
    _write_hex(output_dir / "compute_absmax_fp32.hex",
               np.asarray([compute_absmax]).view(np.uint32), 8)
    _write_hex(output_dir / "package_absmax_fp32.hex",
               np.asarray([package_absmax]).view(np.uint32), 8)
    manifest = {
        "layer": layer,
        "row": row,
        "column_start": column_start,
        "k": int(k),
        "n": LANES,
        "source": "package x_raw stored as FP16; FP32 butterfly transform with final FP16 output",
        "h_block": h_block,
        "hadamard_backend_rounding": "FP32 after each butterfly stage; FP16 after normalization",
        "quantization": "per-token absmax, ties-to-even, signed INT8",
        "scale_fp32": float(hardware_scale),
        "package_output_global_absmax": float(package_absmax),
        "status": "single-token QLinear RTL fixture",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_qlinear_token")
    parser.add_argument("--layer", default="blocks.0.self_attn.qkv")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--column-start", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.output_dir, args.layer,
                             args.row, args.column_start), indent=2))


if __name__ == "__main__":
    main()
