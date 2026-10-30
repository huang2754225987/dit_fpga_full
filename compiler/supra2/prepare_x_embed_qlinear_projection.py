#!/usr/bin/env python3
"""Prepare the archive-z0 256-row quantized x_embed RTL fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .prepare_hadamard import _fp16_hadamard
from .prepare_qlinear_projection import _write_hex


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"


def _patchify_latent(z: np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=np.float16)
    if values.shape != (4, 32, 32):
        raise ValueError("archive z0 must have shape [4,32,32]")
    return np.ascontiguousarray(
        values.reshape(4, 16, 2, 16, 2).transpose(1, 3, 0, 2, 4)
        .reshape(256, 16))


def prepare(export_dir: Path, output_dir: Path) -> dict:
    with np.load(export_dir / "golden_steps.npz", allow_pickle=False) as steps, \
            np.load(export_dir / "weights_quant_int8.npz", allow_pickle=False) as weights:
        raw = _patchify_latent(steps["z0"])
        weight = np.ascontiguousarray(weights["x_embed.qw"], dtype=np.int8)
        weight_scale = np.ascontiguousarray(weights["x_embed.ws"], dtype=np.float32)
        bias = np.ascontiguousarray(weights["x_embed.bias"], dtype=np.float32)
    if weight.shape != (576, 16) or weight_scale.shape != (576,) or bias.shape != (576,):
        raise ValueError("unexpected quantized x_embed tensor shapes")

    rotated = np.stack([_fp16_hadamard(row, 16) for row in raw])
    rotated_fp32 = rotated.astype(np.float32)
    absmax = np.max(np.abs(rotated_fp32), axis=1)
    safe_absmax = np.where(absmax == 0, np.float32(1.0), absmax)
    quantized = np.clip(np.rint(
        rotated_fp32.astype(np.float64) * 127.0 /
        safe_absmax.astype(np.float64)[:, None]), -128, 127).astype(np.int8)
    token_scale = np.asarray([
        np.float32(np.float16(row_max) * np.float16(1.0 / 127.0))
        for row_max in absmax
    ], dtype=np.float32)
    accum64 = quantized.astype(np.int64) @ weight.astype(np.int64).T
    if np.any(accum64 < np.iinfo(np.int32).min) or np.any(accum64 > np.iinfo(np.int32).max):
        raise OverflowError("x_embed accumulator exceeds INT32")
    accum = accum64.astype(np.int32)
    output = ((accum.astype(np.float32) * token_scale[:, None]) *
              weight_scale[None, :] + bias[None, :]).astype(np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "raw_input_fp16.hex", raw.view(np.uint16), 4)
    _write_hex(output_dir / "rotated_fp16.hex", rotated.view(np.uint16), 4)
    _write_hex(output_dir / "expected_quantized_int8.hex", quantized, 2)
    _write_hex(output_dir / "weight_int8.hex", weight, 2)
    _write_hex(output_dir / "accumulator_int32.hex", accum, 8)
    _write_hex(output_dir / "token_scale_fp32.hex", token_scale.view(np.uint32), 8)
    _write_hex(output_dir / "weight_scale_fp32.hex", weight_scale.view(np.uint32), 8)
    _write_hex(output_dir / "bias_fp32.hex", bias.view(np.uint32), 8)
    _write_hex(output_dir / "output_fp16.hex", output.view(np.uint16), 4)
    _write_hex(output_dir / "reference_output_fp16.hex", output.view(np.uint16), 4)
    manifest = {
        "layer": "x_embed",
        "m": 256,
        "n": 576,
        "k": 16,
        "h_block": 16,
        "input_source": "archive golden_steps.z0 patchified without a model forward",
        "patch_order": "row-major patches, then channel, patch-row, patch-column",
        "rtl_lane_padding": "K=16 padded with zero lanes to the 32-lane array beat",
        "activation_path": "FP16 -> FP32 Hadamard -> FP16 -> dynamic INT8",
        "reference": "INT8 GEMM and FP32 scale/bias computed from archive weights",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_x_embed_qlinear")
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
