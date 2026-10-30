#!/usr/bin/env python3
"""Prepare one exact INT8 systolic-array tile from the supplied golden pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
TILE = 32


def _write_hex(path: Path, values: np.ndarray, width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text(
        "".join(f"{int(value) & mask:0{width}x}\n" for value in values.flat),
        encoding="ascii",
    )


def prepare(export_dir: Path, output_dir: Path, layer: str,
            row_start: int = 0, column_start: int = 0) -> dict:
    unit_path = export_dir / "golden_unit_layers.npz"
    weights_path = export_dir / "weights_quant_int8.npz"
    if not unit_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("Supra2 golden INT8 export files are missing")

    with np.load(unit_path, allow_pickle=False) as unit, \
            np.load(weights_path, allow_pickle=False) as weights:
        x = unit[f"{layer}.x_int8"]
        expected_full = unit[f"{layer}.acc_int32"]
        weight = weights[f"{layer}.qw"]
        if x.ndim != 2 or weight.ndim != 2 or expected_full.ndim != 2:
            raise ValueError("golden QLinear tensors must be matrices")
        rows, k = x.shape
        output_features, weight_k = weight.shape
        if weight_k != k or expected_full.shape != (rows, output_features):
            raise ValueError("golden activation, weight, and accumulator shapes disagree")
        if not 0 <= row_start <= rows - TILE:
            raise ValueError("row_start must select a complete 32-row tile")
        if not 0 <= column_start <= output_features - TILE:
            raise ValueError("column_start must select a complete 32-column tile")

        activation_tile = np.ascontiguousarray(x[row_start:row_start + TILE])
        weight_tile = np.ascontiguousarray(
            weight[column_start:column_start + TILE])
        token_scale = np.ascontiguousarray(
            unit[f"{layer}.s_t"][row_start:row_start + TILE], dtype=np.float32)
        weight_scale = np.ascontiguousarray(
            weights[f"{layer}.ws"][column_start:column_start + TILE],
            dtype=np.float32)
        bias = np.ascontiguousarray(
            weights[f"{layer}.bias"][column_start:column_start + TILE],
            dtype=np.float32)
        expected = np.ascontiguousarray(
            expected_full[row_start:row_start + TILE,
                          column_start:column_start + TILE])
        expected_y = np.ascontiguousarray(
            unit[f"{layer}.y_out"][row_start:row_start + TILE,
                                   column_start:column_start + TILE],
            dtype=np.float16)
        full_output_absmax = np.float32(
            np.max(np.abs(unit[f"{layer}.y_out"].astype(np.float32))))
        exact = activation_tile.astype(np.int64) @ weight_tile.astype(np.int64).T
        if not np.array_equal(exact, expected.astype(np.int64)):
            raise ValueError("golden INT32 accumulator is not exact X_int8 @ QW.T")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "activation_int8.hex", activation_tile, 2)
    _write_hex(output_dir / "weight_int8.hex", weight_tile, 2)
    _write_hex(output_dir / "accumulator_int32.hex", expected, 8)
    _write_hex(output_dir / "token_scale_fp32.hex", token_scale.view(np.uint32), 8)
    _write_hex(output_dir / "weight_scale_fp32.hex", weight_scale.view(np.uint32), 8)
    _write_hex(output_dir / "bias_fp32.hex", bias.view(np.uint32), 8)
    _write_hex(output_dir / "output_fp16.hex", expected_y.view(np.uint16), 4)
    _write_hex(output_dir / "output_absmax_fp32.hex",
               np.asarray([full_output_absmax]).view(np.uint32), 8)
    manifest = {
        "layer": layer,
        "row_start": row_start,
        "column_start": column_start,
        "m": TILE,
        "n": TILE,
        "k": k,
        "dtype": "signed_int8",
        "weight_layout": "[N,K] row-major",
        "activation_layout": "[M,K] row-major",
        "accumulator_layout": "[M,N] row-major int32",
        "output_layout": "[M,N] row-major fp16",
        "epilogue": "fp16(fp32(acc_int32) * s_t * weight_scale + bias)",
        "full_output_absmax": float(full_output_absmax),
        "golden_pack": str(export_dir.resolve()),
        "status": "exact golden integer tile prepared",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_int8_golden_tile")
    parser.add_argument("--layer", default="blocks.0.self_attn.qkv")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--column-start", type=int, default=0)
    args = parser.parse_args()
    manifest = prepare(args.export_dir, args.output_dir, args.layer,
                       args.row_start, args.column_start)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
