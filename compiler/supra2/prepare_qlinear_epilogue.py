#!/usr/bin/env python3
"""Export one 32-channel QLinear epilogue tile from the supplied goldens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
LANES = 32


def _write_hex(path: Path, values: np.ndarray, width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text("".join(f"{int(v) & mask:0{width}x}\n"
                             for v in values.flat), encoding="ascii")


def prepare(export_dir: Path, output_dir: Path,
            layer: str = "blocks.0.self_attn.qkv",
            row_start: int = 0, column_start: int = 0) -> dict:
    with np.load(export_dir / "golden_unit_layers.npz", allow_pickle=False) as unit, \
            np.load(export_dir / "weights_quant_int8.npz", allow_pickle=False) as weights:
        acc_all = unit[f"{layer}.acc_int32"]
        token_scale_all = unit[f"{layer}.s_t"]
        weight_scale_all = weights[f"{layer}.ws"]
        bias_all = weights[f"{layer}.bias"]
        output_all = unit[f"{layer}.y_out"]
        if not 0 <= row_start <= acc_all.shape[0] - LANES:
            raise ValueError("row_start must select 32 complete token rows")
        if not 0 <= column_start <= acc_all.shape[1] - LANES:
            raise ValueError("column_start must select 32 complete output channels")

        rows = slice(row_start, row_start + LANES)
        cols = slice(column_start, column_start + LANES)
        acc = np.ascontiguousarray(acc_all[rows, cols])
        token_scale = np.ascontiguousarray(token_scale_all[rows], dtype=np.float32)
        weight_scale = np.ascontiguousarray(weight_scale_all[cols], dtype=np.float32)
        bias = np.ascontiguousarray(bias_all[cols], dtype=np.float32)
        expected = np.ascontiguousarray(output_all[rows, cols], dtype=np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "acc_int32.hex", acc, 8)
    _write_hex(output_dir / "token_scale_fp32.hex", token_scale.view(np.uint32), 8)
    _write_hex(output_dir / "weight_scale_fp32.hex", weight_scale.view(np.uint32), 8)
    _write_hex(output_dir / "bias_fp32.hex", bias.view(np.uint32), 8)
    _write_hex(output_dir / "expected_fp16.hex", expected.view(np.uint16), 4)
    reference_absmax = np.asarray(
        [np.max(np.abs(output_all.astype(np.float32)))], dtype=np.float32)
    _write_hex(output_dir / "reference_absmax_fp32.hex",
               reference_absmax.view(np.uint32), 8)
    manifest = {
        "layer": layer,
        "row_start": row_start,
        "column_start": column_start,
        "lanes": LANES,
        "formula": "fp16(fp32(acc_int32) * s_t * weight_scale + bias)",
        "full_tensor_reference_absmax": float(reference_absmax[0]),
        "golden_pack": str(export_dir.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_epilogue_golden")
    parser.add_argument("--layer", default="blocks.0.self_attn.qkv")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--column-start", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.output_dir, args.layer,
                             args.row_start, args.column_start), indent=2))


if __name__ == "__main__":
    main()
