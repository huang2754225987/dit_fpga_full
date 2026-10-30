#!/usr/bin/env python3
"""Prepare a complete archive-backed QLinear projection fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .prepare_hadamard import _fp16_hadamard


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
TILE = 32


def _write_hex(path: Path, values: np.ndarray, width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text(
        "".join(f"{int(value) & mask:0{width}x}\n" for value in values.flat),
        encoding="ascii",
    )


def prepare(export_dir: Path, output_dir: Path,
            layer: str = "blocks.0.self_attn.qkv",
            fp16_hadamard_frontend: bool = False,
            raw_input_file: Path | None = None,
            row_start: int = 0, row_count: int | None = None) -> dict:
    unit_path = export_dir / "golden_unit_layers.npz"
    weight_path = export_dir / "weights_quant_int8.npz"
    if not unit_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError("Supra2 golden QLinear package is missing")

    with np.load(unit_path, allow_pickle=False) as unit, \
            np.load(weight_path, allow_pickle=False) as weights:
        has_package_golden = all(f"{layer}.{suffix}" in unit.files for suffix in
                                 ("x_int8", "acc_int32", "y_out", "s_t", "x_raw"))
        if has_package_golden:
            package_activation = np.ascontiguousarray(unit[f"{layer}.x_int8"])
            package_acc = np.ascontiguousarray(unit[f"{layer}.acc_int32"])
            output = np.ascontiguousarray(unit[f"{layer}.y_out"], dtype=np.float16)
            package_scale = np.ascontiguousarray(unit[f"{layer}.s_t"], dtype=np.float32)
            raw = np.ascontiguousarray(unit[f"{layer}.x_raw"], dtype=np.float16)
        else:
            package_activation = package_acc = output = package_scale = raw = None
        weight = np.ascontiguousarray(weights[f"{layer}.qw"])
        weight_scale = np.ascontiguousarray(weights[f"{layer}.ws"], dtype=np.float32)
        bias = np.ascontiguousarray(weights[f"{layer}.bias"], dtype=np.float32)

        archive_manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8"))
        layer_entry = next(item for item in archive_manifest["layer_table"]
                           if item["name"] == layer)
        h_block = int(layer_entry["h_block"])

    k = int(weight.shape[1])
    if raw_input_file is not None:
        raw_values = np.fromiter(
            (int(line, 16) for line in raw_input_file.read_text(encoding="ascii").splitlines()),
            dtype=np.uint16,
        )
        if raw_values.size % k:
            raise ValueError("raw RTL activation count is not divisible by projection K")
        raw = raw_values.reshape(-1, k).view(np.float16).copy()
    if raw is not None:
        source_rows = int(raw.shape[0])
    elif package_activation is not None:
        source_rows = int(package_activation.shape[0])
    else:
        raise ValueError("a package golden or --raw-input-file is required")
    if row_start < 0 or row_start >= source_rows:
        raise ValueError(f"row_start must be in [0, {source_rows})")
    row_stop = source_rows if row_count is None else row_start + row_count
    if row_stop <= row_start or row_stop > source_rows:
        raise ValueError("row_count must select a non-empty range within the source")
    row_slice = slice(row_start, row_stop)
    if raw is not None:
        raw = np.ascontiguousarray(raw[row_slice])
    if package_activation is not None:
        package_activation = np.ascontiguousarray(package_activation[row_slice])
    if package_acc is not None:
        package_acc = np.ascontiguousarray(package_acc[row_slice])
    if output is not None:
        output = np.ascontiguousarray(output[row_slice])
    if package_scale is not None:
        package_scale = np.ascontiguousarray(package_scale[row_slice])
    rows = row_stop - row_start
    channels, weight_k = weight.shape
    if rows % TILE or channels % TILE or weight_k != k:
        raise ValueError("projection dimensions must align to 32x32 tiles")
    if package_activation is not None and package_activation.shape != (rows, k):
        raise ValueError("package activation shape disagrees with raw input")
    if package_acc is not None and package_acc.shape != (rows, channels):
        raise ValueError("package accumulator shape disagrees with projection")
    if output is not None and output.shape != (rows, channels):
        raise ValueError("package output shape disagrees with projection")
    if package_scale is not None and package_scale.shape != (rows,):
        raise ValueError("package token scale shape disagrees with projection")
    if weight_scale.shape != (channels,) or bias.shape != (channels,):
        raise ValueError("QLinear scale/bias dimensions disagree")

    if fp16_hadamard_frontend:
        if raw.shape != (rows, k):
            raise ValueError("raw activation shape disagrees with projection K")
        rotated_rows = np.stack([_fp16_hadamard(row, h_block) for row in raw])
        rotated_fp32 = rotated_rows.astype(np.float32)
        absmax = np.max(np.abs(rotated_fp32), axis=1)
        safe_absmax = np.where(absmax == 0, np.float32(1.0), absmax)
        normalized = (rotated_fp32.astype(np.float64) * 127.0) / \
                     safe_absmax.astype(np.float64)[:, None]
        activation = np.clip(np.rint(normalized), -128, 127).astype(np.int8)
        token_scale = np.zeros(rows, dtype=np.float32)
        for row_index, row_absmax in enumerate(absmax):
            token_scale[row_index] = np.float32(
                np.float16(row_absmax) * np.float16(1.0 / 127.0))
        expected_acc = (activation.astype(np.int64) @ weight.astype(np.int64).T)
        if np.any(expected_acc < np.iinfo(np.int32).min) or \
                np.any(expected_acc > np.iinfo(np.int32).max):
            raise OverflowError("frontend-derived accumulator exceeds INT32")
        expected_acc = expected_acc.astype(np.int32)
    else:
        if package_activation is None or package_scale is None or package_acc is None:
            raise ValueError("a package golden is required without --fp16-hadamard-frontend")
        activation = package_activation
        token_scale = package_scale
        expected_acc = package_acc
        # Check the complete package accumulator before exporting the fixture.
        computed_acc = activation.astype(np.int64) @ weight.astype(np.int64).T
        if not np.array_equal(computed_acc, expected_acc.astype(np.int64)):
            raise ValueError("package INT32 golden does not equal X_int8 @ QW.T")
    fp32_chain_output = (
        (expected_acc.astype(np.float32) * token_scale[:, None])
        * weight_scale[None, :] + bias[None, :]
    ).astype(np.float16)
    if output is None or (fp16_hadamard_frontend and raw_input_file is not None):
        # A caller-supplied RTL activation represents a new stage boundary;
        # its projection output must be checked against this input's exact
        # INT8/software chain, not the archive y_out from another input.
        output = fp32_chain_output.copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "activation_int8.hex", activation, 2)
    _write_hex(output_dir / "weight_int8.hex", weight, 2)
    _write_hex(output_dir / "accumulator_int32.hex", expected_acc, 8)
    _write_hex(output_dir / "token_scale_fp32.hex", token_scale.view(np.uint32), 8)
    _write_hex(output_dir / "weight_scale_fp32.hex", weight_scale.view(np.uint32), 8)
    _write_hex(output_dir / "bias_fp32.hex", bias.view(np.uint32), 8)
    _write_hex(output_dir / "output_fp16.hex", output.view(np.uint16), 4)
    _write_hex(output_dir / "reference_output_fp16.hex",
               fp32_chain_output.view(np.uint16), 4)
    if fp16_hadamard_frontend:
        _write_hex(output_dir / "raw_input_fp16.hex", raw.view(np.uint16), 4)
        _write_hex(output_dir / "rotated_fp16.hex", rotated_rows.view(np.uint16), 4)
        _write_hex(output_dir / "expected_quantized_int8.hex", activation, 2)
    manifest = {
        "layer": layer,
        "m": int(rows),
        "n": int(channels),
        "k": int(k),
        "array_tile": [TILE, TILE],
        "activation_layout": "[M,K] signed int8",
        "weight_layout": "[N,K] signed int8",
        "accumulator_layout": "[M,N] signed int32",
        "output_layout": "[M,N] float16",
        "activation_source": ("RTL-produced FP16 activation -> FP32-stage Hadamard -> FP16 -> dynamic INT8"
                              if fp16_hadamard_frontend and raw_input_file is not None else
                              "FP16 x_raw -> FP32-stage Hadamard -> FP16 -> dynamic INT8"
                              if fp16_hadamard_frontend else "package x_int8"),
        "h_block": h_block,
        "row_start": row_start,
        "row_count": rows,
        "integer_golden": ("derived exact INT8 X @ QW.T"
                           if package_acc is None or raw_input_file is not None else
                           "exact full projection X_int8 @ QW.T"),
        "software_epilogue": "fp16(fp32(acc)*s_t*weight_scale+bias)",
        "status": ("QLinear from RTL-produced raw activation; output golden recomputed"
                   if fp16_hadamard_frontend and raw_input_file is not None else
                   "QLinear from RTL-produced raw activation; no archive y_out for this layer"
                   if package_acc is None else
                   "full QLinear projection with RTL activation-front-end vectors"
                   if fp16_hadamard_frontend else
                   "complete projection fixture; runtime data preloaded from archive"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_qkv_projection")
    parser.add_argument("--layer", default="blocks.0.self_attn.qkv")
    parser.add_argument("--fp16-hadamard-frontend", action="store_true",
                        help="derive activations/scales from stored FP16 x_raw")
    parser.add_argument("--raw-input-file", type=Path,
                        help="override x_raw with an RTL-produced row-major FP16 hex file")
    parser.add_argument("--row-start", type=int, default=0,
                        help="first M row to include in this fixture")
    parser.add_argument("--row-count", type=int,
                        help="number of M rows to include; defaults to the remaining rows")
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.output_dir, args.layer,
                             args.fp16_hadamard_frontend, args.raw_input_file,
                             args.row_start, args.row_count), indent=2))


if __name__ == "__main__":
    main()
