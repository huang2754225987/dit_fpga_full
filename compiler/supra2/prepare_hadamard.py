#!/usr/bin/env python3
"""Prepare an archive-backed FP16 block-Hadamard RTL unit fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"


def _write_hex(path: Path, values: np.ndarray, width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text("".join(f"{int(v) & mask:0{width}x}\n" for v in values.flat),
                    encoding="ascii")


def _fp16_hadamard(values: np.ndarray, h_block: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float16).astype(np.float32)
    for offset in range(0, result.size, h_block):
        block = result[offset:offset+h_block].copy()
        stride = 1
        while stride < h_block:
            pairs = block.reshape(h_block // (2*stride), 2, stride).copy()
            block.reshape(h_block // (2*stride), 2, stride)[:, 0, :] = (
                pairs[:, 0, :] + pairs[:, 1, :])
            block.reshape(h_block // (2*stride), 2, stride)[:, 1, :] = (
                pairs[:, 0, :] - pairs[:, 1, :])
            stride *= 2
        result[offset:offset+h_block] = (
            block / np.float32(np.sqrt(h_block)))
    return result.astype(np.float16)


def prepare(export_dir: Path, output_dir: Path,
            layer: str = "blocks.0.self_attn.qkv", row: int = 0) -> dict:
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = next((item for item in manifest["layer_table"]
                  if item["name"] == layer), None)
    if entry is None:
        raise KeyError(f"unknown QLinear layer: {layer}")
    h_block = int(entry["h_block"])
    if h_block not in (16, 64, 256):
        raise ValueError(f"unsupported Supra2 Hadamard block size: {h_block}")
    with np.load(export_dir / "golden_unit_layers.npz", allow_pickle=False) as unit:
        raw = np.ascontiguousarray(unit[f"{layer}.x_raw"][row], dtype=np.float16)
        package_rotated = np.ascontiguousarray(
            unit[f"{layer}.x_rot"][row], dtype=np.float16)
    if raw.size % h_block:
        raise ValueError("input width must be divisible by the Hadamard block size")
    expected = _fp16_hadamard(raw, h_block)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "input_raw_fp16.hex", raw.view(np.uint16), 4)
    _write_hex(output_dir / "expected_rotated_fp16.hex", expected.view(np.uint16), 4)
    _write_hex(output_dir / "package_rotated_fp16.hex",
               package_rotated.view(np.uint16), 4)
    result = {
        "layer": layer,
        "row": row,
        "k": int(raw.size),
        "h_block": h_block,
        "input": "archive x_raw stored as FP16",
        "expected": "FP32 butterfly stages, exact power-of-two normalization, final FP16 rounding",
        "package_storage_note": "x_rot was generated before FP16 NPZ storage rounding",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def prepare_synthetic_partial_k(output_dir: Path, k: int = 16,
                                h_block: int = 16) -> dict:
    if k <= 0 or k % h_block:
        raise ValueError("synthetic K must be a positive multiple of H")
    raw = np.linspace(-0.875, 0.9375, k, dtype=np.float32).astype(np.float16)
    expected = _fp16_hadamard(raw, h_block)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "input_raw_fp16.hex", raw.view(np.uint16), 4)
    _write_hex(output_dir / "expected_rotated_fp16.hex", expected.view(np.uint16), 4)
    _write_hex(output_dir / "package_rotated_fp16.hex", np.zeros(k, dtype=np.uint16), 4)
    result = {
        "layer": "synthetic_partial_k",
        "k": k,
        "h_block": h_block,
        "input": "deterministic synthetic FP16 values",
        "expected": "FP32 butterfly stages, power-of-two normalization, final FP16 rounding",
        "rtl_lane_padding": "zero-fill to the next 32-element beat",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_hadamard")
    parser.add_argument("--layer", default="blocks.0.self_attn.qkv")
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--synthetic-partial-k", action="store_true",
                        help="emit a deterministic K=16/H=16 lane-padding unit fixture")
    args = parser.parse_args()
    if args.synthetic_partial_k:
        result = prepare_synthetic_partial_k(args.output_dir)
    else:
        result = prepare(args.export_dir, args.output_dir, args.layer, args.row)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
