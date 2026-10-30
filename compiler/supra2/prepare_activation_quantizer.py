#!/usr/bin/env python3
"""Export one stored FP16 rotated token for the RTL activation quantizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .activation_quant import quantize_symmetric_int8


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"


def _write_hex(path: Path, values: np.ndarray, width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text("".join(f"{int(v) & mask:0{width}x}\n"
                             for v in values.flat), encoding="ascii")


def prepare(export_dir: Path, output_dir: Path,
            layer: str = "blocks.0.self_attn.qkv", row: int = 0) -> dict:
    with np.load(export_dir / "golden_unit_layers.npz", allow_pickle=False) as unit:
        rotated = np.ascontiguousarray(unit[f"{layer}.x_rot"][row], dtype=np.float16)
    values = rotated.astype(np.float32)
    quantized_rows, scales = quantize_symmetric_int8(values[None, :])
    absmax = np.max(np.abs(values))
    scale = scales[0]
    quantized = quantized_rows[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "input_fp16.hex", rotated.view(np.uint16), 4)
    _write_hex(output_dir / "expected_int8.hex", quantized.view(np.uint8), 2)
    _write_hex(output_dir / "expected_scale_fp32.hex",
               np.asarray([scale], dtype=np.float32).view(np.uint32), 8)
    manifest = {
        "layer": layer,
        "row": row,
        "k": int(rotated.size),
        "source": "package x_rot stored as FP16",
        "reference_scale": float(scale),
        "quantized_values": int(quantized.size),
        "status": "FP16-source quantizer fixture",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_activation_quant")
    parser.add_argument("--layer", default="blocks.0.self_attn.qkv")
    parser.add_argument("--row", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.output_dir,
                             args.layer, args.row), indent=2))


if __name__ == "__main__":
    main()
