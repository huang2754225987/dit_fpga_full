#!/usr/bin/env python3
"""Prepare fixtures for the supra2_layernorm_modulate_32lane RTL unit test.

Slices the first 32 rows of an existing prepare_layernorm_modulate fixture
(LayerNorm is row-independent, so the row-sequential reference rows are exact
references for the tile unit). Defaults point at the fresh block-7
self-attention norm fixture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _read_fp16_hex(path: Path, count: int) -> np.ndarray:
    bits = np.fromiter((int(v, 16) for v in path.read_text(encoding="ascii").splitlines()),
                       dtype=np.uint16)
    if bits.size != count:
        raise ValueError(f"{path} has {bits.size} values; expected {count}")
    return bits.view(np.float16)


def _write_fp16_hex(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bits = np.ascontiguousarray(values).view(np.uint16)
    path.write_text("".join(f"{int(v) & 0xffff:04x}\n" for v in bits.flat),
                    encoding="ascii")


def prepare(input_dir: Path, output_dir: Path, rows: int) -> dict:
    x = _read_fp16_hex(input_dir / "x_fp16.hex", 256 * 576).reshape(256, 576)
    shift = _read_fp16_hex(input_dir / "shift_fp16.hex", 576)
    scale = _read_fp16_hex(input_dir / "scale_fp16.hex", 576)
    reference = _read_fp16_hex(input_dir / "reference_mod_fp16.hex", 256 * 576).reshape(256, 576)

    _write_fp16_hex(output_dir / "x_fp16.hex", x[:rows])
    _write_fp16_hex(output_dir / "shift_fp16.hex", shift)
    _write_fp16_hex(output_dir / "scale_fp16.hex", scale)
    _write_fp16_hex(output_dir / "expected_fp16.hex", reference[:rows])
    manifest = {
        "rows": rows, "features": 576,
        "source": str(input_dir),
        "reference": "rows 0..31 of the row-sequential LayerNorm/AdaLN reference",
        "model_forward": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="ascii")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "build/supra2_rne_b7_self_norm")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_dir, args.output_dir, args.rows), indent=2))


if __name__ == "__main__":
    main()
