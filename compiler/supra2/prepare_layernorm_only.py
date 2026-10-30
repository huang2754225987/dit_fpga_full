#!/usr/bin/env python3
"""Prepare a LayerNorm-only RTL fixture from a row-major FP16 hex tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _read(path: Path, count: int) -> np.ndarray:
    bits = np.fromiter((int(v, 16) for v in path.read_text(encoding="ascii").splitlines()),
                       dtype=np.uint16)
    if bits.size != count:
        raise ValueError(f"{path} contains {bits.size}; expected {count}")
    return bits.view(np.float16)


def _write(path: Path, values: np.ndarray) -> None:
    path.write_text("".join(f"{int(v)&0xffff:04x}\n" for v in values.flat), encoding="ascii")


def prepare(input_file: Path, output_dir: Path, rows: int, features: int = 576,
            shift_file: Path | None = None, scale_file: Path | None = None) -> dict:
    if rows <= 0 or rows % 32 or features != 576:
        raise ValueError("rows must be a positive multiple of 32; features must be 576")
    x = _read(input_file, rows * features).reshape(rows, features)
    x_tensor = torch.from_numpy(x.copy())
    zeros = torch.zeros(features, dtype=torch.float16)
    shift = _read(shift_file, features) if shift_file else zeros.numpy().copy()
    scale = _read(scale_file, features) if scale_file else zeros.numpy().copy()
    normalized = F.layer_norm(x_tensor, (features,), eps=1e-6)
    reference = (normalized * (1.0 + torch.from_numpy(scale.copy())) +
                 torch.from_numpy(shift.copy())).numpy().copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "x_fp16.hex", x.view(np.uint16))
    _write(output_dir / "shift_fp16.hex", shift.view(np.uint16))
    _write(output_dir / "scale_fp16.hex", scale.view(np.uint16))
    _write(output_dir / "reference_fp16.hex", reference.view(np.uint16))
    _write(output_dir / "qkv_reference_fp16.hex", reference.view(np.uint16))
    manifest = {"rows": rows, "features": features, "input": str(input_file),
                "operation": "FP16 LayerNorm(eps=1e-6), zero shift and scale",
                "model_forward": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--shift-file", type=Path)
    parser.add_argument("--scale-file", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_file, args.output_dir, args.rows,
                             shift_file=args.shift_file, scale_file=args.scale_file), indent=2))


if __name__ == "__main__":
    main()
