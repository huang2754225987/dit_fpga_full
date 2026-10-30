#!/usr/bin/env python3
"""Prepare the ungated block-0 cross-attention residual fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path, count: int) -> np.ndarray:
    bits = np.fromiter((int(v, 16) for v in path.read_text(encoding="ascii").splitlines()),
                       dtype=np.uint16)
    if bits.size != count:
        raise ValueError(f"{path} contains {bits.size}; expected {count}")
    return bits.view(np.float16)


def _write(path: Path, values: np.ndarray) -> None:
    path.write_text("".join(f"{int(v)&0xffff:04x}\n" for v in values.flat), encoding="ascii")


def prepare(residual_file: Path, update_file: Path, output_dir: Path,
            rows: int = 256, features: int = 576) -> dict:
    residual = _read(residual_file, rows * features).reshape(rows, features)
    update = _read(update_file, rows * features).reshape(rows, features)
    gate = np.ones(features, dtype=np.float16)
    # The model's cross-attention residual is an ungated addition.
    expected = (torch.from_numpy(residual.copy()) + torch.from_numpy(update.copy())).numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "residual_fp16.hex", residual.view(np.uint16))
    _write(output_dir / "update_fp16.hex", update.view(np.uint16))
    _write(output_dir / "gate_fp16.hex", gate.view(np.uint16))
    _write(output_dir / "expected_fp16.hex", expected.view(np.uint16))
    manifest = {"rows": rows, "features": features, "residual_source": str(residual_file),
                "update_source": str(update_file), "operation": "FP16 x + FP16(cross_attn(x))",
                "model_forward": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-file", type=Path, required=True)
    parser.add_argument("--update-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.residual_file, args.update_file, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
