#!/usr/bin/env python3
"""Generate the model's tanh-approximate GELU ROM and RTL test vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
ROWS, FEATURES = 256, 2304


def _read(path: Path, count: int) -> np.ndarray:
    bits = np.fromiter((int(v, 16) for v in path.read_text(encoding="ascii").splitlines()),
                       dtype=np.uint16)
    if bits.size != count:
        raise ValueError(f"{path} contains {bits.size}; expected {count}")
    return bits.view(np.float16)


def _write_hex(path: Path, values: np.ndarray) -> None:
    path.write_text("".join(f"{int(v)&0xffff:04x}\n" for v in values.flat), encoding="ascii")


def generate_lut(path: Path) -> None:
    x = (np.arange(4096, dtype=np.float32) - 2048.0) / 256.0
    root = np.float32(np.sqrt(np.float32(2.0 / np.pi)))
    cubic = np.float32(0.044715)
    inner = root * (x + cubic * x * x * x)
    y = np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner).astype(np.float32))
    _write_hex(path, y.astype(np.float16).view(np.uint16))


def prepare(input_file: Path, output_dir: Path, lut_file: Path) -> dict:
    x = _read(input_file, ROWS * FEATURES).reshape(ROWS, FEATURES)
    expected = F.gelu(torch.from_numpy(x.copy()), approximate="tanh").numpy().copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_lut(lut_file)
    _write_hex(output_dir / "input_fp16.hex", x.view(np.uint16))
    _write_hex(output_dir / "expected_fp16.hex", expected.view(np.uint16))
    manifest = {"rows": ROWS, "features": FEATURES, "rom_entries": 4096,
                "rom_domain": "[-8, 8) at 1/256 spacing",
                "operation": "0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))",
                "input": str(input_file), "model_forward": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lut-file", type=Path,
                        default=ROOT / "rtl" / "supra2" / "gelu_tanh_lut_fp16.hex")
    args = parser.parse_args()
    print(json.dumps(prepare(args.input_file, args.output_dir, args.lut_file), indent=2))


if __name__ == "__main__":
    main()
