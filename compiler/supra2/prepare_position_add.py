#!/usr/bin/env python3
"""Prepare the 256x576 RTL position-embedding add fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .prepare_final_head import read_fp16_hex, write_fp16_hex


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"


def add_position_embedding(tokens: np.ndarray, position: np.ndarray) -> np.ndarray:
    if tokens.shape != (256, 576):
        raise ValueError(f"expected [256,576] token tensor, got {tokens.shape}")
    if position.shape == (1, 256, 576):
        position = position[0]
    if position.shape != (256, 576):
        raise ValueError(f"expected [256,576] position tensor, got {position.shape}")
    return (tokens.astype(np.float16) + position.astype(np.float16)).astype(np.float16)


def prepare(export_dir: Path, input_file: Path, output_dir: Path) -> dict:
    tokens = read_fp16_hex(input_file, (256, 576))
    with np.load(export_dir / "weights_passthrough_bf16.npz", allow_pickle=False) as weights:
        position = np.asarray(weights["pos_embed"], dtype=np.float16)
    if position.shape == (1, 256, 576):
        position = position[0]
    if position.shape != (256, 576):
        raise ValueError(f"unexpected pos_embed shape {position.shape}")
    result = add_position_embedding(tokens, position)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_fp16_hex(output_dir / "token_fp16.hex", tokens)
    write_fp16_hex(output_dir / "position_fp16.hex", position)
    write_fp16_hex(output_dir / "expected_fp16.hex", result)
    manifest = {
        "shape": [256, 576],
        "input_file": str(input_file),
        "position_source": "weights_passthrough_bf16.npz:pos_embed",
        "operation": "FP16 elementwise add",
        "model_forward": False,
        "rtl_output": str(output_dir / "rtl_output_fp16.hex"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--input-file", type=Path, required=True,
                        help="RTL-produced x_embed [256,576] output")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "supra2_position_add")
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.input_file, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
