#!/usr/bin/env python3
"""Prepare tiled self-attention Q/K tensors and FP16 score goldens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
TOKENS, HEADS, HEAD_DIM = 256, 9, 64


def _read_fp16_hex(path: Path, count: int) -> np.ndarray:
    bits = np.fromiter((int(v, 16) for v in path.read_text(encoding="ascii").splitlines()),
                       dtype=np.uint16)
    if bits.size != count:
        raise ValueError(f"{path} contains {bits.size} elements; expected {count}")
    return bits.view(np.float16)


def _write_hex(path: Path, values: np.ndarray) -> None:
    path.write_text("".join(f"{int(v) & 0xffff:04x}\n" for v in values.flat),
                    encoding="ascii")


def prepare(export_dir: Path, output_dir: Path, qkv_file: Path | None = None) -> dict:
    if qkv_file is None:
        with np.load(export_dir / "golden_unit_layers.npz", allow_pickle=False) as unit:
            qkv = np.asarray(unit["blocks.0.self_attn.qkv.y_out"], dtype=np.float16)
        source = "archive block-0 QKV golden"
    else:
        qkv = _read_fp16_hex(qkv_file, TOKENS*3*HEADS*HEAD_DIM).reshape(TOKENS, -1)
        source = "RTL-produced LayerNorm/AdaLN + QKV projection"
    if qkv.shape != (TOKENS, 3*HEADS*HEAD_DIM):
        raise ValueError("QKV tensor has unexpected dimensions")
    qkv = qkv.reshape(TOKENS, 3, HEADS, HEAD_DIM)
    q = np.ascontiguousarray(qkv[:, 0].transpose(1, 0, 2))
    k = np.ascontiguousarray(qkv[:, 1].transpose(1, 0, 2))
    scores = np.einsum("hqd,hkd->hqk", q.astype(np.float32),
                       k.astype(np.float32), optimize=True) / np.float32(8.0)
    scores = scores.astype(np.float16).transpose(1, 0, 2).copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "q_fp16.hex", q.view(np.uint16))
    _write_hex(output_dir / "k_fp16.hex", k.view(np.uint16))
    _write_hex(output_dir / "expected_scores_fp16.hex", scores.view(np.uint16))
    manifest = {"queries": TOKENS, "heads": HEADS, "keys": TOKENS,
                "head_dim": HEAD_DIM, "array_tile": [32, 32],
                "qkv_source": source,
                "score_formula": "FP32 dot(Q,K)/sqrt(64), rounded to FP16",
                "model_forward": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n",
                                               encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_attention_qk")
    parser.add_argument("--qkv-file", type=Path,
                        help="optional row-major RTL QKV output hex")
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.output_dir, args.qkv_file), indent=2))


if __name__ == "__main__":
    main()
