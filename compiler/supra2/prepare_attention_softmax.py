#!/usr/bin/env python3
"""Prepare block-0 QK score rows for the RTL 256-key online-softmax test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
TOKENS = 256
HEADS = 9
HEAD_DIM = 64
KEY_TILE = 32


def _write_hex(path: Path, values: np.ndarray, width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text("".join(f"{int(v) & mask:0{width}x}\n" for v in values.flat),
                    encoding="ascii")


def prepare(export_dir: Path, output_dir: Path, qkv_file: Path | None = None) -> dict:
    if qkv_file is None:
        with np.load(export_dir / "golden_unit_layers.npz", allow_pickle=False) as unit:
            qkv = np.asarray(unit["blocks.0.self_attn.qkv.y_out"], dtype=np.float16)
        source = "archive block-0 self-attention QKV y_out"
    else:
        raw = np.fromiter(
            (int(line, 16) for line in qkv_file.read_text(encoding="ascii").splitlines()),
            dtype=np.uint16,
        )
        if raw.size != TOKENS * 3 * HEADS * HEAD_DIM:
            raise ValueError("QKV RTL output has the wrong element count")
        qkv = raw.reshape(TOKENS, 3 * HEADS * HEAD_DIM).view(np.float16).copy()
        source = "RTL LayerNorm/AdaLN + QKV projection output"
    if qkv.shape != (TOKENS, 3 * HEADS * HEAD_DIM):
        raise ValueError("block-0 QKV golden has unexpected shape")
    qkv = qkv.reshape(TOKENS, 3, HEADS, HEAD_DIM)
    q = qkv[:, 0].astype(np.float32)
    k = qkv[:, 1].astype(np.float32)
    scores = np.einsum("qhd,khd->qhk", q, k, optimize=True) / np.float32(8.0)
    scores = scores.astype(np.float16)
    scores_f32 = scores.astype(np.float32)
    shifted = scores_f32 - np.max(scores_f32, axis=2, keepdims=True)
    exp = np.exp(shifted, dtype=np.float32)
    probs = (exp / np.sum(exp, axis=2, keepdims=True, dtype=np.float32)).astype(np.float16)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "scores_fp16.hex", scores.view(np.uint16), 4)
    _write_hex(output_dir / "expected_probs_fp16.hex", probs.view(np.uint16), 4)
    manifest = {"queries": TOKENS, "heads": HEADS, "keys": TOKENS,
                "head_dim": HEAD_DIM, "key_tile": KEY_TILE,
                "score_source": source,
                "score_formula": "FP32 dot(Q,K)/sqrt(64), rounded to FP16",
                "reference": "FP32 softmax over the 256 FP16 scores, rounded to FP16",
                "model_forward": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_attention_softmax")
    parser.add_argument("--qkv-file", type=Path,
                        help="optional row-major FP16 hex dumped by the QLinear RTL testbench")
    parser.add_argument("--scores-file", type=Path,
                        help="optional row-major FP16 scores dumped by the RTL QK testbench")
    args = parser.parse_args()
    if args.scores_file is None:
        result = prepare(args.export_dir, args.output_dir, args.qkv_file)
    else:
        raw = np.fromiter(
            (int(line, 16) for line in args.scores_file.read_text(encoding="ascii").splitlines()),
            dtype=np.uint16,
        )
        if raw.size != TOKENS * HEADS * TOKENS:
            raise ValueError("RTL QK score output has the wrong element count")
        scores = raw.reshape(TOKENS, HEADS, TOKENS).view(np.float16).copy()
        scores_f32 = scores.astype(np.float32)
        shifted = scores_f32 - np.max(scores_f32, axis=2, keepdims=True)
        exp = np.exp(shifted, dtype=np.float32)
        probs = (exp / np.sum(exp, axis=2, keepdims=True, dtype=np.float32)).astype(np.float16)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_hex(args.output_dir / "scores_fp16.hex", scores.view(np.uint16), 4)
        _write_hex(args.output_dir / "expected_probs_fp16.hex", probs.view(np.uint16), 4)
        result = {"queries": TOKENS, "heads": HEADS, "keys": TOKENS,
                  "head_dim": HEAD_DIM, "key_tile": KEY_TILE,
                  "score_source": "RTL QK systolic array output",
                  "score_formula": "RTL FP16xFP16 dot, FP32 accumulation, divide by 8",
                  "reference": "FP32 softmax over the 256 RTL FP16 scores, rounded to FP16",
                  "model_forward": False}
        (args.output_dir / "manifest.json").write_text(json.dumps(result, indent=2)+"\n",
                                                          encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
