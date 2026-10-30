#!/usr/bin/env python3
"""Prepare block-0 attention PV inputs and FP32 reference outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
TOKENS, HEADS, HEAD_DIM = 256, 9, 64


def _read(path: Path, count: int) -> np.ndarray:
    bits = np.fromiter((int(v, 16) for v in path.read_text(encoding="ascii").splitlines()),
                       dtype=np.uint16)
    if bits.size != count:
        raise ValueError(f"{path} contains {bits.size} values; expected {count}")
    return bits.view(np.float16)


def _write(path: Path, values: np.ndarray) -> None:
    path.write_text("".join(f"{int(v) & 0xffff:04x}\n" for v in values.flat),
                    encoding="ascii")


def prepare(output_dir: Path, qkv_file: Path, score_file: Path,
            probability_file: Path) -> dict:
    qkv = _read(qkv_file, TOKENS*3*HEADS*HEAD_DIM).reshape(TOKENS, 3, HEADS, HEAD_DIM)
    scores = _read(score_file, TOKENS*HEADS*TOKENS).reshape(TOKENS, HEADS, TOKENS)
    probabilities = _read(probability_file, TOKENS*HEADS*TOKENS).reshape(
        TOKENS, HEADS, TOKENS)
    values = qkv[:, 2].astype(np.float32)
    score32 = scores.astype(np.float32)
    exp = np.exp(score32 - np.max(score32, axis=2, keepdims=True), dtype=np.float32)
    ideal_prob = exp / np.sum(exp, axis=2, keepdims=True, dtype=np.float32)
    expected = np.empty((HEADS, TOKENS, HEAD_DIM), dtype=np.float16)
    for head in range(HEADS):
        expected[head] = (ideal_prob[:, head] @ values[:, head]).astype(np.float16)

    v_transposed = np.ascontiguousarray(values.transpose(1, 2, 0), dtype=np.float16)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "probabilities_fp16.hex", probabilities.view(np.uint16))
    _write(output_dir / "values_transposed_fp16.hex", v_transposed.view(np.uint16))
    _write(output_dir / "expected_attention_fp16.hex", expected.view(np.uint16))
    manifest = {"queries": TOKENS, "heads": HEADS, "keys": TOKENS,
                "head_dim": HEAD_DIM, "array_tile": [32, 32],
                "qkv_source": "RTL QKV projection output",
                "probability_source": "RTL online-softmax output",
                "reference": "FP32 softmax(QK RTL scores) @ V, rounded to FP16",
                "model_forward": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n",
                                               encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_attention_pv")
    parser.add_argument("--qkv-file", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--probability-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.qkv_file, args.score_file,
                             args.probability_file), indent=2))


if __name__ == "__main__":
    main()
