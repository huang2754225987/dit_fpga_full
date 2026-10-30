#!/usr/bin/env python3
"""Prepare cross-attention online-PV vectors from RTL QK and KV outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
QUERIES, KEYS, HEADS, HEAD_DIM = 256, 32, 9, 64


def _read(path: Path, count: int) -> np.ndarray:
    bits = np.fromiter((int(v, 16) for v in path.read_text(encoding="ascii").splitlines()),
                       dtype=np.uint16)
    if bits.size != count:
        raise ValueError(f"{path} contains {bits.size}; expected {count}")
    return bits.view(np.float16)


def _write(path: Path, values: np.ndarray) -> None:
    path.write_text("".join(f"{int(v)&0xffff:04x}\n" for v in values.flat), encoding="ascii")


def prepare(kv_file: Path, score_file: Path, output_dir: Path,
            export_dir: Path = DEFAULT_EXPORT, conditioning: int = 0) -> dict:
    kv = _read(kv_file, KEYS * 2 * HEADS * HEAD_DIM).reshape(KEYS, 2, HEADS, HEAD_DIM)
    scores = _read(score_file, QUERIES * HEADS * KEYS).reshape(QUERIES, HEADS, KEYS)
    with np.load(export_dir / "golden_steps.npz", allow_pickle=False) as steps:
        mask = np.asarray(steps["mask_all"][conditioning])
    valid_keys = int(np.count_nonzero(mask))
    if not 0 < valid_keys <= KEYS or not np.array_equal(np.flatnonzero(mask), np.arange(valid_keys)):
        raise ValueError("context mask must be a non-empty valid prefix within one 32-key tile")
    values = kv[:, 1].astype(np.float32)
    expected = np.empty((HEADS, QUERIES, HEAD_DIM), dtype=np.float16)
    for head in range(HEADS):
        logits = scores[:, head, :valid_keys].astype(np.float32)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True), dtype=np.float32)
        probs = exp / np.sum(exp, axis=1, keepdims=True, dtype=np.float32)
        expected[head] = (probs @ values[:valid_keys, head].astype(np.float32)).astype(np.float16)
    values_transposed = np.ascontiguousarray(values.transpose(1, 2, 0), dtype=np.float16)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "scores_fp16.hex", scores.transpose(1, 0, 2).view(np.uint16))
    _write(output_dir / "values_head_dim_key_fp16.hex", values_transposed.view(np.uint16))
    _write(output_dir / "expected_attention_fp16.hex", expected.view(np.uint16))
    manifest = {"queries": QUERIES, "keys": KEYS, "valid_keys": valid_keys,
                "heads": HEADS,
                "head_dim": HEAD_DIM, "score_source": "RTL cross-attention QK",
                "value_source": "V half of RTL cross-attention KV projection",
                "reference": "FP32 softmax(RTL scores) @ V, rounded to FP16",
                "model_forward": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-file", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--conditioning", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_cross_attention_online_pv")
    args = parser.parse_args()
    print(json.dumps(prepare(args.kv_file, args.score_file, args.output_dir,
                             args.export_dir, args.conditioning), indent=2))


if __name__ == "__main__":
    main()
