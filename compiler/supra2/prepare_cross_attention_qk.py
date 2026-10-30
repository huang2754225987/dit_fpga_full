#!/usr/bin/env python3
"""Prepare block-0 cross-attention Q/K tensors and FP16 score references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
QUERIES, KEYS, HEADS, HEAD_DIM = 256, 32, 9, 64


def _read(path: Path, count: int) -> np.ndarray:
    bits = np.fromiter((int(v, 16) for v in path.read_text(encoding="ascii").splitlines()),
                       dtype=np.uint16)
    if bits.size != count:
        raise ValueError(f"{path} contains {bits.size}; expected {count}")
    return bits.view(np.float16)


def _write(path: Path, values: np.ndarray) -> None:
    path.write_text("".join(f"{int(v)&0xffff:04x}\n" for v in values.flat), encoding="ascii")


def prepare(q_file: Path, kv_file: Path, output_dir: Path,
            export_dir: Path = ROOT / "build" / "supra2_hw_export" / "export",
            conditioning: int = 0) -> dict:
    q = _read(q_file, QUERIES * HEADS * HEAD_DIM).reshape(QUERIES, HEADS, HEAD_DIM)
    kv = _read(kv_file, KEYS * 2 * HEADS * HEAD_DIM).reshape(KEYS, 2, HEADS, HEAD_DIM)
    with np.load(export_dir / "golden_steps.npz", allow_pickle=False) as steps:
        mask = np.asarray(steps["mask_all"][conditioning])
    valid_keys = int(np.count_nonzero(mask))
    if not 0 < valid_keys <= KEYS or not np.array_equal(np.flatnonzero(mask), np.arange(valid_keys)):
        raise ValueError("context mask must be a non-empty valid prefix within one 32-key tile")
    q_head_major = np.ascontiguousarray(q.transpose(1, 0, 2))
    k_head_major = np.ascontiguousarray(kv[:, 0].transpose(1, 0, 2))
    scores = (np.einsum("hqd,hkd->hqk", q_head_major.astype(np.float32),
                        k_head_major.astype(np.float32), optimize=True) / np.float32(8.0))
    scores = scores.astype(np.float16).transpose(1, 0, 2).copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "q_fp16.hex", q_head_major.view(np.uint16))
    _write(output_dir / "k_fp16.hex", k_head_major.view(np.uint16))
    _write(output_dir / "expected_scores_fp16.hex", scores.view(np.uint16))
    manifest = {"queries": QUERIES, "keys": KEYS, "valid_keys": valid_keys,
                "heads": HEADS,
                "head_dim": HEAD_DIM, "q_source": "RTL cross-attention Q projection",
                "kv_source": "RTL cross-attention KV projection",
                "score_formula": "FP32 dot(Q,K)/sqrt(64), rounded to FP16",
                "model_forward": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-file", type=Path, required=True)
    parser.add_argument("--kv-file", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path,
                        default=ROOT / "build" / "supra2_hw_export" / "export")
    parser.add_argument("--conditioning", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_cross_attention_qk")
    args = parser.parse_args()
    print(json.dumps(prepare(args.q_file, args.kv_file, args.output_dir,
                             args.export_dir, args.conditioning), indent=2))


if __name__ == "__main__":
    main()
