#!/usr/bin/env python3
"""Prepare fixtures for the supra2_softmax_2pass RTL unit test.

Scores: 32 query rows x KEYS keys, FP16, row-major. When --scores-file points
at a self-attention QK output (256 queries x 9 heads x 256 keys, query-major
per head as produced by prepare_attention_qk), head --head query rows
[--query-start, --query-start+32) are sliced as a real-data tile; otherwise a
synthetic tile with an attention-like score distribution is generated.

Reference: FP32 softmax over the FP16 scores, rounded to FP16. The RTL uses
the graph exp LUT + 1/16-step interpolation and a recip LUT, so the gate is
tolerance-based, not bitwise.
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


def prepare(output_dir: Path, keys: int, head: int, query_start: int,
            scores_file: Path | None, seed: int) -> dict:
    rows = 32
    source = "synthetic"
    if scores_file is not None:
        # QK score hex files are (queries, keys, heads) with heads innermost
        scores = _read_fp16_hex(scores_file, 256 * 256 * 9).reshape(256, 256, 9)
        if query_start + rows > 256:
            raise ValueError("query slice out of range")
        tile = scores[query_start:query_start + rows, :keys, head].copy()
        source = f"{scores_file} head={head} rows={query_start}..{query_start + rows - 1}"
    else:
        rng = np.random.default_rng(seed)
        # attention-like logits: per-row base spread plus per-key variation
        base = rng.uniform(-6.0, 2.0, size=(rows, 1)).astype(np.float32)
        var = rng.normal(0.0, 3.0, size=(rows, keys)).astype(np.float32)
        tile = (base + var).astype(np.float16)

    x = tile.astype(np.float32)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    probs = (e / e.sum(axis=1, keepdims=True)).astype(np.float16)

    _write_fp16_hex(output_dir / "scores_fp16.hex", tile)
    _write_fp16_hex(output_dir / "expected_fp16.hex", probs)
    manifest = {
        "rows": rows, "keys": keys, "source": str(source),
        "reference": "FP32 softmax over FP16 scores, rounded to FP16",
        "operation": "P = exp(s - max) * recip(sum), per row",
        "model_forward": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="ascii")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keys", type=int, default=256)
    parser.add_argument("--head", type=int, default=0)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--scores-file", type=Path, default=None,
                        help="self-attention QK scores hex (256x9x256) to slice")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.keys, args.head,
                             args.query_start, args.scores_file, args.seed),
                     indent=2))


if __name__ == "__main__":
    main()
