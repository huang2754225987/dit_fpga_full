#!/usr/bin/env python3
"""Merge RTL QLinear M-batches and compare them with the full fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_hex(path: Path) -> np.ndarray:
    return np.fromiter((int(token, 16) for token in
                        path.read_text(encoding="ascii").split()), dtype=np.uint16)


def merge(batch_root: Path, batch_count: int, rows_per_batch: int,
          features: int, expected_file: Path, output_file: Path,
          cycles: int = 0) -> dict:
    pieces = []
    for index in range(batch_count):
        path = batch_root / f"batch_{index:02d}" / "rtl_output_fp16.hex"
        values = _read_hex(path)
        expected_size = rows_per_batch * features
        if values.size != expected_size:
            raise ValueError(f"{path} has {values.size} values; expected {expected_size}")
        pieces.append(values)

    actual_bits = np.concatenate(pieces)
    expected_bits = _read_hex(expected_file)
    if actual_bits.size != expected_bits.size:
        raise ValueError("merged RTL tensor and full reference have different sizes")
    actual = actual_bits.view(np.float16).astype(np.float32)
    expected = expected_bits.view(np.float16).astype(np.float32)
    delta = actual - expected
    peak = float(np.max(np.abs(expected))) if expected.size else 0.0
    metrics = {
        "rows": batch_count * rows_per_batch,
        "features": features,
        "elements": int(actual_bits.size),
        "fp16_bit_mismatches": int(np.count_nonzero(actual_bits != expected_bits)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "relative_to_peak": float(np.max(np.abs(delta)) / max(peak, 1e-30)),
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30)),
        "rtl_cycles_sum": int(cycles),
        "verification": "PASS" if np.array_equal(actual_bits, expected_bits) else "FAIL",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(f"{int(value):04x}\n" for value in actual_bits),
                           encoding="ascii")
    output_file.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if metrics["verification"] != "PASS":
        raise ValueError(f"merged QLinear output differs from full reference: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--batch-count", type=int, required=True)
    parser.add_argument("--rows-per-batch", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--expected-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=0)
    args = parser.parse_args()
    metrics = merge(args.batch_root, args.batch_count, args.rows_per_batch,
                    args.features, args.expected_file, args.output_file,
                    args.cycles)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
