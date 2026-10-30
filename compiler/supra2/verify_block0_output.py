#!/usr/bin/env python3
"""Compare an RTL-produced block output with its archived golden tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"


def verify(actual_file: Path, export_dir: Path, gate: float = 0.02,
           block: int = 0, allow_missing: bool = False) -> dict:
    bits = np.fromiter((int(v, 16) for v in actual_file.read_text(encoding="ascii").splitlines()),
                       dtype=np.uint16)
    actual = bits.view(np.float16).astype(np.float32)
    with np.load(export_dir / "golden_block.npz", allow_pickle=False) as blocks:
        golden_key = f"b{block}.y"
        if golden_key not in blocks.files:
            if allow_missing:
                return {"block": block, "status": "package_golden_unavailable",
                        "golden": f"golden_block.{golden_key}"}
            raise KeyError(f"archive does not contain golden_block.{golden_key}")
        expected = np.asarray(blocks[golden_key], dtype=np.float16).astype(np.float32).reshape(-1)
    if actual.shape != expected.shape:
        raise ValueError(f"RTL output has {actual.size} elements; expected {expected.size}")
    delta = actual - expected
    peak = float(np.max(np.abs(expected)))
    metrics = {"elements": int(actual.size), "max_abs": float(np.max(np.abs(delta))),
               "rel_to_peak": float(np.max(np.abs(delta)) / peak),
               "rmse": float(np.sqrt(np.mean(delta * delta))),
               "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(expected)),
               "gate_rel_to_peak": gate, "golden": f"golden_block.b{block}.y",
               "block": block}
    if metrics["rel_to_peak"] > gate:
        raise ValueError(f"block-{block} output relative-to-peak error {metrics['rel_to_peak']:.6%} exceeds {gate:.6%}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-file", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--gate", type=float, default=0.02)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true",
                        help="report unavailable block goldens without treating them as failures")
    args = parser.parse_args()
    print(json.dumps(verify(args.actual_file, args.export_dir, args.gate, args.block,
                            args.allow_missing), indent=2))


if __name__ == "__main__":
    main()
