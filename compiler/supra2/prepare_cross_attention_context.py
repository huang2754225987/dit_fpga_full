#!/usr/bin/env python3
"""Crop padded context projections to one physical key tile using archive mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
CONTEXT_TOKENS, FEATURES, TILE = 128, 576, 32


def prepare(projection_file: Path, export_dir: Path, output_dir: Path,
            conditioning: int = 0) -> dict:
    bits = np.fromiter((int(v, 16) for v in projection_file.read_text(encoding="ascii").splitlines()),
                       dtype=np.uint16)
    if bits.size != CONTEXT_TOKENS * FEATURES:
        raise ValueError("context projection must contain 128x576 FP16 elements")
    rows = bits.view(np.float16).reshape(CONTEXT_TOKENS, FEATURES)
    with np.load(export_dir / "golden_steps.npz", allow_pickle=False) as steps:
        mask = np.asarray(steps["mask_all"][conditioning], dtype=np.float16)
    valid = np.flatnonzero(mask != 0)
    if valid.size == 0 or not np.array_equal(valid, np.arange(valid.size)):
        raise ValueError("context mask must be non-empty and contiguous from token zero")
    if valid.size > TILE:
        raise ValueError(f"{valid.size} valid context tokens exceed one {TILE}-key tile")
    packed = np.zeros((TILE, FEATURES), dtype=np.float16)
    packed[:valid.size] = rows[:valid.size]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "valid_context_fp16.hex").write_text(
        "".join(f"{int(v):04x}\n" for v in packed.view(np.uint16).flat), encoding="ascii")
    manifest = {"source_rows": CONTEXT_TOKENS, "physical_rows": TILE,
                "valid_keys": int(valid.size), "features": FEATURES,
                "conditioning": conditioning,
                "operation": "preserve valid prefix and zero-pad to one 32-key tile",
                "model_forward": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-file", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditioning", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.projection_file, args.export_dir, args.output_dir,
                             args.conditioning), indent=2))


if __name__ == "__main__":
    main()
