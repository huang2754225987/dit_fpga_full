#!/usr/bin/env python3
"""Compare a FakeDDR x_embed output dump with its single-linear golden file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_dirty_range(path: Path, base: int, size: int) -> bytes:
    payload = bytearray(size)
    present = bytearray(size)
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"invalid FakeDDR dump line {line_number}")
        address, word, mask = int(fields[0], 16), int(fields[1], 16), int(fields[2], 16)
        beat = word.to_bytes(16, byteorder="little")
        for lane in range(16):
            absolute = address + lane
            if mask & (1 << lane) and base <= absolute < base + size:
                offset = absolute - base
                payload[offset] = beat[lane]
                present[offset] = 1
    missing = present.find(0)
    if missing >= 0:
        raise ValueError(f"FakeDDR output byte missing at 0x{base + missing:08x}")
    return bytes(payload)


def verify(dump_path: Path, output_dir: Path, tolerance: float) -> dict:
    manifest = json.loads(
        (output_dir / "projection_smoke_manifest.json").read_text(encoding="utf-8"))
    rows, features = manifest["output_shape"]
    output_bytes = rows * features * 2
    batch_tokens = manifest["token_batch"]
    batch_bytes = batch_tokens * features * 2
    batch_count = manifest["batch_count"]
    actual_raw = b"".join(
        read_dirty_range(
            dump_path,
            manifest["output_ddr_base"] + index * manifest["output_window_bytes"],
            batch_bytes,
        )
        for index in range(batch_count)
    )
    expected_raw = (output_dir / manifest["golden_file"]).read_bytes()
    if len(expected_raw) != output_bytes:
        raise ValueError("golden output has an unexpected byte length")
    actual = np.frombuffer(actual_raw, dtype="<f2").astype(np.float32)
    expected = np.frombuffer(expected_raw, dtype="<f2").astype(np.float32)
    error = np.abs(actual - expected)
    result = {
        "elements": int(actual.size),
        "max_abs_error": float(error.max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(error * error))) if error.size else 0.0,
        "tolerance": tolerance,
        "dump": str(dump_path.resolve()),
        "status": "pass" if float(error.max(initial=0.0)) <= tolerance else "fail",
    }
    if result["status"] != "pass":
        raise SystemExit(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path,
                        default=Path("build/supra2_projection_smoke/layer0_dirty_output.hex"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("build/supra2_projection_smoke"))
    parser.add_argument("--tolerance", type=float, default=0.02)
    args = parser.parse_args()
    result = verify(args.dump, args.output_dir, args.tolerance)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
