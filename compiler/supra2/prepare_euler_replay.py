#!/usr/bin/env python3
"""Export the archive's 50 golden velocities for an RTL Euler-update replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
ELEMENTS = 4 * 32 * 32
LANES = 32
STEP_SIZE = np.float16(1.0 / 50.0)


def _write_hex(path: Path, values: np.ndarray) -> None:
    path.write_text("".join(f"{int(v):04x}\n" for v in values.flat),
                    encoding="ascii")


def prepare(export_dir: Path, output_dir: Path) -> dict:
    with np.load(export_dir / "golden_steps.npz", allow_pickle=False) as steps:
        latent = np.ascontiguousarray(steps["z0"], dtype=np.float16)
        velocity = np.ascontiguousarray(steps["v_per_step"], dtype=np.float16)
        expected = np.ascontiguousarray(steps["z_final"], dtype=np.float16)
    if latent.size != ELEMENTS or velocity.shape != (50, 4, 32, 32) or \
            expected.shape != latent.shape or ELEMENTS % LANES:
        raise ValueError("unexpected Euler golden tensor geometry")

    reference = latent.copy()
    for step_velocity in velocity:
        delta = (step_velocity.astype(np.float32) * STEP_SIZE).astype(np.float16)
        reference = (reference.astype(np.float32) +
                     delta.astype(np.float32)).astype(np.float16)
    error = reference.astype(np.float32) - expected.astype(np.float32)
    relative_l2 = float(np.linalg.norm(error) /
                        max(np.linalg.norm(expected.astype(np.float32)), 1.0e-30))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "z0_fp16.hex", latent.view(np.uint16))
    _write_hex(output_dir / "velocity_steps_fp16.hex", velocity.view(np.uint16))
    _write_hex(output_dir / "z_final_fp16.hex", expected.view(np.uint16))
    _write_hex(output_dir / "reference_final_fp16.hex", reference.view(np.uint16))
    manifest = {
        "steps": 50,
        "elements": ELEMENTS,
        "lanes": LANES,
        "step_size_fp16_bits": f"{int(STEP_SIZE.view(np.uint16)):04x}",
        "step_size_value": float(STEP_SIZE),
        "input": "archive golden_steps.z0 and v_per_step; no model forward",
        "reference_arithmetic": "FP16-rounded multiply then FP16-rounded add each step",
        "software_reference_rel_l2_vs_z_final": relative_l2,
        "limit": "golden-velocity Euler replay only; denoiser not evaluated",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_euler_replay")
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
