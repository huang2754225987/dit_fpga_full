#!/usr/bin/env python3
"""Stage block-13 RTL fixtures from the existing serial block chain."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
OUT = BUILD / "supra2_block13_top"

STAGES = {
    "qkv": "supra2_b13_qkv_after_layernorm",
    "sproj": "supra2_b13_self_attn_proj",
    "cq": "supra2_b13_cross_q_proj",
    "ckv": "supra2_b13_cross_kv_proj",
    "cproj": "supra2_b13_cross_attention_proj",
    "mlp0": "supra2_b13_block13_mlp0",
    "mlp2": "supra2_b13_block13_mlp2",
}


def copy(source: Path, name: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copyfile(source, OUT / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe").mkdir(exist_ok=True)

    copy(BUILD / "supra2_b12_block12_mlp_residual/rtl_block12_output_fp16.hex",
         "x_input.hex")
    copy(BUILD / "supra2_b13_block13_mlp_residual/rtl_block13_output_fp16.hex",
         "golden_y.hex")
    copy(BUILD / "supra2_b13_cross_context_packed/valid_context_fp16.hex",
         "ctx_packed.hex")

    ln = BUILD / "supra2_b13_layernorm_modulate"
    cross_ln = BUILD / "supra2_b13_cross_q_norm"
    for source, name in (
        (ln / "scale_fp16.hex", "ln0_scale.hex"),
        (ln / "shift_fp16.hex", "ln0_shift.hex"),
        (cross_ln / "scale_fp16.hex", "ln1_scale.hex"),
        (cross_ln / "shift_fp16.hex", "ln1_shift.hex"),
        (ln / "scale_mlp_fp16.hex", "ln2_scale.hex"),
        (ln / "shift_mlp_fp16.hex", "ln2_shift.hex"),
        (ln / "gate_sa_fp16.hex", "gate0.hex"),
        (ln / "gate_mlp_fp16.hex", "gate1.hex"),
    ):
        copy(source, name)

    for stage, directory in STAGES.items():
        base = BUILD / directory
        copy(base / "weight_int8.hex", f"w_{stage}_int8.hex")
        copy(base / "weight_scale_fp32.hex", f"ws_{stage}_fp32.hex")
        copy(base / "bias_fp32.hex", f"b_{stage}_fp32.hex")

    manifest = {
        "block": 13,
        "input": "RTL block-12 output (rtl_block12_output_fp16.hex)",
        "reference": "RTL serial-chain block-13 output (rtl_block13_output_fp16.hex)",
        "reference_kind": "serial-chain reference, not an archive golden",
        "context": "block-13 packed context fixture",
        "adaln": "block-13 host-precomputed FP16 fixtures",
        "stages": list(STAGES),
        "valid_context_keys": 15,
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(f"staged block-13 fixtures into {OUT}")


if __name__ == "__main__":
    main()
