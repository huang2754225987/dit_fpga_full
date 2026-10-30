#!/usr/bin/env python3
"""Stage fixtures for the connected one-block RTL datapath simulation.

Collects everything `tb_supra2_block_top` needs into
`build/supra2_block7_top/` with uniform names:

  x_input.hex / golden_y.hex   - archive golden_block.npz b7.x / b7.y
  ctx_packed.hex               - packed 32x576 context tile (15 valid keys)
  ln{0,1,2}_{scale,shift}.hex  - AdaLN params (site 0=self, 1=cross, 2=mlp)
  gate{0,1}.hex                - per-feature residual gates (self, mlp)
  w_<stage>_int8.hex / ws_<stage>_fp32.hex / b_<stage>_fp32.hex
                               - INT8 weights (K-major per n), weight scales,
                                 biases for qkv, sproj, cq, ckv, cproj,
                                 mlp0, mlp2

Weight/parameter fixtures are input-independent and are copied from the
existing block-7 stage directories. Block input/golden come straight from
the archive so the connected run is comparable to the recorded 0.3685%
archived-b7.x chain result.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
OUT = BUILD / "supra2_block7_top"
EXPORT = BUILD / "supra2_hw_export" / "export"

STAGE_DIRS = {
    "qkv": "supra2_b7_qkv_after_layernorm",
    "sproj": "supra2_b7_self_attn_proj",
    "cq": "supra2_b7_cross_q_proj",
    "ckv": "supra2_b7_cross_kv_proj",
    "cproj": "supra2_b7_cross_attention_proj",
    "mlp0": "supra2_b7_block7_mlp0",
    "mlp2": "supra2_b7_block7_mlp2",
}

LN_DIR = BUILD / "supra2_b7_layernorm_modulate"
CROSS_LN_DIR = BUILD / "supra2_b7_cross_q_norm"
CTX_DIR = BUILD / "supra2_b7_cross_context_packed"


def dump_fp16_hex(path: Path, arr: np.ndarray) -> None:
    flat = np.asarray(arr, dtype=np.float16).reshape(-1)
    with path.open("w") as f:
        for v in flat:
            f.write(f"{np.uint16(v.view(np.uint16)):04x}\n")


def copy(src: Path, dst_name: str) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copyfile(src, OUT / dst_name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    golden = np.load(EXPORT / "golden_block.npz")
    dump_fp16_hex(OUT / "x_input.hex", golden["b7.x"])
    dump_fp16_hex(OUT / "golden_y.hex", golden["b7.y"])

    copy(CTX_DIR / "valid_context_fp16.hex", "ctx_packed.hex")

    # AdaLN scale/shift per site; gates for the two gated residuals.
    copy(LN_DIR / "scale_fp16.hex", "ln0_scale.hex")
    copy(LN_DIR / "shift_fp16.hex", "ln0_shift.hex")
    copy(CROSS_LN_DIR / "scale_fp16.hex", "ln1_scale.hex")
    copy(CROSS_LN_DIR / "shift_fp16.hex", "ln1_shift.hex")
    copy(LN_DIR / "scale_mlp_fp16.hex", "ln2_scale.hex")
    copy(LN_DIR / "shift_mlp_fp16.hex", "ln2_shift.hex")
    copy(LN_DIR / "gate_sa_fp16.hex", "gate0.hex")
    copy(LN_DIR / "gate_mlp_fp16.hex", "gate1.hex")

    for stage, d in STAGE_DIRS.items():
        base = BUILD / d
        copy(base / "weight_int8.hex", f"w_{stage}_int8.hex")
        copy(base / "weight_scale_fp32.hex", f"ws_{stage}_fp32.hex")
        copy(base / "bias_fp32.hex", f"b_{stage}_fp32.hex")

    manifest = {
        "block": 7,
        "input": "archive golden_block.npz b7.x",
        "golden": "archive golden_block.npz b7.y",
        "adaln_source": "host-precomputed FP16 (same fixtures as the serial chain)",
        "stages": list(STAGE_DIRS),
        "valid_context_keys": 15,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"staged fixtures into {OUT}")


if __name__ == "__main__":
    main()
