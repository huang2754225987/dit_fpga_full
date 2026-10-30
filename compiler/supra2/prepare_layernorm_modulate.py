#!/usr/bin/env python3
"""Prepare archive-backed LayerNorm/AdaLN RTL vectors without a model forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"


def _write_hex(path: Path, values: np.ndarray, width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text("".join(f"{int(v) & mask:0{width}x}\n" for v in values.flat),
                    encoding="ascii")


def prepare(export_dir: Path, output_dir: Path, block: int = 0,
            input_file: Path | None = None, c_file: Path | None = None,
            branch: str = "self_attn") -> dict:
    if branch not in {"self_attn", "mlp"}:
        raise ValueError("branch must be 'self_attn' or 'mlp'")
    with np.load(export_dir / "golden_block.npz", allow_pickle=False) as blocks, \
            np.load(export_dir / "weights_passthrough_bf16.npz", allow_pickle=False) as weights:
        if input_file is None:
            x = np.ascontiguousarray(blocks[f"b{block}.x"], dtype=np.float16)
            input_source = f"archive block input b{block}.x"
        else:
            input_bits = np.fromiter(
                (int(value, 16) for value in input_file.read_text(encoding="ascii").splitlines()),
                dtype=np.uint16,
            )
            if input_bits.size != 256 * 576:
                raise ValueError(f"{input_file} must contain 256x576 FP16 values")
            x = np.ascontiguousarray(input_bits.view(np.float16).reshape(256, 576))
            input_source = f"RTL-produced activation: {input_file}"
        c_key = f"b{block}.c"
        if c_file is not None:
            c_bits = np.fromiter(
                (int(value, 16) for value in c_file.read_text(encoding="ascii").splitlines()),
                dtype=np.uint16,
            )
            if c_bits.size != 576:
                raise ValueError(f"{c_file} must contain 576 FP16 timestep values")
            c = np.ascontiguousarray(c_bits.view(np.float16))
            c_source = f"explicit FP16 timestep vector: {c_file}"
        elif c_key in blocks.files:
            c = np.ascontiguousarray(blocks[c_key], dtype=np.float16)
            c_source = f"archive {c_key}"
        else:
            c = np.ascontiguousarray(blocks["b0.c"], dtype=np.float16)
            c_source = "archive b0.c (shared timestep vector; b0.c equals b7.c)"
        weight = np.ascontiguousarray(weights[f"blocks.{block}.adaln.1.weight"], dtype=np.float16)
        bias = np.ascontiguousarray(weights[f"blocks.{block}.adaln.1.bias"], dtype=np.float16)
    if x.shape != (256, 576) or c.shape != (576,) or weight.shape != (3456, 576):
        raise ValueError("archive tensors do not match Supra2 block dimensions")

    # Reproduce only the timestep MLP and its LN/modulation boundary. No DiT
    # block, attention, denoising step, or image model forward is executed.
    c_t = torch.from_numpy(c.copy())
    w_t = torch.from_numpy(weight.copy())
    b_t = torch.from_numpy(bias.copy())
    silu = F.silu(c_t)
    ada = F.linear(silu, w_t, b_t)
    shift = ada[:576].numpy().copy()
    scale = ada[576:1152].numpy().copy()
    gate = ada[1152:1728].numpy().copy()
    shift_mlp = ada[1728:2304].numpy().copy()
    scale_mlp = ada[2304:2880].numpy().copy()
    gate_mlp = ada[2880:3456].numpy().copy()
    x_t = torch.from_numpy(x.copy())
    selected_shift = shift if branch == "self_attn" else shift_mlp
    selected_scale = scale if branch == "self_attn" else scale_mlp
    target_mod = (F.layer_norm(x_t, (576,), eps=1e-6) *
                  (1.0 + torch.from_numpy(selected_scale)) +
                  torch.from_numpy(selected_shift)).numpy().copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_hex(output_dir / "x_fp16.hex", x.view(np.uint16), 4)
    _write_hex(output_dir / "shift_fp16.hex", selected_shift.view(np.uint16), 4)
    _write_hex(output_dir / "scale_fp16.hex", selected_scale.view(np.uint16), 4)
    _write_hex(output_dir / "gate_sa_fp16.hex", gate.view(np.uint16), 4)
    _write_hex(output_dir / "shift_mlp_fp16.hex", shift_mlp.view(np.uint16), 4)
    _write_hex(output_dir / "scale_mlp_fp16.hex", scale_mlp.view(np.uint16), 4)
    _write_hex(output_dir / "gate_mlp_fp16.hex", gate_mlp.view(np.uint16), 4)
    _write_hex(output_dir / "reference_mod_fp16.hex", target_mod.view(np.uint16), 4)
    with np.load(export_dir / "golden_unit_layers.npz", allow_pickle=False) as unit:
        qkv_key = (f"blocks.{block}.self_attn.qkv.x_raw" if branch == "self_attn"
                   else f"blocks.{block}.mlp.0.x_raw")
        qkv_raw = (np.asarray(unit[qkv_key], dtype=np.float16)
                   if qkv_key in unit.files else target_mod)
    _write_hex(output_dir / "qkv_xraw_fp16.hex", qkv_raw.view(np.uint16), 4)
    manifest = {"block": block, "branch": branch, "rows": 256, "features": 576,
                "adaln_source": "CPU FP16 SiLU + FP16 linear using the selected timestep vector",
                "activation_source": input_source, "timestep_source": c_source,
                "operation": "FP16 LayerNorm(eps=1e-6), then FP16 modulation",
                "model_forward": False}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_layernorm_modulate")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--input-file", type=Path,
                        help="optional RTL-produced 256x576 FP16 activation")
    parser.add_argument("--c-file", type=Path,
                        help="optional 576-element FP16 timestep vector")
    parser.add_argument("--branch", choices=("self_attn", "mlp"), default="self_attn",
                        help="which AdaLN modulation to emit")
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.output_dir, args.block,
                             args.input_file, args.c_file, args.branch), indent=2))


if __name__ == "__main__":
    main()
