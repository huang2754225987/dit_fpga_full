#!/usr/bin/env python3
"""Prepare the final Supra2 token head from an RTL-produced block-13 tensor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
TOKENS = 256
WIDTH = 576
PATCH_VALUES = 16


def read_fp16_hex(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.fromiter(
        (int(value, 16) for value in path.read_text(encoding="ascii").splitlines()),
        dtype=np.uint16,
    )
    expected = int(np.prod(shape))
    if bits.size != expected:
        raise ValueError(f"{path} has {bits.size} values; expected {expected}")
    return bits.view(np.float16).reshape(shape).copy()


def write_fp16_hex(path: Path, values: np.ndarray) -> None:
    path.write_text(
        "".join(f"{int(value):04x}\n" for value in values.view(np.uint16).flat),
        encoding="ascii",
    )


def unpatchify_tokens(tokens: np.ndarray) -> np.ndarray:
    """Map [256,16] patch vectors to the model latent [4,32,32]."""
    if tokens.shape != (TOKENS, PATCH_VALUES):
        raise ValueError(f"expected [256,16] tokens, got {tokens.shape}")
    return tokens.reshape(16, 16, 4, 2, 2).transpose(2, 0, 3, 1, 4).reshape(4, 32, 32)


def prepare(export_dir: Path, input_file: Path, output_dir: Path) -> dict:
    x = read_fp16_hex(input_file, (TOKENS, WIDTH))
    with np.load(export_dir / "golden_block.npz", allow_pickle=False) as blocks, \
            np.load(export_dir / "weights_passthrough_bf16.npz", allow_pickle=False) as weights:
        c = np.asarray(blocks["b0.c"], dtype=np.float16).copy()
        adaln_w = np.asarray(weights["final.adaln.1.weight"], dtype=np.float16)
        adaln_b = np.asarray(weights["final.adaln.1.bias"], dtype=np.float16)
        linear_w = np.asarray(weights["final.linear.weight"], dtype=np.float16)
        linear_b = np.asarray(weights["final.linear.bias"], dtype=np.float16)

    if c.shape != (WIDTH,) or adaln_w.shape != (2 * WIDTH, WIDTH) or \
            adaln_b.shape != (2 * WIDTH,) or linear_w.shape != (PATCH_VALUES, WIDTH) or \
            linear_b.shape != (PATCH_VALUES,):
        raise ValueError("final head tensor shapes do not match the export contract")

    # The archive's b0.c is the shared t=0 timestep embedding. Compute only
    # final AdaLN and final head; no denoiser/model forward is invoked.
    c_t = torch.from_numpy(c)
    ada = F.linear(F.silu(c_t), torch.from_numpy(adaln_w), torch.from_numpy(adaln_b))
    shift = ada[:WIDTH].numpy().copy()
    scale = ada[WIDTH:].numpy().copy()
    x_t = torch.from_numpy(x)
    normalized = F.layer_norm(x_t, (WIDTH,), eps=1e-6)
    modulated = normalized * (1.0 + torch.from_numpy(scale)) + torch.from_numpy(shift)
    output = F.linear(modulated, torch.from_numpy(linear_w), torch.from_numpy(linear_b))
    output_np = output.detach().cpu().numpy().astype(np.float16, copy=True)
    latent = unpatchify_tokens(output_np)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_fp16_hex(output_dir / "final_norm_x_fp16.hex", x)
    write_fp16_hex(output_dir / "final_shift_fp16.hex", shift)
    write_fp16_hex(output_dir / "final_scale_fp16.hex", scale)
    write_fp16_hex(output_dir / "normalized_modulated_fp16.hex", modulated.numpy())
    write_fp16_hex(output_dir / "linear_weight_fp16.hex", linear_w)
    write_fp16_hex(output_dir / "linear_bias_fp16.hex", linear_b)
    write_fp16_hex(output_dir / "reference_tokens_fp16.hex", output_np)
    write_fp16_hex(output_dir / "reference_latent_fp16.hex", latent)
    manifest = {
        "tokens": TOKENS,
        "width": WIDTH,
        "head_out": PATCH_VALUES,
        "latent_shape": [4, 32, 32],
        "input": str(input_file),
        "timestep_source": "golden_block.b0.c (shared t=0 embedding)",
        "operations": ["RTL LayerNorm(eps=1e-6) + final AdaLN shift/scale",
                       "FP16 linear 576->16", "unpatchify"],
        "model_forward": False,
        "note": "FP16 reference fixture from RTL block-13 activation; not an end-to-end velocity golden",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--input-file", type=Path, required=True,
                        help="RTL-produced block-13 [256,576] FP16 tensor")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_final_head")
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.input_file, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
