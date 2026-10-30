#!/usr/bin/env python3
"""Precompute the package's fixed 50-step timestep/AdaLN tables for DDR.

This performs only timestep embeddings and per-block/final AdaLN linears; it
does not execute a DiT block, denoising step, or image-model forward pass.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
DEFAULT_OUTPUT = ROOT / "build" / "supra2_timestep_adaln"
WIDTH = 576
BLOCKS = 14
STEPS = 50


def _fp16_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float16))


def _sinusoidal(t: float) -> torch.Tensor:
    half = 128
    indices = torch.arange(half, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * indices / half)
    phase = torch.tensor([t], dtype=torch.float32)[:, None] * frequencies[None] * 1000.0
    return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1).to(torch.float16)


def prepare(export_dir: Path, output_dir: Path) -> dict:
    with np.load(export_dir / "golden_steps.npz", allow_pickle=False) as steps, \
            np.load(export_dir / "weights_passthrough_bf16.npz", allow_pickle=False) as weights, \
            np.load(export_dir / "golden_block.npz", allow_pickle=False) as golden:
        schedule = np.asarray(steps["t_schedule"], dtype=np.float32)
        reference_c = np.asarray(golden["b0.c"], dtype=np.float16).reshape(WIDTH)
        if schedule.shape != (STEPS,):
            raise ValueError(f"expected {STEPS} timesteps, got {schedule.shape}")

        w0 = _fp16_tensor(weights["t_embed.mlp.0.weight"])
        b0 = _fp16_tensor(weights["t_embed.mlp.0.bias"])
        w2 = _fp16_tensor(weights["t_embed.mlp.2.weight"])
        b2 = _fp16_tensor(weights["t_embed.mlp.2.bias"])
        block_params = np.empty((STEPS, BLOCKS, 8, WIDTH), dtype="<f2")
        final_params = np.empty((STEPS, 2, WIDTH), dtype="<f2")
        c_values = np.empty((STEPS, WIDTH), dtype="<f2")

        for step, timestep in enumerate(schedule):
            hidden = F.silu(F.linear(_sinusoidal(float(timestep)), w0, b0))
            c = F.linear(hidden, w2, b2)
            c_values[step] = c.cpu().numpy().reshape(WIDTH)
            c_mod = F.silu(c)

            for block in range(BLOCKS):
                prefix = f"blocks.{block}.adaln.1"
                ada = F.linear(c_mod,
                               _fp16_tensor(weights[f"{prefix}.weight"]),
                               _fp16_tensor(weights[f"{prefix}.bias"]))
                values = ada.cpu().numpy().reshape(6, WIDTH)
                # block_top param_buf contract: LN0 scale/shift, LN1
                # scale/shift (plain norm), LN2 scale/shift, gate_sa, gate_mlp.
                block_params[step, block, 0] = values[1]
                block_params[step, block, 1] = values[0]
                block_params[step, block, 2:4] = 0.0
                block_params[step, block, 4] = values[4]
                block_params[step, block, 5] = values[3]
                block_params[step, block, 6] = values[2]
                block_params[step, block, 7] = values[5]

            final_ada = F.linear(c_mod,
                                 _fp16_tensor(weights["final.adaln.1.weight"]),
                                 _fp16_tensor(weights["final.adaln.1.bias"]))
            final_values = final_ada.cpu().numpy().reshape(2, WIDTH)
            final_params[step] = final_values

    # The supplied b0.c is the known t=0 MLP reference. Gate table generation
    # on agreement so weight/layout or precision changes cannot silently ship.
    generated_c0 = c_values[0]
    max_abs = float(np.max(np.abs(generated_c0.astype(np.float32) -
                                  reference_c.astype(np.float32))))
    peak = float(np.max(np.abs(reference_c.astype(np.float32))))
    rel_peak = max_abs / max(peak, 1e-30)
    if rel_peak > 0.01:
        raise ValueError(f"t=0 timestep MLP reference error {rel_peak:.6%} exceeds 1%")

    output_dir.mkdir(parents=True, exist_ok=True)
    block_path = output_dir / "block_adaln_params_fp16.bin"
    final_path = output_dir / "final_adaln_params_fp16.bin"
    c_path = output_dir / "timestep_embedding_fp16.bin"
    block_params.tofile(block_path)
    final_params.tofile(final_path)
    c_values.tofile(c_path)
    manifest = {
        "steps": STEPS,
        "blocks": BLOCKS,
        "width": WIDTH,
        "block_param_vectors": 8,
        "block_param_order": ["ln0_scale", "ln0_shift", "ln1_scale", "ln1_shift",
                              "ln2_scale", "ln2_shift", "gate_sa", "gate_mlp"],
        "final_param_order": ["shift", "scale"],
        "block_file": str(block_path.resolve()),
        "block_bytes": block_path.stat().st_size,
        "final_file": str(final_path.resolve()),
        "final_bytes": final_path.stat().st_size,
        "timestep_embedding_file": str(c_path.resolve()),
        "timestep_embedding_bytes": c_path.stat().st_size,
        "t0_c_reference_rel_peak": rel_peak,
        "model_forward": False,
        "precision": "FP16 tensors and PyTorch FP16 linear/SiLU operations",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.export_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
