#!/usr/bin/env python3
"""Validate Supra2-IMG checkpoint shapes and emit an FPGA projection plan.

This is a planning/export front end. It does not yet emit executable ucode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "model" / "model_final_ema.pt"
TILE = 32
TOKEN_BATCH = 32
L1_OUTPUT_PAGE_BYTES = 64 * 1024


def choose_token_batch(tokens: int, output_features: int) -> tuple[int, int]:
    """Choose a power-of-two M batch that fits one current L1 output page."""
    output_tiles = (output_features + TILE - 1) // TILE
    output_row_stride = output_tiles * TILE * 2
    max_rows_by_l1 = L1_OUTPUT_PAGE_BYTES // output_row_stride
    limit = min(tokens, TOKEN_BATCH, max_rows_by_l1)
    if limit < 1:
        raise ValueError("one output row does not fit the current L1 output page")
    batch = 1 << (limit.bit_length() - 1)
    return batch, output_row_stride


def projection_specs(depth: int) -> list[tuple[str, str, str]]:
    specs = [
        ("x_embed", "latent_patch_embedding", "latent_tokens"),
        ("t_embed.mlp.0", "timestep_embedding_0", "timestep"),
        ("t_embed.mlp.2", "timestep_embedding_2", "timestep"),
        ("ctx_proj", "text_context_projection", "text_context"),
    ]
    for index in range(depth):
        prefix = f"blocks.{index}."
        specs.extend([
            (prefix + "self_attn.qkv", "self_attention_qkv", "latent_tokens"),
            (prefix + "self_attn.proj", "self_attention_output", "latent_tokens"),
            (prefix + "cross_attn.q", "cross_attention_q", "latent_tokens"),
            (prefix + "cross_attn.kv", "cross_attention_kv", "text_context"),
            (prefix + "cross_attn.proj", "cross_attention_output", "latent_tokens"),
            (prefix + "mlp.0", "mlp_expand", "latent_tokens"),
            (prefix + "mlp.2", "mlp_contract", "latent_tokens"),
            (prefix + "adaln.1", "adaln_parameters", "timestep"),
        ])
    specs.extend([
        ("final.adaln.1", "final_adaln_parameters", "timestep"),
        ("final.linear", "latent_patch_output", "latent_tokens"),
    ])
    return specs


def load_state(checkpoint: Path):
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required to read the checkpoint") from exc

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise SystemExit("Unsupported checkpoint: expected a dictionary")
    weights = state.get("ema", state.get("model"))
    if not isinstance(weights, dict):
        raise SystemExit("Checkpoint has neither an 'ema' nor a 'model' state dict")
    return state, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "build" / "projection_plan.json")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    state, weights = load_state(args.checkpoint)
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    if config.get("patch", 2) != 2 or config.get("ctx_len", 128) != 128:
        raise SystemExit("Only patch=2 and ctx_len=128 are supported")

    records = []
    missing = []
    for prefix, role, input_kind in projection_specs(depth=14):
        weight_key, bias_key = prefix + ".weight", prefix + ".bias"
        weight, bias = weights.get(weight_key), weights.get(bias_key)
        if (weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2):
            missing.append(weight_key)
            continue
        out_features, in_features = map(int, weight.shape)
        if (bias is None or not hasattr(bias, "numel") or
                int(bias.numel()) != out_features):
            missing.append(bias_key)
            continue

        rows = {"timestep": 1, "text_context": 128, "latent_tokens": 256}[input_kind]
        token_batch, output_row_stride = choose_token_batch(rows, out_features)
        records.append({
            "name": prefix,
            "role": role,
            "input_kind": input_kind,
            "input_tokens": rows,
            "input_features": in_features,
            "output_features": out_features,
            "dtype": "fp16",
            "token_batch": token_batch,
            "token_batches": (rows + token_batch - 1) // token_batch,
            "output_row_stride_bytes": output_row_stride,
            "output_page_bytes": L1_OUTPUT_PAGE_BYTES,
            "k_tiles": (in_features + TILE - 1) // TILE,
            "n_tiles": (out_features + TILE - 1) // TILE,
            "weight_key": weight_key,
            "bias_key": bias_key,
        })

    if missing:
        raise SystemExit("Missing expected checkpoint tensors:\n  " + "\n  ".join(missing))

    plan = {
        "model_id": "SupraLabs/Supra2-IMG",
        "checkpoint": os.path.relpath(args.checkpoint.resolve(), ROOT),
        "parameter_count": sum(int(value.numel()) for value in weights.values()
                               if hasattr(value, "numel")),
        "hardware": {"rtl_top": None,
                     "rtl_operator_filelist": "rtl/supra2/supra2_model_operators.f",
                     "array": [32, 32],
                     "dtype": "fp16", "tile": TILE,
                     "token_batch": TOKEN_BATCH},
        "sampling": {"steps": 50, "latent_tokens": 256,
                     "cfg": "two denoiser calls per step when enabled"},
        "fpga_candidate": "all listed FP16 linear projections",
        "host_initially": [
            "Flan-T5 text encoder",
            "attention score, softmax, and value mixing",
            "full-width LayerNorm and AdaLN elementwise operations",
            "GELU(tanh)", "Euler flow update", "SD-VAE decoder",
        ],
        "projections": records,
        "status": "plan only; executable microcode and numerical validation pending",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} projection records to {args.output}")
    print("No executable microcode was generated.")


if __name__ == "__main__":
    main()
