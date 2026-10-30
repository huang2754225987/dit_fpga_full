#!/usr/bin/env python3
"""Inspect a local Supra2-IMG checkpoint without changing it."""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="model/model_final_ema.pt",
        help="local model_final_ema.pt path",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download it with the official inference.py first."
        )

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for checkpoint inspection") from exc

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {args.checkpoint}")
    print(f"size_bytes: {os.path.getsize(args.checkpoint)}")
    if isinstance(state, dict):
        print(f"top_level_keys: {sorted(state.keys())}")
        config = state.get("config")
        if isinstance(config, dict):
            print(f"config_fields: {sorted(config.keys())}")
            for key in ("patch", "ctx_len"):
                if key in config:
                    print(f"config.{key}: {config[key]}")
        weights = state.get("ema", state.get("model", state))
    else:
        weights = state

    if not isinstance(weights, dict):
        raise SystemExit("Unsupported checkpoint payload: expected a state dict")

    tensors = [value for value in weights.values() if hasattr(value, "numel")]
    print(f"tensor_count: {len(tensors)}")
    print(f"parameter_count: {sum(int(value.numel()) for value in tensors)}")
    print("first_keys:")
    for key in list(weights)[:20]:
        value = weights[key]
        shape = tuple(value.shape) if hasattr(value, "shape") else "<non-tensor>"
        print(f"  {key}: {shape}")


if __name__ == "__main__":
    main()
