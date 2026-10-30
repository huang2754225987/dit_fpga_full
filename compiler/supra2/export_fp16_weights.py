#!/usr/bin/env python3
"""Export Supra2 checkpoint linear weights in the NPU's FP16 tile layout.

This performs CPU-side serialization only; it does not instantiate or execute
the model. The WRF layout is output-tile-major, then K-tile-major, with each
32x32 tile serialized as 32 K rows of 32 output lanes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .compile_projection_plan import (
        DEFAULT_CHECKPOINT, ROOT, load_state, projection_specs,
    )
except ImportError:
    from compile_projection_plan import (
        DEFAULT_CHECKPOINT, ROOT, load_state, projection_specs,
    )


TILE = 32
WEIGHT_TILE_BYTES = TILE * TILE * 2
WEIGHT_ALIGNMENT = 4096
BIAS_ALIGNMENT = 64


def pack_fp16_weight_tiles(weight) -> bytes:
    """Pack an [out_features, in_features] tensor into hardware WRF order."""
    import torch
    import torch.nn.functional as functional

    if not hasattr(weight, "shape") or len(weight.shape) != 2:
        raise ValueError("weight must be a rank-2 tensor")
    out_features, in_features = map(int, weight.shape)
    if out_features < 1 or in_features < 1:
        raise ValueError("weight dimensions must be positive")

    n_tiles = (out_features + TILE - 1) // TILE
    k_tiles = (in_features + TILE - 1) // TILE
    padded = functional.pad(
        weight.detach().to(device="cpu", dtype=torch.float16),
        (0, k_tiles * TILE - in_features, 0, n_tiles * TILE - out_features),
    ).contiguous()
    tiled = padded.reshape(n_tiles, TILE, k_tiles, TILE)
    # RTL gathers each K row into the 32 systolic output lanes.
    payload = tiled.permute(0, 2, 3, 1).contiguous().numpy()
    packed = payload.astype("<f2", copy=False).tobytes(order="C")
    expected_bytes = n_tiles * k_tiles * WEIGHT_TILE_BYTES
    if len(packed) != expected_bytes:
        raise RuntimeError("internal FP16 weight packing size mismatch")
    return packed


def pack_fp16_bias(bias, padded_features: int) -> bytes:
    """Pack a bias as contiguous 32-lane vectors, zero-padding the final tile."""
    import torch
    import torch.nn.functional as functional

    if not hasattr(bias, "shape") or len(bias.shape) != 1:
        raise ValueError("bias must be a rank-1 tensor")
    if int(bias.numel()) > padded_features or padded_features % TILE:
        raise ValueError("padded bias width must cover the vector and align to 32")
    padded = functional.pad(
        bias.detach().to(device="cpu", dtype=torch.float16),
        (0, padded_features - int(bias.numel())),
    ).contiguous()
    packed = padded.numpy().astype("<f2", copy=False).tobytes(order="C")
    if len(packed) % BIAS_ALIGNMENT:
        raise RuntimeError("internal FP16 bias alignment mismatch")
    return packed


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(checkpoint: Path, output_dir: Path) -> dict:
    state, weights = load_state(checkpoint)
    config = state.get("config", {})
    if not isinstance(config, dict):
        config = {}
    if config.get("patch", 2) != 2 or config.get("ctx_len", 128) != 128:
        raise SystemExit("Only patch=2 and ctx_len=128 are supported")

    tensors = []
    for name, role, input_kind in projection_specs(depth=14):
        weight_key = name + ".weight"
        bias_key = name + ".bias"
        weight, bias = weights.get(weight_key), weights.get(bias_key)
        if (weight is None or not hasattr(weight, "shape") or len(weight.shape) != 2 or
                bias is None or not hasattr(bias, "numel")):
            raise SystemExit(f"Missing or invalid checkpoint projection: {name}")
        out_features, in_features = map(int, weight.shape)
        if int(bias.numel()) != out_features:
            raise SystemExit(f"Bias shape mismatch for {name}")
        tensors.append((name, role, input_kind, weight, bias,
                        out_features, in_features))

    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = output_dir / "weights_fp16.bin"
    bias_path = output_dir / "biases_fp16.bin"
    weight_cursor = 0
    bias_cursor = 0
    records = []
    with weight_path.open("wb") as weight_file, bias_path.open("wb") as bias_file:
        for name, role, input_kind, weight, bias, out_features, in_features in tensors:
            aligned_weight = align_up(weight_cursor, WEIGHT_ALIGNMENT)
            weight_file.write(b"\0" * (aligned_weight - weight_cursor))
            packed_weight = pack_fp16_weight_tiles(weight)
            weight_file.write(packed_weight)
            weight_cursor = aligned_weight + len(packed_weight)

            padded_out = ((out_features + TILE - 1) // TILE) * TILE
            aligned_bias = align_up(bias_cursor, BIAS_ALIGNMENT)
            bias_file.write(b"\0" * (aligned_bias - bias_cursor))
            packed_bias = pack_fp16_bias(bias, padded_out)
            bias_file.write(packed_bias)
            bias_cursor = aligned_bias + len(packed_bias)

            records.append({
                "name": name,
                "role": role,
                "input_kind": input_kind,
                "input_features": in_features,
                "output_features": out_features,
                "padded_input_features": ((in_features + TILE - 1) // TILE) * TILE,
                "padded_output_features": padded_out,
                "k_tiles": (in_features + TILE - 1) // TILE,
                "n_tiles": (out_features + TILE - 1) // TILE,
                "dtype": "fp16",
                "weight_offset": aligned_weight,
                "weight_bytes": len(packed_weight),
                "bias_offset": aligned_bias,
                "bias_bytes": len(packed_bias),
                "weight_key": name + ".weight",
                "bias_key": name + ".bias",
            })

    manifest = {
        "model_id": "SupraLabs/Supra2-IMG",
        "checkpoint": str(checkpoint.resolve()),
        "format": {
            "dtype": "little-endian-fp16",
            "weight_file": weight_path.name,
            "bias_file": bias_path.name,
            "weight_alignment_bytes": WEIGHT_ALIGNMENT,
            "bias_alignment_bytes": BIAS_ALIGNMENT,
            "tile_shape_k_n": [TILE, TILE],
            "tile_order": "n_tile, k_tile, k_row, n_lane",
            "weight_matrix_layout": "pytorch[out_features, in_features]",
        },
        "weight_file_bytes": weight_cursor,
        "bias_file_bytes": bias_cursor,
        "sha256": {
            weight_path.name: sha256_file(weight_path),
            bias_path.name: sha256_file(bias_path),
        },
        "projection_count": len(records),
        "projections": records,
        "status": "serialized for FPGA; RTL numerical validation pending",
    }
    manifest_path = output_dir / "weight_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_checkpoint_export(checkpoint: Path, output_dir: Path) -> int:
    """Verify hashes, sizes, offsets, and every exported tensor byte-for-byte."""
    manifest_path = output_dir / "weight_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Export manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("projection_count") != 118:
        raise SystemExit("Unexpected projection count in export manifest")

    weight_path = output_dir / manifest["format"]["weight_file"]
    bias_path = output_dir / manifest["format"]["bias_file"]
    for path, expected_size in (
        (weight_path, manifest["weight_file_bytes"]),
        (bias_path, manifest["bias_file_bytes"]),
    ):
        if not path.is_file() or path.stat().st_size != expected_size:
            raise SystemExit(f"Export size mismatch: {path}")
        expected_hash = manifest.get("sha256", {}).get(path.name)
        if not expected_hash or sha256_file(path) != expected_hash:
            raise SystemExit(f"Export hash mismatch: {path}")

    _, weights = load_state(checkpoint)
    records = {record["name"]: record for record in manifest["projections"]}
    if len(records) != 118:
        raise SystemExit("Duplicate or missing projection records")
    with weight_path.open("rb") as weight_file, bias_path.open("rb") as bias_file:
        for name, _, _ in projection_specs(depth=14):
            record = records.get(name)
            if record is None:
                raise SystemExit(f"Missing export record for {name}")
            packed_weight = pack_fp16_weight_tiles(weights[name + ".weight"])
            weight_file.seek(record["weight_offset"])
            if weight_file.read(record["weight_bytes"]) != packed_weight:
                raise SystemExit(f"Weight image differs from checkpoint tensor: {name}")
            packed_bias = pack_fp16_bias(
                weights[name + ".bias"], record["padded_output_features"])
            bias_file.seek(record["bias_offset"])
            if bias_file.read(record["bias_bytes"]) != packed_bias:
                raise SystemExit(f"Bias image differs from checkpoint tensor: {name}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_fp16_export")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify an existing export against the checkpoint")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.verify_only:
        count = verify_checkpoint_export(args.checkpoint, args.output_dir)
        print(f"Verified all {count} FP16 projection weights and biases byte-for-byte")
        print("No model forward or Transformer inference was executed.")
        return
    manifest = export_checkpoint(args.checkpoint, args.output_dir)
    print(f"Exported {manifest['projection_count']} FP16 projections to {args.output_dir}")
    print(f"Weight image: {manifest['weight_file_bytes']} bytes")
    print("No model forward or Transformer inference was executed.")


if __name__ == "__main__":
    main()
