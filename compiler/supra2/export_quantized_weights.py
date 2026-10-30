#!/usr/bin/env python3
"""Pack the supplied Supra2 INT8/FP16 export for tiled FPGA weight access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT = ROOT / "build" / "supra2_hw_export" / "export"
DEFAULT_OUTPUT = ROOT / "build" / "supra2_quant_export"
TILE = 32
WEIGHT_TILE_BYTES = TILE * TILE


def _align_file(stream, alignment: int) -> int:
    position = stream.tell()
    aligned = (position + alignment - 1) // alignment * alignment
    if aligned != position:
        stream.write(bytes(aligned - position))
    return aligned


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pad_vector(values: np.ndarray, padded_size: int, pad_value: float) -> bytes:
    padded = np.full((padded_size,), pad_value, dtype="<f4")
    padded[:values.size] = np.asarray(values, dtype="<f4")
    return padded.tobytes()


def export(export_dir: Path, output_dir: Path) -> dict:
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Supra2 package manifest not found: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layer_table = source_manifest["layer_table"]
    records = []
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(export_dir / "weights_quant_int8.npz", allow_pickle=False) as quant, \
            np.load(export_dir / "weights_passthrough_bf16.npz",
                    allow_pickle=False) as passthrough, \
            (output_dir / "weights_int8_tiled.bin").open("wb") as weight_file, \
            (output_dir / "weight_scales_fp32.bin").open("wb") as scale_file, \
            (output_dir / "weight_biases_fp32.bin").open("wb") as bias_file:
        for spec in layer_table:
            name = spec["name"]
            weight = np.asarray(quant[name + ".qw"], dtype=np.int8)
            scales = np.asarray(quant[name + ".ws"], dtype="<f4")
            bias = np.asarray(quant[name + ".bias"], dtype="<f4")
            out_features, in_features = map(int, weight.shape)
            if (out_features, in_features) != (spec["out"], spec["d_in"]):
                raise ValueError(f"manifest/weight shape mismatch for {name}")
            if scales.shape != (out_features,) or bias.shape != (out_features,):
                raise ValueError(f"scale/bias shape mismatch for {name}")

            padded_n = ((out_features + TILE - 1) // TILE) * TILE
            padded_k = ((in_features + TILE - 1) // TILE) * TILE
            n_tiles, k_tiles = padded_n // TILE, padded_k // TILE
            weight_offset = _align_file(weight_file, 4096)
            for n_tile in range(n_tiles):
                n0 = n_tile * TILE
                for k_tile in range(k_tiles):
                    k0 = k_tile * TILE
                    tile = np.zeros((TILE, TILE), dtype=np.int8)
                    n_valid = max(0, min(TILE, out_features - n0))
                    k_valid = max(0, min(TILE, in_features - k0))
                    if n_valid and k_valid:
                        tile[:k_valid, :n_valid] = weight[
                            n0:n0 + n_valid, k0:k0 + k_valid].T
                    weight_file.write(tile.tobytes(order="C"))

            padded_scales = ((out_features + TILE - 1) // TILE) * TILE
            scale_offset = _align_file(scale_file, 64)
            scale_file.write(_pad_vector(scales, padded_scales, 1.0))
            bias_offset = _align_file(bias_file, 64)
            bias_file.write(_pad_vector(bias, padded_scales, 0.0))
            records.append({
                **spec,
                "padded_out": padded_n,
                "padded_in": padded_k,
                "n_tiles": n_tiles,
                "k_tiles": k_tiles,
                "weight_offset": weight_offset,
                "weight_tile_bytes": WEIGHT_TILE_BYTES,
                "weight_bytes": n_tiles * k_tiles * WEIGHT_TILE_BYTES,
                "scale_offset": scale_offset,
                "bias_offset": bias_offset,
                "padded_vector_bytes": padded_scales * 4,
            })

        passthrough_records = []
        with (output_dir / "passthrough_fp16.bin").open("wb") as data_file:
            for name in passthrough.files:
                tensor = np.asarray(passthrough[name], dtype="<f2")
                offset = _align_file(data_file, 64)
                payload = tensor.tobytes(order="C")
                data_file.write(payload)
                passthrough_records.append({
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": "float16",
                    "offset": offset,
                    "bytes": len(payload),
                })

    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in output_dir.glob("*.bin")
    }
    result = {
        "model": source_manifest["model"],
        "quantization_recipe": source_manifest["recipe"],
        "tile": [TILE, TILE],
        "weight_tile_layout": "[K,N] signed int8, row-major within each [N_tile,K_tile]",
        "weight_tile_order": "N tile, then K tile",
        "metadata_layout": "FP32 per-output-channel scale and bias; vectors padded to 32 lanes",
        "quantized_layers": records,
        "passthrough_tensors": passthrough_records,
        "files": files,
        "source_export": str(export_dir.resolve()),
    }
    (output_dir / "weight_image_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _unpack_layer(weight_file, record: dict) -> np.ndarray:
    n, k = int(record["out"]), int(record["d_in"])
    result = np.zeros((n, k), dtype=np.int8)
    with weight_file.open("rb") as stream:
        for n_tile in range(record["n_tiles"]):
            n0 = n_tile * TILE
            for k_tile in range(record["k_tiles"]):
                k0 = k_tile * TILE
                offset = (record["weight_offset"] +
                          (n_tile * record["k_tiles"] + k_tile) * WEIGHT_TILE_BYTES)
                stream.seek(offset)
                tile = np.frombuffer(stream.read(WEIGHT_TILE_BYTES), dtype=np.int8)
                tile = tile.reshape(TILE, TILE)
                n_valid = min(TILE, n - n0)
                k_valid = min(TILE, k - k0)
                result[n0:n0 + n_valid, k0:k0 + k_valid] = \
                    tile[:k_valid, :n_valid].T
    return result


def verify(export_dir: Path, output_dir: Path) -> dict:
    image_manifest = json.loads(
        (output_dir / "weight_image_manifest.json").read_text(encoding="utf-8"))
    files = {name: output_dir / name for name in image_manifest["files"]}
    for name, record in image_manifest["files"].items():
        path = files[name]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"packed artifact changed: {name}")

    compared = 0
    with np.load(export_dir / "weights_quant_int8.npz", allow_pickle=False) as quant, \
            np.load(export_dir / "weights_passthrough_bf16.npz",
                    allow_pickle=False) as passthrough:
        for record in image_manifest["quantized_layers"]:
            name = record["name"]
            expected_weight = quant[name + ".qw"]
            actual_weight = _unpack_layer(files["weights_int8_tiled.bin"], record)
            if not np.array_equal(actual_weight, expected_weight):
                raise ValueError(f"INT8 tiled weight round-trip failed: {name}")
            with files["weight_scales_fp32.bin"].open("rb") as stream:
                stream.seek(record["scale_offset"])
                actual_scale = np.frombuffer(
                    stream.read(record["padded_vector_bytes"]), dtype="<f4")
            with files["weight_biases_fp32.bin"].open("rb") as stream:
                stream.seek(record["bias_offset"])
                actual_bias = np.frombuffer(
                    stream.read(record["padded_vector_bytes"]), dtype="<f4")
            n = record["out"]
            if (not np.array_equal(actual_scale[:n], quant[name + ".ws"]) or
                    not np.array_equal(actual_bias[:n], quant[name + ".bias"]) or
                    np.any(actual_scale[n:] != 1.0) or np.any(actual_bias[n:] != 0.0)):
                raise ValueError(f"FP32 scale/bias round-trip failed: {name}")
            compared += 1

        pass_file = files["passthrough_fp16.bin"]
        with pass_file.open("rb") as stream:
            for record in image_manifest["passthrough_tensors"]:
                stream.seek(record["offset"])
                expected = np.asarray(passthrough[record["name"]], dtype="<f2")
                actual = np.frombuffer(stream.read(record["bytes"]), dtype="<f2")
                if not np.array_equal(actual, expected.reshape(-1)):
                    raise ValueError(
                        f"FP16 passthrough round-trip failed: {record['name']}")

    return {"quantized_layers_verified": compared,
            "passthrough_tensors_verified": len(image_manifest["passthrough_tensors"]),
            "status": "pass"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.export_dir, args.output_dir)
    else:
        manifest = export(args.export_dir, args.output_dir)
        result = verify(args.export_dir, args.output_dir)
        print(f"Packed {len(manifest['quantized_layers'])} INT8 layers and "
              f"{len(manifest['passthrough_tensors'])} FP16 tensors")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
