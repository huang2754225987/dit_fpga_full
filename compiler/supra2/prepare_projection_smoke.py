#!/usr/bin/env python3
"""Prepare a real-checkpoint x_embed projection run for the FPGA RTL top.

The generated test uses synthetic latent tokens and the real checkpoint's
x_embed weights and bias. It runs the NPU top in FakeDDR simulation; it
does not execute the Transformer/model forward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .compile_projection_plan import DEFAULT_CHECKPOINT, ROOT, load_state
    from .export_fp16_weights import pack_fp16_weight_tiles
except ImportError:
    from compile_projection_plan import DEFAULT_CHECKPOINT, ROOT, load_state
    from export_fp16_weights import pack_fp16_weight_tiles

from compiler.compiler_backend_py import isa, static_linear


ACTIVATION_DDR_BASE = 0x00410000
OUTPUT_DDR_BASE = 0x00080000
ACTIVATION_L1_BASE = 0x4000
OUTPUT_PAGE = 1
TOKEN_COUNT = 256
BATCH_TOKENS = 32
INPUT_FEATURES = 16
PADDED_INPUT_FEATURES = 32
INPUT_SLOT_BYTES = 256
OUTPUT_FEATURES = 576
OUTPUT_ROW_BYTES = OUTPUT_FEATURES * 2


def build_projection_smoke_program(weight_offset: int = 0,
                                  bias_offset: int = 0,
                                  tokens: int = TOKEN_COUNT) -> list:
    """Compile the complete 256-token x_embed projection in M=32 batches."""
    if weight_offset < 0 or weight_offset % 16:
        raise ValueError("weight offset must be non-negative and 16-byte aligned")
    if tokens < 1 or tokens % BATCH_TOKENS:
        raise ValueError("smoke token count must be a positive multiple of 32")
    batch_count = tokens // BATCH_TOKENS
    if batch_count > 16:
        raise ValueError("the current DMA output-window encoding supports 16 batches")
    region = static_linear.LinearRegion(
        weight=weight_offset,
        scale_zero=None,
        rows=OUTPUT_FEATURES,
        cols=PADDED_INPUT_FEATURES,
        nbits=16,
        tile_bytes=2048,
        groups=0,
    )
    if region.rows != OUTPUT_FEATURES or region.cols != PADDED_INPUT_FEATURES:
        raise AssertionError("unexpected x_embed hardware geometry")

    program = list(isa.static_dma(bias_offset, l1_bank=0, fp16=True))
    program.append(isa.barrier())
    for batch_index, first_token in enumerate(range(0, tokens, BATCH_TOKENS)):
        program.extend([
            isa.token_loop_begin(BATCH_TOKENS),
            isa.dma_rd_token_group(
                first_token * INPUT_SLOT_BYTES,
                PADDED_INPUT_FEATURES,
                ACTIVATION_L1_BASE >> 12,
                BATCH_TOKENS,
            ),
            isa.token_loop_end(),
        ])
        program.extend(static_linear.emit_fp16_batch(
            region,
            activation_base=ACTIVATION_L1_BASE,
            output_base=0,
            tokens=BATCH_TOKENS,
            activation_page=0,
            output_page=OUTPUT_PAGE,
            bias_base=0,
            bias_page=0,
            activation_row_stride_bytes=INPUT_SLOT_BYTES,
            output_row_stride_bytes=OUTPUT_ROW_BYTES,
            wait_after_store=True,
        ))
        # One 64-KiB DDR output window per token batch. The 32x576 matrix is
        # exactly 36 1-KiB writes, with no tail bytes or row padding.
        program.append(isa.barrier())
        for offset in range(0, BATCH_TOKENS * OUTPUT_ROW_BYTES, 4096):
            size = min(4096, BATCH_TOKENS * OUTPUT_ROW_BYTES - offset)
            program.append(isa.dma_wr_linear_burst(
                offset, size, output_window=batch_index, page=OUTPUT_PAGE))
        # DMA_WR is scoreboard-serialized. One fence after the complete batch
        # is sufficient before the next batch reuses the output page.
        program.append(isa.barrier())
    program.append(isa.end())
    return program


def _write_u16_tensor(path: Path, tensor) -> None:
    import torch

    raw = tensor.detach().to(device="cpu", dtype=torch.float16).contiguous()
    path.write_bytes(raw.numpy().astype("<f2", copy=False).tobytes(order="C"))


def prepare(checkpoint: Path, output_dir: Path) -> dict:
    import torch

    _, weights = load_state(checkpoint)
    weight = weights.get("x_embed.weight")
    bias = weights.get("x_embed.bias")
    if weight is None or tuple(map(int, weight.shape)) != (OUTPUT_FEATURES, INPUT_FEATURES):
        raise SystemExit("Expected x_embed.weight with shape [576, 16]")
    if bias is None or tuple(map(int, bias.shape)) != (OUTPUT_FEATURES,):
        raise SystemExit("Expected x_embed.bias with shape [576]")

    # Deterministic patch-token-like inputs, padded to the hardware's K=32.
    indices = torch.arange(TOKEN_COUNT * INPUT_FEATURES, dtype=torch.int32)
    activations = ((indices.reshape(TOKEN_COUNT, INPUT_FEATURES) % 31) - 15)
    activations = activations.to(torch.float32).div(32).to(torch.float16)
    activation_slots = torch.zeros((TOKEN_COUNT, INPUT_SLOT_BYTES // 2),
                                   dtype=torch.float16)
    activation_slots[:, :INPUT_FEATURES] = activations

    padded_weight_image = pack_fp16_weight_tiles(weight)
    bias_offset = (len(padded_weight_image) + 4095) & ~4095
    padded_bias_image = bytearray(4096)
    bias_bytes = bias.detach().to(device="cpu", dtype=torch.float16).contiguous()
    padded_bias_image[:bias_bytes.numel() * 2] = bias_bytes.numpy().astype(
        "<f2", copy=False).tobytes()
    program = build_projection_smoke_program(bias_offset=bias_offset)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / "x_embed_projection_image.bin"
    with image_path.open("wb") as image:
        image.write(padded_weight_image)
        image.seek(bias_offset)
        image.write(padded_bias_image)
        image.seek(ACTIVATION_DDR_BASE)
        image.write(activation_slots.numpy().astype("<f2", copy=False).tobytes())

    _write_u16_tensor(output_dir / "x_embed_projection_input.bin", activations)
    # A single linear-layer reference is used for numerical checking; this is
    # not a model/Transformer forward pass. FP32 accumulation mirrors the RTL.
    accumulator = (activations.to(torch.float32) @
                   weight.detach().to(device="cpu", dtype=torch.float32).transpose(0, 1))
    # AC2V materializes FP16 before the vector ADD applies the FP16 bias.
    golden = accumulator.to(torch.float16).to(torch.float32) + bias_bytes.to(torch.float32)
    _write_u16_tensor(output_dir / "x_embed_projection_golden.bin", golden)

    prefix = output_dir / "supra2_x_embed_smoke"
    (output_dir / f"{prefix.name}.stage00.bin").write_bytes(
        b"".join(instruction.bytes_le() for instruction in program))
    # This generic NPU testbench requires a descriptor-table image even though
    # the program contains no routing or expert instructions.
    (output_dir / "unused_expert_descriptors.bin").write_bytes(bytes(180 * 12))

    manifest = {
        "model_id": "SupraLabs/Supra2-IMG",
        "projection": "x_embed",
        "checkpoint": str(checkpoint.resolve()),
        "uses_real_checkpoint_weights": True,
        "uses_synthetic_activation_tokens": True,
        "bias_applied": True,
        "input_shape": [TOKEN_COUNT, INPUT_FEATURES],
        "token_batch": BATCH_TOKENS,
        "batch_count": TOKEN_COUNT // BATCH_TOKENS,
        "padded_input_features": PADDED_INPUT_FEATURES,
        "output_shape": [TOKEN_COUNT, OUTPUT_FEATURES],
        "activation_ddr_base": ACTIVATION_DDR_BASE,
        "weight_ddr_base": 0,
        "output_ddr_base": OUTPUT_DDR_BASE,
        "output_window_bytes": 65536,
        "weight_image_bytes": len(padded_weight_image),
        "bias_ddr_offset": bias_offset,
        "bias_image_bytes": len(padded_bias_image),
        "instruction_count": len(program),
        "testbench": "rtl/sim/tb_layer0_fake_ddr.sv",
        "golden_file": "x_embed_projection_golden.bin",
        "status": "prepared; run the FakeDDR testbench and verifier separately",
    }
    (output_dir / "projection_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "build" / "supra2_projection_smoke")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    manifest = prepare(args.checkpoint, args.output_dir)
    print(f"Prepared {manifest['instruction_count']} instructions for x_embed")
    print(f"Activation DDR base: 0x{manifest['activation_ddr_base']:08x}")
    print("No Transformer/model forward was executed.")


if __name__ == "__main__":
    main()
