"""Manifest-driven real-weight Layer-0 machine-code lowering."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from . import (attention, isa, normalization, routed_moe, shared_expert,
                   static_linear, static_linear_fp16_batch,
                   static_linear_quant_batch, static_linear_quant_ops)
    from .static_linear_fp16_batch_validate import tiles_per_gemm
except ImportError:
    import attention
    import isa
    import normalization
    import routed_moe
    import shared_expert
    import static_linear
    import static_linear_fp16_batch
    import static_linear_quant_batch
    import static_linear_quant_ops
    from static_linear_fp16_batch_validate import tiles_per_gemm

TILE = 32
VECTOR_BYTES = TILE * 2
MAX_PREFILL_INPUT_TOKENS = 256  # DMA_RD_TOKEN_GROUP has a 1-MiB byte window.

# Layer-local L1 allocation. Regions are deliberately reused between phases.
# Keep the input residual in bank 1; mixed-precision K loops ping-pong weights
# through banks 2 and 3 for the lifetime of this residual.
L1_WEIGHT = 0x1000
L1_RESIDUAL_IN = L1_WEIGHT
L1_NORM = 0x4000
L1_ATTN_VALUE = 0x5000
L1_ATTN_PROJ = 0x6000
L1_NORM1_GAMMA = 0x7000
L1_SHARED_GATE = 0x5000
L1_SHARED_UP = 0x8000
L1_NORM2_GAMMA = 0x9000
L1_V_BIAS = 0xA000
L1_ROUTED_OUT = 0xC000
L1_ATTN_RESIDUAL = 0xD000
L1_FINAL = 0xE000
L1_O_BIAS = 0xF000
L1_ROUTER_BIAS = 0x0000
L1_SCALAR = 0x3FC0
# Page 0 keeps the DMA token scratch and up to eight 4-KiB source rows. Page 1
# holds the packed activation matrix and a token-window-aligned output region.
L1_PREFILL_GROUP = 0x4000
PREFILL_GROUP_ROWS = 8
PREFILL_TOKEN_SLOT_BYTES = 4096
PREFILL_OUTPUT_WINDOW_BYTES = 65536
PREFILL_HEAD_Q_BASE = 0x8000
PREFILL_HEAD_K_BASE = 0xA000
PREFILL_HEAD_V_BASE = 0xC000
PREFILL_HEAD_PAIR_V_BASE = 0x4000
PREFILL_HEAD_SCORE_BASE = 0xE000
PREFILL_HEAD_OUTPUT_BASE = 0xE040
PREFILL_HEAD_BYTES = 128 * 2
PREFILL_ROPE_BASE = 0x0000
PREFILL_ROPE_ROW_BYTES = PREFILL_HEAD_BYTES
PREFILL_QUERY_PACK_BASE = 0x3000
DECODE_K_CACHE_L1_BASE = 0x4000
DECODE_V_CACHE_L1_BASE = 0x8000
DECODE_V_CURRENT_L1_BASE = 0x7000
DECODE_SCORE_BASE = 0xE000
DECODE_ATTENTION_OUT_BASE = 0xC000
DECODE_QUERY_BASE = 0xD000
DECODE_HEAD_PAIR_BYTES = 2 * PREFILL_HEAD_BYTES


def _default_prefill_attention_rows(tokens):
    """Use the measured full M tile only when at least 17 query rows exist."""
    return 32 if tokens > 16 else 16


def _tensor_name(manifest, suffix):
    name = f"model.layers.{int(manifest['layer'])}.{suffix}"
    if name not in manifest["inventory"]:
        raise KeyError(f"manifest inventory is missing {name}")
    return name


def _regions_for(manifest, tensor_name):
    return {
        region["role"]: region
        for region in manifest["regions"]
        if region.get("source_tensor") == tensor_name
    }


def _linear_region(manifest, suffix):
    name = _tensor_name(manifest, suffix)
    spec = manifest["inventory"][name]
    regions = _regions_for(manifest, name)
    if spec["kind"] not in ("quantized_linear", "fp16_linear"):
        raise ValueError(f"{name} is not a linear weight")
    scale = regions.get("scale_zero_fp16")
    bias = regions.get("bias_fp16")
    rows, cols = (int(value) for value in spec["shape"])
    return static_linear.LinearRegion(
        weight=int(regions["npu_wrf_quantized" if spec["nbits"] != 16
                           else "npu_wrf_fp16"]["byte_base"]),
        scale_zero=None if scale is None else int(scale["byte_base"]),
        rows=rows,
        cols=cols,
        nbits=int(spec["nbits"]),
        tile_bytes=int(spec["wrf_tile_bytes"]),
        groups=int(spec.get("groups", 0)),
        bias=None if bias is None else int(bias["byte_base"]),
    )


def _routed_intermediate_size(manifest):
    """Resolve routed-expert width independently of shared-expert width."""
    dimensions = manifest.get("demo_dimensions", {})
    configured = dimensions.get("routed_intermediate")
    if configured is not None:
        return int(configured)

    gate_name = _tensor_name(manifest, "mlp.experts.0.gate_proj")
    up_name = _tensor_name(manifest, "mlp.experts.0.up_proj")
    down_name = _tensor_name(manifest, "mlp.experts.0.down_proj")
    inventory = manifest["inventory"]
    gate_shape = tuple(map(int, inventory[gate_name]["shape"]))
    up_shape = tuple(map(int, inventory[up_name]["shape"]))
    down_shape = tuple(map(int, inventory[down_name]["shape"]))
    if (len(gate_shape) != 2 or up_shape != gate_shape or
            down_shape != (gate_shape[1], gate_shape[0])):
        raise ValueError("routed expert Gate/Up/Down dimensions are inconsistent")
    return gate_shape[0]


def _vector_region(manifest, suffix):
    name = _tensor_name(manifest, suffix)
    regions = _regions_for(manifest, name)
    return int(regions["raw_fp16"]["byte_base"])


def _linear(region, act_base, out_base, bias_base=None,
            residual_base=None, rmsnorm_accumulate=False,
            configure_cfg_k=True, restore_cfg_k=True,
            wait_postproc_before_gld_a=False, output_page=0,
            input_page=0):
    if region.nbits == 16:
        return static_linear.emit_fp16(
            region, act_base, out_base, bias_base=bias_base,
            residual_base=residual_base,
            rmsnorm_accumulate=rmsnorm_accumulate,
            configure_cfg_k=configure_cfg_k,
            restore_cfg_k=restore_cfg_k,
            wait_postproc_before_gld_a=wait_postproc_before_gld_a,
            output_page=output_page,input_page=input_page)
    if wait_postproc_before_gld_a:
        raise ValueError("marked GLD.A waits are only supported for FP16 projections")
    if residual_base is not None or rmsnorm_accumulate:
        raise ValueError("fused residual/RMSNorm epilogues require FP16 weights")
    wide_group_bytes = static_linear_quant_ops.k128_weight_group_bytes(region)
    wide_quantized_k = (
        wide_group_bytes != 0 and region.k_tiles % 4 == 0 and
        region.weight % wide_group_bytes == 0)
    program = ([isa.set_cfg_k(32)]
               if configure_cfg_k and not wide_quantized_k else [])
    program.extend(static_linear.emit_quantized(
        region, act_base, out_base, output_page=output_page,
        input_page=input_page)
    )
    if bias_base is not None:
        _add_bias(program, region, out_base,
                  preloaded_bias_base=bias_base,
                  output_page=output_page)
    if (restore_cfg_k and not configure_cfg_k and
            not wide_quantized_k):
        program.append(isa.set_cfg_k(32))
    return program


def _can_chain_fp16_projections(first, second):
    if first.nbits != 16 or second.nbits != 16:
        return False
    return tiles_per_gemm(first) == tiles_per_gemm(second)


def _can_defer_rmsnorm_fence(consumer_region):
    """Only the FP16 static emitter supports the marked first-GLD.A wait."""
    return consumer_region.nbits == 16


def _prepare_projection_k(program, region, current_k=None):
    """Select the projection's K geometry before its first activation read."""
    gemm_k = (32 * tiles_per_gemm(region) if region.nbits == 16 else 32)
    if current_k is None or current_k != gemm_k:
        program.append(isa.set_cfg_k(gemm_k))
    return gemm_k


def _load_4k(ddr_addr, l1_bank, wait=True):
    program = isa.static_dma(ddr_addr, l1_bank=l1_bank, fp16=True)
    if wait:
        program.append(isa.barrier())
    return program


def _add_bias(program, region, value_base, scratch_bank=2,
              preloaded_bias_base=None, wait_after_store=True,
              output_page=0):
    if region.bias is None:
        return
    if preloaded_bias_base is None:
        program.extend(_load_4k(region.bias, scratch_bank))
        bias_base = scratch_bank * 4096
    else:
        bias_base = preloaded_bias_base
    for tile in range(region.n_tiles):
        offset = tile * VECTOR_BYTES
        if output_page == 0:
            program.extend([
                isa.vld(0, value_base + offset),
                isa.vld(1, bias_base + offset),
                isa.add(0, 0, 1),
                isa.vst(0, value_base + offset),
            ])
        else:
            program.extend([
                isa.vld_page(0, value_base + offset, output_page),
                isa.vld(1, bias_base + offset),
                isa.add(0, 0, 1),
                isa.vst_page(0, value_base + offset, output_page),
            ])
    if wait_after_store:
        program.append(isa.barrier())


def _add_bias_residual(program, region, value_base, residual_base, out_base,
                       hidden_size, scratch_bank=2,
                       preloaded_bias_base=None):
    """Fuse ``(value + bias) + residual`` and write each tile only once."""
    if region.bias is None:
        program.extend(_add_vectors(
            value_base, residual_base, out_base, hidden_size))
        return

    if preloaded_bias_base is None:
        program.extend(_load_4k(region.bias, scratch_bank))
        bias_base = scratch_bank * 4096
    else:
        bias_base = preloaded_bias_base
    for offset in range(0, hidden_size * 2, VECTOR_BYTES):
        program.extend([
            isa.vld(0, value_base + offset),
            isa.vld(1, bias_base + offset),
            isa.add(0, 0, 1),
            isa.vld(1, residual_base + offset),
            isa.add(0, 0, 1),
            isa.vst(0, out_base + offset),
        ])
    program.append(isa.barrier())


def _add_vectors(lhs_base, rhs_base, out_base, hidden_size):
    program = []
    for offset in range(0, hidden_size * 2, VECTOR_BYTES):
        program.extend([
            isa.vld(0, lhs_base + offset),
            isa.vld(1, rhs_base + offset),
            isa.add(0, 0, 1),
            isa.vst(0, out_base + offset),
        ])
    program.append(isa.barrier())
    return program


def load_manifest(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != "npu-real-layer-ddr-manifest-v1":
        raise ValueError("unsupported real-layer manifest schema")
    return manifest


def emit_batched_fp16_projection(manifest, suffix, activation_base,
                                 output_base, tokens, activation_page=0,
                                 output_page=0):
    """Emit one shared-weight projection from compact K-tile activation tiles."""
    region = _linear_region(manifest, suffix)
    if region.nbits != 16:
        raise ValueError("batched static projection requires FP16 weights")
    return static_linear.emit_fp16_batch(
        region, activation_base, output_base, tokens,
        activation_page=activation_page, output_page=output_page,
        activation_row_stride_bytes=4096)


def _prefill_group_layout(first_token, remaining_tokens):
    """Place compact activations on page 0 and aligned outputs on page 1."""
    token_in_window = first_token % 16
    output_base = token_in_window * PREFILL_TOKEN_SLOT_BYTES
    max_rows = min(PREFILL_GROUP_ROWS, remaining_tokens, 16 - token_in_window)
    activation_end = L1_PREFILL_GROUP + max_rows * PREFILL_TOKEN_SLOT_BYTES
    if max_rows < 1 or activation_end > PREFILL_OUTPUT_WINDOW_BYTES:
        raise ValueError("prefill activation group exceeds page-0 capacity")
    return max_rows, L1_PREFILL_GROUP, output_base


def emit_prefill_fp16_projection(manifest, suffix, tokens=32,
                                 input_start_token=0,
                                 output_window_base=0,
                                 output_start_token=None,
                                 input_source_windows=None,
                                 configure_shared_cfg=True,
                                 restore_shared_cfg=True):
    """Emit a 2048-wide, token-batched FP16 projection into OUT DDR.

    Up to eight tokens are gathered at a time and kept in compact token-major
    rows. Batched GLD.A transposes those rows directly into the ARF, and one
    shared weight traversal serves each output tile. This is intended for static
    attention projections; routed experts remain on their descriptor-driven
    mixed-precision path. Inputs come from ACT DDR by default; OUT source
    entries may be a window number or ``(window, first_token)`` pair so the
    projection can read 8-row slices from 16-row attention output windows.

    The caller must configure ACT/OUT DDR bases and provide contiguous 4096-
    byte FP16 activation/output token slots. The emitter deliberately rejects
    reduced dimensions because it uses four fixed 1-KiB legacy DMA writes per
    result row.
    """
    region = _linear_region(manifest, suffix)
    if region.nbits not in (1, 2, 3, 4, 16) or (
            region.rows, region.cols) != (2048, 2048):
        raise ValueError(
            "batched prefill projection requires a supported 2048x2048 matrix")
    if not 1 <= tokens <= 32:
        raise ValueError("batched prefill projection supports 1..32 tokens")
    if output_start_token is None:
        output_start_token = input_start_token
    if not 0 <= input_start_token <= 0xFFFF:
        raise ValueError("input_start_token must fit the token activation address")
    if (input_source_windows is None and
            input_start_token + tokens > MAX_PREFILL_INPUT_TOKENS):
        raise ValueError(
            "input token range exceeds the 256-token activation DMA window")
    if not 0 <= output_start_token <= 32 - tokens:
        raise ValueError("output token range must fit the 32-token activation buffer")
    normalized_input_groups = None
    if input_source_windows is not None:
        expected_groups = tokens // PREFILL_GROUP_ROWS
        if (input_start_token != 0 or output_start_token % 16 != 0 or
                tokens % PREFILL_GROUP_ROWS != 0 or
                len(input_source_windows) != expected_groups or
                any(not 0 <= (group[0] if isinstance(group, (tuple, list))
                              else group) < 16
                    for group in input_source_windows)):
            raise ValueError(
                "OUT-sourced prefill projections require aligned full 8-row groups")
        normalized_input_groups = []
        for group in input_source_windows:
            if isinstance(group, (tuple, list)):
                if len(group) != 2:
                    raise ValueError(
                        "OUT source groups must be a window or (window, first_token)")
                source_window, source_first_token = map(int, group)
            else:
                source_window, source_first_token = int(group), 0
            if (not 0 <= source_window < 16 or
                    source_first_token < 0 or
                    source_first_token + PREFILL_GROUP_ROWS > 16):
                raise ValueError("OUT source token subrange exceeds its 16-row window")
            normalized_input_groups.append((source_window, source_first_token))
    last_output_window = output_window_base + (output_start_token + tokens - 1) // 16
    if output_window_base < 0 or last_output_window >= 16:
        raise ValueError("prefill projection output windows must fit OUT DDR")

    program = []
    bias_base = None
    if region.bias is not None:
        program.extend(_load_4k(region.bias, l1_bank=2))
        bias_source = 2 * 4096
        for offset in range(0, 4096, VECTOR_BYTES):
            program.extend([
                isa.vld(0, bias_source + offset),
                isa.vst_page(0, offset, page=2),
            ])
        program.append(isa.barrier())
        bias_base = 0

    group_start = 0
    grouped_output_dma = output_start_token % 16 == 0
    output_window_start = output_start_token
    output_window_rows = 0
    # A token range aligned to eight rows is emitted as full 8-row groups.
    # Share the globally-idle configuration writes across those groups; short
    # tails retain the standalone emitter's per-group M setting.
    shared_group_cfg = (output_start_token % PREFILL_GROUP_ROWS == 0 and
                        tokens % PREFILL_GROUP_ROWS == 0)
    if shared_group_cfg:
        group_k_tiles = tiles_per_gemm(region)
        if configure_shared_cfg:
            program.extend([isa.set_cfg_m(PREFILL_GROUP_ROWS),
                            isa.set_cfg_k(32 * group_k_tiles)])
    while group_start < tokens:
        first_input_token = input_start_token + group_start
        first_token = output_start_token + group_start
        rows, activation_base, output_base = _prefill_group_layout(
            first_token, tokens - group_start)
        if input_source_windows is None:
            source_dma = isa.dma_rd_token_group(
                first_input_token * 4096, 2048,
                l1_bank=activation_base >> 12, tokens=rows)
        else:
            source_window, source_first_token = normalized_input_groups[
                group_start // PREFILL_GROUP_ROWS]
            source_dma = isa.dma_rd_out_token_group(
                (source_window * PREFILL_OUTPUT_WINDOW_BYTES +
                 source_first_token * PREFILL_TOKEN_SLOT_BYTES),
                PREFILL_TOKEN_SLOT_BYTES,
                l1_bank=activation_base >> 12,
                tokens=rows)
        program.extend([isa.token_loop_begin(rows), source_dma,
                        isa.token_loop_end()])
        # DMA_RD_TOKEN_GROUP writes each full token directly at
        # activation_base + token_iter * 4096, in the compact row layout
        # consumed by batched transpose GLD.A.
        batch_options = dict(
            activation_page=0, output_page=1,
            activation_row_stride_bytes=PREFILL_TOKEN_SLOT_BYTES,
            bias_base=bias_base, bias_page=2,
            configure_cfg=not shared_group_cfg,
            restore_cfg=not shared_group_cfg,
            # Aligned output is retired by the following all-idle token loop;
            # the misaligned per-row DMA path needs an explicit producer fence.
            wait_after_store=not grouped_output_dma)
        if region.nbits == 16:
            projection_program = static_linear.emit_fp16_batch(
                region, activation_base, output_base, rows, **batch_options)
        else:
            projection_program = static_linear_quant_batch.emit_quantized_batch(
                region, activation_base, output_base, rows, **batch_options)
        program.extend(projection_program)

        # An aligned 16-token output window occupies one contiguous 64-KiB
        # L1 page. Defer its writeback until both 8-row compute groups are
        # present, then use one token-strided DMA command per row instead of
        # four 1-KiB commands. Misaligned subranges keep the legacy offsets.
        last_token = first_token + rows - 1
        output_window = first_token // 16
        if output_window >= 16 or last_token // 16 != output_window:
            raise ValueError("prefill projection group crosses a DDR output window")
        if grouped_output_dma:
            if output_base != (first_token % 16) * PREFILL_TOKEN_SLOT_BYTES:
                raise ValueError("prefill grouped output row base is misaligned")
            output_window_rows += rows
        else:
            for row in range(rows):
                token = first_token + row
                for chunk in range(4):
                    output_imm = (token % 16) * 4096 + chunk * 1024
                    source = (output_base + row * PREFILL_TOKEN_SLOT_BYTES +
                              chunk * 1024)
                    if source != output_imm:
                        raise ValueError("prefill output L1 row is not DDR-window aligned")
                    program.append(isa.dma_wr(
                        output_imm,
                        output_window=output_window_base + output_window,
                        page=1))

        group_start += rows
        window_end_token = output_start_token + group_start
        if (grouped_output_dma and
                (window_end_token % 16 == 0 or group_start == tokens)):
            program.extend(isa.dma_wr_group_tokens(
                0, 2048, output_window_rows,
                output_window=(output_window_base +
                               output_window_start // 16)))
            output_window_start = window_end_token
            output_window_rows = 0

    if shared_group_cfg and restore_shared_cfg:
        program.extend([isa.set_cfg_m(1), isa.set_cfg_k(32)])
    # The aligned grouped writer retires at TOKEN_END. Per-row DMA_WR writes
    # for a misaligned range are asynchronous, so fence the final write before
    # a caller reuses page 1 for another projection.
    if not grouped_output_dma:
        program.append(isa.barrier())
    program.append(isa.end())
    if len(program) > 262144:
        raise ValueError(f"batched prefill projection has {len(program)} instructions")
    return program


def emit_prefill_input_rmsnorm(manifest, tokens=32, input_start_token=0,
                               output_window_base=12):
    """Normalize ACT token rows and write compact 16-token OUT windows.

    This is the full-hidden input RMSNorm prologue for prefill attention.
    The returned fragment has no END so it can be composed with following
    stages. Each output window is consumed only after its token loop retires.
    """
    if (not 1 <= tokens <= 32 or
            not 0 <= input_start_token <= 32 - tokens):
        raise ValueError("prefill RMSNorm input range must fit 1..32 tokens")
    output_window_count = (tokens + 15) // 16
    if (output_window_base < 0 or
            output_window_base + output_window_count > 16):
        raise ValueError("prefill RMSNorm outputs exceed OUT DDR windows")

    gamma = _vector_region(manifest, "input_layernorm")
    program = _load_4k(gamma, l1_bank=L1_NORM1_GAMMA >> 12)
    group_start = 0
    while group_start < tokens:
        group_tokens = min(16, tokens - group_start)
        first_token = input_start_token + group_start
        program.extend([
            isa.token_loop_begin(group_tokens),
            isa.dma_rd_token(first_token * PREFILL_TOKEN_SLOT_BYTES,
                             hidden_size=2048, l1_bank=1),
            isa.barrier(),
        ])
        program.extend(normalization.emit_rmsnorm(
            L1_RESIDUAL_IN, L1_NORM1_GAMMA, L1_NORM, hidden_size=2048))
        program.extend([
            isa.dma_wr_token(
                L1_NORM, hidden_size=2048,
                output_window=output_window_base + group_start // 16),
            isa.token_loop_end(),
        ])
        group_start += group_tokens
    if len(program) > 262144:
        raise ValueError(f"prefill input RMSNorm has {len(program)} instructions")
    return program


def emit_prefill_qkv_projections(manifest, tokens=32, input_start_token=0,
                                 first_output_window=0,
                                 input_source_windows=None,
                                 output_start_token=None):
    """Emit batched Q/K/V projections into three non-overlapping OUT regions.

    Each 2048-wide FP16 projection occupies two 64-KiB output windows for a
    32-token sequence. The returned program reuses the same ACT input range
    and assigns Q, K, and V consecutive two-window regions beginning at
    ``first_output_window``. ACT must contain input-RMSNorm outputs. The Q/K
    manifest entries must declare ``qwen_neox_rope_interleaved``: the DDR
    packer has already reordered each head's projection rows into the adjacent
    pair layout consumed by the hardware RoPE engine. Q/K still require the
    position-specific RoPE operation before attention consumes them. The
    output remains token-major within each projection.
    """
    if not 1 <= tokens <= 32:
        raise ValueError("prefill QKV supports 1..32 tokens")
    if (input_start_token < 0 or
            input_start_token + tokens > MAX_PREFILL_INPUT_TOKENS):
        raise ValueError(
            "prefill QKV input range must fit the 256-token activation DMA window")
    if output_start_token is None:
        output_start_token = input_start_token
    if not 0 <= output_start_token <= 32 - tokens:
        raise ValueError("prefill QKV output rows must fit a 32-token chunk")
    if not 0 <= first_output_window <= 10:
        raise ValueError("prefill QKV needs six consecutive OUT windows")

    for suffix in ("self_attn.q_proj", "self_attn.k_proj"):
        tensor_name = _tensor_name(manifest, suffix)
        inventory = manifest["inventory"][tensor_name]
        if (inventory.get("npu_output_layout") !=
                "qwen_neox_rope_interleaved" or
                int(inventory.get("rope_head_dim", 0)) != 128):
            raise ValueError(
                f"{tensor_name} must be packed in the Qwen NeoX-to-RoPE interleaved layout")

    program = []
    qkv_regions = [_linear_region(manifest, suffix) for suffix in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")]
    share_qkv_cfg = (
        all(region.nbits == 16 for region in qkv_regions) and
        len({tiles_per_gemm(region) for region in qkv_regions}) == 1)
    for projection_index, suffix in enumerate((
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")):
        projection = emit_prefill_fp16_projection(
            manifest, suffix, tokens=tokens,
            input_start_token=input_start_token,
            output_window_base=first_output_window + projection_index * 2,
            output_start_token=output_start_token,
            input_source_windows=input_source_windows,
            configure_shared_cfg=(not share_qkv_cfg or projection_index == 0),
            restore_shared_cfg=(not share_qkv_cfg or projection_index == 2))
        if not projection or projection[-1].opcode != isa.UOpcode.END:
            raise ValueError(f"{suffix} emitter did not terminate with END")
        program.extend(projection[:-1])
    program.append(isa.end())
    if len(program) > 262144:
        raise ValueError(f"batched QKV program has {len(program)} instructions")
    return program


def emit_prefill_o_projection(manifest, tokens=32,
                              attention_output_window_base=6,
                              output_window_base=10,
                              attention_rows_per_group=16):
    """Apply O projection to attention outputs stored as compact OUT groups."""
    if (not 1 <= tokens <= 32 or tokens % PREFILL_GROUP_ROWS or
            attention_rows_per_group not in (8, 16, 32)):
        raise ValueError("O projection requires complete 8-row source groups")
    group_count = tokens // PREFILL_GROUP_ROWS
    input_windows = []
    if attention_rows_per_group == 32:
        # M=32 attention scatters compact head pairs into two adjacent
        # 16-token OUT windows; O projection still consumes 8-row groups.
        input_windows.extend(
            (attention_output_window_base + token_start // 16,
             token_start % 16)
            for token_start in range(0, tokens, PREFILL_GROUP_ROWS))
        attention_window_count = (tokens + 15) // 16
    else:
        attention_group_count = (
            (tokens + attention_rows_per_group - 1) // attention_rows_per_group)
        for group_index in range(attention_group_count):
            window = attention_output_window_base + group_index
            group_tokens = min(attention_rows_per_group,
                               tokens - group_index * attention_rows_per_group)
            input_windows.extend(
                (window, first_token)
                for first_token in range(0, group_tokens,
                                         PREFILL_GROUP_ROWS))
        attention_window_count = attention_group_count
    if (attention_output_window_base < 0 or
            attention_output_window_base + attention_window_count > 16):
        raise ValueError("attention output groups exceed OUT DDR windows")
    return emit_prefill_fp16_projection(
        manifest, "self_attn.o_proj", tokens=tokens,
        input_start_token=0, output_window_base=output_window_base,
        output_start_token=0, input_source_windows=input_windows)


def emit_prefill_attention_stage(
        manifest, tokens=32, input_start_token=0,
        qkv_window_base=0, attention_output_window_base=6,
        output_window_base=10, attention_rows_per_group=None,
        first_position=0, input_source_windows=None):
    """Emit Q/K/V projection, causal attention, and O projection in order.

    Without ``input_source_windows``, input RMSNorm activations must already
    be present in ACT DDR. Otherwise those 8-row source groups are gathered
    from OUT DDR before Q/K/V projections. Q/K/V, compact attention outputs,
    and O-projection outputs occupy disjoint OUT windows. The returned
    instruction stream contains exactly one END.
    """
    if not 1 <= tokens <= 32 or tokens % PREFILL_GROUP_ROWS:
        raise ValueError(
            "composed prefill attention requires complete 8-row source groups")
    if attention_rows_per_group is None:
        attention_rows_per_group = _default_prefill_attention_rows(tokens)
    if attention_rows_per_group not in (8, 16, 32):
        raise ValueError("attention query tile must use 8, 16, or 32 rows")

    qkv_window_count = 6
    attention_window_count = (
        (tokens + 15) // 16 if attention_rows_per_group == 32 else
        (tokens + attention_rows_per_group - 1) // attention_rows_per_group)
    output_window_count = (tokens + 15) // 16
    ranges = (
        (qkv_window_base, qkv_window_base + qkv_window_count),
        (attention_output_window_base,
         attention_output_window_base + attention_window_count),
        (output_window_base, output_window_base + output_window_count),
    )
    if any(start < 0 or stop > 16 for start, stop in ranges):
        raise ValueError("composed prefill attention exceeds OUT DDR windows")
    for index, (start, stop) in enumerate(ranges):
        for other_start, other_stop in ranges[index + 1:]:
            if start < other_stop and other_start < stop:
                raise ValueError("prefill attention phases must use disjoint OUT windows")

    if input_source_windows is not None:
        qkv_start, qkv_stop = ranges[0]
        source_windows = set()
        for source in input_source_windows:
            source_window = int(source[0] if isinstance(source, (tuple, list))
                                else source)
            if not 0 <= source_window < 16:
                raise ValueError("prefill QKV input window is outside OUT DDR")
            source_windows.add(source_window)
        if any(qkv_start <= window < qkv_stop for window in source_windows):
            raise ValueError("prefill QKV inputs must not overlap QKV outputs")

    program = emit_prefill_qkv_projections(
        manifest, tokens=tokens,
        input_start_token=(input_start_token if input_source_windows is None
                           else 0),
        first_output_window=qkv_window_base,
        input_source_windows=input_source_windows)
    if not program or program[-1].opcode != isa.UOpcode.END:
        raise ValueError("QKV projection emitter did not terminate with END")
    program.pop()
    program.extend(emit_prefill_causal_attention(
        manifest, tokens=tokens, rows_per_group=attention_rows_per_group,
        qkv_window_base=qkv_window_base,
        output_window_base=attention_output_window_base,
        first_position=first_position))
    o_projection = emit_prefill_o_projection(
        manifest, tokens=tokens,
        attention_output_window_base=attention_output_window_base,
        output_window_base=output_window_base,
        attention_rows_per_group=attention_rows_per_group)
    if not o_projection or o_projection[-1].opcode != isa.UOpcode.END:
        raise ValueError("O projection emitter did not terminate with END")
    program.extend(o_projection)
    if len(program) > 262144:
        raise ValueError(f"composed prefill attention has {len(program)} instructions")
    return program


def emit_decode_attention_from_cache(
        manifest, position, q_base=DECODE_QUERY_BASE,
        output_base=DECODE_ATTENTION_OUT_BASE,
        k_cache_window_base=None, v_cache_window_base=None,
        current_k_base=None, current_v_base=None,
        current_k_page=0, current_v_page=0,
        defer_output_fence=False, pair_head_tiles=True,
        head_group_size=3, prefetch_kv_tiles=False,
        serialize_pv_rf_transfers=True,
        defer_online_normalization=False,
        keep_output_page1=False, kv_source="out",
        cached_keys_rotated=True,
        k_source_base=None, v_source_base=None):
    """Emit one real decode attention stage using a resident OUT DDR KV cache.

    ``q_base`` must contain the current token's already-RoPE-rotated 2048-wide
    Q vector. K and V cache rows must be laid out as 4-KiB token records in OUT
    DDR, with K and V each starting at a two-window-aligned region. Adjacent
    128-wide heads are gathered as compact per-token fields across positions
    ``0..position``; each head then reads its own 256-byte slice from L1. The
    caller may provide the live current K/V vectors to avoid reading that row
    back from DDR; the cache write must still complete for later tokens. This
    is deliberately a one-token program, not nestable inside
    :func:`emit_tokens`.
    ``pair_head_tiles=False`` emits the single-head gather schedule as a
    same-math comparison baseline; production callers should keep the default.
    ``head_group_size`` may be 2, 3, or 4 for multi-tile contexts. Four-head
    tiles use page 0 for the full K/V tile and relocate Q/output/scale into
    pages 1/2; the existing M32 GEMMs and online-Softmax instructions are
    unchanged. Single-tile contexts retain the established two-head path.
    ``prefetch_kv_tiles`` double-buffers successive K/V tiles in single-head
    mode. ``serialize_pv_rf_transfers=False`` removes the optional PV-side
    barrier and relies on the GEMM scoreboard for the TGLD-to-GEMM dependency.
    ``keep_output_page1=True`` leaves four-head output compact on page 1 for a
    page-aware immediate consumer; standalone callers retain the page-0 copy.
    Set ``defer_output_fence`` only when the immediate consumer uses a
    dependency-gated PostProc/GEMM load; standalone callers wait for the final
    head's output stores by default. ``kv_source='act'`` or ``'wgt'`` enables
    long-context cache streaming through the existing 16-MiB head-gather DMA;
    in that mode K/V are full-hidden token rows at the supplied DDR-relative
    source bases, and the OUT-window cache limit does not apply.
    """
    valid_tokens = position + 1
    if kv_source not in ("out", "act", "wgt"):
        raise ValueError("decode KV source must be out, act, or wgt")
    external_kv = kv_source != "out"
    if not isinstance(cached_keys_rotated, bool):
        raise ValueError("cached_keys_rotated must be a boolean")
    if not cached_keys_rotated and (
            external_kv or position < 1 or position > TILE):
        raise ValueError(
            "on-the-fly cached-key RoPE currently supports OUT caches through position 32")
    if external_kv:
        if (not 0 <= position < attention.MAX_FLASH_SEQUENCE_LENGTH or
                valid_tokens <= TILE):
            raise ValueError(
                "external Decode KV streaming requires a multi-tile context "
                f"of at most {attention.MAX_FLASH_SEQUENCE_LENGTH} tokens")
        if current_k_base is not None:
            raise ValueError(
                "external Decode KV streaming cannot stage a resident current row")
        if (k_source_base is None or v_source_base is None or
                any(base < 0 or base % 256 or base >= (1 << 24)
                    for base in (k_source_base, v_source_base))):
            raise ValueError(
                "external Decode K/V source bases must be 256-byte aligned "
                "and encodable")
        source_extent = valid_tokens * PREFILL_TOKEN_SLOT_BYTES
        if (k_source_base < v_source_base + source_extent and
                v_source_base < k_source_base + source_extent):
            raise ValueError("external Decode K and V source ranges must not overlap")
        cache_windows = 0
        max_cache_base = 0
        k_cache_window_base = v_cache_window_base = 0
    else:
        if not 0 <= position < 128:
            raise ValueError("OUT-window Decode cache supports positions 0..127")
        cache_windows = (2 if valid_tokens <= 32 else
                         4 if valid_tokens <= 64 else
                         5 if valid_tokens <= 80 else
                         (valid_tokens + 15) // 16)
        max_cache_base = 16 - cache_windows
        if k_cache_window_base is None:
            k_cache_window_base = 0
        if v_cache_window_base is None:
            v_cache_window_base = cache_windows
        if (not 0 <= k_cache_window_base <= max_cache_base or
                not 0 <= v_cache_window_base <= max_cache_base):
            raise ValueError("decode KV cache exceeds its OUT window allocation")
        if (k_cache_window_base < v_cache_window_base + cache_windows and
                v_cache_window_base < k_cache_window_base + cache_windows):
            raise ValueError("decode K and V cache windows must not overlap")
    if head_group_size not in (2, 3, 4):
        raise ValueError("decode head groups must contain two, three, or four heads")
    if (current_k_base is None) != (current_v_base is None):
        raise ValueError("direct decode attention requires both current K and V")
    if current_k_base is not None:
        for base, page in ((current_k_base, current_k_page),
                           (current_v_base, current_v_page)):
            if (page not in (0, 1) or base < 0 or base % 32 or
                    base + 4096 > 65536):
                raise ValueError("direct current K/V vectors must fit an aligned L1 page")
        if (current_k_page == current_v_page and
                current_k_base < current_v_base + 4096 and
                current_v_base < current_k_base + 4096):
            raise ValueError("direct current K and V vectors must not overlap")
    if (q_base < 0 or q_base % 32 or q_base + 4096 > 65536 or
            output_base < 0 or output_base % 32 or
            output_base + 4096 > 65536):
        raise ValueError("decode Q/output vectors must fit page 0")

    if prefetch_kv_tiles and (valid_tokens <= TILE or pair_head_tiles):
        raise ValueError(
            "decode K/V tile prefetch requires multi-tile single-head mode")
    grouped_head_count = head_group_size if pair_head_tiles else 1
    if valid_tokens <= TILE:
        grouped_head_count = 2 if pair_head_tiles else 1
    triple_head_tiles = valid_tokens > TILE and grouped_head_count == 3
    quad_head_tiles = valid_tokens > TILE and grouped_head_count == 4
    kv_k_l1_base = 0x0000 if triple_head_tiles or quad_head_tiles else DECODE_K_CACHE_L1_BASE
    kv_v_l1_base = (0x8000 if quad_head_tiles else
                    0x6000 if triple_head_tiles else DECODE_V_CACHE_L1_BASE)
    scale_l1_base = (0x2000 if quad_head_tiles else
                     0xF000 if triple_head_tiles else 0x2000)
    query_l1_base = 0x0000 if quad_head_tiles else q_base
    query_l1_page = 1 if quad_head_tiles else 0
    output_l1_base = 0x1000 if quad_head_tiles else output_base
    output_l1_page = 1 if quad_head_tiles else 0
    score_l1_base = 0x0000 if quad_head_tiles else DECODE_SCORE_BASE
    score_l1_page = 2 if quad_head_tiles else 0
    scale_l1_page = 1 if quad_head_tiles else 0
    alpha_l1_base = 0x0100 if quad_head_tiles else 0x2000
    kv_staging_page = 2
    current_k_stage = 0x1000 if quad_head_tiles else 0x0000
    current_v_stage = 0x2000 if quad_head_tiles else 0x1000
    live_ranges = [
        (query_l1_page, query_l1_base, query_l1_base + 4096),
        (output_l1_page, output_l1_base, output_l1_base + 4096),
        (score_l1_page, score_l1_base,
         score_l1_base + PREFILL_HEAD_BYTES),
        (scale_l1_page, scale_l1_base, scale_l1_base + 256),
    ]
    if quad_head_tiles:
        live_ranges.extend((
            (2, alpha_l1_base, alpha_l1_base + 4 * PREFILL_HEAD_BYTES),
            (kv_staging_page, current_k_stage, current_k_stage + 4096),
            (kv_staging_page, current_v_stage, current_v_stage + 4096),
        ))
    if valid_tokens > TILE:
        tile_row_bytes = grouped_head_count * PREFILL_HEAD_BYTES
        # FlashAttention streams one 32-key tile through compact head buffers;
        # tile size stays bounded as the context grows.
        if prefetch_kv_tiles:
            live_ranges.extend((
                (0, 0x0000, 0x2000), (0, 0x4000, 0x6000),
                (0, 0x6000, 0x8000), (0, 0xA000, 0xC000),
            ))
        else:
            live_ranges.extend((
                (0, kv_k_l1_base, kv_k_l1_base + TILE * tile_row_bytes),
                (0, kv_v_l1_base, kv_v_l1_base + TILE * tile_row_bytes),
            ))
    else:
        live_ranges.extend((
            (0, DECODE_K_CACHE_L1_BASE,
             DECODE_K_CACHE_L1_BASE +
             valid_tokens * DECODE_HEAD_PAIR_BYTES),
            (0, DECODE_V_CACHE_L1_BASE,
             DECODE_V_CACHE_L1_BASE +
             valid_tokens * DECODE_HEAD_PAIR_BYTES),
        ))
    for index, left in enumerate(live_ranges):
        for right in live_ranges[index + 1:]:
            if (left[0] == right[0] and left[1] < right[2] and
                    right[1] < left[2]):
                raise ValueError("decode attention live L1 ranges overlap")

    def kv_gather(source_base, cache_window_base, head_offset, tile_start,
                  transfer_bytes, l1_bank, tokens):
        if external_kv:
            return isa.dma_rd_head_token_group(
                source_base + head_offset + tile_start * PREFILL_TOKEN_SLOT_BYTES,
                transfer_bytes, l1_bank=l1_bank, tokens=tokens,
                source=kv_source)
        return isa.dma_rd_out_token_group(
            cache_window_base * PREFILL_OUTPUT_WINDOW_BYTES + head_offset +
            tile_start * PREFILL_TOKEN_SLOT_BYTES,
            transfer_bytes, l1_bank=l1_bank, tokens=tokens)

    program = [
        isa.set_cfg_k(TILE),
        isa.set_cfg_m(1),
    ]
    if current_k_base is not None:
        # Stage the complete current rows outside page 0 before historical DMA
        # begins. The live V row can alias either compact cache destination.
        for offset in range(0, PREFILL_TOKEN_SLOT_BYTES, VECTOR_BYTES):
            program.extend([
                isa.vld_page(0, current_k_base + offset, current_k_page),
                isa.vst_page(0, current_k_stage + offset, kv_staging_page),
                isa.vld_page(0, current_v_base + offset, current_v_page),
                isa.vst_page(0, current_v_stage + offset, kv_staging_page),
            ])
    program.extend(emit_prefill_attention_scale(manifest))
    if not cached_keys_rotated:
        # Prompt K rows are captured before attention and remain unrotated in
        # the FakeDDR backing store. Keep the 32 position vectors in page 1,
        # outside the page-0 K/V tile, then rotate each gathered historical K
        # tile before its QK GEMM. The just-projected current K row is already
        # RoPE-rotated by emit_decode_token and is appended after this step.
        program.extend(emit_prefill_rope_table(
            manifest, tokens=position, first_position=0,
            l1_destination_base=0x0000))
        for offset in range(0, position * PREFILL_ROPE_ROW_BYTES, VECTOR_BYTES):
            program.extend([
                isa.vld_page(0, offset, 0),
                isa.vst_page(1, 0x6000 + offset, 0),
            ])
        program.append(isa.barrier())

    def rotate_cached_key_rows(tile_start, tokens, group_heads,
                               transfer_bytes, l1_base):
        if cached_keys_rotated or not tokens:
            return []
        operations = []
        for head_lane in range(group_heads):
            operations.extend(emit_prefill_rope_component(
                tokens,
                l1_base + head_lane * PREFILL_HEAD_BYTES,
                first_position=tile_start,
                row_stride_bytes=transfer_bytes,
                rope_table_first_position=0,
                rope_table_base=0x6000,
                rope_table_page=1))
        return operations

    if quad_head_tiles:
        # Preserve the score scale in page 1, then pack all 16 Q head vectors
        # there before page 0 is filled by the four-head K/V tiles.
        for offset in range(0, 256, VECTOR_BYTES):
            program.extend([
                isa.vld_page(1, 0x2000 + offset, 0),
                isa.vst_page(1, 0x2000 + offset, 1),
            ])
        for head_index in range(16):
            for offset in range(0, PREFILL_HEAD_BYTES, VECTOR_BYTES):
                program.extend([
                    isa.vld_page(
                        0, q_base + head_index * PREFILL_HEAD_BYTES + offset, 0),
                    isa.vst_page(
                        0, head_index * PREFILL_HEAD_BYTES + offset, 1),
                ])
        program.append(isa.barrier())

    if triple_head_tiles:
        # The three-head K tile occupies 24 KiB from page-0 offset zero,
        # including the regular scale slot. Preserve the scale in a disjoint
        # L1 range before the first gathered tile overwrites the source.
        for offset in range(0, 256, VECTOR_BYTES):
            program.extend([
                isa.vld_page(0, 0x2000 + offset, 0),
                isa.vst_page(0, scale_l1_base + offset, 0),
            ])
        program.append(isa.barrier())

    if valid_tokens > TILE and pair_head_tiles:
        # Gather adjacent heads per 32-token K/V tile. The 512- or 768-byte
        # token row is consumed as independent 256-byte head streams.
        for first_head in range(0, 16, grouped_head_count):
            group_heads = min(grouped_head_count, 16 - first_head)
            transfer_bytes = group_heads * PREFILL_HEAD_BYTES
            head_offset = first_head * PREFILL_HEAD_BYTES
            tile_loaders = []
            for tile_start in range(0, valid_tokens, TILE):
                tile_count = min(TILE, valid_tokens - tile_start)
                loader = []
                if current_k_base is not None and tile_start + tile_count > position:
                    historical_tokens = position - tile_start
                else:
                    historical_tokens = tile_count
                if historical_tokens:
                    loader.extend([
                        isa.token_loop_begin(historical_tokens, allow_overlap=True),
                        kv_gather(
                            k_source_base, k_cache_window_base,
                            head_offset, tile_start, transfer_bytes,
                            l1_bank=kv_k_l1_base >> 12,
                            tokens=historical_tokens),
                        kv_gather(
                            v_source_base, v_cache_window_base,
                            head_offset, tile_start, transfer_bytes,
                            l1_bank=kv_v_l1_base >> 12,
                            tokens=historical_tokens),
                        isa.token_loop_end(),
                    ])
                    loader.extend(rotate_cached_key_rows(
                        tile_start, historical_tokens, group_heads,
                        transfer_bytes, kv_k_l1_base))
                if current_k_base is not None and tile_start + tile_count > position:
                    # The just-projected current row may still be in flight to
                    # DDR. Append it directly after the historical rows in
                    # this final compact tile instead of reading it back.
                    for offset in range(0, transfer_bytes, VECTOR_BYTES):
                        loader.extend([
                            isa.vld_page(
                                0, current_k_stage + head_offset + offset, 2),
                            isa.vst_page(
                                0, kv_k_l1_base +
                                historical_tokens * transfer_bytes + offset, 0),
                            isa.vld_page(
                                0, current_v_stage + head_offset + offset, 2),
                            isa.vst_page(
                                0, kv_v_l1_base +
                                historical_tokens * transfer_bytes + offset, 0),
                        ])
                tile_loaders.append(loader)

            program.extend(attention.emit_flash_attention_query_block(
                q_base=query_l1_base + head_offset,
                k_base=kv_k_l1_base,
                v_base=kv_v_l1_base,
                score_base=score_l1_base,
                alpha_base=alpha_l1_base,
                output_base=output_l1_base + head_offset,
                query_start=position,
                query_count=1,
                sequence_length=valid_tokens,
                key_tile_loaders=tile_loaders,
                head_dim=128,
                query_row_stride_bytes=transfer_bytes,
                output_row_stride=PREFILL_TOKEN_SLOT_BYTES,
                output_feature_offset=0,
                score_scale_base=scale_l1_base,
                output_page=output_l1_page,
                alpha_page=2,
                state_slot_base=0,
                query_page=query_l1_page,
                score_page=score_l1_page,
                score_scale_page=scale_l1_page,
                k_token_row_stride_bytes=transfer_bytes,
                v_token_row_stride_bytes=transfer_bytes,
                additional_heads=tuple((
                    query_l1_base + head_offset +
                    head_lane * PREFILL_HEAD_BYTES,
                    kv_k_l1_base + head_lane * PREFILL_HEAD_BYTES,
                    kv_v_l1_base + head_lane * PREFILL_HEAD_BYTES,
                    output_l1_base + head_offset +
                    head_lane * PREFILL_HEAD_BYTES,
                    head_lane,
                ) for head_lane in range(1, group_heads)),
                serialize_pv_rf_transfers=serialize_pv_rf_transfers,
                defer_online_normalization=defer_online_normalization,
                wait_after_store=(first_head + group_heads == 16 and
                                  not defer_output_fence),
            ))
        if quad_head_tiles:
            program.append(isa.barrier())
            if not keep_output_page1:
                for head_index in range(16):
                    for offset in range(0, PREFILL_HEAD_BYTES, VECTOR_BYTES):
                        program.extend([
                            isa.vld_page(
                                0, output_l1_base +
                                head_index * PREFILL_HEAD_BYTES + offset, 1),
                            isa.vst_page(
                                0, output_base +
                                head_index * PREFILL_HEAD_BYTES + offset, 0),
                        ])
                program.append(isa.barrier())
        return program

    if valid_tokens > TILE:
        # Reference schedule: load one 128-wide head field per key tile and
        # execute the same online-softmax/GEMM work without pair gathering.
        for head_index in range(16):
            head_offset = head_index * PREFILL_HEAD_BYTES
            tile_loaders = []
            tile_bases = []
            for tile_start in range(0, valid_tokens, TILE):
                tile_count = min(TILE, valid_tokens - tile_start)
                if prefetch_kv_tiles:
                    tile_slot = (tile_start // TILE) & 1
                    tile_k_base = (0x0000, 0x4000)[tile_slot]
                    tile_v_base = (0x6000, 0xA000)[tile_slot]
                    tile_bases.append((tile_k_base, tile_v_base))
                else:
                    tile_k_base, tile_v_base = (
                        DECODE_K_CACHE_L1_BASE, DECODE_V_CACHE_L1_BASE)
                loader = []
                if current_k_base is not None and tile_start + tile_count > position:
                    historical_tokens = position - tile_start
                else:
                    historical_tokens = tile_count
                if historical_tokens:
                    loader.extend([
                        isa.token_loop_begin(historical_tokens, allow_overlap=True),
                        kv_gather(
                            k_source_base, k_cache_window_base,
                            head_offset, tile_start, PREFILL_HEAD_BYTES,
                            l1_bank=tile_k_base >> 12,
                            tokens=historical_tokens),
                        kv_gather(
                            v_source_base, v_cache_window_base,
                            head_offset, tile_start, PREFILL_HEAD_BYTES,
                            l1_bank=tile_v_base >> 12,
                            tokens=historical_tokens),
                        isa.token_loop_end(),
                    ])
                    loader.extend(rotate_cached_key_rows(
                        tile_start, historical_tokens, 1,
                        PREFILL_HEAD_BYTES, tile_k_base))
                if current_k_base is not None and tile_start + tile_count > position:
                    for offset in range(0, PREFILL_HEAD_BYTES, VECTOR_BYTES):
                        loader.extend([
                            isa.vld_page(
                                0, current_k_stage + head_offset + offset, 2),
                            isa.vst_page(
                                0, tile_k_base +
                                historical_tokens * PREFILL_HEAD_BYTES + offset, 0),
                            isa.vld_page(
                                0, current_v_stage + head_offset + offset, 2),
                            isa.vst_page(
                                0, tile_v_base +
                                historical_tokens * PREFILL_HEAD_BYTES + offset, 0),
                        ])
                tile_loaders.append(loader)

            program.extend(attention.emit_flash_attention_query_block(
                q_base=q_base + head_offset,
                k_base=DECODE_K_CACHE_L1_BASE,
                v_base=DECODE_V_CACHE_L1_BASE,
                score_base=DECODE_SCORE_BASE,
                alpha_base=0x2000,
                output_base=output_base + head_offset,
                query_start=position,
                query_count=1,
                sequence_length=valid_tokens,
                key_tile_loaders=tile_loaders,
                head_dim=128,
                query_row_stride_bytes=PREFILL_HEAD_BYTES,
                output_row_stride=PREFILL_TOKEN_SLOT_BYTES,
                output_feature_offset=0,
                score_scale_base=0x2000,
                output_page=0,
                alpha_page=2,
                state_slot_base=0,
                key_tile_bases=(tile_bases if prefetch_kv_tiles else None),
                key_tile_prefetchers=(tile_loaders[1:]
                                      if prefetch_kv_tiles else None),
                serialize_pv_rf_transfers=serialize_pv_rf_transfers,
                defer_online_normalization=defer_online_normalization,
                wait_after_store=(head_index == 15 and
                                  not defer_output_fence),
            ))
        return program

    for first_head in range(0, 16, 2):
        head_offset = first_head * PREFILL_HEAD_BYTES
        if current_k_base is not None:
            current_row_offset = position * DECODE_HEAD_PAIR_BYTES
            for offset in range(0, DECODE_HEAD_PAIR_BYTES, VECTOR_BYTES):
                program.extend([
                    isa.vld_page(0, current_k_stage + head_offset + offset, 2),
                    isa.vst_page(0, DECODE_K_CACHE_L1_BASE +
                                 current_row_offset + offset, 0),
                    isa.vld_page(0, current_v_stage + head_offset + offset, 2),
                    isa.vst_page(0, DECODE_V_CACHE_L1_BASE +
                                 current_row_offset + offset, 0),
                ])
        historical_tokens = (position if current_k_base is not None
                             else valid_tokens)
        if historical_tokens:
            k_byte_offset = (k_cache_window_base * PREFILL_OUTPUT_WINDOW_BYTES +
                             head_offset)
            v_byte_offset = (v_cache_window_base * PREFILL_OUTPUT_WINDOW_BYTES +
                             head_offset)
            # Gather only history when the current projected row is already
            # resident. TGLD consumes the compact pair stride directly.
            program.extend([
                isa.token_loop_begin(historical_tokens, allow_overlap=True),
                isa.dma_rd_out_token_group(
                    k_byte_offset, DECODE_HEAD_PAIR_BYTES,
                    l1_bank=DECODE_K_CACHE_L1_BASE >> 12,
                    tokens=historical_tokens),
                isa.dma_rd_out_token_group(
                    v_byte_offset, DECODE_HEAD_PAIR_BYTES,
                    l1_bank=DECODE_V_CACHE_L1_BASE >> 12,
                    tokens=historical_tokens),
                isa.token_loop_end(),
            ])

        for pair_lane in range(2):
            head_index = first_head + pair_lane
            pair_head_offset = pair_lane * PREFILL_HEAD_BYTES
            program.extend(attention.emit_attention_head(
                q_base=q_base + head_index * PREFILL_HEAD_BYTES,
                k_base=DECODE_K_CACHE_L1_BASE + pair_head_offset,
                v_base=DECODE_V_CACHE_L1_BASE + pair_head_offset,
                score_base=DECODE_SCORE_BASE,
                out_base=output_base + head_index * PREFILL_HEAD_BYTES,
                query_token=0,
                valid_tokens=valid_tokens,
                head_dim=128,
                token_row_stride=DECODE_HEAD_PAIR_BYTES,
                causal=False,
                score_scale_base=0x2000,
                output_page=0,
                wait_after_store=(head_index == 15 and
                                  not defer_output_fence),
            ))
    return program


def emit_decode_row_store(source_base, position, output_window_base,
                          source_page=0, source_ready=False,
                          defer_dma_fence=False):
    """Store one 2048-wide FP16 row in a position-indexed OUT DDR window.

    DMA_WR's legacy address uses the same immediate for its L1 source and OUT
    byte offset. A source already at its cache-slot offset can be written
    directly from page 0 or page 1; other sources are staged in page 1. Four
    1-KiB bursts cover the row, and the final barrier makes the slot safe to
    reuse.
    ``defer_dma_fence`` is for a caller that overlaps independent work with the
    write and inserts a completion fence before reusing the page-1 cache slot.
    """
    if not 0 <= position < 80 or not 0 <= output_window_base <= 15:
        raise ValueError("decode cache position/window is outside the 80-token layout")
    if source_page not in (0, 1) or source_base < 0 or source_base % 32:
        raise ValueError("decode KV source must be 32-byte aligned on page 0 or 1")
    if source_base + PREFILL_TOKEN_SLOT_BYTES > 65536:
        raise ValueError("decode KV source row exceeds its L1 page")
    output_window = output_window_base + position // 16
    if output_window >= 16:
        raise ValueError("decode KV cache exceeds the OUT window address space")

    slot_base = (position % 16) * PREFILL_TOKEN_SLOT_BYTES
    direct_source = source_base == slot_base
    if (source_page == 1 and not direct_source and
            source_base < slot_base + PREFILL_TOKEN_SLOT_BYTES and
            slot_base < source_base + PREFILL_TOKEN_SLOT_BYTES):
        raise ValueError("decode KV source overlaps the page-1 DMA staging slot")

    program = []
    if not direct_source:
        for offset in range(0, PREFILL_TOKEN_SLOT_BYTES, 64):
            program.extend([
                isa.vld_page(0, source_base + offset, source_page),
                isa.vst_page(0, slot_base + offset, 1),
            ])
    # A staged row must retire before DMA. A direct row on either L1 page may
    # skip this wait only when its producer has already fenced the final VST.
    if not direct_source or not source_ready:
        program.append(isa.barrier())
    for offset in range(0, PREFILL_TOKEN_SLOT_BYTES, 1024):
        program.append(isa.dma_wr(
            slot_base + offset, output_window=output_window,
            page=(source_page if direct_source else 1)))
    if not defer_dma_fence:
        program.append(isa.barrier())
    return program


def _emit_decode_rope_heads(output_base, output_page, heads=16,
                            use_token_loop=True):
    """Apply one position's RoPE row across head-strided decode vectors."""
    if (output_page not in (0, 1) or not 1 <= heads <= 16 or
            output_base < 0 or output_base % VECTOR_BYTES or
            output_base + heads * PREFILL_HEAD_BYTES > 65536):
        raise ValueError("decode RoPE head rows must fit an aligned L1 page")

    program = []
    for chunk in range(128 // TILE):
        offset = chunk * VECTOR_BYTES
        # Every head shares this chunk of the current position's RoPE row.
        # Hoisting its load reduces 16 duplicate vector loads per chunk.
        program.append(isa.vld(3, PREFILL_ROPE_BASE + offset))
        if use_token_loop:
            # With no following bias DMA to overlap, stride over head rows in
            # hardware to reduce the fetched microcode to four short bodies.
            program.extend([
                isa.token_loop_begin(heads),
                isa.vld_token(0, output_base + offset,
                              PREFILL_HEAD_BYTES, page=output_page),
                isa.rope(0, 0, 3),
                isa.vst_token(0, output_base + offset,
                              PREFILL_HEAD_BYTES, page=output_page),
                isa.token_loop_end(),
            ])
        else:
            # Q/K keep straight-line vector issue so their work can hide the
            # next projection's bias DMA without a loop-entry all-idle gate.
            for head in range(heads):
                head_addr = output_base + head * PREFILL_HEAD_BYTES + offset
                program.extend([
                    isa.vld_page(0, head_addr, output_page),
                    isa.rope(0, 0, 3),
                    isa.vst_page(0, head_addr, output_page),
                ])
    return program


def _decode_v_output_base(position):
    """Choose a direct-write V slot when it does not alias live page-0 data."""
    slot_base = (position % 16) * PREFILL_TOKEN_SLOT_BYTES
    live_ranges = [
        (L1_RESIDUAL_IN, L1_RESIDUAL_IN + PREFILL_TOKEN_SLOT_BYTES),
        (0x2000, 0x4000),  # static FP16 weight ping-pong staging
        (L1_NORM2_GAMMA, L1_NORM2_GAMMA + PREFILL_TOKEN_SLOT_BYTES),
        (L1_V_BIAS, L1_V_BIAS + PREFILL_TOKEN_SLOT_BYTES),
        (DECODE_QUERY_BASE, DECODE_QUERY_BASE + PREFILL_TOKEN_SLOT_BYTES),
        (DECODE_SCORE_BASE, DECODE_SCORE_BASE + PREFILL_HEAD_BYTES),
        (L1_O_BIAS, L1_O_BIAS + PREFILL_TOKEN_SLOT_BYTES),
    ]
    if position == 0:
        live_ranges.append((L1_ATTN_VALUE,
                            L1_ATTN_VALUE + PREFILL_TOKEN_SLOT_BYTES))
    if any(slot_base < end and start < slot_base + PREFILL_TOKEN_SLOT_BYTES
           for start, end in live_ranges):
        return DECODE_V_CURRENT_L1_BASE
    return slot_base


def emit_decode_kv_cache_store(source_base, position, cache_window_base,
                               source_page=0, source_ready=False,
                               defer_dma_fence=False):
    """Append one 2048-wide FP16 K or V row to the decode OUT DDR cache."""
    return emit_decode_row_store(
        source_base, position, cache_window_base, source_page=source_page,
        source_ready=source_ready, defer_dma_fence=defer_dma_fence)


def _prefill_head_dma_instruction(projection_index, first_token, tokens,
                                 head_index, first_output_window,
                                 l1_destination_base=None,
                                 transfer_bytes=PREFILL_HEAD_BYTES):
    if projection_index not in (0, 1, 2):
        raise ValueError("projection_index must select Q, K, or V")
    if (not 0 <= first_token < 32 or not 1 <= tokens <= 32 or
            first_token + tokens > 32):
        raise ValueError("head group must contain 1..32 rows within 32 tokens")
    if not 0 <= first_output_window <= 10:
        raise ValueError("QKV output allocation must fit six OUT windows")
    if (not 0 <= head_index < 16 or
            (transfer_bytes == DECODE_HEAD_PAIR_BYTES and head_index % 2) or
            transfer_bytes not in (PREFILL_HEAD_BYTES,
                                   DECODE_HEAD_PAIR_BYTES)):
        raise ValueError("Qwen 2048/128 head index must be in [0,15]")

    projection_window = first_output_window + projection_index * 2
    source_offset = (projection_window * PREFILL_OUTPUT_WINDOW_BYTES +
                     first_token * PREFILL_TOKEN_SLOT_BYTES +
                     head_index * PREFILL_HEAD_BYTES)
    if (source_offset + (tokens - 1) * PREFILL_TOKEN_SLOT_BYTES +
            transfer_bytes >
            (projection_window + 2) * PREFILL_OUTPUT_WINDOW_BYTES):
        raise ValueError("head-group read exceeds its Q/K/V output allocation")
    compact_bases = (PREFILL_HEAD_Q_BASE, PREFILL_HEAD_K_BASE,
                     PREFILL_HEAD_V_BASE)
    destination_base = (compact_bases[projection_index]
                        if l1_destination_base is None
                        else l1_destination_base)
    if destination_base < 0 or destination_base % 4096:
        raise ValueError("Q/K/V L1 destination must be 4-KiB aligned")
    return isa.dma_rd_out_token_group(
        source_offset, transfer_bytes,
        l1_bank=destination_base >> 12, tokens=tokens)


def emit_prefill_head_group(projection_index, first_token, tokens,
                            head_index, first_output_window=0):
    """Gather one Q/K/V head directly into its compact page-0 L1 matrix.

    OUT DDR stores each projection as full 4-KiB token rows. A token-strided
    read copies only the selected 256-byte head field from each row into a
    compact destination. The 32-token Q/K/V arrays occupy 8 KiB each at
    ``PREFILL_HEAD_Q/K/V_BASE`` and can be consumed by
    ``attention.emit_attention_head``.
    """

    return [
        isa.token_loop_begin(tokens),
        _prefill_head_dma_instruction(
            projection_index, first_token, tokens, head_index,
            first_output_window),
        isa.token_loop_end(),
    ]


def emit_prefill_attention_head_inputs(head_index, first_output_window=0,
                                       tokens=32, first_token=0,
                                       query_tokens=None,
                                       query_first_token=None,
                                       query_l1_base=PREFILL_HEAD_Q_BASE,
                                       include_query=True, kv_tokens=None):
    """Gather raw projected Q/K rows and V for one 128-wide attention head.

    Q and K rows are already in the adjacent-pair feature order declared by
    the manifest packer, but must receive their position-dependent RoPE
    transform before the caller emits ``attention.emit_attention_head``.
    """
    if not 0 <= head_index < 16:
        raise ValueError("Qwen 2048/128 head index must be in [0,15]")
    if not 1 <= tokens <= 32 or not 0 <= first_token <= 32 - tokens:
        raise ValueError("attention head token range must fit the 32-token sequence")
    if not 0 <= first_output_window <= 10:
        raise ValueError("QKV output allocation must fit six OUT windows")
    query_tokens = tokens if query_tokens is None else query_tokens
    kv_tokens = tokens if kv_tokens is None else kv_tokens
    query_first_token = (first_token if query_first_token is None
                         else query_first_token)
    if (not 1 <= query_tokens <= 32 or
            not 0 <= query_first_token <= 32 - query_tokens):
        raise ValueError("attention query block must fit the 32-token sequence")
    if not 1 <= kv_tokens <= tokens or first_token + kv_tokens > 32:
        raise ValueError("attention K/V prefix must fit the selected token range")
    if query_first_token + query_tokens > kv_tokens:
        raise ValueError("attention K/V prefix must cover every query in the block")
    if include_query:
        if query_l1_base < 0 or query_l1_base % 4096:
            raise ValueError("attention query L1 base must be 4-KiB aligned")
        if query_l1_base + query_tokens * PREFILL_HEAD_BYTES > 65536:
            raise ValueError("attention query block exceeds page 0")
    if (include_query and query_tokens == tokens and
            query_first_token == first_token and
            query_l1_base == PREFILL_HEAD_Q_BASE and
            kv_tokens == tokens):
        # Preserve the compact legacy form when Q, K, and V share one range.
        program = [isa.token_loop_begin(tokens, allow_overlap=True)]
        for projection_index in range(3):
            program.append(_prefill_head_dma_instruction(
                projection_index, first_token=first_token, tokens=tokens,
                head_index=head_index,
                first_output_window=first_output_window))
        program.append(isa.token_loop_end())
        return program
    # Gather Q/K/V into page 0 while the previous head's result drains from
    # page 1. Q is compacted to only this query block; K/V keep the full range.
    program = []
    if include_query:
        program.extend([
            isa.token_loop_begin(query_tokens, allow_overlap=True),
            _prefill_head_dma_instruction(
                0, first_token=query_first_token, tokens=query_tokens,
                head_index=head_index,
                first_output_window=first_output_window,
                l1_destination_base=query_l1_base),
            isa.token_loop_end(),
        ])
    program.extend([
        isa.token_loop_begin(kv_tokens, allow_overlap=True),
        _prefill_head_dma_instruction(
            1, first_token=first_token, tokens=kv_tokens,
            head_index=head_index, first_output_window=first_output_window),
        _prefill_head_dma_instruction(
            2, first_token=first_token, tokens=kv_tokens,
            head_index=head_index, first_output_window=first_output_window),
        isa.token_loop_end(),
    ])
    return program


def _rope_table_region(manifest):
    matches = [region for region in manifest["regions"]
               if region.get("role") == "rope_embedding_table_fp16"]
    if len(matches) != 1:
        raise ValueError("manifest must contain exactly one FP16 RoPE embedding table")
    region = matches[0]
    shape = region.get("shape", [])
    if (region.get("packing") != "position,chunk,cos[16],sin[16]" or
            len(shape) != 2 or int(shape[1]) != 128):
        raise ValueError("manifest RoPE table does not match the 128-dim RTL layout")
    return region


def _attention_scale_region(manifest):
    matches = [region for region in manifest["regions"]
               if region.get("role") == "attention_scale_fp16"]
    if len(matches) != 1:
        raise ValueError("manifest must contain exactly one FP16 attention scale")
    region = matches[0]
    if (region.get("packing") != "repeated_fp16_scalar" or
            region.get("shape") != [128]):
        raise ValueError("manifest attention scale must be a 128-lane FP16 vector")
    return region


def _prefill_activation_ddr_base(manifest):
    activation_base = next(
        (int(candidate["byte_base"]) for candidate in manifest["regions"]
         if (candidate.get("name") == "activation.input_layernorm" or
             str(candidate.get("source_tensor", "")).startswith(
                 "activation.input_layernorm"))),
        None,
    )
    if activation_base is None:
        raise ValueError("manifest is missing the prefill ACT base region")
    return activation_base


def emit_prefill_attention_scale(manifest, l1_destination_base=0x2000):
    """Load the 1/sqrt(128) vector used to scale QK logits before softmax."""
    if (l1_destination_base < 0 or l1_destination_base % 4096 or
            l1_destination_base + 256 > 65536):
        raise ValueError("attention scale destination must be a valid L1 bank")
    region = _attention_scale_region(manifest)
    source_offset = int(region["byte_base"]) - _prefill_activation_ddr_base(manifest)
    if source_offset < 0 or source_offset + 256 > (1 << 20):
        raise ValueError("attention scale exceeds the ACT DMA immediate window")
    return [
        isa.token_loop_begin(1),
        isa.dma_rd_token_group(
            source_offset, hidden_size=128,
            l1_bank=l1_destination_base >> 12, tokens=1),
        isa.token_loop_end(),
    ]


def emit_prefill_rope_table(manifest, tokens=32, first_position=0,
                            allow_postproc_overlap=False,
                            l1_destination_base=PREFILL_ROPE_BASE):
    """Load position-specific RoPE vectors into compact L1 rows once per batch.

    ``allow_postproc_overlap`` is safe when the caller's outstanding PostProc
    stores target the disjoint normalized-activation region in page 0.
    """
    if (not 1 <= tokens <= 32 or first_position < 0 or
            l1_destination_base < 0 or l1_destination_base % 4096 or
            l1_destination_base + tokens * PREFILL_ROPE_ROW_BYTES > 65536):
        raise ValueError("RoPE prefill range must contain 1..32 non-negative positions")
    region = _rope_table_region(manifest)
    max_positions = int(region["shape"][0])
    if first_position + tokens > max_positions:
        raise ValueError("RoPE prefill range exceeds the manifest embedding table")
    activation_base = _prefill_activation_ddr_base(manifest)
    rope_byte_offset = int(region["byte_base"]) - activation_base
    source_offset = rope_byte_offset + first_position * PREFILL_ROPE_ROW_BYTES
    if (source_offset + tokens * PREFILL_ROPE_ROW_BYTES) > (1 << 20):
        raise ValueError("RoPE table rows exceed the ACT DMA immediate window")
    return [
        isa.token_loop_begin(tokens,
                             allow_overlap=allow_postproc_overlap),
        isa.dma_rd_token_group(
            source_offset, hidden_size=128,
            l1_bank=l1_destination_base >> 12,
            tokens=tokens),
        isa.token_loop_end(),
    ]


def emit_prefill_rope_head(tokens=32):
    """Apply Qwen positions to compact Q/K head rows using the hardware RoPE.

    The caller loads the matching embedding rows once with
    :func:`emit_prefill_rope_table` and gathers Q/K with
    :func:`emit_prefill_attention_head_inputs`. Each 128-wide head is already
    in adjacent-pair order from the DDR packer; this loop reuses each position
    embedding for both Q and K and adds no software lane-shuffle stage.
    """
    if not 1 <= tokens <= 32:
        raise ValueError("RoPE head processing supports 1..32 tokens")
    program = [isa.token_loop_begin(tokens)]
    for chunk in range(128 // TILE):
        offset = chunk * VECTOR_BYTES
        rope_offset = PREFILL_ROPE_BASE + offset
        program.extend([
            isa.vld_token(0, PREFILL_HEAD_Q_BASE + offset,
                          PREFILL_HEAD_BYTES, page=0),
            isa.vld_token(3, rope_offset,
                          PREFILL_ROPE_ROW_BYTES, page=0),
            isa.rope(0, 0, 3),
            isa.vst_token(0, PREFILL_HEAD_Q_BASE + offset,
                          PREFILL_HEAD_BYTES, page=0),
            isa.vld_token(0, PREFILL_HEAD_K_BASE + offset,
                          PREFILL_HEAD_BYTES, page=0),
            isa.rope(0, 0, 3),
            isa.vst_token(0, PREFILL_HEAD_K_BASE + offset,
                          PREFILL_HEAD_BYTES, page=0),
        ])
    program.append(isa.token_loop_end())
    return program


def emit_prefill_rope_component(tokens, data_base, first_position=0,
                               row_stride_bytes=PREFILL_HEAD_BYTES,
                               rope_table_first_position=0,
                               rope_table_base=PREFILL_ROPE_BASE,
                               rope_table_page=0):
    """Apply RoPE using a compact L1 table whose row zero has a given position."""
    table_row = first_position - rope_table_first_position
    rope_table_end = (rope_table_base +
                      (table_row + tokens) * PREFILL_ROPE_ROW_BYTES)
    if (not 1 <= tokens <= 32 or table_row < 0 or
            table_row + tokens > 64 or rope_table_end > 65536 or
            rope_table_base < 0 or rope_table_base % VECTOR_BYTES or
            rope_table_page not in (0, 1) or
            data_base < 0 or data_base % 32 or
            row_stride_bytes < PREFILL_HEAD_BYTES or
            row_stride_bytes > 4096 or row_stride_bytes % 32 or
            data_base + (tokens - 1) * row_stride_bytes +
            PREFILL_HEAD_BYTES > 65536):
        raise ValueError("RoPE head component exceeds its L1/sequence range")
    program = [isa.token_loop_begin(tokens)]
    for chunk in range(128 // TILE):
        offset = chunk * VECTOR_BYTES
        program.extend([
            isa.vld_token(0, data_base + offset,
                          row_stride_bytes, page=0),
            isa.vld_token(
                3, rope_table_base + table_row * PREFILL_ROPE_ROW_BYTES +
                offset, PREFILL_ROPE_ROW_BYTES, page=rope_table_page),
            isa.rope(0, 0, 3),
            isa.vst_token(0, data_base + offset,
                          row_stride_bytes, page=0),
        ])
    program.append(isa.token_loop_end())
    return program


def emit_prefill_causal_attention(manifest, tokens=32, rows_per_group=None,
                                  qkv_window_base=0,
                                  output_window_base=6,
                                  first_position=0,
                                  reuse_qk_rope=True):
    """Emit full-sequence causal attention, storing compact token groups to OUT.

    Q/K/V projections must already be in the QKV windows. RoPE constants are
    loaded once, then adjacent 128-wide heads are gathered together into
    512-byte token rows and rotated independently. QK and AV use M-row GEMMs
    over each consecutive query block rather than serial M=1 GEMMs.
    Eight- and sixteen-query blocks use full-hidden L1 output rows. A
    32-query block writes head results to a compact page-1 tile and scatters
    that tile into token-major OUT windows.
    """
    if not 1 <= tokens <= 32:
        raise ValueError("causal prefill attention supports 1..32 tokens")
    if rows_per_group is None:
        rows_per_group = _default_prefill_attention_rows(tokens)
    if rows_per_group not in (8, 16, 32):
        raise ValueError("causal attention query tiles must use 8, 16, or 32 rows")
    group_count = (tokens + rows_per_group - 1) // rows_per_group
    attention_window_count = (
        (tokens + 15) // 16 if rows_per_group == 32 else group_count)
    if (not 0 <= qkv_window_base <= 10 or
            not 0 <= output_window_base <= 16 - attention_window_count):
        raise ValueError("causal attention QKV/output windows exceed OUT DDR")

    program = [
        *emit_prefill_attention_scale(manifest),
        *emit_prefill_rope_table(
            manifest, tokens=tokens, first_position=first_position),
        isa.set_cfg_k(TILE),
    ]
    configured_m = None
    for group_index, query_start in enumerate(range(0, tokens, rows_per_group)):
        group_tokens = min(rows_per_group, tokens - query_start)
        if group_tokens != configured_m:
            program.append(isa.set_cfg_m(group_tokens))
            configured_m = group_tokens
        for first_head in range(0, 16, 2):
            if rows_per_group == 32:
                # M=32 needs three non-overlapping compact head matrices.
                # The paired 512-byte layout would make Q and K overlap in
                # page 0 at this row count, so gather one head at a time.
                for head_index in (first_head, first_head + 1):
                    q_head_base = 0x8000
                    k_head_base = 0x6000
                    v_head_base = 0x4000
                    program.extend([
                        isa.token_loop_begin(group_tokens, allow_overlap=True),
                        _prefill_head_dma_instruction(
                            0, first_token=query_start, tokens=group_tokens,
                            head_index=head_index,
                            first_output_window=qkv_window_base,
                            l1_destination_base=q_head_base,
                            transfer_bytes=PREFILL_HEAD_BYTES),
                        isa.token_loop_end(),
                        isa.token_loop_begin(group_tokens, allow_overlap=True),
                        _prefill_head_dma_instruction(
                            1, first_token=0, tokens=group_tokens,
                            head_index=head_index,
                            first_output_window=qkv_window_base,
                            l1_destination_base=k_head_base,
                            transfer_bytes=PREFILL_HEAD_BYTES),
                        _prefill_head_dma_instruction(
                            2, first_token=0, tokens=group_tokens,
                            head_index=head_index,
                            first_output_window=qkv_window_base,
                            l1_destination_base=v_head_base,
                            transfer_bytes=PREFILL_HEAD_BYTES),
                        isa.token_loop_end(),
                    ])
                    if reuse_qk_rope:
                        program.extend(attention.emit_rope_qk_components(
                            group_tokens, q_head_base, k_head_base,
                            first_position=first_position + query_start,
                            q_row_stride=PREFILL_HEAD_BYTES,
                            k_row_stride=PREFILL_HEAD_BYTES,
                            table_first_position=first_position))
                    else:
                        program.extend(emit_prefill_rope_component(
                            group_tokens, q_head_base,
                            first_position=first_position + query_start,
                            row_stride_bytes=PREFILL_HEAD_BYTES,
                            rope_table_first_position=first_position))
                        program.extend(emit_prefill_rope_component(
                            group_tokens, k_head_base,
                            first_position=first_position,
                            row_stride_bytes=PREFILL_HEAD_BYTES,
                            rope_table_first_position=first_position))
                    program.extend(attention.emit_batched_causal_attention_head_queries(
                        q_base=q_head_base,
                        k_base=k_head_base,
                        v_base=v_head_base,
                        score_base=PREFILL_HEAD_SCORE_BASE,
                        query_pack_base=PREFILL_QUERY_PACK_BASE,
                        output_base=0,
                        query_start=query_start,
                        query_count=group_tokens,
                        q_block_packed=True,
                        valid_tokens=group_tokens,
                        head_dim=128,
                        token_row_stride=PREFILL_HEAD_BYTES,
                        query_row_stride_bytes=PREFILL_HEAD_BYTES,
                        output_row_stride=PREFILL_HEAD_BYTES,
                        output_page=1,
                        score_scale_base=0x2000,
                        wait_after_store=False,
                    ))
                    program.extend(isa.dma_wr_head_group_scatter(
                        hidden_size=128, tokens=group_tokens,
                        output_window=output_window_base + query_start // 16,
                        output_offset=head_index * PREFILL_HEAD_BYTES,
                        page=1))
                continue

            # Adjacent Q heads are contiguous in the token-major projection.
            # Gather both in one DMA into a 512-byte-strided query matrix.
            program.extend([
                isa.token_loop_begin(group_tokens, allow_overlap=True),
                _prefill_head_dma_instruction(
                    0, first_token=query_start, tokens=group_tokens,
                    head_index=first_head,
                    first_output_window=qkv_window_base,
                    l1_destination_base=PREFILL_HEAD_Q_BASE,
                    transfer_bytes=DECODE_HEAD_PAIR_BYTES),
                isa.token_loop_end(),
            ])
            kv_tokens = min(tokens, query_start + group_tokens)
            # Keep both adjacent K/V heads resident as one token-major tile.
            # The two 128-wide streams are consumed separately below with a
            # 512-byte row stride, avoiding a second DDR gather per head.
            program.extend([
                isa.token_loop_begin(kv_tokens, allow_overlap=True),
                _prefill_head_dma_instruction(
                    1, first_token=0, tokens=kv_tokens,
                    head_index=first_head,
                    first_output_window=qkv_window_base,
                    l1_destination_base=PREFILL_HEAD_K_BASE,
                    transfer_bytes=DECODE_HEAD_PAIR_BYTES),
                _prefill_head_dma_instruction(
                    2, first_token=0, tokens=kv_tokens,
                    head_index=first_head,
                    first_output_window=qkv_window_base,
                    l1_destination_base=PREFILL_HEAD_PAIR_V_BASE,
                    transfer_bytes=DECODE_HEAD_PAIR_BYTES),
                isa.token_loop_end(),
            ])
            for pair_lane in range(2):
                head_index = first_head + pair_lane
                q_head_base = (PREFILL_HEAD_Q_BASE +
                               pair_lane * PREFILL_HEAD_BYTES)
                k_head_base = (PREFILL_HEAD_K_BASE +
                               pair_lane * PREFILL_HEAD_BYTES)
                v_head_base = (PREFILL_HEAD_PAIR_V_BASE +
                               pair_lane * PREFILL_HEAD_BYTES)
                if reuse_qk_rope:
                    if query_start:
                        program.extend(emit_prefill_rope_component(
                            query_start, k_head_base,
                            first_position=first_position,
                            row_stride_bytes=DECODE_HEAD_PAIR_BYTES,
                            rope_table_first_position=first_position))
                    program.extend(attention.emit_rope_qk_components(
                        group_tokens, q_head_base,
                        k_head_base + query_start * DECODE_HEAD_PAIR_BYTES,
                        first_position=first_position + query_start,
                        q_row_stride=DECODE_HEAD_PAIR_BYTES,
                        k_row_stride=DECODE_HEAD_PAIR_BYTES,
                        table_first_position=first_position))
                else:
                    program.extend(emit_prefill_rope_component(
                        group_tokens, q_head_base,
                        first_position=first_position + query_start,
                        row_stride_bytes=DECODE_HEAD_PAIR_BYTES,
                        rope_table_first_position=first_position))
                    program.extend(emit_prefill_rope_component(
                        kv_tokens, k_head_base,
                        first_position=first_position,
                        row_stride_bytes=DECODE_HEAD_PAIR_BYTES,
                        rope_table_first_position=first_position))
                program.extend(attention.emit_batched_causal_attention_head_queries(
                    q_base=q_head_base,
                    k_base=k_head_base,
                    v_base=v_head_base,
                    score_base=PREFILL_HEAD_SCORE_BASE,
                    query_pack_base=PREFILL_QUERY_PACK_BASE,
                    output_base=0,
                    query_start=query_start,
                    query_count=group_tokens,
                    q_block_packed=True,
                    valid_tokens=kv_tokens,
                    head_dim=128,
                    token_row_stride=DECODE_HEAD_PAIR_BYTES,
                    query_row_stride_bytes=DECODE_HEAD_PAIR_BYTES,
                    output_row_stride=PREFILL_TOKEN_SLOT_BYTES,
                    output_feature_offset=head_index * PREFILL_HEAD_BYTES,
                    output_page=1,
                    score_scale_base=0x2000,
                    # The following group-output TOKEN_BEGIN is all-idle gated;
                    # it already fences page-1 stores before the next DMA.
                    wait_after_store=False,
                ))
        if rows_per_group != 32:
            program.extend(isa.dma_wr_group_tokens(
                0, hidden_size=2048, tokens=group_tokens,
                output_window=output_window_base + group_index))
    program.append(isa.set_cfg_m(1))
    return program


def _emit_prefill_flash_attention_paired_head_block(
        manifest, query_start, sequence_length, qkv_window_bases,
        first_position, qkv_already_rope, serialize_pv_rf_transfers,
        defer_online_normalization):
    """Emit one production M32 block with two independent head streams."""
    if query_start % TILE:
        raise ValueError("paired production FlashAttention requires an M32-aligned query tile")
    query_local_start = query_start % 32
    chunk_index = query_start // 32
    qkv_base = qkv_window_bases[chunk_index]
    query_temp_base = 0x2000
    key_base, value_base = 0x4000, 0x8000
    score_base, scale_base = 0xE000, 0xD000
    query_page_bases = (0x4000, 0x6000)
    output_page_bases = (0, 0x2000)

    program = [isa.set_cfg_k(TILE),
               *emit_prefill_attention_scale(
                   manifest, l1_destination_base=scale_base)]
    if not qkv_already_rope:
        program.extend(emit_prefill_rope_table(
            manifest, tokens=TILE,
            first_position=first_position + query_start,
            l1_destination_base=PREFILL_ROPE_BASE))

    for first_head in range(0, 16, 2):
        program.extend([
            isa.token_loop_begin(TILE, allow_overlap=True),
            _prefill_head_dma_instruction(
                0, first_token=query_local_start, tokens=TILE,
                head_index=first_head,
                first_output_window=qkv_base,
                l1_destination_base=query_temp_base,
                transfer_bytes=DECODE_HEAD_PAIR_BYTES),
            isa.token_loop_end(),
        ])
        if not qkv_already_rope:
            for pair_lane in range(2):
                program.extend(emit_prefill_rope_component(
                    TILE, query_temp_base + pair_lane * PREFILL_HEAD_BYTES,
                    first_position=first_position + query_start,
                    row_stride_bytes=DECODE_HEAD_PAIR_BYTES,
                    rope_table_first_position=first_position + query_start))

        for pair_lane in range(2):
            program.extend(attention.emit_pack_head_rows(
                source_base=query_temp_base,
                destination_base=query_page_bases[pair_lane],
                tokens=TILE, hidden_size=256, head_index=pair_lane,
                head_dim=128, source_row_stride=DECODE_HEAD_PAIR_BYTES,
                source_page=0, destination_page=1,
                wait_after_store=(pair_lane == 1)))

        key_tile_loaders = []
        for tile_start in range(0, sequence_length, TILE):
            tile_tokens = min(TILE, sequence_length - tile_start)
            tile_window = qkv_window_bases[tile_start // TILE]
            loader = []
            if not qkv_already_rope:
                loader.extend(emit_prefill_rope_table(
                    manifest, tokens=tile_tokens,
                    first_position=first_position + tile_start,
                    l1_destination_base=PREFILL_ROPE_BASE))
            loader.extend([
                isa.token_loop_begin(tile_tokens, allow_overlap=True),
                _prefill_head_dma_instruction(
                    1, first_token=0, tokens=tile_tokens,
                    head_index=first_head,
                    first_output_window=tile_window,
                    l1_destination_base=key_base,
                    transfer_bytes=DECODE_HEAD_PAIR_BYTES),
                _prefill_head_dma_instruction(
                    2, first_token=0, tokens=tile_tokens,
                    head_index=first_head,
                    first_output_window=tile_window,
                    l1_destination_base=value_base,
                    transfer_bytes=DECODE_HEAD_PAIR_BYTES),
                isa.token_loop_end(),
            ])
            if not qkv_already_rope:
                for pair_lane in range(2):
                    loader.extend(emit_prefill_rope_component(
                        tile_tokens, key_base + pair_lane * PREFILL_HEAD_BYTES,
                        first_position=first_position + tile_start,
                        row_stride_bytes=DECODE_HEAD_PAIR_BYTES,
                        rope_table_first_position=first_position + tile_start))
            key_tile_loaders.append(loader)

        additional_heads = ((
            query_page_bases[1], key_base + PREFILL_HEAD_BYTES,
            value_base + PREFILL_HEAD_BYTES, output_page_bases[1],
            TILE,
        ),)
        program.extend(attention.emit_flash_attention_query_block(
            q_base=query_page_bases[0], k_base=key_base, v_base=value_base,
            score_base=score_base, alpha_base=0,
            output_base=output_page_bases[0],
            query_start=query_start, query_count=TILE,
            sequence_length=sequence_length,
            key_tile_loaders=key_tile_loaders, head_dim=128,
            query_row_stride_bytes=PREFILL_HEAD_BYTES,
            output_row_stride=PREFILL_HEAD_BYTES,
            score_scale_base=scale_base, output_page=1, alpha_page=2,
            state_slot_base=0, wait_after_store=False,
            k_token_row_stride_bytes=DECODE_HEAD_PAIR_BYTES,
            v_token_row_stride_bytes=DECODE_HEAD_PAIR_BYTES,
            additional_heads=additional_heads,
            serialize_pv_rf_transfers=serialize_pv_rf_transfers,
            query_page=1,
            defer_online_normalization=defer_online_normalization))
        for pair_lane in range(2):
            head_index = first_head + pair_lane
            program.extend(isa.dma_wr_head_group_scatter(
                hidden_size=128, tokens=TILE,
                output_window=qkv_base + query_local_start // 16,
                output_offset=head_index * PREFILL_HEAD_BYTES,
                page=1, source_l1_offset=output_page_bases[pair_lane]))

    program.extend([isa.barrier(), isa.end()])
    if len(program) > 262144:
        raise ValueError(
            f"paired production FlashAttention block has {len(program)} instructions")
    return program


def emit_prefill_flash_attention_block(
        manifest, query_start, query_count=32, sequence_length=64,
        qkv_window_bases=(0, 6), first_position=0,
        qkv_already_rope=False, reuse_rope_table=True,
        pair_adjacent_heads=False,
        serialize_pv_rf_transfers=True,
        defer_online_normalization=False):
    """Emit one production-layout Prefill query block over 32-key tiles.

    Q/K/V for each 32-token projection chunk occupy six OUT windows. Attention
    output overwrites that chunk's two Q windows only after each head's Q has
    been gathered; its K/V windows stay live for later query blocks. The
    caller must run every QKV projection chunk first, then all attention
    blocks, before reusing the K/V windows for O-projection output.
    ``qkv_already_rope`` is for callers whose Q/K projections were rotated in
    an earlier staged program; V is never RoPE-transformed.

    The current OUT window budget supports a complete 64-token, FP16 Q/K/V
    working set (two six-window QKV chunks) plus head-scattered outputs in the
    respective Q windows. Longer contexts need a larger/paged cache allocation
    and are intentionally rejected here rather than silently aliasing data.
    """
    if sequence_length != 64:
        raise ValueError("production tiled Prefill currently requires 64 tokens")
    if (query_count not in (16, 32) or query_start < 0 or
            query_start % 16 or
            query_start + query_count > sequence_length):
        raise ValueError("Prefill FlashAttention blocks must be aligned 16/32-row tiles")
    query_local_start = query_start % 32
    if (query_local_start + query_count > 32 or
            (query_count == 32 and query_local_start != 0)):
        raise ValueError("query block must fit its 32-token QKV chunk")
    if len(qkv_window_bases) != 2:
        raise ValueError("64-token Prefill requires two QKV window groups")
    normalized_bases = tuple(int(base) for base in qkv_window_bases)
    if normalized_bases != (0, 6):
        raise ValueError("64-token QKV windows must use the validated 0/6 layout")
    if first_position < 0:
        raise ValueError("first_position must be non-negative")
    if pair_adjacent_heads:
        if query_count != TILE:
            raise ValueError("paired production FlashAttention requires an M32 query tile")
        return _emit_prefill_flash_attention_paired_head_block(
            manifest, query_start=query_start,
            sequence_length=sequence_length,
            qkv_window_bases=normalized_bases,
            first_position=first_position,
            qkv_already_rope=qkv_already_rope,
            serialize_pv_rf_transfers=serialize_pv_rf_transfers,
            defer_online_normalization=defer_online_normalization)
    chunk_index = query_start // 32
    qkv_base = normalized_bases[chunk_index]
    scale_base = (0x4000 if reuse_rope_table and not qkv_already_rope
                  else 0x2000)
    program = [isa.set_cfg_k(TILE), *emit_prefill_attention_scale(
        manifest, l1_destination_base=scale_base)]
    if not qkv_already_rope and reuse_rope_table:
        # A causal block never consumes K positions beyond its last query.
        # Cache only that prefix, in 32-position chunks, for all heads.
        rope_tokens = min(sequence_length, query_start + query_count)
        for tile_start in range(0, rope_tokens, TILE):
            tile_tokens = min(TILE, rope_tokens - tile_start)
            program.extend(emit_prefill_rope_table(
                manifest, tokens=tile_tokens,
                first_position=first_position + tile_start,
                l1_destination_base=(PREFILL_ROPE_BASE +
                                     tile_start * PREFILL_ROPE_ROW_BYTES)))

    for head_index in range(16):
        query_base, key_base, value_base = 0x8000, 0xA000, 0xC000
        if not qkv_already_rope and not reuse_rope_table:
            program.extend(emit_prefill_rope_table(
                manifest, tokens=query_count,
                first_position=first_position + query_start))
        program.extend([
            isa.token_loop_begin(query_count, allow_overlap=True),
            _prefill_head_dma_instruction(
                0, first_token=query_local_start, tokens=query_count,
                head_index=head_index, first_output_window=qkv_base,
                l1_destination_base=query_base),
            isa.token_loop_end(),
        ])
        if not qkv_already_rope:
            program.extend(emit_prefill_rope_component(
                query_count, query_base,
                first_position=first_position + query_start,
                row_stride_bytes=PREFILL_HEAD_BYTES,
                rope_table_first_position=(first_position if reuse_rope_table
                                           else first_position + query_start)))

        key_tile_loaders = []
        for tile_start in range(0, sequence_length, TILE):
            tile_chunk = tile_start // 32
            tile_base = normalized_bases[tile_chunk]
            tile_tokens = min(TILE, sequence_length - tile_start)
            loader = [
                isa.token_loop_begin(tile_tokens, allow_overlap=True),
                _prefill_head_dma_instruction(
                    1, first_token=0, tokens=tile_tokens,
                    head_index=head_index, first_output_window=tile_base,
                    l1_destination_base=key_base),
                _prefill_head_dma_instruction(
                    2, first_token=0, tokens=tile_tokens,
                    head_index=head_index, first_output_window=tile_base,
                    l1_destination_base=value_base),
                isa.token_loop_end(),
            ]
            if not qkv_already_rope and not reuse_rope_table:
                loader.extend(emit_prefill_rope_table(
                    manifest, tokens=tile_tokens,
                    first_position=first_position + tile_start))
            if not qkv_already_rope:
                loader.extend(emit_prefill_rope_component(
                    tile_tokens, key_base,
                    first_position=first_position + tile_start,
                    row_stride_bytes=PREFILL_HEAD_BYTES,
                    rope_table_first_position=(first_position if reuse_rope_table
                                               else first_position + tile_start)))
            key_tile_loaders.append(loader)

        program.extend(attention.emit_flash_attention_query_block(
            q_base=query_base, k_base=key_base, v_base=value_base,
            score_base=0xE000, alpha_base=0,
            output_base=0, query_start=query_start,
            query_count=query_count, sequence_length=sequence_length,
            key_tile_loaders=key_tile_loaders,
            head_dim=128, query_row_stride_bytes=PREFILL_HEAD_BYTES,
            output_row_stride=PREFILL_HEAD_BYTES,
            score_scale_base=scale_base, output_page=1, alpha_page=2,
            state_slot_base=0, wait_after_store=False,
            defer_online_normalization=defer_online_normalization))
        program.extend(isa.dma_wr_head_group_scatter(
            hidden_size=128, tokens=query_count,
            output_window=qkv_base + query_local_start // 16,
            output_offset=head_index * PREFILL_HEAD_BYTES,
            page=1))
    program.extend([isa.barrier(), isa.end()])
    if len(program) > 262144:
        raise ValueError(
            f"64-token Prefill FlashAttention block has {len(program)} instructions")
    return program


def emit_prefill_flash_attention_kv_source(
        manifest, query_start, sequence_length, head_index,
        query_count=32, q_source="act", k_source="act", v_source="wgt",
        q_source_base=0, k_source_base=0, v_source_base=0,
        query_l1_base=0x0000, key_l1_base=0x4000, value_l1_base=0x6000,
        score_base=0xE000, scale_base=0x2000, output_base=0,
        output_page=1, alpha_base=0, output_window=None,
        output_offset=None, load_scale=True, finish_program=True,
        prefetch_kv_tiles=False, serialize_pv_rf_transfers=True,
        defer_online_normalization=False):
    """Emit one causal M<=32 FlashAttention tile over DDR-resident Q/K/V.

    Q is gathered once into L1. K/V are gathered as 32-token tiles from their
    token-major DDR arrays and consumed by the existing online-softmax engine
    before the next tile replaces them. This is the long-context path: context
    length changes the number of K/V tiles, never the fixed 32-row hardware
    tile or its bounded L1 footprint. Source bases are relative to the selected
    ACT/WGT DDR window and each token occupies a 4-KiB row.
    """
    if not 1 <= sequence_length <= attention.MAX_FLASH_SEQUENCE_LENGTH:
        raise ValueError(
            "FlashAttention sequence length must be in "
            f"[1,{attention.MAX_FLASH_SEQUENCE_LENGTH}]")
    if (query_start < 0 or query_count < 1 or query_count > TILE or
            query_start + query_count > sequence_length):
        raise ValueError("FlashAttention query tile must fit the causal sequence")
    if not 0 <= head_index < 16:
        raise ValueError("Qwen attention head index must be in [0,15]")
    sources = (q_source, k_source, v_source)
    if any(source not in ("act", "wgt") for source in sources):
        raise ValueError("long-context Q/K/V sources must select ACT or WGT DDR")
    bases = (q_source_base, k_source_base, v_source_base)
    for base in bases:
        if base < 0 or base % 256:
            raise ValueError("long-context Q/K/V source bases must be 256-byte aligned")
    if output_page not in (0, 1, 2):
        raise ValueError("FlashAttention output page must be 0, 1, or 2")
    if output_window is not None and not 0 <= output_window < 16:
        raise ValueError("FlashAttention output window must be in [0,15]")
    if output_offset is None:
        output_offset = head_index * PREFILL_HEAD_BYTES
    if output_window is not None and (
            output_offset < 0 or output_offset % 32 or
            output_offset + PREFILL_HEAD_BYTES > PREFILL_TOKEN_SLOT_BYTES):
        raise ValueError("FlashAttention output head must fit a token row")

    head_offset = head_index * PREFILL_HEAD_BYTES
    for base, first_token, last_token in (
            (q_source_base, query_start, query_start + query_count - 1),
            (k_source_base, 0, sequence_length - 1),
            (v_source_base, 0, sequence_length - 1)):
        source_end = (base + last_token * PREFILL_TOKEN_SLOT_BYTES +
                      head_offset + PREFILL_HEAD_BYTES)
        if (base + first_token * PREFILL_TOKEN_SLOT_BYTES + head_offset >=
                1 << 24 or source_end > 1 << 24):
            raise ValueError("Q/K/V token range exceeds the 16-MiB gather window")

    page_bytes = {0: 65536, 1: 65536, 2: 16384}
    l1_bases = (query_l1_base, key_l1_base, value_l1_base,
                score_base, scale_base)
    if any(base < 0 or base % 4096 for base in l1_bases):
        raise ValueError("Q/K/V/score/scale buffers must start on an L1 bank")
    q_end = query_l1_base + (query_count - 1) * PREFILL_HEAD_BYTES + PREFILL_HEAD_BYTES
    k_end = key_l1_base + (TILE - 1) * PREFILL_HEAD_BYTES + PREFILL_HEAD_BYTES
    v_end = value_l1_base + (TILE - 1) * PREFILL_HEAD_BYTES + PREFILL_HEAD_BYTES
    score_end = score_base + query_count * TILE * 2
    scale_end = scale_base + 256
    live_ranges = ((query_l1_base, q_end), (key_l1_base, k_end),
                   (value_l1_base, v_end), (score_base, score_end),
                   (scale_base, scale_end))
    if (any(end > 65536 for _, end in live_ranges) or
            any(left[0] < right[1] and right[0] < left[1]
                for index, left in enumerate(live_ranges)
                for right in live_ranges[index + 1:]) or
            output_base < 0 or output_base % 32 or
            output_base + query_count * PREFILL_HEAD_BYTES > page_bytes[output_page]):
        raise ValueError("long-context FlashAttention L1 tiles overlap or overflow")

    query_offset = (q_source_base + query_start * PREFILL_TOKEN_SLOT_BYTES +
                    head_offset)
    program = [isa.set_cfg_k(TILE)]
    if load_scale:
        program.extend(emit_prefill_attention_scale(manifest, scale_base))
    program.extend([
        isa.token_loop_begin(query_count, allow_overlap=True),
        isa.dma_rd_head_token_group(
            query_offset, PREFILL_HEAD_BYTES,
            l1_bank=query_l1_base >> 12, tokens=query_count,
            source=q_source),
        isa.token_loop_end(),
    ])

    tile_count = (sequence_length + TILE - 1) // TILE
    if prefetch_kv_tiles and tile_count <= 1:
        raise ValueError("K/V tile prefetch requires at least two key tiles")
    key_tile_loaders = []
    key_tile_bases = []
    for tile_index, tile_start in enumerate(range(0, sequence_length, TILE)):
        tile_tokens = min(TILE, sequence_length - tile_start)
        if prefetch_kv_tiles:
            tile_slot = tile_index & 1
            tile_key_base, tile_value_base = (
                ((0x4000, 0x6000), (0x8000, 0xA000))[tile_slot])
            key_tile_bases.append((tile_key_base, tile_value_base))
        else:
            tile_key_base, tile_value_base = key_l1_base, value_l1_base
        key_offset = (k_source_base + tile_start * PREFILL_TOKEN_SLOT_BYTES +
                      head_offset)
        value_offset = (v_source_base + tile_start * PREFILL_TOKEN_SLOT_BYTES +
                        head_offset)
        key_tile_loaders.append([
            isa.token_loop_begin(tile_tokens, allow_overlap=True),
            isa.dma_rd_head_token_group(
                key_offset, PREFILL_HEAD_BYTES,
                l1_bank=tile_key_base >> 12, tokens=tile_tokens,
                source=k_source),
            isa.dma_rd_head_token_group(
                value_offset, PREFILL_HEAD_BYTES,
                l1_bank=tile_value_base >> 12, tokens=tile_tokens,
                source=v_source),
            isa.token_loop_end(),
        ])

    program.extend(attention.emit_flash_attention_query_block(
        q_base=query_l1_base, k_base=key_l1_base, v_base=value_l1_base,
        score_base=score_base, alpha_base=alpha_base,
        output_base=output_base, query_start=query_start,
        query_count=query_count, sequence_length=sequence_length,
        key_tile_loaders=key_tile_loaders,
        key_tile_bases=(key_tile_bases if prefetch_kv_tiles else None),
        key_tile_prefetchers=(key_tile_loaders[1:]
                              if prefetch_kv_tiles else None),
        head_dim=128, query_row_stride_bytes=PREFILL_HEAD_BYTES,
        output_row_stride=PREFILL_HEAD_BYTES,
        score_scale_base=scale_base, output_page=output_page,
        alpha_page=2, state_slot_base=0, wait_after_store=False,
        serialize_pv_rf_transfers=serialize_pv_rf_transfers,
        defer_online_normalization=defer_online_normalization))
    if output_window is not None:
        program.extend(isa.dma_wr_head_group_scatter(
            hidden_size=128, tokens=query_count,
            output_window=output_window, output_offset=output_offset,
            page=output_page))
    if finish_program:
        program.extend([isa.barrier(), isa.end()])
    if len(program) > 262144:
        raise ValueError(
            f"FlashAttention query tile has {len(program)} instructions")
    return program


def emit_prefill_flash_attention_kv_source_group(
        manifest, query_start, sequence_length, query_count=32,
        head_start=0, head_count=16, q_source="act", k_source="act",
        v_source="wgt", q_source_base=0, k_source_base=0,
        v_source_base=0, output_window=None, prefetch_kv_tiles=False,
        pair_adjacent_heads=False, serialize_pv_rf_transfers=True,
        defer_online_normalization=False):
    """Emit one hidden-row query tile, reusing L1 across attention heads.

    ``pair_adjacent_heads`` gathers two adjacent head fields in one 512-byte
    token transfer and reuses each K/V tile for both independent attention
    streams. It is limited to the M32 compact-output layout; both head results
    are scattered to their original feature ranges before L1 is reused.
    """
    if head_start < 0 or head_count < 1 or head_start + head_count > 16:
        raise ValueError("attention head group must fit the 16-head projection")
    if output_window is not None and not 0 <= output_window < 16:
        raise ValueError("FlashAttention output window must be in [0,15]")

    if pair_adjacent_heads:
        if (head_start % 2 or head_count % 2 or query_count > TILE or
                output_window is None or prefetch_kv_tiles):
            raise ValueError(
                "paired-head tiles require even head ranges, M<=32, an output window, and no K/V prefetch")
        query_base, key_base, value_base = 0x0000, 0x4000, 0x8000
        score_base, scale_base = 0xE000, 0xF000
        program = [isa.set_cfg_k(TILE),
                   *emit_prefill_attention_scale(manifest, scale_base)]
        for first_head in range(head_start, head_start + head_count, 2):
            head_offset = first_head * PREFILL_HEAD_BYTES
            query_offset = (q_source_base + query_start *
                            PREFILL_TOKEN_SLOT_BYTES + head_offset)
            program.extend([
                isa.token_loop_begin(query_count, allow_overlap=True),
                isa.dma_rd_head_token_group(
                    query_offset, 2 * PREFILL_HEAD_BYTES,
                    l1_bank=query_base >> 12, tokens=query_count,
                    source=q_source),
                isa.token_loop_end(),
            ])

            loaders = []
            for tile_start in range(0, sequence_length, TILE):
                tile_tokens = min(TILE, sequence_length - tile_start)
                key_offset = (k_source_base + tile_start *
                              PREFILL_TOKEN_SLOT_BYTES + head_offset)
                value_offset = (v_source_base + tile_start *
                                PREFILL_TOKEN_SLOT_BYTES + head_offset)
                loaders.append([
                    isa.token_loop_begin(tile_tokens, allow_overlap=True),
                    isa.dma_rd_head_token_group(
                        key_offset, 2 * PREFILL_HEAD_BYTES,
                        l1_bank=key_base >> 12, tokens=tile_tokens,
                        source=k_source),
                    isa.dma_rd_head_token_group(
                        value_offset, 2 * PREFILL_HEAD_BYTES,
                        l1_bank=value_base >> 12, tokens=tile_tokens,
                        source=v_source),
                    isa.token_loop_end(),
                ])

            program.extend(attention.emit_flash_attention_query_block(
                q_base=query_base, k_base=key_base, v_base=value_base,
                score_base=score_base, alpha_base=0, output_base=0,
                query_start=query_start, query_count=query_count,
                sequence_length=sequence_length,
                key_tile_loaders=loaders, head_dim=128,
                query_row_stride_bytes=2 * PREFILL_HEAD_BYTES,
                output_row_stride=PREFILL_HEAD_BYTES,
                output_page=1, alpha_page=2, state_slot_base=0,
                wait_after_store=False,
                k_token_row_stride_bytes=2 * PREFILL_HEAD_BYTES,
                v_token_row_stride_bytes=2 * PREFILL_HEAD_BYTES,
                score_scale_base=scale_base,
                serialize_pv_rf_transfers=serialize_pv_rf_transfers,
                defer_online_normalization=defer_online_normalization,
                additional_heads=((
                    query_base + PREFILL_HEAD_BYTES,
                    key_base + PREFILL_HEAD_BYTES,
                    value_base + PREFILL_HEAD_BYTES,
                    0x2000, query_count,
                ),)))
            for head_lane in range(2):
                program.extend(isa.dma_wr_head_group_scatter(
                    hidden_size=128, tokens=query_count,
                    output_window=output_window,
                    output_offset=(first_head + head_lane) *
                                  PREFILL_HEAD_BYTES,
                    page=1, source_l1_offset=head_lane * 0x2000))
        program.extend([isa.barrier(), isa.end()])
        if len(program) > 262144:
            raise ValueError(
                f"paired-head FlashAttention tile has {len(program)} instructions")
        return program

    program = [isa.set_cfg_k(TILE),
               *emit_prefill_attention_scale(manifest, 0x2000)]
    for head_index in range(head_start, head_start + head_count):
        program.extend(emit_prefill_flash_attention_kv_source(
            manifest, query_start=query_start,
            sequence_length=sequence_length, head_index=head_index,
            query_count=query_count, q_source=q_source, k_source=k_source,
            v_source=v_source, q_source_base=q_source_base,
            k_source_base=k_source_base, v_source_base=v_source_base,
            output_window=output_window,
            output_offset=head_index * PREFILL_HEAD_BYTES,
            load_scale=False, finish_program=False,
            prefetch_kv_tiles=prefetch_kv_tiles,
            serialize_pv_rf_transfers=serialize_pv_rf_transfers,
            defer_online_normalization=defer_online_normalization))
    program.extend([isa.barrier(), isa.end()])
    if len(program) > 262144:
        raise ValueError(
            f"FlashAttention hidden query tile has {len(program)} instructions")
    return program


def emit_prefill_flash_attention_kv_source_tiled_programs(
        manifest, sequence_length, query_start=0, query_count=None,
        query_tile_rows=32, head_start=0, head_count=16,
        q_source="act", k_source="act", v_source="wgt",
        q_source_base=0, k_source_base=0, v_source_base=0,
        output_window_base=0, pair_adjacent_heads=None,
        serialize_pv_rf_transfers=True,
        defer_online_normalization=False):
    """Emit separate M16/M32 FlashAttention launches for a query range.

    Q/K/V are preprojected, token-major DDR arrays. Each returned program is
    one query tile over a stream of 32-key tiles and uses the hardware online
    Softmax state; no sequence-sized score matrix or widened GEMM is created.
    The output range is scattered to consecutive 16-token OUT windows.
    """
    if not 1 <= sequence_length <= attention.MAX_FLASH_SEQUENCE_LENGTH:
        raise ValueError(
            "FlashAttention sequence length must be in "
            f"[1,{attention.MAX_FLASH_SEQUENCE_LENGTH}]")
    if query_count is None:
        query_count = sequence_length - query_start
    if (query_start < 0 or query_start % 16 or query_count < 1 or
            query_start + query_count > sequence_length):
        raise ValueError("query range must be nonempty and 16-token aligned")
    if query_tile_rows not in (16, 32):
        raise ValueError("FlashAttention query tiles must use M16 or M32")
    if pair_adjacent_heads is None:
        pair_adjacent_heads = query_tile_rows == 32
    if not isinstance(pair_adjacent_heads, bool):
        raise ValueError("paired-head scheduling must be selected with a boolean")
    if (head_start < 0 or head_count < 1 or
            head_start + head_count > 16):
        raise ValueError("attention head group must fit the 16-head projection")
    if pair_adjacent_heads and (head_start % 2 or head_count % 2):
        raise ValueError("paired-head FlashAttention requires even head ranges")
    output_windows = (query_count + 15) // 16
    if not 0 <= output_window_base <= 16 - output_windows:
        raise ValueError("FlashAttention output range exceeds OUT DDR windows")

    programs = []
    for query_offset in range(0, query_count, query_tile_rows):
        tile_start = query_start + query_offset
        tile_rows = min(query_tile_rows, query_count - query_offset)
        if tile_start % 16:
            raise ValueError("query tile starts must align to 16-token OUT rows")
        programs.append(emit_prefill_flash_attention_kv_source_group(
            manifest, query_start=tile_start, sequence_length=sequence_length,
            query_count=tile_rows, head_start=head_start,
            head_count=head_count, q_source=q_source, k_source=k_source,
            v_source=v_source, q_source_base=q_source_base,
            k_source_base=k_source_base, v_source_base=v_source_base,
            output_window=output_window_base + query_offset // 16,
            pair_adjacent_heads=pair_adjacent_heads,
            serialize_pv_rf_transfers=serialize_pv_rf_transfers,
            defer_online_normalization=defer_online_normalization))
    return tuple(programs)


def emit_prefill_flash_attention_kv_source_reuse_group(
        manifest, query_start, sequence_length, query_count=64,
        head_start=0, head_count=16, q_source="act", k_source="act",
        v_source="wgt", q_source_base=0, k_source_base=0,
        v_source_base=0, output_window_base=None,
        serialize_pv_rf_transfers=True):
    """Emit up to four sequential physical M32 query tiles as one K/V-reuse group.

    The hardware GEMM and online-softmax datapaths still operate on physical
    M32xK32 tiles. The compiler keeps each gathered K/V tile resident while
    processing up to four consecutive M32 query tiles, reducing repeated DDR
    gathers without constructing or issuing a wider query matrix.
    """
    if not 1 <= sequence_length <= attention.MAX_FLASH_SEQUENCE_LENGTH:
        raise ValueError(
            "FlashAttention sequence length must be in "
            f"[1,{attention.MAX_FLASH_SEQUENCE_LENGTH}]")
    if (query_start < 0 or query_start % 64 or query_count < 1 or
            query_count > 128 or query_start + query_count > sequence_length):
        raise ValueError(
            "K/V-reuse query supertiles must start on an M64 boundary and contain 1..128 rows")
    if head_start < 0 or head_count < 1 or head_start + head_count > 16:
        raise ValueError("attention head group must fit the 16-head projection")
    if any(source not in ("act", "wgt")
           for source in (q_source, k_source, v_source)):
        raise ValueError("long-context Q/K/V sources must select ACT or WGT DDR")
    for base in (q_source_base, k_source_base, v_source_base):
        if base < 0 or base % 256:
            raise ValueError("long-context Q/K/V source bases must be 256-byte aligned")
    if output_window_base is None:
        output_window_base = (query_start // 16) % 16
    if not 0 <= output_window_base < 16:
        raise ValueError("attention output window base must be in [0,15]")
    for base, first_token, last_token in (
            (q_source_base, query_start, query_start + query_count - 1),
            (k_source_base, 0, sequence_length - 1),
            (v_source_base, 0, sequence_length - 1)):
        source_end = (base + last_token * PREFILL_TOKEN_SLOT_BYTES +
                      (head_start + head_count) * PREFILL_HEAD_BYTES)
        if (base + first_token * PREFILL_TOKEN_SLOT_BYTES +
                head_start * PREFILL_HEAD_BYTES >= 1 << 24 or
                source_end > 1 << 24):
            raise ValueError("Q/K/V token range exceeds the 16-MiB gather window")

    if query_count <= 64:
        q_l1_base, k_l1_base, v_l1_base = 0x0000, 0x4000, 0x6000
        score_base, scale_base = 0xE000, 0xF000
    else:
        q_l1_base, k_l1_base, v_l1_base = 0x0000, 0x8000, 0xA000
        score_base, scale_base = 0xC000, 0xD000
    program = [isa.set_cfg_k(TILE),
               *emit_prefill_attention_scale(manifest, scale_base)]
    for head_index in range(head_start, head_start + head_count):
        head_offset = head_index * PREFILL_HEAD_BYTES
        for query_offset in range(0, query_count, TILE):
            query_rows = min(TILE, query_count - query_offset)
            query_bank = (q_l1_base + query_offset * PREFILL_HEAD_BYTES) >> 12
            query_ddr_offset = (q_source_base +
                                (query_start + query_offset) *
                                PREFILL_TOKEN_SLOT_BYTES + head_offset)
            program.extend([
                isa.token_loop_begin(query_rows, allow_overlap=True),
                isa.dma_rd_head_token_group(
                    query_ddr_offset, PREFILL_HEAD_BYTES,
                    l1_bank=query_bank, tokens=query_rows,
                    source=q_source),
                isa.token_loop_end(),
            ])

        loaders = []
        for tile_start in range(0, sequence_length, TILE):
            tile_rows = min(TILE, sequence_length - tile_start)
            key_ddr_offset = (k_source_base +
                              tile_start * PREFILL_TOKEN_SLOT_BYTES +
                              head_offset)
            value_ddr_offset = (v_source_base +
                                tile_start * PREFILL_TOKEN_SLOT_BYTES +
                                head_offset)
            loaders.append([
                isa.token_loop_begin(tile_rows, allow_overlap=True),
                isa.dma_rd_head_token_group(
                    key_ddr_offset, PREFILL_HEAD_BYTES,
                    l1_bank=k_l1_base >> 12, tokens=tile_rows,
                    source=k_source),
                isa.dma_rd_head_token_group(
                    value_ddr_offset, PREFILL_HEAD_BYTES,
                    l1_bank=v_l1_base >> 12, tokens=tile_rows,
                    source=v_source),
                isa.token_loop_end(),
            ])

        program.extend(attention.emit_flash_attention_query_block_reusing_kv_tiles(
            q_base=q_l1_base, k_base=k_l1_base, v_base=v_l1_base,
            score_base=score_base, alpha_base=0, output_base=0,
            query_start=query_start, query_count=query_count,
            sequence_length=sequence_length,
            key_tile_loaders=loaders, head_dim=128,
            query_row_stride_bytes=PREFILL_HEAD_BYTES,
            output_row_stride=PREFILL_HEAD_BYTES,
            score_scale_base=scale_base, output_page=1, alpha_page=2,
            state_slot_base=0, wait_after_store=False,
            serialize_pv_rf_transfers=serialize_pv_rf_transfers))
        for query_offset in range(0, query_count, TILE):
            query_rows = min(TILE, query_count - query_offset)
            program.extend(isa.dma_wr_head_group_scatter(
                hidden_size=128, tokens=query_rows,
                output_window=(output_window_base + query_offset // 16) % 16,
                output_offset=head_index * PREFILL_HEAD_BYTES,
                page=1, source_l1_offset=query_offset * PREFILL_HEAD_BYTES))
    program.extend([isa.barrier(), isa.end()])
    if len(program) > 262144:
        raise ValueError(
            f"K/V-reuse FlashAttention supertile has {len(program)} instructions")
    return program


def emit_prefill_flash_attention_programs(
        manifest, input_start_token=0, first_position=0,
        query_tile_rows=32, pair_adjacent_heads=None):
    """Emit staged QKV -> tiled FlashAttention -> O projection for 64 tokens.

    Programs use OUT windows 0..11 for the two QKV blocks. Attention results
    are scattered over the corresponding Q windows, preserving all K/V data
    until both query blocks finish. O projections then reuse the expired K
    windows for output, yielding token-major 32-row chunks in windows 2 and 8.
    """
    if (input_start_token < 0 or
            input_start_token + 64 > MAX_PREFILL_INPUT_TOKENS):
        raise ValueError("64-token Prefill exceeds the ACT activation DMA window")
    if query_tile_rows not in (16, 32):
        raise ValueError("Prefill query tile rows must be M16 or M32")
    if pair_adjacent_heads is None:
        pair_adjacent_heads = query_tile_rows == TILE
    if not isinstance(pair_adjacent_heads, bool):
        raise ValueError("pair_adjacent_heads must be a bool")
    if pair_adjacent_heads and query_tile_rows != TILE:
        raise ValueError("paired-head Prefill requires M32 query tiles")
    if first_position < 0:
        raise ValueError("first_position must be non-negative")

    qkv_programs = tuple(
        emit_prefill_qkv_projections(
            manifest, tokens=32,
            input_start_token=input_start_token + chunk_start,
            first_output_window=qkv_base, output_start_token=0)
        for chunk_start, qkv_base in ((0, 0), (32, 6)))
    attention_programs = tuple(
        emit_prefill_flash_attention_block(
            manifest, query_start=query_start,
            query_count=min(query_tile_rows, 64 - query_start),
            sequence_length=64, qkv_window_bases=(0, 6),
            first_position=first_position,
            pair_adjacent_heads=pair_adjacent_heads)
        for query_start in range(0, 64, query_tile_rows))
    o_projection_programs = tuple(
        emit_prefill_fp16_projection(
            manifest, "self_attn.o_proj", tokens=32,
            output_window_base=qkv_base + 2, output_start_token=0,
            input_source_windows=[
                (qkv_base + output_subwindow, first_token)
                for output_subwindow in (0, 1)
                for first_token in (0, 8)])
        for qkv_base in (0, 6))
    return {
        "qkv": qkv_programs,
        "attention": attention_programs,
        "o_projection": o_projection_programs,
    }


def emit_grouped_moe_prefill_stage(manifest, tokens=32, rows_per_group=None,
                                   cache_routes=True,
                                   activation_source_window=None):
    """Lower the grouped routed-MoE stage for a prepared prefill batch.

    The caller must have written each token's post-attention MoE activation to
    ACT DDR, or to a contiguous 4-KiB/token OUT window when
    ``activation_source_window`` is provided, and produced that token's router
    logits in the TopK ACC row. This stage can optionally cache all Top-4 results,
    then executes descriptor-driven experts in compact row groups. It is a
    routed-MoE stage, not a complete Layer-0 prefill schedule.
    """
    router = _linear_region(manifest, "mlp.gate")
    if router.rows != 60:
        raise ValueError("routed prefill requires logits for all 60 experts")
    intermediate_size = _routed_intermediate_size(manifest)
    if (router.cols < 32 or router.cols > 2048 or router.cols % 32 or
            intermediate_size < 32 or intermediate_size % 32):
        raise ValueError("grouped prefill dimensions exceed routed-MoE L1 geometry")
    if activation_source_window is not None and (
            router.cols != 2048 or not 0 <= activation_source_window < 16 or
            activation_source_window * PREFILL_OUTPUT_WINDOW_BYTES +
            (tokens - 1) * PREFILL_TOKEN_SLOT_BYTES +
            PREFILL_TOKEN_SLOT_BYTES > (1 << 20)):
        raise ValueError("OUT activation source must be a valid full-width token window")
    return routed_moe.emit_all_experts_grouped_batches(
        tokens=tokens, rows_per_group=rows_per_group,
        hidden_size=router.cols, intermediate_size=intermediate_size,
        num_experts=60, cache_routes=cache_routes,
        scale_metadata_n_invariant=bool(manifest.get(
            "routed_expert_scale_metadata_n_invariant", False)),
        activation_source=("act" if activation_source_window is None else "out"),
        activation_byte_offset=(0 if activation_source_window is None else
                                activation_source_window *
                                PREFILL_OUTPUT_WINDOW_BYTES))


def emit_prefill_router_topk_cache(manifest, tokens=32,
                                   activation_source_window=None):
    """Compute and cache Top-4 routes from normalized prefill token rows.

    By default, ACT rows contain the activations at token offsets
    ``0..tokens-1``. ``activation_source_window`` selects full-width 4-KiB
    token rows in OUT DDR. Eight token rows share each Router weight
    traversal; Top-4 then reads each token's own accumulator row.
    """
    if not 1 <= tokens <= 32:
        raise ValueError("prefill router cache supports 1..32 tokens")
    router = _linear_region(manifest, "mlp.gate")
    if router.nbits != 16 or (router.rows, router.cols) != (60, 2048):
        raise ValueError("prefill Top-4 routing requires an FP16 [60,2048] router")
    if activation_source_window is not None and (
            not 0 <= activation_source_window < 16 or
            activation_source_window + (tokens - 1) // 16 >= 16):
        raise ValueError("OUT activation source window exceeds the token DDR range")
    program = []
    configured_m = None
    configured_k = None
    router_cfg_k = 32 * tiles_per_gemm(router)
    for token_start in range(0, tokens, PREFILL_GROUP_ROWS):
        group_tokens = min(PREFILL_GROUP_ROWS, tokens - token_start)
        if activation_source_window is None:
            input_dma = isa.dma_rd_token_group(
                token_start * PREFILL_TOKEN_SLOT_BYTES, router.cols,
                l1_bank=L1_NORM >> 12, tokens=group_tokens)
        else:
            source_window = activation_source_window + token_start // 16
            source_token = token_start % 16
            group_tokens = min(group_tokens, 16 - source_token)
            input_dma = isa.dma_rd_out_token_group(
                source_window * PREFILL_OUTPUT_WINDOW_BYTES +
                source_token * PREFILL_TOKEN_SLOT_BYTES,
                transfer_bytes=PREFILL_TOKEN_SLOT_BYTES,
                l1_bank=L1_NORM >> 12, tokens=group_tokens)
        program.extend([isa.token_loop_begin(group_tokens), input_dma,
                        isa.token_loop_end()])
        if group_tokens != configured_m:
            program.append(isa.set_cfg_m(group_tokens))
            configured_m = group_tokens
        if router_cfg_k != configured_k:
            program.append(isa.set_cfg_k(router_cfg_k))
            configured_k = router_cfg_k
        program.extend(static_linear_fp16_batch.emit_fp16_batch(
            router, L1_NORM, output_base=0, tokens=group_tokens,
            activation_page=0, output_page=2,
            activation_row_stride_bytes=PREFILL_TOKEN_SLOT_BYTES,
            output_row_stride_bytes=router.n_tiles * 64,
            configure_cfg=False, restore_cfg=False,
            configure_cfg_m=False, restore_cfg_m=False,
            configure_cfg_k=False, restore_cfg_k=False,
            acc_only=True))
        # TOPK consumes the two ACC sets directly; draining logits to L1 would
        # only add conversion, vector, and store work before these commands.
        for local_row in range(group_tokens):
            program.append(isa.topk_acc_row(
                source_row=local_row, token=token_start + local_row))
    if configured_m is not None and configured_m != 1:
        program.append(isa.set_cfg_m(1))
    if configured_k is not None and configured_k != 32:
        program.append(isa.set_cfg_k(32))
    return program


def emit_prefill_routed_moe_stage(manifest, tokens=32, rows_per_group=None,
                                  activation_source_window=None):
    """Emit Router/Top-4 followed by descriptor-driven grouped routed experts.

    Post-attention normalized activations must already be stored in ACT DDR at
    token offsets ``0..tokens-1``, or in a full-width OUT DDR window selected
    by ``activation_source_window``. The result is the routed-expert contribution
    in compact OUT windows; shared-expert and residual assembly remain separate
    stages. Every expert projection continues to select its 1/2/3-bit format
    from the active hardware descriptor.
    """
    program = emit_prefill_router_topk_cache(
        manifest, tokens=tokens,
        activation_source_window=activation_source_window)
    program.extend(emit_grouped_moe_prefill_stage(
        manifest, tokens=tokens, rows_per_group=rows_per_group,
        cache_routes=False,
        activation_source_window=activation_source_window))
    program.append(isa.end())
    if len(program) > 262144:
        raise ValueError(f"prefill routed-MoE program has {len(program)} instructions")
    return program


def emit_prefill_post_attention_norm(
        manifest, tokens=32, input_start_token=0,
        attention_output_window_base=10, norm_output_window_base=12,
        residual_output_window_base=14):
    """Add the attention residual, RMS-normalize it, and preserve both tensors.

    The raw residual input is read from ACT DDR. The O-projection result is
    read from its token-major OUT windows. The normalized sum is written to
    ``norm_output_window_base`` and the unnormalized residual sum is retained
    in ``residual_output_window_base`` for the final MLP residual epilogue.
    The returned fragment has no END and can be composed with later stages.
    """
    if (not 1 <= tokens <= 32 or
            not 0 <= input_start_token <= 32 - tokens):
        raise ValueError("post-attention normalization range must fit 1..32 tokens")
    window_count = (tokens + 15) // 16
    windows = (
        (attention_output_window_base,
         attention_output_window_base + window_count),
        (norm_output_window_base, norm_output_window_base + window_count),
        (residual_output_window_base,
         residual_output_window_base + window_count),
    )
    if any(start < 0 or stop > 16 for start, stop in windows):
        raise ValueError("post-attention normalization windows exceed OUT DDR")
    for index, (start, stop) in enumerate(windows):
        for other_start, other_stop in windows[index + 1:]:
            if start < other_stop and other_start < stop:
                raise ValueError("post-attention inputs and outputs must use disjoint windows")

    gamma_base = _vector_region(manifest, "post_attention_layernorm")
    program = _load_4k(
        gamma_base, l1_bank=L1_NORM2_GAMMA >> 12)
    residual_base = 0x5000
    attention_base = 0x2000
    normalized_base = 0x6000
    body = [
        isa.dma_rd_token(input_start_token * PREFILL_TOKEN_SLOT_BYTES,
                         hidden_size=2048, l1_bank=L1_RESIDUAL_IN >> 12),
        isa.dma_rd_out_token(
            attention_output_window_base * PREFILL_OUTPUT_WINDOW_BYTES,
            hidden_size=2048, l1_bank=attention_base >> 12),
        isa.barrier(),
    ]
    for chunk in range(2048 // TILE):
        offset = chunk * VECTOR_BYTES
        body.extend([
            isa.vld(0, L1_RESIDUAL_IN + offset),
            isa.vld(1, attention_base + offset),
            isa.add(0, 0, 1),
            isa.rmsnorm_accumulate(0, reset=chunk == 0),
            isa.vst(0, residual_base + offset),
        ])
    body.extend(normalization.emit_rmsnorm_apply(
        residual_base, L1_NORM2_GAMMA, normalized_base,
        hidden_size=2048, finalize=True, wait_after_store=True))
    body.extend([
        isa.dma_wr_token(
            residual_base, hidden_size=2048,
            output_window=residual_output_window_base),
        isa.dma_wr_token(
            normalized_base, hidden_size=2048,
            output_window=norm_output_window_base),
    ])
    program.extend([isa.token_loop_begin(tokens), *body,
                    isa.token_loop_end()])
    return program


def emit_prefill_shared_expert_stage(
        manifest, tokens=32, rows_per_group=PREFILL_GROUP_ROWS,
        norm_input_window_base=12, routed_output_window_base=0,
        residual_window_base=14, output_window_base=4,
        input_start_token=0, routed_rows_per_group=PREFILL_GROUP_ROWS):
    """Run the shared expert in L1-safe groups and fuse route plus residual.

    Routed results are stored one token group per OUT window, matching
    ``emit_prefill_routed_moe_stage``. Batched FP16 or quantized shared
    Gate/Up/Down projections reuse each weight traversal across the group.
    Final output uses one OUT window per token group.
    """
    if (not 1 <= tokens <= 32 or rows_per_group not in (4, 8) or
            tokens % rows_per_group or
            not 0 <= input_start_token <= 32 - tokens or
            input_start_token % rows_per_group):
        raise ValueError("shared prefill requires complete 4- or 8-row groups")
    if not 1 <= routed_rows_per_group <= 32:
        raise ValueError("routed group size must be in [1,32]")
    group_count = tokens // rows_per_group
    token_end = input_start_token + tokens
    first_norm_group = input_start_token // 16
    last_norm_group = (token_end - 1) // 16
    first_route_group = input_start_token // routed_rows_per_group
    last_route_group = (token_end - 1) // routed_rows_per_group
    source_ranges = (
        (norm_input_window_base + first_norm_group,
         norm_input_window_base + last_norm_group + 1),
        (routed_output_window_base + first_route_group,
         routed_output_window_base + last_route_group + 1),
        (residual_window_base + first_norm_group,
         residual_window_base + last_norm_group + 1),
        (output_window_base, output_window_base + group_count),
    )
    if any(start < 0 or stop > 16 for start, stop in source_ranges):
        raise ValueError("shared prefill tensors exceed OUT DDR windows")
    for index, (start, stop) in enumerate(source_ranges):
        for other_start, other_stop in source_ranges[index + 1:]:
            if start < other_stop and other_start < stop:
                raise ValueError("shared prefill input/output windows must not overlap")

    gate = _linear_region(manifest, "mlp.shared_expert.gate_proj")
    up = _linear_region(manifest, "mlp.shared_expert.up_proj")
    down = _linear_region(manifest, "mlp.shared_expert.down_proj")
    scalar = _linear_region(manifest, "mlp.shared_expert_gate")
    hidden_size = gate.cols
    if hidden_size != 2048:
        raise ValueError("prefill shared-expert path requires hidden_size=2048")
    batch_shared = (scalar.nbits == 16 and all(
        region.nbits in (1, 2, 3, 4, 16) for region in (gate, up, down)))
    if (not batch_shared and
            (routed_rows_per_group < rows_per_group or
             routed_rows_per_group % rows_per_group)):
        raise ValueError(
            "non-batched shared prefill requires aligned routed groups")

    norm_base = L1_NORM
    route_base = L1_ROUTED_OUT
    residual_base = L1_ATTN_RESIDUAL
    program = []
    for group_index in range(group_count):
        token_start = input_start_token + group_index * rows_per_group
        norm_window = norm_input_window_base + token_start // 16
        residual_window = residual_window_base + token_start // 16
        route_window = (routed_output_window_base +
                        token_start // routed_rows_per_group)
        output_window = output_window_base + group_index
        if batch_shared:
            batch_input_base = (0x10000 -
                                rows_per_group * PREFILL_TOKEN_SLOT_BYTES)
            gate_base = 0x4000
            intermediate_bytes = rows_per_group * gate.rows * 2
            down_output_bytes = rows_per_group * hidden_size * 2
            up_page1_base = (max(gate_base + intermediate_bytes,
                                 down_output_bytes) + 63) & ~63
            if up_page1_base + intermediate_bytes <= 0x10000:
                up_base = up_page1_base
                up_page = 1
            else:
                # Up consumes each Gate tile before storing its fused result.
                # Reuse Gate's exact page-1 range in place so the Down input
                # stays outside page-0 static-weight staging banks.
                up_base = gate_base
                up_page = 1
            if up_base + intermediate_bytes > 0x10000:
                raise ValueError(
                    "shared-expert Gate/Up matrices exceed available L1 pages")
            down_output_base = 0
            program.extend([
                isa.token_loop_begin(rows_per_group),
                isa.dma_rd_out_token_group(
                    norm_window * PREFILL_OUTPUT_WINDOW_BYTES +
                    (token_start % 16) * PREFILL_TOKEN_SLOT_BYTES,
                    PREFILL_TOKEN_SLOT_BYTES, l1_bank=batch_input_base >> 12,
                    tokens=rows_per_group),
                # TOKEN_END is all-idle gated and retires the input DMA.
                isa.token_loop_end(),
            ])
            program.extend(shared_expert.emit_shared_expert_batch(
                batch_input_base, gate, up, down, scalar, down_output_base,
                tokens=rows_per_group, gate_base=gate_base, up_base=up_base,
                scalar_base=0, activation_page=0,
                activation_row_stride_bytes=PREFILL_TOKEN_SLOT_BYTES,
                intermediate_page=1, scalar_page=2, output_page=up_page,
                down_output_page=1))

            # Down has retired, so page 0 is free for two full token matrices.
            # The route and residual gathers occupy disjoint low/high halves;
            # the batched Down result remains in page 1 until the epilogue.
            route_segments = []
            route_token = token_start
            destination_row = 0
            rows_left = rows_per_group
            while rows_left:
                source_group = route_token // routed_rows_per_group
                source_row = route_token % routed_rows_per_group
                segment_rows = min(
                    rows_left, routed_rows_per_group - source_row)
                source_window = routed_output_window_base + source_group
                source_offset = (
                    source_window * PREFILL_OUTPUT_WINDOW_BYTES +
                    source_row * PREFILL_TOKEN_SLOT_BYTES)
                route_segments.append((source_offset, destination_row,
                                       segment_rows))
                route_token += segment_rows
                destination_row += segment_rows
                rows_left -= segment_rows

            if len(route_segments) == 1:
                source_offset, destination_row, segment_rows = route_segments[0]
                program.extend([
                    isa.token_loop_begin(rows_per_group),
                    isa.dma_rd_out_token_group(
                        source_offset, hidden_size * 2, l1_bank=0,
                        tokens=segment_rows),
                    isa.dma_rd_out_token_group(
                        residual_window * PREFILL_OUTPUT_WINDOW_BYTES +
                        (token_start % 16) * PREFILL_TOKEN_SLOT_BYTES,
                        hidden_size * 2, l1_bank=8, tokens=rows_per_group),
                    isa.token_loop_end(),
                ])
            else:
                # A shared-expert batch may straddle two compact routed-output
                # windows. Gather each contiguous source segment into the
                # matching destination rows, then gather residuals separately.
                for source_offset, destination_row, segment_rows in route_segments:
                    program.extend([
                        isa.token_loop_begin(segment_rows),
                        isa.dma_rd_out_token_group(
                            source_offset, hidden_size * 2,
                            l1_bank=destination_row, tokens=segment_rows),
                        isa.token_loop_end(),
                    ])
                program.extend([
                    isa.token_loop_begin(rows_per_group),
                    isa.dma_rd_out_token_group(
                        residual_window * PREFILL_OUTPUT_WINDOW_BYTES +
                        (token_start % 16) * PREFILL_TOKEN_SLOT_BYTES,
                        hidden_size * 2, l1_bank=8, tokens=rows_per_group),
                    isa.token_loop_end(),
                ])

            program.extend([
                isa.token_loop_begin(rows_per_group),
                isa.vld_token(2, 0, row_stride_bytes=64, page=2),
            ])
            for n_tile in range(hidden_size // TILE):
                output_addr = n_tile * VECTOR_BYTES
                program.extend([
                    isa.vld_token(0, down_output_base + output_addr,
                                  row_stride_bytes=PREFILL_TOKEN_SLOT_BYTES,
                                  page=1),
                    isa.mul(0, 0, 2),
                    isa.vld_token(1, output_addr,
                                  row_stride_bytes=PREFILL_TOKEN_SLOT_BYTES,
                                  page=0),
                    isa.add(0, 1, 0),
                    isa.vld_token(1, 0x8000 + output_addr,
                                  row_stride_bytes=PREFILL_TOKEN_SLOT_BYTES,
                                  page=0),
                    isa.add(0, 0, 1),
                    isa.vst_token(0, down_output_base + output_addr,
                                  row_stride_bytes=PREFILL_TOKEN_SLOT_BYTES,
                                  page=1),
                ])
            program.append(isa.token_loop_end())
            program.extend([
                isa.barrier(),
                *isa.dma_wr_group_tokens(
                    down_output_base, hidden_size, rows_per_group,
                    output_window=output_window),
            ])
        else:
            body = [
                isa.dma_rd_out_token(
                    norm_window * PREFILL_OUTPUT_WINDOW_BYTES +
                    (token_start % 16) * PREFILL_TOKEN_SLOT_BYTES,
                    hidden_size, l1_bank=norm_base >> 12),
                isa.dma_rd_out_token(
                    route_window * PREFILL_OUTPUT_WINDOW_BYTES +
                    (token_start % PREFILL_GROUP_ROWS) *
                    PREFILL_TOKEN_SLOT_BYTES,
                    hidden_size, l1_bank=route_base >> 12),
                isa.dma_rd_out_token(
                    residual_window * PREFILL_OUTPUT_WINDOW_BYTES +
                    (token_start % 16) * PREFILL_TOKEN_SLOT_BYTES,
                    hidden_size, l1_bank=residual_base >> 12),
                isa.barrier(),
            ]
            body.extend(shared_expert.emit_shared_expert(
                norm_base, gate, up, down, scalar, L1_FINAL,
                gate_base=L1_SHARED_GATE, up_base=L1_SHARED_UP,
                scalar_base=L1_SCALAR, route_base=route_base,
                residual_base=residual_base, defer_output_gate=True))
            body.append(isa.dma_wr_token(
                L1_FINAL, hidden_size, output_window=output_window))
            program.extend([isa.token_loop_begin(rows_per_group), *body,
                            isa.token_loop_end()])
    return program


def _pack_prefill_shared_groups(group_programs, instruction_limit=262144):
    """Pack independent shared-expert token groups into bounded launches."""
    if instruction_limit < 2:
        raise ValueError("prefill instruction limit must leave room for END")

    launches = []
    current = []
    for group_index, group in enumerate(group_programs):
        if not group:
            raise ValueError("shared-expert group program cannot be empty")
        if any(instruction.opcode == isa.UOpcode.END for instruction in group):
            raise ValueError("shared-expert group body must not contain END")
        if len(group) + 1 > instruction_limit:
            raise ValueError(
                f"shared-expert group {group_index} exceeds the instruction store")

        boundary_barrier = 1 if current else 0
        if (current and len(current) + boundary_barrier + len(group) + 1 >
                instruction_limit):
            current.append(isa.end())
            launches.append(current)
            current = []
            boundary_barrier = 0

        if boundary_barrier:
            # A prior standalone launch ended only after all units were idle.
            # Keep the same retirement boundary between groups in a packed launch.
            current.append(isa.barrier())
        current.extend(group)

    if current:
        current.append(isa.end())
        launches.append(current)
    return launches


def emit_prefill_layer0_stage(
        manifest, tokens=32, input_start_token=0, rows_per_group=8,
        attention_rows_per_group=None, first_position=None):
    """Emit a complete 32-token Layer-0 prefill schedule.

    The output is grouped by eight tokens into OUT windows 4 onward. QKV and
    attention scratch windows are reused only after their consumers retire;
    normalized MLP input and the pre-MLP residual remain in windows 12-15.
    Routed experts retain descriptor-selected 1/2/3-bit weights and hardware
    Top-4 routing. Shared-expert projections use batched FP16 or quantized
    weights, with L1-safe groups selected for the intermediate width.
    """
    if (not 1 <= tokens <= 32 or tokens % PREFILL_GROUP_ROWS or
            not 0 <= input_start_token <= 32 - tokens or
            rows_per_group != PREFILL_GROUP_ROWS):
        raise ValueError("full prefill requires aligned 8-row groups within 32 tokens")
    if first_position is None:
        first_position = input_start_token
    if first_position < 0:
        raise ValueError("first_position must be non-negative")

    norm_window = 12
    residual_window = 14
    qkv_input_groups = [
        (norm_window + token // 16, token % 16)
        for token in range(0, tokens, PREFILL_GROUP_ROWS)
    ]
    program = emit_prefill_input_rmsnorm(
        manifest, tokens=tokens, input_start_token=input_start_token,
        output_window_base=norm_window)
    attention_program = emit_prefill_attention_stage(
        manifest, tokens=tokens, qkv_window_base=0,
        attention_output_window_base=6, output_window_base=10,
        attention_rows_per_group=attention_rows_per_group,
        first_position=first_position,
        input_source_windows=qkv_input_groups)
    if not attention_program or attention_program[-1].opcode != isa.UOpcode.END:
        raise ValueError("prefill attention emitter did not terminate with END")
    program.extend(attention_program[:-1])
    program.extend(emit_prefill_post_attention_norm(
        manifest, tokens=tokens, input_start_token=input_start_token,
        attention_output_window_base=10,
        norm_output_window_base=norm_window,
        residual_output_window_base=residual_window))
    program.extend(emit_prefill_routed_moe_stage(
        manifest, tokens=tokens, rows_per_group=rows_per_group,
        activation_source_window=norm_window)[:-1])
    shared_gate = _linear_region(manifest, "mlp.shared_expert.gate_proj")
    shared_page_capacity = 0x10000 - 0x4000
    max_shared_rows = shared_page_capacity // (shared_gate.rows * 2)
    if rows_per_group >= 8 and max_shared_rows >= 8:
        shared_rows_per_group = 8
    elif max_shared_rows >= 4:
        shared_rows_per_group = 4
    else:
        raise ValueError("shared-expert intermediate does not fit a supported L1 group")
    program.extend(emit_prefill_shared_expert_stage(
        manifest, tokens=tokens, rows_per_group=shared_rows_per_group,
        norm_input_window_base=norm_window,
        routed_output_window_base=0,
        residual_window_base=residual_window,
        output_window_base=4))
    program.append(isa.end())
    if len(program) > 262144:
        raise ValueError(f"full prefill program has {len(program)} instructions")
    return program


def emit_prefill_layer0_programs(
        manifest, tokens=32, input_start_token=0,
        attention_rows_per_group=None, first_position=None):
    """Split a complete prefill into instruction-store-sized launch programs.

    Program order is input-norm/attention/post-norm, router plus grouped MoE,
    then instruction-store-sized bundles of independent shared-expert groups.
    OUT DDR carries activations, residuals, and routed results between launches;
    each group reloads its inputs, so L1 scratch is not carried between groups.
    The caller must preserve DDR contents and loaded expert descriptors between
    launches. Top-4 selection and mixed-precision dequantization remain in the
    routed-MoE program and execute entirely in hardware.
    """
    if (not 1 <= tokens <= 32 or tokens % PREFILL_GROUP_ROWS or
            not 0 <= input_start_token <= 32 - tokens or
            input_start_token % PREFILL_GROUP_ROWS):
        raise ValueError("full prefill requires aligned 8-row groups within 32 tokens")
    if first_position is None:
        first_position = input_start_token
    if first_position < 0:
        raise ValueError("first_position must be non-negative")

    norm_window = 12
    residual_window = 14
    qkv_input_groups = [
        (norm_window + token // 16, token % 16)
        for token in range(0, tokens, PREFILL_GROUP_ROWS)
    ]
    input_program = emit_prefill_input_rmsnorm(
        manifest, tokens=tokens, input_start_token=input_start_token,
        output_window_base=norm_window)
    attention_program = emit_prefill_attention_stage(
        manifest, tokens=tokens, qkv_window_base=0,
        attention_output_window_base=6, output_window_base=10,
        attention_rows_per_group=attention_rows_per_group,
        first_position=first_position,
        input_source_windows=qkv_input_groups)
    if (not attention_program or
            attention_program[-1].opcode != isa.UOpcode.END):
        raise ValueError("prefill attention emitter did not terminate with END")
    input_program.extend(attention_program[:-1])
    input_program.extend(emit_prefill_post_attention_norm(
        manifest, tokens=tokens, input_start_token=input_start_token,
        attention_output_window_base=10,
        norm_output_window_base=norm_window,
        residual_output_window_base=residual_window))
    input_program.append(isa.end())

    shared_gate = _linear_region(manifest, "mlp.shared_expert.gate_proj")
    shared_page_capacity = 0x10000 - 0x4000
    max_shared_rows = shared_page_capacity // (shared_gate.rows * 2)
    shared_rows = 8 if max_shared_rows >= 8 else 4 if max_shared_rows >= 4 else 0
    if shared_rows == 0:
        raise ValueError("shared-expert intermediate does not fit a supported L1 group")

    routed_intermediate = _routed_intermediate_size(manifest)
    shared_regions = (
        shared_gate,
        _linear_region(manifest, "mlp.shared_expert.up_proj"),
        _linear_region(manifest, "mlp.shared_expert.down_proj"),
        _linear_region(manifest, "mlp.shared_expert_gate"),
    )
    batch_shared = (shared_regions[3].nbits == 16 and all(
        region.nbits in (1, 2, 3, 4, 16) for region in shared_regions[:3]))
    route_rows = next((candidate for candidate in range(tokens, 0, -1)
                       if (batch_shared or candidate % shared_rows == 0) and
                       (tokens + candidate - 1) // candidate <= 4 and
                       all(routed_moe.route_group_l1_fits(
                           min(candidate, tokens - start), 2048,
                           routed_intermediate)
                           for start in range(0, tokens, candidate))), None)
    if route_rows is None:
        # Retain the prior aligned schedule for dimensions whose compact
        # routed outputs need more than the four DDR windows reserved before
        # the final outputs. This is the full-width routed fallback.
        route_rows = next((candidate for candidate in range(tokens, 0, -1)
                           if candidate % shared_rows == 0 and
                           (tokens + candidate - 1) // candidate <= 16 and
                           all(routed_moe.route_group_l1_fits(
                               min(candidate, tokens - start), 2048,
                               routed_intermediate)
                               for start in range(0, tokens, candidate))), None)
    if route_rows is None:
        raise ValueError("no route-group size aligns with shared groups and L1 capacity")
    routed_program = emit_prefill_routed_moe_stage(
        manifest, tokens=tokens, rows_per_group=route_rows,
        activation_source_window=norm_window)

    programs = [input_program, routed_program]
    shared_group_count = tokens // shared_rows
    shared_group_programs = []
    for group_index in range(shared_group_count):
        token_start = input_start_token + group_index * shared_rows
        shared_program = emit_prefill_shared_expert_stage(
            manifest, tokens=shared_rows, rows_per_group=shared_rows,
            norm_input_window_base=norm_window,
            routed_output_window_base=0,
            residual_window_base=residual_window,
            output_window_base=4 + group_index,
            input_start_token=token_start,
            routed_rows_per_group=route_rows)
        shared_group_programs.append(shared_program)
    programs.extend(_pack_prefill_shared_groups(shared_group_programs))

    for index, program in enumerate(programs):
        if len(program) > 262144:
            raise ValueError(
                f"prefill launch program {index} has {len(program)} instructions")
    return programs


def _emit_serial_token_body(manifest):
    """Emit one token-loop body using the currently selected token index.

    The current implementation retains the token-0 causal-attention shortcut:
    attention output is V. Repeating this body is therefore a resident-weight
    serialized timing smoke, not numerical 32-token causal attention.
    """
    v_proj = _linear_region(manifest, "self_attn.v_proj")
    o_proj = _linear_region(manifest, "self_attn.o_proj")
    router = _linear_region(manifest, "mlp.gate")
    shared_gate = _linear_region(manifest, "mlp.shared_expert.gate_proj")
    shared_up = _linear_region(manifest, "mlp.shared_expert.up_proj")
    shared_down = _linear_region(manifest, "mlp.shared_expert.down_proj")
    shared_scalar = _linear_region(manifest, "mlp.shared_expert_gate")
    hidden_size = v_proj.cols
    intermediate_size = _routed_intermediate_size(manifest)

    if (v_proj.rows, v_proj.cols) != (hidden_size, hidden_size):
        raise ValueError("V projection must be square at hidden_size")
    if (o_proj.rows, o_proj.cols) != (hidden_size, hidden_size):
        raise ValueError("O projection must be square at hidden_size")
    if (router.rows, router.cols) != (60, hidden_size):
        raise ValueError("router projection must have shape [60, hidden_size]")
    if intermediate_size < 32 or intermediate_size % 32:
        raise ValueError("routed intermediate size must be a multiple of 32")

    chain_attention_cfg_k = _can_chain_fp16_projections(v_proj, o_proj)
    defer_v_norm_fence = _can_defer_rmsnorm_fence(v_proj)
    defer_router_norm_fence = _can_defer_rmsnorm_fence(router)

    # The persistent input residual uses bank 1; mixed-precision weight
    # ping-pong is isolated to banks 2 and 3. RMSNorm reads the token
    # immediately, so wait for its token transfer before consuming it.
    program = [isa.dma_rd_token(0, hidden_size, l1_bank=1), isa.barrier()]
    # Configure the first projection before RMSNorm starts its asynchronous
    # PostProc stores, allowing weight DMA to begin before the next GLD.A wait.
    _prepare_projection_k(program, v_proj)
    program.extend(normalization.emit_rmsnorm(
        L1_RESIDUAL_IN, L1_NORM1_GAMMA, L1_NORM, hidden_size,
        # Only the FP16 emitter carries the one-shot GLD.A producer wait.
        # Quantized projection emitters retain the global producer fence.
        wait_after_store=not defer_v_norm_fence))

    # For the first causal token softmax has one valid key, so attention is
    # exactly V. Q/K/RoPE do not affect the numerical result for this token.
    if v_proj.nbits == 16:
        program.extend(_linear(
            v_proj, L1_NORM, L1_ATTN_VALUE,
            bias_base=(L1_V_BIAS if v_proj.bias is not None else None),
            configure_cfg_k=False,
            restore_cfg_k=not chain_attention_cfg_k,
            wait_postproc_before_gld_a=defer_v_norm_fence))
    else:
        program.extend(_linear(v_proj, L1_NORM, L1_ATTN_VALUE))
        _add_bias(program, v_proj, L1_ATTN_VALUE,
                  preloaded_bias_base=(L1_V_BIAS if v_proj.bias is not None
                                       else None))

    if o_proj.nbits == 16:
        program.extend(_linear(
            o_proj, L1_ATTN_VALUE, L1_ATTN_RESIDUAL,
            bias_base=(L1_O_BIAS if o_proj.bias is not None else None),
            residual_base=L1_RESIDUAL_IN,
            rmsnorm_accumulate=True,
            configure_cfg_k=not chain_attention_cfg_k))
    else:
        program.extend(_linear(o_proj, L1_ATTN_VALUE, L1_ATTN_PROJ))
        _add_bias_residual(
            program, o_proj, L1_ATTN_PROJ, L1_RESIDUAL_IN,
            L1_ATTN_RESIDUAL, hidden_size,
            preloaded_bias_base=(L1_O_BIAS if o_proj.bias is not None else None))

    # The router projection follows the second normalization. Preconfigure its
    # K geometry so the first weight transfer can overlap the final norm stores.
    _prepare_projection_k(program, router)
    if o_proj.nbits == 16:
        program.extend(normalization.emit_rmsnorm_apply(
            L1_ATTN_RESIDUAL, L1_NORM2_GAMMA, L1_NORM, hidden_size,
            finalize=True, wait_after_store=not defer_router_norm_fence))
    else:
        program.extend(normalization.emit_rmsnorm(
            L1_ATTN_RESIDUAL, L1_NORM2_GAMMA, L1_NORM, hidden_size,
            wait_after_store=not defer_router_norm_fence))

    program.extend(_linear(
        router, L1_NORM, L1_SHARED_GATE,
        configure_cfg_k=router.nbits != 16,
        wait_postproc_before_gld_a=defer_router_norm_fence))
    _add_bias(program, router, L1_SHARED_GATE,
              preloaded_bias_base=(L1_ROUTER_BIAS if router.bias is not None
                                   else None),
              # TOPK is all-idle gated next, so it already retires these
              # vector stores without a separate global barrier.
              wait_after_store=False)
    # TOPK consumes the current token's ACC row. Its all-idle gate covers
    # router completion; the following RWEIGHT commands wait for TopK.
    program.append(isa.topk_current_token())
    program.extend(routed_moe.emit_token(
        0, hidden_size, intermediate_size, current_token=True,
        output_base=L1_ROUTED_OUT, restore_cached_route=False,
        scale_metadata_n_invariant=bool(manifest.get(
            "routed_expert_scale_metadata_n_invariant", False)),
        configure_cfg_m=False, defer_output_fence=True))
    program.extend(shared_expert.emit_shared_expert(
        L1_NORM, shared_gate, shared_up, shared_down, shared_scalar,
        L1_FINAL, gate_base=L1_SHARED_GATE, up_base=L1_SHARED_UP,
        scalar_base=L1_SCALAR, scalar_reg=2, defer_output_gate=True,
        route_base=L1_ROUTED_OUT, residual_base=L1_ATTN_RESIDUAL))
    program.append(isa.dma_wr_token(L1_FINAL, hidden_size))
    return program


def emit_tokens(manifest, tokens=1):
    """Emit a hardware-looped serialized real-weight Layer-0 timing smoke."""
    if not 1 <= tokens <= 32:
        raise ValueError("real Layer-0 serialized token loop supports 1..32 tokens")
    norm1_gamma = _vector_region(manifest, "input_layernorm")
    norm2_gamma = _vector_region(manifest, "post_attention_layernorm")
    program = [
        # Prologue reads share RES_DMA_RD and are serialized by the decoder.
        # TOKEN_LOOP_BEGIN then waits for all of them before the body starts.
        *_load_4k(norm1_gamma, l1_bank=L1_NORM1_GAMMA >> 12, wait=False),
        *_load_4k(norm2_gamma, l1_bank=L1_NORM2_GAMMA >> 12, wait=False),
    ]
    v_proj = _linear_region(manifest, "self_attn.v_proj")
    o_proj = _linear_region(manifest, "self_attn.o_proj")
    router = _linear_region(manifest, "mlp.gate")
    if router.bias is not None:
        program.extend(_load_4k(
            router.bias, l1_bank=L1_ROUTER_BIAS >> 12, wait=False))
    if v_proj.bias is not None:
        program.extend(_load_4k(v_proj.bias, l1_bank=L1_V_BIAS >> 12,
                                 wait=False))
    if o_proj.bias is not None:
        program.extend(_load_4k(o_proj.bias, l1_bank=L1_O_BIAS >> 12,
                                 wait=False))
    # M stays at one throughout the serialized token body. Configure once
    # before the loop instead of issuing an all-idle SET_CFG_M per token.
    program.append(isa.set_cfg_m(1))
    body = _emit_serial_token_body(manifest)
    program.extend([isa.token_loop_begin(tokens), *body,
                    isa.token_loop_end(), isa.end()])
    if len(program) > 262144:
        raise ValueError(f"real Layer-0 program has {len(program)} instructions")
    return program


def emit_decode_token(manifest, position, input_byte_offset,
                      k_cache_window_base=None, v_cache_window_base=None,
                      output_window_base=None,
                      cached_keys_rotated=True):
    """Emit one position-specialized, real-weight Layer-0 decode program.

    Unlike :func:`emit_tokens`, this entry point is not a repeated-token timing
    smoke. It computes Q/K/V, applies position-specific RoPE when needed,
    appends the current K/V rows, gathers the causal KV prefix, computes
    attention and O projection, then executes router/Top-4/routed and shared
    experts before storing the final token. At position zero, the single-key
    attention and zero-position RoPE identities skip Q and attention work.
    Earlier KV positions must already be present in the configured OUT DDR
    cache windows.
    """
    if not 0 <= position < 80:
        raise ValueError("position-specialized decode supports cache positions 0..79")
    cache_windows = 5 if position >= 64 else 4 if position >= 32 else 2
    max_cache_base = 16 - cache_windows
    if k_cache_window_base is None:
        k_cache_window_base = 0
    if v_cache_window_base is None:
        v_cache_window_base = 5 if cache_windows == 5 else (
            4 if cache_windows == 4 else 2)
    if output_window_base is None:
        output_window_base = 10 if cache_windows == 5 else (
            8 if cache_windows == 4 else 4)
    if (not 0 <= k_cache_window_base <= max_cache_base or
            not 0 <= v_cache_window_base <= max_cache_base or
            not 0 <= output_window_base <= max_cache_base):
        raise ValueError("decode K/V/output rows exceed the OUT window allocation")
    if (k_cache_window_base < v_cache_window_base + cache_windows and
            v_cache_window_base < k_cache_window_base + cache_windows):
        raise ValueError("decode K and V cache windows must not overlap")
    if (output_window_base < k_cache_window_base + cache_windows and
            k_cache_window_base < output_window_base + cache_windows) or (
            output_window_base < v_cache_window_base + cache_windows and
            v_cache_window_base < output_window_base + cache_windows):
        raise ValueError("decode output windows must not overwrite the KV cache")

    projections = {
        suffix: _linear_region(manifest, f"self_attn.{suffix}_proj")
        for suffix in ("q", "k", "v", "o")
    }
    q_proj, k_proj, v_proj, o_proj = (projections[key]
                                       for key in ("q", "k", "v", "o"))
    hidden_size = q_proj.cols
    if (hidden_size != 2048 or any(
            (region.rows, region.cols) != (hidden_size, hidden_size) or
            region.nbits not in (1, 2, 3, 4, 16)
            for region in projections.values())):
        raise ValueError("decode attention projections must be supported 2048x2048 weights")
    for suffix in ("q_proj", "k_proj"):
        name = _tensor_name(manifest, f"self_attn.{suffix}")
        inventory = manifest["inventory"][name]
        if (inventory.get("npu_output_layout") !=
                "qwen_neox_rope_interleaved" or
                int(inventory.get("rope_head_dim", 0)) != 128):
            raise ValueError(
                f"{name} must use the 128-wide interleaved RoPE output layout")

    router = _linear_region(manifest, "mlp.gate")
    shared_gate = _linear_region(manifest, "mlp.shared_expert.gate_proj")
    shared_up = _linear_region(manifest, "mlp.shared_expert.up_proj")
    shared_down = _linear_region(manifest, "mlp.shared_expert.down_proj")
    shared_scalar = _linear_region(manifest, "mlp.shared_expert_gate")
    if (router.rows, router.cols) != (60, hidden_size):
        raise ValueError("decode router projection must have shape [60, 2048]")
    intermediate_size = _routed_intermediate_size(manifest)
    if intermediate_size < 32 or intermediate_size > 1408 or intermediate_size % 32:
        raise ValueError("decode routed intermediate size exceeds the L1 geometry")

    norm1_gamma = _vector_region(manifest, "input_layernorm")
    norm2_gamma = _vector_region(manifest, "post_attention_layernorm")
    first_qkv_projection = k_proj if position == 0 else q_proj
    defer_norm1_fence = _can_defer_rmsnorm_fence(first_qkv_projection)
    program = [
        # Static vectors and the input row use the same DMA-read resource.
        # The input DMA's barrier below is the common completion fence.
        *_load_4k(norm1_gamma, l1_bank=L1_NORM1_GAMMA >> 12, wait=False),
        *_load_4k(norm2_gamma, l1_bank=L1_NORM2_GAMMA >> 12, wait=False),
        isa.set_cfg_k(TILE),
        isa.set_cfg_m(1),
        isa.dma_rd_token(input_byte_offset, hidden_size, l1_bank=1),
        isa.barrier(),
    ]
    # For the first causal position, attention is exactly V, so Q is unused.
    # Track only the projections that will execute and avoid configuring Q's
    # K geometry on this path.
    projection_cfg_k = [32]
    if position != 0:
        projection_cfg_k[0] = _prepare_projection_k(
            program, q_proj, current_k=32)
    # These independent static reads overlap with the full-hidden RMSNorm
    # PostProc work. Its trailing barrier fences them before epilogue use.
    if position != 0 and q_proj.bias is not None:
        program.extend(_load_4k(
            q_proj.bias, l1_bank=L1_V_BIAS >> 12, wait=False))
    if position == 0 and k_proj.bias is not None:
        # Q no longer provides a projection slot in which to prefetch K bias.
        program.extend(_load_4k(
            k_proj.bias, l1_bank=L1_V_BIAS >> 12, wait=False))
    if o_proj.bias is not None:
        program.extend(_load_4k(
            o_proj.bias, l1_bank=L1_O_BIAS >> 12, wait=False))
    program.extend(normalization.emit_rmsnorm(
        L1_RESIDUAL_IN, L1_NORM1_GAMMA, L1_NORM, hidden_size,
        wait_after_store=not defer_norm1_fence))

    # Position-zero RoPE is the identity. The first decode token therefore
    # needs neither a RoPE-table DMA nor vector RoPE work for its cached K.
    if position != 0:
        # Q and K use the same statically selected row and interleaved pairing
        # convention as prefill.
        program.extend(emit_prefill_rope_table(
            manifest, tokens=1, first_position=position,
            allow_postproc_overlap=defer_norm1_fence))

    def emit_project(region, output_base, apply_rope=False,
                     cache_window_base=None, output_page=0,
                     bias_preloaded=False, prefetch_bias=None,
                     defer_cache_fence=False,
                     wait_postproc_before_gld_a=False):
        bias_base = None
        if region.bias is not None:
            if not bias_preloaded:
                program.extend(_load_4k(
                    region.bias, l1_bank=L1_V_BIAS >> 12, wait=False))
            bias_base = L1_V_BIAS
        if region is not q_proj:
            projection_cfg_k[0] = _prepare_projection_k(
                program, region, current_k=projection_cfg_k[0])
        program.extend(_linear(
            region, L1_NORM, output_base, bias_base=bias_base,
            configure_cfg_k=False, restore_cfg_k=False,
            output_page=output_page,
            wait_postproc_before_gld_a=wait_postproc_before_gld_a))
        # The previous projection has fully drained its bias before returning.
        # Start the next read before RoPE; vector work can overlap RES_DMA_RD,
        # and the RoPE fence retires the prefetch before the next projection.
        if prefetch_bias is not None and prefetch_bias.bias is not None:
            program.extend(_load_4k(
                prefetch_bias.bias, l1_bank=L1_V_BIAS >> 12, wait=False))
        if apply_rope:
            use_token_loop = prefetch_bias is None
            program.extend(_emit_decode_rope_heads(
                output_base, output_page,
                heads=hidden_size // 128,
                use_token_loop=use_token_loop))
            if not use_token_loop:
                program.append(isa.barrier())
        if cache_window_base is not None:
            program.extend(emit_decode_kv_cache_store(
                output_base, position, cache_window_base,
                source_page=output_page, source_ready=True,
                defer_dma_fence=defer_cache_fence))

    v_output_base = _decode_v_output_base(position)
    v_cache_slot = ((position % 16) * PREFILL_TOKEN_SLOT_BYTES)
    v_cache_uses_direct_slot = v_output_base == v_cache_slot
    if position != 0:
        emit_project(q_proj, DECODE_QUERY_BASE, apply_rope=True,
                     bias_preloaded=True, prefetch_bias=k_proj,
                     wait_postproc_before_gld_a=defer_norm1_fence)
    kv_output_page = 0 if position == 0 else 1
    kv_output_base = (L1_ATTN_VALUE if position == 0 else
                      (position % 16) * PREFILL_TOKEN_SLOT_BYTES)
    emit_project(k_proj, kv_output_base, apply_rope=(position != 0),
                 cache_window_base=k_cache_window_base,
                 output_page=kv_output_page, bias_preloaded=True,
                 prefetch_bias=v_proj,
                 defer_cache_fence=v_cache_uses_direct_slot,
                 wait_postproc_before_gld_a=(
                     position == 0 and defer_norm1_fence))
    # Keep V in page 0 while the K-cache DMA drains from page 1. The K fence
    # is deferred only for a direct V slot; staged V rows wait for K first so
    # their page-1 writeback cannot overwrite K's in-flight source row.
    emit_project(v_proj, v_output_base,
                 cache_window_base=v_cache_window_base,
                 bias_preloaded=True,
                 output_page=0)

    if position == 0:
        # With one valid key, softmax([q·k]) is exactly [1], so the attention
        # result is the already projected V vector. The position-zero K is
        # cached unchanged because its RoPE transform is the identity and later
        # decode positions consume that cache row.
        if projection_cfg_k[0] != 32:
            program.append(isa.set_cfg_k(32))
            projection_cfg_k[0] = 32
        attention_input_base = v_output_base
        attention_input_page = 0
    else:
        # Keep the hardware 32-key tile as the scheduling unit. Adjacent
        # heads share a compact gather, while each head keeps its own GEMM
        # tile and online-softmax state.
        program.extend(emit_decode_attention_from_cache(
            manifest, position, q_base=DECODE_QUERY_BASE,
            output_base=DECODE_ATTENTION_OUT_BASE,
            k_cache_window_base=k_cache_window_base,
            v_cache_window_base=v_cache_window_base,
            current_k_base=kv_output_base,
            current_v_base=v_output_base,
            current_k_page=1,
            current_v_page=0,
            defer_output_fence=True,
            head_group_size=4,
            # The GEMM scoreboard enforces the PV TGLD->GEMM dependency, so
            # the explicit transfer barrier is redundant on the production
            # tiled Decode path.
            serialize_pv_rf_transfers=False,
            keep_output_page1=True,
            cached_keys_rotated=cached_keys_rotated))
        # Decode attention configures K=32 for its head GEMMs.
        projection_cfg_k[0] = 32
        # Four-head tiled Decode keeps its compact attention result on page 1.
        # Feed O projection directly from that page instead of copying the
        # 2,048-wide vector back to page 0 first.
        attention_input_base = 0x1000 if position >= TILE else DECODE_ATTENTION_OUT_BASE
        attention_input_page = 1 if position >= TILE else 0

    if o_proj.nbits == 16:
        program.extend(_linear(
            o_proj, attention_input_base, L1_ATTN_RESIDUAL,
            bias_base=L1_O_BIAS if o_proj.bias is not None else None,
            residual_base=L1_RESIDUAL_IN,
            rmsnorm_accumulate=True,
            wait_postproc_before_gld_a=True,
            input_page=attention_input_page))
    else:
        program.extend(_linear(
            o_proj, attention_input_base, L1_ATTN_PROJ,
            bias_base=L1_O_BIAS if o_proj.bias is not None else None,
            configure_cfg_k=False, restore_cfg_k=False,
            input_page=attention_input_page))
        for chunk in range(hidden_size // TILE):
            offset = chunk * VECTOR_BYTES
            program.extend([
                isa.vld(0, L1_ATTN_PROJ + offset),
                isa.vld(1, L1_RESIDUAL_IN + offset),
                isa.add(0, 0, 1),
                isa.rmsnorm_accumulate(0, reset=chunk == 0),
                isa.vst(0, L1_ATTN_RESIDUAL + offset),
            ])

    projection_cfg_k[0] = _prepare_projection_k(
        program, router, current_k=projection_cfg_k[0])
    defer_router_norm_fence = _can_defer_rmsnorm_fence(router)
    # Only the quantized router's vector epilogue consumes a preloaded bias;
    # FP16 Top-K reads ACC directly and skips that unused bias transfer.
    if router.nbits != 16 and router.bias is not None:
        program.extend(_load_4k(
            router.bias, l1_bank=L1_ROUTER_BIAS >> 12, wait=False))
    program.extend(normalization.emit_rmsnorm_apply(
        L1_ATTN_RESIDUAL, L1_NORM2_GAMMA, L1_NORM, hidden_size,
        finalize=True, wait_after_store=not defer_router_norm_fence))
    if router.nbits == 16:
        # Top-K reads router logits from ACC directly; the vector drain and
        # L1 bias/store epilogue does not feed its input and only adds TPOT.
        program.extend(static_linear.emit_fp16(
            router, L1_NORM, L1_SHARED_GATE,
            configure_cfg_k=False, restore_cfg_k=True,
            wait_postproc_before_gld_a=defer_router_norm_fence,
            acc_only=True))
    else:
        program.extend(_linear(
            router, L1_NORM, L1_SHARED_GATE,
            configure_cfg_k=False, restore_cfg_k=False,
            wait_postproc_before_gld_a=defer_router_norm_fence))
        _add_bias(program, router, L1_SHARED_GATE,
                  preloaded_bias_base=(L1_ROUTER_BIAS
                                       if router.bias is not None else None),
                  wait_after_store=False)
    program.append(isa.topk(0))
    program.extend(routed_moe.emit_token(
        0, hidden_size, intermediate_size,
        output_base=L1_ROUTED_OUT, restore_cached_route=True,
        scale_metadata_n_invariant=bool(manifest.get(
            "routed_expert_scale_metadata_n_invariant", False)),
        configure_cfg_m=False, defer_output_fence=True))
    final_slot_base = ((position % 16) * PREFILL_TOKEN_SLOT_BYTES)
    final_output_page = 1 if shared_down.nbits == 16 else 0
    final_output_base = (final_slot_base if final_output_page == 1
                         else L1_FINAL)
    program.extend(shared_expert.emit_shared_expert(
        L1_NORM, shared_gate, shared_up, shared_down, shared_scalar,
        final_output_base, gate_base=L1_SHARED_GATE, up_base=L1_SHARED_UP,
        scalar_base=L1_SCALAR, scalar_reg=2, defer_output_gate=True,
        route_base=L1_ROUTED_OUT, residual_base=L1_ATTN_RESIDUAL,
        output_page=final_output_page))
    program.extend(emit_decode_row_store(
        final_output_base, position, output_window_base,
        source_page=final_output_page,
        source_ready=(final_output_page == 1)))
    program.append(isa.end())
    if len(program) > 262144:
        raise ValueError(f"one-token decode program has {len(program)} instructions")
    return program


def emit_token0(manifest):
    """Backward-compatible one-token entry point."""
    return emit_tokens(manifest, tokens=1)
