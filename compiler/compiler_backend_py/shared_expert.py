"""Static shared-expert lowering with sigmoid scalar gating."""
try:
    from . import (isa, static_linear, static_linear_quant_batch,
                   static_linear_quant_ops)
    from .static_linear_fp16_batch_validate import tiles_per_gemm
except ImportError:
    import isa
    import static_linear
    import static_linear_quant_batch
    import static_linear_quant_ops
    from static_linear_fp16_batch_validate import tiles_per_gemm

TILE = 32
VECTOR_BYTES = TILE * 2


def _linear(region, act_base, out_base, silu_base=None, route_base=None,
            scale_reg=None, route_first=False, residual_base=None,
            configure_cfg_k=True, restore_cfg_k=True, output_page=0,
            wait_after_store=True,
            wait_postproc_before_gld_a=False):
    if region.nbits == 16:
        return static_linear.emit_fp16(
            region, act_base, out_base, silu_base=silu_base,
            route_base=route_base, scale_reg=scale_reg,
            route_first=route_first, residual_base=residual_base,
            configure_cfg_k=configure_cfg_k,
            restore_cfg_k=restore_cfg_k, output_page=output_page,
            wait_after_store=wait_after_store,
            wait_postproc_before_gld_a=wait_postproc_before_gld_a)
    if output_page != 0:
        raise ValueError("quantized shared projections only support page-0 output")
    program = [isa.set_cfg_k(32)] if configure_cfg_k else []
    program.extend(static_linear.emit_quantized(
        region, act_base, out_base, silu_base=silu_base,
        route_base=route_base, scale_reg=scale_reg,
        route_first=route_first, residual_base=residual_base,
        configure_cfg_k=configure_cfg_k,
        restore_cfg_k=restore_cfg_k,
        wait_after_store=wait_after_store,
        wait_postproc_before_gld_a=wait_postproc_before_gld_a))
    if restore_cfg_k and not configure_cfg_k:
        program.append(isa.set_cfg_k(32))
    return program


def _fp16_k_config(region):
    if region.nbits != 16:
        return None
    return 32 * tiles_per_gemm(region)


def _can_chain_fp16_k(regions):
    configs = [_fp16_k_config(region) for region in regions]
    return (all(config is not None for config in configs) and
            len(set(configs)) == 1 and configs[0] > 32)


def _batch_cfg_k(region):
    """Return the CFG_K selected by the batched projection emitter."""
    if region.nbits == 16:
        return 32 * tiles_per_gemm(region)
    group_bytes = static_linear_quant_ops.k128_weight_group_bytes(region)
    wide_gemm = (group_bytes != 0 and region.k_tiles % 4 == 0 and
                 region.weight % group_bytes == 0)
    return 128 if wide_gemm else 32


def emit_shared_expert(input_base, gate_region, up_region, down_region,
                       scalar_gate_region, output_base, gate_base=0x5000,
                       up_base=0x8000, scalar_base=0x3FC0,
                       gate_reg=0, up_reg=1, scalar_reg=2,
                       defer_output_gate=False, route_base=None,
                       residual_base=None, output_page=0):
    """Emit SwiGLU shared expert and sigmoid-gate its hidden-size output."""
    if output_page not in (0, 1, 2):
        raise ValueError("shared output page must be 0, 1, or 2")
    hidden_size = gate_region.cols
    intermediate_size = gate_region.rows
    if (up_region.rows, up_region.cols) != (intermediate_size, hidden_size):
        raise ValueError("shared gate/up projection shapes must match")
    if (down_region.rows, down_region.cols) != (hidden_size, intermediate_size):
        raise ValueError("shared down projection shape is incompatible")
    if scalar_gate_region.rows != 1 or scalar_gate_region.cols != hidden_size:
        raise ValueError("shared scalar gate must have shape [1, hidden_size]")
    if len({gate_reg, up_reg, scalar_reg}) != 3:
        raise ValueError("shared expert requires three distinct PRF registers")
    if (route_base is None) != (residual_base is None):
        raise ValueError("fused shared output requires both route and residual vectors")
    if route_base is not None and not defer_output_gate:
        raise ValueError("fused shared output requires deferred output gating")

    scalar_cfg_k = _batch_cfg_k(scalar_gate_region)
    gate_cfg_k = _batch_cfg_k(gate_region)
    up_cfg_k = _batch_cfg_k(up_region)
    down_cfg_k = _batch_cfg_k(down_region)
    active_cfg_k = 32
    program = []
    if scalar_cfg_k != active_cfg_k:
        program.append(isa.set_cfg_k(scalar_cfg_k))
        active_cfg_k = scalar_cfg_k
    program.extend(_linear(
        scalar_gate_region, input_base, scalar_base,
        configure_cfg_k=False, restore_cfg_k=False))
    program.extend([
        isa.vld(scalar_reg, scalar_base),
        isa.sigmoid(scalar_reg, scalar_reg),
        isa.broadcast(scalar_reg, scalar_reg, lane=0),
    ])

    if gate_cfg_k != active_cfg_k:
        program.append(isa.set_cfg_k(gate_cfg_k))
        active_cfg_k = gate_cfg_k
    program.extend(_linear(
        gate_region, input_base, gate_base,
        configure_cfg_k=False, restore_cfg_k=False,
        wait_after_store=False))
    if up_cfg_k != active_cfg_k:
        program.append(isa.set_cfg_k(up_cfg_k))
        active_cfg_k = up_cfg_k
    program.extend(_linear(up_region, input_base, up_base,
                           silu_base=gate_base,
                           configure_cfg_k=False, restore_cfg_k=False,
                           wait_after_store=False))

    # Down's first activation load waits for the Up epilogue to retire. Its
    # independent weight/scale preparation may start while Up writes its
    # output, avoiding a global barrier without weakening the data dependency.
    if down_cfg_k != active_cfg_k:
        program.append(isa.set_cfg_k(down_cfg_k))
        active_cfg_k = down_cfg_k
    program.extend(_linear(
        down_region, up_base, output_base,
        route_base=route_base,
        scale_reg=scalar_reg if defer_output_gate else None,
        route_first=route_base is not None,
        residual_base=residual_base,
        configure_cfg_k=False, restore_cfg_k=False,
        output_page=output_page,
        wait_postproc_before_gld_a=True))
    if active_cfg_k != 32:
        program.append(isa.set_cfg_k(32))
    if defer_output_gate:
        # The caller either fuses the pending route/residual epilogue itself,
        # or this emitter already wrote the fused output directly at the drain.
        return program
    for chunk in range((hidden_size + TILE - 1) // TILE):
        offset = chunk * VECTOR_BYTES
        if output_page == 0:
            program.extend([
                isa.vld(gate_reg, output_base + offset),
                isa.mul(gate_reg, gate_reg, scalar_reg),
                isa.vst(gate_reg, output_base + offset),
            ])
        else:
            program.extend([
                isa.vld_page(gate_reg, output_base + offset, output_page),
                isa.mul(gate_reg, gate_reg, scalar_reg),
                isa.vst_page(gate_reg, output_base + offset, output_page),
            ])
    return program


def emit_shared_expert_batch(input_base, gate_region, up_region, down_region,
                            scalar_gate_region, output_base, tokens=8,
                            gate_base=0, up_base=0x6000,
                            scalar_base=0xC000, activation_page=0,
                            activation_row_stride_bytes=4096,
                            intermediate_page=1, scalar_page=1,
                            output_page=0, scalar_reg=2,
                            down_output_page=None):
    """Emit a mixed-precision shared expert for a compact token-row group.

    The scalar-gate projection is batched with a padded 64-byte row stride;
    its sigmoid/broadcast epilogue remains per row. Wide projections use
    batched FP16 or 1-4-bit GEMM so each static weight tile is fetched once
    per group. Gate output is rounded to FP16 before SiLU.
    """
    hidden_size = gate_region.cols
    intermediate_size = gate_region.rows
    if (tokens < 1 or tokens > 32 or
            (up_region.rows, up_region.cols) !=
            (intermediate_size, hidden_size) or
            (down_region.rows, down_region.cols) !=
            (hidden_size, intermediate_size) or
            scalar_gate_region.rows != 1 or
            scalar_gate_region.cols != hidden_size):
        raise ValueError("incompatible shared-expert batch geometry")
    if (scalar_gate_region.nbits != 16 or
            any(region.nbits not in (1, 2, 3, 4, 16) for region in
                (gate_region, up_region, down_region))):
        raise ValueError("batched shared expert requires supported static weights")
    if any(page not in (0, 1, 2) for page in
           (activation_page, intermediate_page, scalar_page, output_page)):
        raise ValueError("shared-expert batch pages must be in [0,2]")
    if down_output_page is None:
        down_output_page = intermediate_page
    if down_output_page not in (0, 1, 2):
        raise ValueError("shared-expert Down output page must be in [0,2]")
    scalar_limit = 16384 if scalar_page == 2 else 65536
    if (scalar_base < 0 or scalar_base % 64 or
            scalar_base + tokens * 64 > scalar_limit):
        raise ValueError("shared-expert scalar outputs exceed their L1 page")
    intermediate_bytes = tokens * intermediate_size * 2
    scalar_end = scalar_base + tokens * 64
    for base in (gate_base, up_base):
        if (scalar_page == intermediate_page and
                scalar_base < base + intermediate_bytes and
                base < scalar_end):
            raise ValueError("shared scalar outputs alias a projection tensor")

    program = [isa.set_cfg_m(tokens)]
    program.extend(static_linear.emit_fp16_batch(
        scalar_gate_region, input_base, scalar_base, tokens,
        activation_page=activation_page, output_page=scalar_page,
        activation_row_stride_bytes=activation_row_stride_bytes,
        output_row_stride_bytes=64, configure_cfg_m=False,
        restore_cfg_m=False))
    # TOKEN_END is all-idle gated, so it retires the per-row scalar stores.
    program.extend([
        isa.token_loop_begin(tokens),
        isa.vld_token(scalar_reg, scalar_base, row_stride_bytes=64,
                      page=scalar_page),
        isa.sigmoid(scalar_reg, scalar_reg),
        isa.broadcast(scalar_reg, scalar_reg, lane=0),
        isa.vst_token(scalar_reg, scalar_base, row_stride_bytes=64,
                      page=scalar_page),
        isa.token_loop_end(),
    ])

    def emit_projection(region, activation_base, destination_base,
                        source_page, destination_page, source_stride,
                        silu_base=None, silu_page=None,
                        configure_cfg_k=True, restore_cfg_k=False,
                        wait_after_store=True,
                        wait_postproc_before_gld_a=False,
                        allow_silu_output_alias=False):
        options = dict(
            activation_page=source_page, output_page=destination_page,
            activation_row_stride_bytes=source_stride,
            silu_base=silu_base, silu_page=silu_page,
            configure_cfg=False, restore_cfg=False,
            configure_cfg_m=False, restore_cfg_m=False,
            configure_cfg_k=configure_cfg_k,
            restore_cfg_k=restore_cfg_k,
            wait_after_store=wait_after_store,
            wait_postproc_before_gld_a=wait_postproc_before_gld_a,
            allow_silu_output_alias=allow_silu_output_alias)
        if region.nbits == 16:
            return static_linear.emit_fp16_batch(
                region, activation_base, destination_base, tokens, **options)
        return static_linear_quant_batch.emit_quantized_batch(
            region, activation_base, destination_base, tokens, **options)

    gate_cfg_k = _batch_cfg_k(gate_region)
    up_cfg_k = _batch_cfg_k(up_region)
    down_cfg_k = _batch_cfg_k(down_region)
    program.extend(emit_projection(
        gate_region, input_base, gate_base, activation_page,
        intermediate_page, activation_row_stride_bytes,
        configure_cfg_k=True, wait_after_store=False))
    program.extend(emit_projection(
        up_region, input_base, up_base, activation_page,
        output_page, activation_row_stride_bytes,
        silu_base=gate_base, silu_page=intermediate_page,
        configure_cfg_k=(up_cfg_k != gate_cfg_k), wait_after_store=False,
        allow_silu_output_alias=(
            up_base == gate_base and output_page == intermediate_page)))
    program.extend(emit_projection(
        down_region, up_base, output_base, output_page,
        down_output_page, intermediate_size * 2,
        configure_cfg_k=(down_cfg_k != up_cfg_k), wait_after_store=True,
        wait_postproc_before_gld_a=True))
    program.extend((isa.set_cfg_m(1), isa.set_cfg_k(32)))
    return program
