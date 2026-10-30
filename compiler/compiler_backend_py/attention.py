"""Attention microcode fragments for sequence tiles of at most 32 tokens."""
try:
    from . import isa
except ImportError:
    import isa

TILE = 32
MAX_FLASH_SEQUENCE_LENGTH = 1536


def emit_rope_qk_components(tokens, q_base, k_base, first_position=0,
                           q_row_stride=256, k_row_stride=256,
                           table_first_position=0):
    """Apply RoPE to same-position Q/K rows while loading each table row once.

    Q and K use the same positional embedding for a given token. Keeping the
    pair of transforms in one token-loop body lets the second ROPE consume the
    still-live embedding vector from PostProc RF instead of issuing another
    VLD for it. ``table_first_position`` identifies the absolute position in
    slot zero of the compact RoPE table loaded in L1.
    """
    table_row = first_position - table_first_position
    if (not 1 <= tokens <= 32 or table_row < 0 or
            table_row + tokens > 32):
        raise ValueError("Q/K RoPE token range must fit the 32-position table")
    for base, stride in ((q_base, q_row_stride), (k_base, k_row_stride)):
        if (base < 0 or base % 32 or stride < 256 or stride > 512 or
                stride % 32 or base + (tokens - 1) * stride + 256 > 65536):
            raise ValueError("Q/K RoPE rows exceed their L1 page")
    if (q_base < k_base + (tokens - 1) * k_row_stride + 256 and
            k_base < q_base + (tokens - 1) * q_row_stride + 256):
        raise ValueError("Q and K RoPE ranges must not overlap")

    program = [isa.token_loop_begin(tokens)]
    for chunk in range(128 // TILE):
        offset = chunk * TILE * 2
        for base, stride in ((q_base, q_row_stride),
                             (k_base, k_row_stride)):
            program.append(isa.vld_token(0, base + offset, stride, page=0))
            if base == q_base:
                program.append(isa.vld_token(
                    3, table_row * 256 + offset, 256, page=0))
            program.append(isa.rope(0, 0, 3))
            program.append(isa.vst_token(0, base + offset, stride, page=0))
    program.append(isa.token_loop_end())
    return program


def emit_pack_head_rows(source_base, destination_base, tokens, hidden_size,
                        head_index, head_dim=128, first_token=0,
                        source_row_stride=None, source_page=1,
                        destination_page=0, wait_after_store=True):
    """Pack one head from full-width token rows into a compact token sequence.

    The source is a token-major ``[tokens, hidden_size]`` FP16 matrix in one
    L1 page. The destination is a compact ``[sequence, head_dim]`` FP16 matrix;
    ``first_token`` selects where this chunk belongs in that sequence. This is
    useful when attention projections are computed in row groups: each group
    can be packed into its final token positions without materializing a
    second full-hidden matrix.

    The caller is responsible for loading the source rows and for preserving
    the destination across groups. If several chunks are packed before any
    GLD.A/TGLD consumer, set ``wait_after_store=False`` and fence once after
    the final chunk.
    """
    if (not 1 <= tokens <= 32 or not 0 <= first_token < 32 or
            first_token + tokens > 32):
        raise ValueError("head-pack token range must fit the 32-token sequence")
    if (hidden_size < TILE or hidden_size > 2048 or hidden_size % TILE or
            head_dim < TILE or head_dim > 256 or head_dim % TILE or
            hidden_size % head_dim):
        raise ValueError("invalid hidden or attention-head dimension")
    head_count = hidden_size // head_dim
    if not 0 <= head_index < head_count:
        raise ValueError("head index is outside the hidden dimension")
    if source_row_stride is None:
        source_row_stride = hidden_size * 2
    if (source_row_stride < hidden_size * 2 or
            source_row_stride > 4096 or source_row_stride % 32):
        raise ValueError("source row stride must fit one FP16 hidden row and be 32-byte aligned")
    if source_page not in (0, 1, 2) or destination_page not in (0, 1, 2):
        raise ValueError("L1 page must be 0, 1, or 2")

    page_bytes = {0: 65536, 1: 65536, 2: 16384}
    source_head_offset = head_index * head_dim * 2
    source_start = source_base + source_head_offset
    source_end = (source_base + (tokens - 1) * source_row_stride +
                  source_head_offset + head_dim * 2)
    destination_start = destination_base + first_token * head_dim * 2
    destination_end = destination_start + tokens * head_dim * 2
    if (source_base < 0 or source_base % 32 or destination_base < 0 or
            destination_base % 32 or source_end > page_bytes[source_page] or
            destination_end > page_bytes[destination_page]):
        raise ValueError("head-pack source or destination exceeds its L1 page")
    if (source_page == destination_page and source_start < destination_end and
            destination_start < source_end):
        raise ValueError("head-pack source and destination ranges overlap")

    program = []
    for row in range(tokens):
        src_row = source_base + row * source_row_stride + source_head_offset
        dst_row = destination_base + (first_token + row) * head_dim * 2
        for chunk in range(head_dim // TILE):
            offset = chunk * TILE * 2
            program.extend([
                isa.vld_page(0, src_row + offset, source_page),
                isa.vst_page(0, dst_row + offset, destination_page),
            ])
    if wait_after_store:
        program.append(isa.barrier())
    return program


def emit_qk_score(q_base, k_base, score_base, query_token,
                  head_dim=128, token_row_stride=None,
                  wait_after_store=True):
    """Emit Q[token] @ K.T using TGLD; K remains token-major in L1.

    ``k_base`` points at a [32, head_dim] FP16 buffer. TGLD gathers one
    [32,32] K slice and transposes it directly into WRF. The caller may pad
    unused token rows with zero and apply its causal/valid-token mask before
    softmax.
    """
    if head_dim < TILE or head_dim > 256 or head_dim % TILE:
        raise ValueError("head_dim must be a multiple of 32 in [32,256]")
    if not 0 <= query_token < 32:
        raise ValueError("query_token must be in [0,31]")
    stride = head_dim * 2 if token_row_stride is None else token_row_stride
    if stride < head_dim * 2 or stride > 512 or stride % 32:
        raise ValueError("token row stride must be 32-byte aligned and at most 512 bytes")

    program = []
    q_row = q_base + query_token * stride
    for chunk in range(head_dim // TILE):
        offset = chunk * TILE * 2
        program.extend([
            isa.transpose_load(k_base + offset, stride),
            isa.gld_a(q_row + offset),
            # GEMM is scoreboard-gated on both inactive RF writes; no global
            # fence is needed between the producers and their consumer.
            isa.gemm(acc=chunk != 0, skip_store=chunk != head_dim // TILE - 1,
                     imm=TILE, prec=isa.Precision.FP16),
        ])
    # AC2V waits for GEMM completion; VST is serialized by PostProc. A caller
    # that sends scores to GLD.A needs a fence because that reader is separate;
    # vector consumers can instead rely on the shared PostProc dependency.
    program.extend([isa.ac2v(0, row=0), isa.vst(0, score_base)])
    if wait_after_store:
        program.append(isa.barrier())
    return program


def emit_attention_head(q_base, k_base, v_base, score_base, out_base,
                        query_token, valid_tokens, head_dim=128,
                        token_row_stride=None, causal=True,
                        score_scale_base=None, output_page=0,
                        wait_after_store=True):
    """Emit masked score softmax and value aggregation for one attention head.

    Q, K, and V are token-major FP16 rows in L1. The output is one contiguous
    ``head_dim`` FP16 row at ``out_base``. Causal mode exposes tokens through
    the current query only; non-causal mode exposes all ``valid_tokens`` rows.
    If supplied, ``score_scale_base`` points to a 32-lane vector containing
    the attention scale (normally ``1/sqrt(head_dim)``); otherwise Q is
    expected to have been pre-scaled. Set ``wait_after_store=False`` only
    when the caller can defer the output-consumer fence until a later point.
    """
    if not 1 <= valid_tokens <= TILE:
        raise ValueError("valid_tokens must be in [1,32]")
    if not 0 <= query_token < valid_tokens:
        raise ValueError("query_token must select a valid token")
    stride = head_dim * 2 if token_row_stride is None else token_row_stride
    valid_lanes = min(valid_tokens, query_token + 1) if causal else valid_tokens
    if output_page not in (0, 1, 2):
        raise ValueError("attention output page must be 0, 1, or 2")

    program = emit_qk_score(q_base, k_base, score_base, query_token,
                            head_dim, stride, wait_after_store=False)
    program.append(isa.vld(0, score_base))
    if score_scale_base is not None:
        program.extend([isa.vld(1, score_scale_base), isa.mul(0, 0, 1)])
    program.extend([isa.softmax(0, 0, valid_lanes=valid_lanes),
                    isa.vst(0, score_base)])

    for chunk in range(head_dim // TILE):
        offset = chunk * TILE * 2
        program.extend([
            isa.transpose_load(v_base + offset, stride, transpose=False),
            # The first score load waits for its PostProc producer; later
            # K chunks reuse the same already-published score row.
            isa.gld_a(score_base, wait_postproc=(chunk == 0)),
            # The GEMM scoreboard waits for TGLD/GLD.A to publish both RFs.
            isa.gemm(imm=TILE, prec=isa.Precision.FP16),
            isa.ac2v(0, row=0),
            isa.vst_page(0, out_base + offset, output_page),
        ])
        if (chunk == head_dim // TILE - 1 and wait_after_store):
            # Consumers using the independent GLD.A/DMA path need a fence.
            # Consecutive heads write disjoint output ranges, so callers may
            # defer this fence until the final head and overlap its VST with
            # the next head's QK/PV work.
            program.append(isa.barrier())
    return program


def emit_flash_attention_query_tiles(
    q_base, k_base, v_base, score_base, out_base,
    query_position, sequence_length, key_tile_loaders,
    head_dim=128, score_scale_base=None, output_page=0,
    output_rescale_vd=3, wait_after_store=True,
):
    """Emit one-query FlashAttention over streamed 32-key tiles.

    ``key_tile_loaders`` contains one instruction sequence per key tile; each
    sequence places at most 32 token-major K/V rows in the same L1 buffers.
    The online softmax instruction updates the engine's running max/sum and
    returns globally normalized tile weights plus the rescale factor for the
    output accumulated from prior tiles.
    """
    if not 1 <= sequence_length <= MAX_FLASH_SEQUENCE_LENGTH:
        raise ValueError(
            f"FlashAttention sequence length must be in [1,{MAX_FLASH_SEQUENCE_LENGTH}]")
    if not 0 <= query_position < sequence_length:
        raise ValueError("query_position must be within the sequence")
    if head_dim < TILE or head_dim > 256 or head_dim % TILE:
        raise ValueError("head_dim must be a multiple of 32 in [32,256]")
    if output_page not in (0, 1, 2):
        raise ValueError("attention output page must be 0, 1, or 2")
    tile_count = (sequence_length + TILE - 1) // TILE
    if len(key_tile_loaders) != tile_count:
        raise ValueError("one K/V loader is required for each sequence tile")
    if (output_rescale_vd < 0 or output_rescale_vd > 15 or
            output_rescale_vd in (0, 1, 2)):
        raise ValueError("output-rescale register must not alias working registers")
    if (q_base < 0 or q_base % 32 or q_base + head_dim * 2 > 65536 or
            k_base < 0 or k_base % 32 or
            k_base + TILE * head_dim * 2 > 65536 or
            v_base < 0 or v_base % 32 or
            v_base + TILE * head_dim * 2 > 65536 or
            score_base < 0 or score_base % 32 or score_base + 64 > 65536 or
            out_base < 0 or out_base % 32 or
            out_base + head_dim * 2 > {0: 65536, 1: 65536, 2: 16384}[output_page]):
        raise ValueError("FlashAttention tile buffers exceed their L1 page")
    if score_scale_base is not None and (
            score_scale_base < 0 or score_scale_base % 32 or
            score_scale_base + 64 > 65536):
        raise ValueError("attention scale vector exceeds its L1 page")

    program = []
    first_tile = True
    for tile_index, loader in enumerate(key_tile_loaders):
        tile_start = tile_index * TILE
        valid_lanes = min(TILE, sequence_length - tile_start,
                          query_position - tile_start + 1)
        if valid_lanes <= 0:
            break
        program.extend(loader)
        program.extend(emit_qk_score(
            q_base, k_base, score_base, query_token=0,
            head_dim=head_dim, token_row_stride=head_dim * 2,
            wait_after_store=False))
        program.append(isa.vld(0, score_base))
        if score_scale_base is not None:
            program.extend([isa.vld(1, score_scale_base),
                            isa.mul(0, 0, 1)])
        program.extend([
            isa.online_softmax(0, 0, output_rescale_vd,
                               valid_lanes=valid_lanes, reset=first_tile),
            isa.vst(0, score_base),
        ])
        for feature_tile in range(head_dim // TILE):
            offset = feature_tile * TILE * 2
            program.extend([
                isa.transpose_load(v_base + offset, head_dim * 2,
                                   transpose=False),
                isa.gld_a(score_base, wait_postproc=(feature_tile == 0)),
                isa.gemm(imm=TILE, prec=isa.Precision.FP16),
                isa.ac2v(0, row=0),
            ])
            if not first_tile:
                program.extend([
                    isa.vld_page(2, out_base + offset, output_page),
                    isa.mul(2, 2, output_rescale_vd),
                    isa.add(0, 0, 2),
                ])
            program.append(isa.vst_page(0, out_base + offset, output_page))
        first_tile = False
    if first_tile:
        raise ValueError("query has no causal K/V tile")
    if wait_after_store:
        program.append(isa.barrier())
    return program


def emit_flash_attention_query_block(
    q_base, k_base, v_base, score_base, alpha_base, output_base,
    query_start, query_count, sequence_length, key_tile_loaders,
    head_dim=128, query_row_stride_bytes=None,
    output_row_stride=4096, output_feature_offset=0,
    score_scale_base=None, output_page=1, alpha_page=2,
    state_slot_base=0, wait_after_store=True,
    k_token_row_stride_bytes=None, v_token_row_stride_bytes=None,
    additional_heads=(), key_tile_bases=None,
    key_tile_prefetchers=None, serialize_pv_rf_transfers=True,
    query_page=0, score_page=0, score_scale_page=0,
    defer_online_normalization=False,
    causal=True,
):
    """Emit a FlashAttention Q block over streamed 32-key tiles.

    Each K/V loader fills a compact L1 tile. For every key tile, QK and
    PV are batched across the active query rows. The online-softmax state slot
    is query-local, and alpha scalars are spilled to page 2 so all query rows
    can rescale their prior output after the batched PV GEMM. Optional
    ``additional_heads`` consume the same gathered tile sequentially while
    retaining independent Q/K/V addresses, output buffers, online states, and
    alpha spill rows. Each extra stream is a five-tuple
    ``(q, k, v, out, state_slot)``; its alpha spill range is allocated after
    the primary range unless a sixth explicit alpha-base field is supplied.
    QK-side barriers serialize TGLD and GLD.A because they share the L1 read
    path. ``serialize_pv_rf_transfers`` retains the legacy barrier between
    PV TGLD and GEMM for schedule comparisons; the GEMM scoreboard dependency
    protects the default PV schedule.
    """
    if not 1 <= sequence_length <= MAX_FLASH_SEQUENCE_LENGTH:
        raise ValueError(
            f"FlashAttention sequence length must be in [1,{MAX_FLASH_SEQUENCE_LENGTH}]")
    if defer_online_normalization and sequence_length <= TILE:
        raise ValueError(
            "deferred online normalization requires multiple key tiles")
    if (query_start < 0 or query_count < 1 or query_count > 32 or
            query_start + query_count > sequence_length):
        raise ValueError("FlashAttention query block must contain 1..32 valid rows")
    if head_dim < TILE or head_dim > 256 or head_dim % TILE:
        raise ValueError("head_dim must be a multiple of 32 in [32,256]")
    query_stride = (head_dim * 2 if query_row_stride_bytes is None
                    else query_row_stride_bytes)
    if (query_stride < head_dim * 2 or
            query_stride > 2048 or query_stride % 32):
        raise ValueError("query row stride must be 32-byte aligned in [head_dim*2,2048]")
    if len(key_tile_loaders) != (sequence_length + TILE - 1) // TILE:
        raise ValueError("one K/V loader is required for each sequence tile")
    tile_count = len(key_tile_loaders)
    if key_tile_bases is not None and (
            len(key_tile_bases) != tile_count or additional_heads or
            any(len(pair) != 2 for pair in key_tile_bases)):
        raise ValueError("per-tile K/V bases require one primary head stream")
    if key_tile_prefetchers is not None and (
            len(key_tile_prefetchers) != tile_count - 1 or
            additional_heads or key_tile_bases is None):
        raise ValueError("tile prefetch requires per-tile bases and one head stream")
    if not 0 <= state_slot_base <= 128 - query_count:
        raise ValueError("query block exceeds online-softmax state slots")
    if (query_page not in (0, 1) or score_page not in (0, 1, 2) or
            score_scale_page not in (0, 1, 2) or
            output_page not in (0, 1, 2) or alpha_page not in (0, 1, 2)):
        raise ValueError("FlashAttention buffers must select L1 page 0..2")
    page_bytes = {0: 65536, 1: 65536, 2: 16384}
    score_row_bytes = TILE * 2
    k_stride = (head_dim * 2 if k_token_row_stride_bytes is None
                else k_token_row_stride_bytes)
    v_stride = (head_dim * 2 if v_token_row_stride_bytes is None
                else v_token_row_stride_bytes)
    alpha_span = query_count * score_row_bytes
    head_streams = [(
        q_base, k_base, v_base, output_base, state_slot_base, alpha_base)]
    for stream_index, stream in enumerate(additional_heads, start=1):
        if len(stream) == 5:
            stream = (*stream, alpha_base + stream_index * alpha_span)
        elif len(stream) != 6:
            raise ValueError("additional head streams need five or six fields")
        head_streams.append(stream)
    if additional_heads and sequence_length <= TILE:
        raise ValueError("additional head streams require multi-tile online softmax")
    for (stream_q, stream_k, stream_v, stream_out,
         stream_state_slot, stream_alpha_base) in head_streams:
        if (not 0 <= stream_state_slot <= 128 - query_count or
                stream_alpha_base < 0 or stream_alpha_base % 32 or
                stream_alpha_base + alpha_span > page_bytes[alpha_page]):
            raise ValueError("head stream exceeds online-softmax state slots")
        if (stream_q < 0 or stream_q % 32 or
                stream_q + (query_count - 1) * query_stride +
                head_dim * 2 > page_bytes[query_page] or
                stream_k < 0 or stream_k % 32 or
                k_stride < head_dim * 2 or k_stride > 2048 or
                k_stride % 32 or
                stream_k + (TILE - 1) * k_stride + head_dim * 2 > 65536 or
                stream_v < 0 or stream_v % 32 or
                v_stride < head_dim * 2 or v_stride > 2048 or
                v_stride % 32 or
                stream_v + (TILE - 1) * v_stride + head_dim * 2 > 65536 or
                stream_out < 0 or stream_out % 32 or
                stream_out + (query_count - 1) * output_row_stride +
                output_feature_offset + head_dim * 2 > page_bytes[output_page]):
            raise ValueError("FlashAttention head-stream buffers exceed their L1 pages")
    if (q_base < 0 or q_base % 32 or
            score_base < 0 or score_base % 32 or
            score_base + query_count * score_row_bytes > page_bytes[score_page] or
            alpha_base < 0 or alpha_base % 32 or
            alpha_base + query_count * score_row_bytes > page_bytes[alpha_page] or
            output_feature_offset < 0 or output_feature_offset % 32 or
            output_row_stride < output_feature_offset + head_dim * 2 or
            output_row_stride % 32):
        raise ValueError("FlashAttention block buffers exceed their L1 pages")
    if key_tile_bases is not None:
        tile_ranges = []
        fixed_ranges = [
            (q_base, q_base + (query_count - 1) * query_stride + head_dim * 2),
            (score_base, score_base + query_count * score_row_bytes),
        ]
        if score_scale_base is not None:
            fixed_ranges.append((score_scale_base, score_scale_base + 64))
        if output_page == 0:
            fixed_ranges.append((
                output_base,
                output_base + (query_count - 1) * output_row_stride +
                output_feature_offset + head_dim * 2))
        for tile_k_base, tile_v_base in key_tile_bases:
            if (tile_k_base < 0 or tile_k_base % 32 or
                    tile_k_base + (TILE - 1) * k_stride + head_dim * 2 > 65536 or
                    tile_v_base < 0 or tile_v_base % 32 or
                    tile_v_base + (TILE - 1) * v_stride + head_dim * 2 > 65536):
                raise ValueError("per-tile K/V buffer exceeds its L1 page")
            ranges = (
                (tile_k_base,
                 tile_k_base + (TILE - 1) * k_stride + head_dim * 2),
                (tile_v_base,
                 tile_v_base + (TILE - 1) * v_stride + head_dim * 2),
            )
            if any(left[0] < right[1] and right[0] < left[1]
                   for left, right in ((ranges[0], ranges[1]),
                                       *((rng, fixed) for rng in ranges
                                         for fixed in fixed_ranges))):
                raise ValueError("per-tile K/V buffers overlap another live L1 range")
            tile_ranges.append(ranges)
        if key_tile_prefetchers is not None:
            for current, following in zip(tile_ranges, tile_ranges[1:]):
                if any(left[0] < right[1] and right[0] < left[1]
                       for left in current for right in following):
                    raise ValueError("prefetched K/V tiles must use disjoint L1 buffers")
    alpha_ranges = [(stream[5], stream[5] + alpha_span)
                    for stream in head_streams]
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(alpha_ranges)
           for right in alpha_ranges[index + 1:]):
        raise ValueError("FlashAttention head streams need disjoint alpha spills")
    state_ranges = [(stream[4], stream[4] + query_count)
                    for stream in head_streams]
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(state_ranges)
           for right in state_ranges[index + 1:]):
        raise ValueError("FlashAttention head streams need disjoint online states")
    if score_scale_base is not None and (
            score_scale_base < 0 or score_scale_base % 32 or
            score_scale_base + 64 > page_bytes[score_scale_page]):
        raise ValueError("attention scale vector exceeds its L1 page")

    # A single key tile has no cross-tile state to maintain. Keep the existing
    # batched causal-softmax path to avoid online-state and alpha spill work.
    if sequence_length <= TILE:
        if query_page != 0 or score_page != 0 or score_scale_page != 0:
            raise ValueError("single-tile attention uses page 0 for Q, score, and scale")
        if k_stride != v_stride:
            raise ValueError("single-tile attention requires matching K/V row strides")
        program = list(key_tile_loaders[0])
        program.extend([isa.set_cfg_k(TILE), isa.set_cfg_m(query_count)])
        program.extend(emit_batched_causal_attention_head_queries(
            q_base=q_base, k_base=k_base, v_base=v_base,
            score_base=score_base, query_pack_base=alpha_base,
            output_base=output_base, query_start=query_start,
            query_count=query_count, valid_tokens=sequence_length,
            head_dim=head_dim, token_row_stride=k_stride,
            query_row_stride_bytes=query_stride,
            output_row_stride=output_row_stride,
            output_feature_offset=output_feature_offset,
            output_page=output_page, score_scale_base=score_scale_base,
            wait_after_store=wait_after_store, q_block_packed=True))
        return program

    program = [isa.set_cfg_k(TILE)]
    if score_scale_base is not None:
        # Online Softmax writes RF0/3; AC2V and output merge use RF0/2/3.
        # Keep the invariant scale resident in RF1 across every row/tile.
        program.append(
            isa.vld(1, score_scale_base) if score_scale_page == 0 else
            isa.vld_page(1, score_scale_base, score_scale_page))
    cfg_m = None
    for tile_index, loader in enumerate(key_tile_loaders):
        tile_start = tile_index * TILE
        active_start = (max(0, tile_start - query_start) if causal else 0)
        active_count = query_count - active_start
        if active_count <= 0:
            break
        if key_tile_prefetchers is None or tile_index == 0:
            program.extend(loader)
        if cfg_m != active_count:
            program.append(isa.set_cfg_m(active_count))
            cfg_m = active_count

        for (stream_q, stream_k, stream_v, stream_out,
             stream_state_slot, stream_alpha_base) in head_streams:
            if key_tile_bases is None:
                tile_k_base, tile_v_base = stream_k, stream_v
            else:
                tile_k_base, tile_v_base = key_tile_bases[tile_index]
            for feature_tile in range(head_dim // TILE):
                feature_offset = feature_tile * TILE * 2
                q_address = (stream_q + active_start * query_stride +
                             feature_offset)
                program.append(
                    isa.transpose_load(tile_k_base + feature_offset, k_stride))
                program.append(isa.barrier())
                program.extend([
                    isa.gld_a_batch_transpose_strided(
                        q_address, query_stride, page=query_page),
                    isa.gemm(acc=feature_tile != 0,
                             skip_store=feature_tile != head_dim // TILE - 1,
                             imm=TILE, prec=isa.Precision.FP16),
                ])

            # Fetch into the alternate L1 tile while the current QK GEMM
            # drains. The prefetch TOKEN_END fences before Softmax/PV consumes
            # the current accumulator, while the two buffer ranges stay disjoint.
            next_active_start = max(0, (tile_index + 1) * TILE - query_start)
            if (key_tile_prefetchers is not None and
                    tile_index < len(key_tile_prefetchers) and
                    next_active_start < query_count):
                program.extend(key_tile_prefetchers[tile_index])

            for active_row in range(active_count):
                query_slot = active_start + active_row
                query_position = query_start + query_slot
                valid_lanes = min(
                    TILE, sequence_length - tile_start,
                    query_position - tile_start + 1 if causal else TILE)
                program.append(isa.ac2v(0, row=active_row))
                if score_scale_base is not None:
                    program.append(isa.mul(0, 0, 1))
                program.extend([
                    isa.online_softmax(
                        0, 0, 3, valid_lanes=valid_lanes,
                        reset=(tile_index == 0),
                        state_slot=stream_state_slot + query_slot,
                        raw_output=defer_online_normalization),
                    isa.vst_page(0, score_base + active_row * score_row_bytes,
                                 score_page),
                    isa.vst_page(3, stream_alpha_base +
                                 query_slot * score_row_bytes,
                                 alpha_page),
                ])

            if score_page == 2:
                # Page 2 shares the GLD.A control fields with the PostProc
                # wait bit, so retire online-softmax score stores first.
                program.append(isa.barrier())
            program.append(isa.gld_a_batch_transpose(
                score_base, page=score_page,
                wait_postproc=(score_page != 2)))
            program.append(isa.barrier())
            for feature_tile in range(head_dim // TILE):
                feature_offset = feature_tile * TILE * 2
                program.append(isa.transpose_load(
                    tile_v_base + feature_offset, v_stride, transpose=False))
                if serialize_pv_rf_transfers:
                    program.append(isa.barrier())
                program.extend([
                    isa.gemm(acc=False, skip_store=False,
                             imm=TILE, prec=isa.Precision.FP16),
                ])
                for active_row in range(active_count):
                    query_slot = active_start + active_row
                    out_addr = (stream_out + query_slot * output_row_stride +
                                output_feature_offset + feature_offset)
                    program.append(isa.ac2v(0, row=active_row))
                    if tile_index != 0:
                        program.extend([
                            isa.vld_page(2, out_addr, output_page),
                            isa.vld_page(3,
                                         stream_alpha_base +
                                         query_slot * score_row_bytes,
                                         alpha_page),
                            isa.mul(2, 2, 3),
                            isa.add(0, 0, 2),
                        ])
                    program.append(isa.vst_page(0, out_addr, output_page))
    if defer_online_normalization:
        # Keep the output accumulator unnormalized between key tiles. Only
        # after the final tile, retrieve 1/sum(exp(score - max)) from the
        # online-softmax state and apply it once to the accumulated numerator.
        for (stream_q, _stream_k, _stream_v, stream_out,
             stream_state_slot, _stream_alpha_base) in head_streams:
            for query_slot in range(query_count):
                program.append(isa.finalize_online_softmax(
                    alpha_vd=3, state_slot=stream_state_slot + query_slot))
                for feature_tile in range(head_dim // TILE):
                    out_addr = (stream_out + query_slot * output_row_stride +
                                output_feature_offset + feature_tile * TILE * 2)
                    program.extend([
                        isa.vld_page(2, out_addr, output_page),
                        isa.mul(2, 2, 3),
                        isa.vst_page(2, out_addr, output_page),
                    ])
        if wait_after_store and tile_index == len(key_tile_loaders) - 1:
            program.append(isa.barrier())
    program.append(isa.set_cfg_m(1))
    return program


def emit_flash_attention_query_block_reusing_kv_tiles(
    q_base, k_base, v_base, score_base, alpha_base, output_base,
    query_start, query_count, sequence_length, key_tile_loaders,
    head_dim=128, query_row_stride_bytes=None,
    output_row_stride=256, score_scale_base=None,
    output_page=1, alpha_page=2, state_slot_base=0,
    wait_after_store=True, k_token_row_stride_bytes=None,
    v_token_row_stride_bytes=None, serialize_pv_rf_transfers=True,
):
    """Emit up to four hardware-M32 query tiles with K/V loaded once per tile.

    The key-tile loop is outermost, so a gathered K/V tile is reused by each
    M<=32 query sub-block in this logical query block. Each query row still
    has its own online-softmax state; this does not widen the systolic M tile.
    """
    if not 1 <= sequence_length <= MAX_FLASH_SEQUENCE_LENGTH:
        raise ValueError(
            f"FlashAttention sequence length must be in [1,{MAX_FLASH_SEQUENCE_LENGTH}]")
    if (query_count < 1 or query_count > 128 or query_start < 0 or
            query_start + query_count > sequence_length):
        raise ValueError("KV-reuse query block must contain 1..128 causal rows")
    if head_dim < TILE or head_dim > 256 or head_dim % TILE:
        raise ValueError("head_dim must be a multiple of 32 in [32,256]")
    query_stride = (head_dim * 2 if query_row_stride_bytes is None
                    else query_row_stride_bytes)
    k_stride = (head_dim * 2 if k_token_row_stride_bytes is None
                else k_token_row_stride_bytes)
    v_stride = (head_dim * 2 if v_token_row_stride_bytes is None
                else v_token_row_stride_bytes)
    if (len(key_tile_loaders) != (sequence_length + TILE - 1) // TILE or
            not 0 <= state_slot_base <= 128 - query_count):
        raise ValueError("KV-reuse tile count or online-softmax state range is invalid")
    if output_page not in (0, 1, 2) or alpha_page not in (0, 1, 2):
        raise ValueError("FlashAttention buffers must select L1 page 0..2")
    page_bytes = {0: 65536, 1: 65536, 2: 16384}
    alpha_end = alpha_base + query_count * TILE * 2
    score_end = score_base + TILE * TILE * 2
    q_end = q_base + (query_count - 1) * query_stride + head_dim * 2
    k_end = k_base + (TILE - 1) * k_stride + head_dim * 2
    v_end = v_base + (TILE - 1) * v_stride + head_dim * 2
    output_end = (output_base + (query_count - 1) * output_row_stride +
                  head_dim * 2)
    if (any(base < 0 or base % 32 for base in
            (q_base, k_base, v_base, score_base, alpha_base, output_base)) or
            query_stride < head_dim * 2 or query_stride % 32 or
            k_stride < head_dim * 2 or k_stride > 2048 or k_stride % 32 or
            v_stride < head_dim * 2 or v_stride > 2048 or v_stride % 32 or
            output_row_stride < head_dim * 2 or output_row_stride % 32 or
            score_end > page_bytes[0] or alpha_end > page_bytes[alpha_page] or
            q_end > page_bytes[0] or k_end > page_bytes[0] or
            v_end > page_bytes[0] or output_end > page_bytes[output_page]):
        raise ValueError("KV-reuse FlashAttention buffers exceed their L1 pages")
    live_ranges = ((q_base, q_end), (k_base, k_end), (v_base, v_end),
                   (score_base, score_end))
    if score_scale_base is not None:
        if (score_scale_base < 0 or score_scale_base % 32 or
                score_scale_base + TILE * 2 > page_bytes[0]):
            raise ValueError("attention scale vector exceeds its L1 page")
        live_ranges += ((score_scale_base, score_scale_base + TILE * 2),)
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(live_ranges)
           for right in live_ranges[index + 1:]):
        raise ValueError("KV-reuse query, key, value, score, and scale buffers overlap")

    program = [isa.set_cfg_k(TILE)]
    if score_scale_base is not None:
        program.append(isa.vld(1, score_scale_base))
    for tile_index, loader in enumerate(key_tile_loaders):
        tile_start = tile_index * TILE
        program.extend(loader)
        for query_offset in range(0, query_count, TILE):
            block_count = min(TILE, query_count - query_offset)
            block_query_start = query_start + query_offset
            active_start = max(0, tile_start - block_query_start)
            active_count = block_count - active_start
            if active_count <= 0:
                continue
            program.extend([isa.set_cfg_m(active_count)])
            query_address = (q_base +
                             (query_offset + active_start) * query_stride)
            for feature_tile in range(head_dim // TILE):
                feature_offset = feature_tile * TILE * 2
                program.extend([
                    isa.transpose_load(k_base + feature_offset, k_stride),
                    isa.barrier(),
                    isa.gld_a_batch_transpose_strided(
                        query_address + feature_offset, query_stride, page=0),
                    isa.gemm(acc=feature_tile != 0,
                             skip_store=feature_tile != head_dim // TILE - 1,
                             imm=TILE, prec=isa.Precision.FP16),
                ])
            for active_row in range(active_count):
                query_slot = query_offset + active_start + active_row
                query_position = query_start + query_slot
                valid_lanes = min(TILE, sequence_length - tile_start,
                                  query_position - tile_start + 1)
                program.extend([
                    isa.ac2v(0, row=active_row),
                ])
                if score_scale_base is not None:
                    program.append(isa.mul(0, 0, 1))
                program.extend([
                    isa.online_softmax(
                        0, 0, 3, valid_lanes=valid_lanes,
                        reset=(tile_index == 0),
                        state_slot=state_slot_base + query_slot),
                    isa.vst_page(0, score_base + active_row * TILE * 2, 0),
                    isa.vst_page(3, alpha_base + query_slot * TILE * 2,
                                 alpha_page),
                ])
            program.extend([
                isa.gld_a_batch_transpose(
                    score_base, page=0, wait_postproc=True),
                isa.barrier(),
            ])
            for feature_tile in range(head_dim // TILE):
                feature_offset = feature_tile * TILE * 2
                program.append(isa.transpose_load(
                    v_base + feature_offset, v_stride, transpose=False))
                if serialize_pv_rf_transfers:
                    program.append(isa.barrier())
                program.append(isa.gemm(
                    acc=False, skip_store=False, imm=TILE,
                    prec=isa.Precision.FP16))
                for active_row in range(active_count):
                    query_slot = query_offset + active_start + active_row
                    out_address = (output_base + query_slot * output_row_stride +
                                   feature_offset)
                    program.append(isa.ac2v(0, row=active_row))
                    if tile_index != 0:
                        program.extend([
                            isa.vld_page(2, out_address, output_page),
                            isa.vld_page(3,
                                         alpha_base + query_slot * TILE * 2,
                                         alpha_page),
                            isa.mul(2, 2, 3),
                            isa.add(0, 0, 2),
                        ])
                    program.append(isa.vst_page(0, out_address, output_page))
    if wait_after_store:
        program.append(isa.barrier())
    program.append(isa.set_cfg_m(1))
    return program


def emit_causal_attention_head_queries(
    q_base, k_base, v_base, score_base, output_base,
    query_start, query_count, valid_tokens, head_dim=128,
    token_row_stride=None, output_row_stride=4096,
    output_feature_offset=0, output_page=1,
    score_scale_base=None,
):
    """Compute a consecutive causal query block for one head.

    Results are written into padded full-hidden token rows so independently
    computed heads can be assembled in one L1 page before a grouped DMA write.
    ``query_start`` is the sequence position; output rows are local to this
    block and use ``output_row_stride`` bytes.
    """
    if not 1 <= valid_tokens <= TILE:
        raise ValueError("valid_tokens must be in [1,32]")
    if (query_start < 0 or query_count < 1 or
            query_start + query_count > valid_tokens):
        raise ValueError("causal query block must fit the valid sequence")
    if head_dim < TILE or head_dim > 256 or head_dim % TILE:
        raise ValueError("head_dim must be a multiple of 32 in [32,256]")
    if output_page not in (0, 1, 2):
        raise ValueError("attention output page must be 0, 1, or 2")
    stride = head_dim * 2 if token_row_stride is None else token_row_stride
    if stride < head_dim * 2 or stride > 512 or stride % 32:
        raise ValueError("token row stride must be 32-byte aligned and at most 512 bytes")
    if output_row_stride < output_feature_offset + head_dim * 2:
        raise ValueError("attention output row does not contain the selected head")
    page_bytes = {0: 65536, 1: 65536, 2: 16384}[output_page]
    output_end = (output_base + (query_count - 1) * output_row_stride +
                  output_feature_offset + head_dim * 2)
    if (output_base < 0 or output_feature_offset < 0 or
            output_base % 32 or output_feature_offset % 32 or
            output_end > page_bytes):
        raise ValueError("causal attention output exceeds its selected L1 page")

    program = []
    for local_query, query_token in enumerate(
            range(query_start, query_start + query_count)):
        out_base = (output_base + local_query * output_row_stride +
                    output_feature_offset)
        program.extend(emit_attention_head(
            q_base, k_base, v_base, score_base, out_base,
            query_token=query_token, valid_tokens=valid_tokens,
            head_dim=head_dim, token_row_stride=stride, causal=True,
            score_scale_base=score_scale_base, output_page=output_page,
            wait_after_store=(local_query == query_count - 1)))
    return program


def emit_batched_causal_attention_head_queries(
    q_base, k_base, v_base, score_base, query_pack_base, output_base,
    query_start, query_count, valid_tokens, head_dim=128,
    token_row_stride=None, output_row_stride=4096,
    output_feature_offset=0, output_page=1,
    score_scale_base=None, wait_after_store=True,
    q_block_packed=False, query_row_stride_bytes=None,
):
    """Compute a causal query block as two batched GEMM phases.

    The caller must set ``CFG_M=query_count`` and ``CFG_K=32`` before this
    fragment. Q rows are transposed directly from their token-major L1 matrix
    for each 32-feature slice, then one GEMM computes scores for all queries
    in the block. ``query_pack_base`` remains accepted for compatibility and
    validation, but the current lowering does not access that scratch region.
    The score GEMM is followed by per-row causal softmax, one batched
    activation load, and a second GEMM for all value rows. This raises systolic
    M utilization versus issuing one M=1 GEMM per query without changing the
    numerical attention operation.
    Callers may defer the final output fence when the next operation accesses
    disjoint buffers.
    """
    if not 1 <= valid_tokens <= TILE:
        raise ValueError("valid_tokens must be in [1,32]")
    if (query_start < 0 or query_count < 1 or
            query_start + query_count > valid_tokens or query_count > TILE):
        raise ValueError("causal query block must fit the valid sequence")
    if head_dim < TILE or head_dim > 256 or head_dim % TILE:
        raise ValueError("head_dim must be a multiple of 32 in [32,256]")
    if output_page not in (0, 1, 2):
        raise ValueError("attention output page must be 0, 1, or 2")
    stride = head_dim * 2 if token_row_stride is None else token_row_stride
    if stride < head_dim * 2 or stride > 512 or stride % 32:
        raise ValueError("token row stride must be 32-byte aligned and at most 512 bytes")
    query_stride = (stride if query_row_stride_bytes is None
                    else query_row_stride_bytes)
    if (query_stride < head_dim * 2 or query_stride > 512 or
            query_stride % 32):
        raise ValueError("query row stride must be 32-byte aligned and at most 512 bytes")
    if (output_row_stride < output_feature_offset + head_dim * 2 or
            output_row_stride % 32):
        raise ValueError("attention output row does not contain the selected head")

    page_bytes = {0: 65536, 1: 65536, 2: 16384}
    input_ranges = []
    q_rows = query_count if q_block_packed else valid_tokens
    for base, row_count, row_stride in (
            (q_base, q_rows, query_stride),
            (k_base, valid_tokens, stride),
            (v_base, valid_tokens, stride)):
        end = base + (row_count - 1) * row_stride + head_dim * 2
        if base < 0 or base % 32 or end > page_bytes[0]:
            raise ValueError("attention input exceeds page 0")
        input_ranges.append((base, end))
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(input_ranges)
           for right in input_ranges[index + 1:]):
        raise ValueError("Q, K, and V input buffers must not overlap")

    tile_row_bytes = TILE * 2
    scratch_end = query_pack_base + query_count * tile_row_bytes
    score_end = score_base + query_count * tile_row_bytes
    if (query_pack_base < 0 or query_pack_base % 32 or
            scratch_end > page_bytes[0] or score_base < 0 or
            score_base % 32 or score_end > page_bytes[0]):
        raise ValueError("batched attention scratch exceeds page 0")
    for scratch_range in ((query_pack_base, scratch_end),
                          (score_base, score_end)):
        if any(scratch_range[0] < source[1] and source[0] < scratch_range[1]
               for source in input_ranges):
            raise ValueError("attention scratch overlaps a live Q/K/V buffer")
    if (query_pack_base < score_end and score_base < scratch_end):
        raise ValueError("query-pack and score scratch buffers must not overlap")
    if score_scale_base is not None and (
            score_scale_base < 0 or score_scale_base % 32 or
            score_scale_base + 256 > page_bytes[0] or
            any(score_scale_base < live[1] and
                live[0] < score_scale_base + 256
                for live in input_ranges + [
                    (query_pack_base, scratch_end), (score_base, score_end)])):
        raise ValueError("attention scale vector overlaps or exceeds live page-0 data")

    output_end = (output_base + (query_count - 1) * output_row_stride +
                  output_feature_offset + head_dim * 2)
    if (output_base < 0 or output_feature_offset < 0 or
            output_base % 32 or output_feature_offset % 32 or
            output_end > page_bytes[output_page]):
        raise ValueError("causal attention output exceeds its selected L1 page")

    program = []
    feature_tiles = head_dim // TILE
    for feature_tile in range(feature_tiles):
        feature_offset = feature_tile * tile_row_bytes
        # Read the token-major Q slice directly. The row-strided transpose
        # removes the per-query VLD/VST packing loop without changing Q order.
        q_row = 0 if q_block_packed else query_start
        q_address = q_base + q_row * query_stride + feature_offset
        q_load = (isa.gld_a_batch_transpose(q_address, page=0)
                  if query_stride == tile_row_bytes else
                  isa.gld_a_batch_transpose_strided(
                      q_address, query_stride, page=0))
        program.extend([
            isa.transpose_load(k_base + feature_offset, stride),
            # TGLD and GLD.A share the single L1 read port. Retire the
            # transpose load before starting the activation read; the GEMM
            # scoreboard still handles RF readiness after GLD.A is issued.
            isa.barrier(),
            q_load,
            isa.gemm(acc=feature_tile != 0,
                     skip_store=feature_tile != feature_tiles - 1,
                     imm=TILE, prec=isa.Precision.FP16),
        ])

    if score_scale_base is not None:
        program.append(isa.vld(1, score_scale_base))
    for local_query in range(query_count):
        query_token = query_start + local_query
        program.append(isa.ac2v(0, row=local_query))
        if score_scale_base is not None:
            program.append(isa.mul(0, 0, 1))
        program.extend([
            isa.softmax(0, 0, valid_lanes=min(valid_tokens, query_token + 1)),
            isa.vst_page(0, score_base + local_query * tile_row_bytes, 0),
        ])
    # Wait only for score VSTs before the batched ARF transpose; unrelated
    # resources need not retire at this producer-consumer boundary. TGLD also
    # uses the L1 read port, so retire this GLD.A before the first V transpose.
    program.append(isa.gld_a_batch_transpose(
        score_base, page=0, wait_postproc=True))
    program.append(isa.barrier())
    for feature_tile in range(feature_tiles):
        feature_offset = feature_tile * tile_row_bytes
        program.extend([
            isa.transpose_load(v_base + feature_offset, stride,
                               transpose=False),
            # Each tile is a distinct N slice of the head output, not a K
            # partial sum. Retire it independently before loading the next
            # slice so all head features reach their own output offsets.
            isa.gemm(acc=False, skip_store=False,
                     imm=TILE, prec=isa.Precision.FP16),
        ])
        for local_query in range(query_count):
            out_base = (output_base + local_query * output_row_stride +
                        output_feature_offset + feature_offset)
            program.extend([
                isa.ac2v(0, row=local_query),
                isa.vst_page(0, out_base, output_page),
            ])
    if wait_after_store:
        program.append(isa.barrier())
    return program
