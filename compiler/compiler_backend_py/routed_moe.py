"""Double-buffered routed-expert body for the runtime-free Layer-0 path.

Requires cfg_M=1, cfg_N=32, base cfg_K=32 and compact resident descriptors.
Router logits must already be in ACC. The hardware token loop loads each
post-attention-normalized activation from ACT DDR, evaluates the routed FFN,
and stores the weighted result contiguously in OUT DDR.
"""
try:
    from . import isa
except ImportError:  # driver.py also supports direct execution from this directory
    import isa

INPUT = 0x4000       # 2048 FP16 values
GATE = 0x5000        # 1408 FP16 values
OUTPUT = 0x7000      # 2048 FP16 values; weighted sum of four experts
TOKEN_UP_BASES = (0x8000, 0x9000, 0xA000, 0xB000)
UP = TOKEN_UP_BASES[0]  # First expert's activated intermediate vector
WEIGHT_BANK = 2      # 0x2000/0x3000 ping-pong staging pair
GROUP_WEIGHT_BANK = 1  # bank 1/0 ping-pong pair; bank 0 is expert-DMA addressed
GROUP_ACTIVATION = 0x2000
GROUP_INTERMEDIATE_SPILL_BASE = 0xA000
GROUP_INTERMEDIATE_EXTRA_BYTES = 0x4000
GROUP_OUTPUT_ACCUM = 0  # token-major accumulator in the high 64-KiB L1 page
TOKEN_ROUTE_OUTPUT_REG = 0
TOKEN_ROUTE_WEIGHT_REG = 1
TOKEN_ROUTE_WEIGHT_REGS = (1, 2, 3, 4)
TOKEN_ROUTE_SUM_REG_BASE = 5
TOKEN_ROUTE_SUM_REG_COUNT = 11
L1_BANK_BYTES = 4096


def _activation_dma_disjoint(base, size):
    """Prove a page-0 activation range avoids all weight staging banks."""
    end = base + size
    if base < 0 or size <= 0 or end > 0x10000:
        return False
    return all(end <= bank * L1_BANK_BYTES or
               base >= (bank + 1) * L1_BANK_BYTES
               for bank in range(4))


def select_expert_gemm_k(k_tiles, other_tiles=None):
    """Choose the widest K block aligned to one projection's K dimension."""
    if not (1 <= k_tiles <= 255):
        raise ValueError("expert K tile counts must be in [1,255]")
    if other_tiles is not None and not 1 <= other_tiles <= 255:
        raise ValueError("expert K tile counts must be in [1,255]")
    candidates = (k_tiles,) if other_tiles is None else (k_tiles, other_tiles)
    if all(tiles % 8 == 0 for tiles in candidates):
        return 256
    if all(tiles % 4 == 0 for tiles in candidates):
        return 128
    if all(tiles % 2 == 0 for tiles in candidates):
        return 64
    return 32


def split_expert_k_tiles(k_tiles, max_gemm_k=256):
    """Partition a K dimension into the widest legal 32-aligned GEMMs."""
    if not 1 <= k_tiles <= 255 or max_gemm_k not in (32, 64, 128, 256):
        raise ValueError("invalid expert K dimension or maximum GEMM K")
    remaining = k_tiles
    chunks = []
    max_tiles_per_chunk = max_gemm_k // 32
    legal_chunk_tiles = (max_tiles_per_chunk,) + tuple(
        tiles for tiles in (4, 2, 1) if tiles < max_tiles_per_chunk)
    for tiles_per_chunk in legal_chunk_tiles:
        while remaining >= tiles_per_chunk:
            chunks.append(tiles_per_chunk * 32)
            remaining -= tiles_per_chunk
    if remaining:
        raise ValueError("expert K dimension cannot be partitioned into legal GEMMs")
    return chunks


def group_down_weight_bank(n_tile, chunk_order, chunk_count):
    """Select the safe ping-pong bank for a grouped Down K chunk."""
    if n_tile < 0 or chunk_count < 1 or not 0 <= chunk_order < chunk_count:
        raise ValueError("invalid grouped Down weight-bank coordinates")
    tile_origin = GROUP_WEIGHT_BANK ^ ((n_tile * chunk_count) & 1)
    return tile_origin ^ (chunk_order & 1)


def route_group_l1_fits(tokens, hidden_size, intermediate_size):
    """Check the grouped-FFN page layout without emitting its instruction body."""
    if (not 1 <= tokens <= 32 or hidden_size < 32 or hidden_size > 2048 or
            hidden_size % 32 or intermediate_size < 32 or
            intermediate_size > 8160 or intermediate_size % 32):
        return False

    hidden_tiles = hidden_size // 32
    intermediate_tiles = intermediate_size // 32
    gemm_k = select_expert_gemm_k(hidden_tiles)
    if hidden_tiles % (gemm_k // 32):
        return False

    activation_end = GROUP_ACTIVATION + tokens * hidden_size * 2
    output_bytes = tokens * hidden_size * 2
    weight_pair_end = (max(GROUP_WEIGHT_BANK, GROUP_WEIGHT_BANK ^ 1) + 1) * 4096
    if (activation_end > 0x10000 or output_bytes > 0x10000 or
            output_bytes < weight_pair_end or output_bytes % 2048):
        return False

    page1_capacity = 0x10000 - output_bytes
    page0_spill_capacity = 0x10000 - GROUP_INTERMEDIATE_SPILL_BASE
    page0_available = GROUP_INTERMEDIATE_SPILL_BASE >= activation_end
    page2_spill_capacity = GROUP_INTERMEDIATE_EXTRA_BYTES
    chunk_cursor = 0
    page0_spill_used = 0
    page2_spill_used = 0
    for chunk_k in split_expert_k_tiles(intermediate_tiles, 256):
        chunk_bytes = tokens * chunk_k * 2
        if chunk_cursor + chunk_bytes <= page1_capacity:
            pass
        elif (page0_available and
              page0_spill_used + chunk_bytes <= page0_spill_capacity):
            page0_spill_used += chunk_bytes
        elif page2_spill_used + chunk_bytes <= page2_spill_capacity:
            page2_spill_used += chunk_bytes
        else:
            return False
        chunk_cursor += chunk_bytes
    return True


def projection(nt, kt_total, slot, proj, act_base, gemm_k=32,
              scale_preloaded=False, active_cfg_k=None,
              trailing_cfg_k=None, prefetch_scale_nt=None):
    """Compute one N tile with K loops and an explicit mixed-width tail."""
    if gemm_k not in (32, 64, 128, 256):
        raise ValueError("expert GEMM K must be 32, 64, 128, or 256")
    chunks = split_expert_k_tiles(kt_total, gemm_k)
    p = []
    # INT1 is only the instruction fallback here. For expert_sel != 0, the
    # hardware descriptor overrides both GLD.W and GEMM precision per expert
    # and projection, preserving the routed 1/2/3-bit mixed-precision layout.
    if len(set(chunks)) == 1:
        chunk_k = chunks[0]
        gemm_count = len(chunks)
        allow_dma_overlap = _activation_dma_disjoint(
            act_base, gemm_count * chunk_k * 2)
        if active_cfg_k is not None and active_cfg_k != chunk_k:
            p.append(isa.set_cfg_k(chunk_k))
        if not scale_preloaded:
            p.extend([isa.nop_scale(nt), isa.barrier()])
        p.extend([isa.hw_loop(gemm_count, 4),
                  # The opt-in is emitted only when the complete activation
                  # range is disjoint from all low-page weight DMA banks.
                  isa.gld_a(act_base,
                            allow_dma_overlap=allow_dma_overlap),
                  isa.expert_dma(nt, 0, kt_total, slot, proj, WEIGHT_BANK),
                  isa.gld_w(isa.Precision.INT1, WEIGHT_BANK * 4096,
                            group=0, expert_sel=slot, proj=proj),
                  isa.gemm(imm=chunk_k, prec=isa.Precision.INT1,
                           expert_sel=slot, proj=proj)])
    else:
        if active_cfg_k is None:
            active_cfg_k = gemm_k
        if trailing_cfg_k is not None and trailing_cfg_k not in (32, 64, 128, 256):
            raise ValueError("trailing expert CFG_K must be 32, 64, 128, or 256")
        first_k = chunks[0]
        if active_cfg_k != first_k:
            p.append(isa.set_cfg_k(first_k))
            active_cfg_k = first_k
        leading_count = 1
        while (leading_count < len(chunks) and
               chunks[leading_count] == first_k):
            leading_count += 1
        loop_prefix = scale_preloaded and leading_count > 1
        if loop_prefix:
            # Compact the repeated wide-K prefix into the existing K-loop
            # sequencer. Its DMA/GLD_A/GLD_W address updates match the explicit
            # tile offsets below; the loop's final GEMM stores a partial sum,
            # which the following tail GEMM continues with acc=True.
            p.extend([
                isa.hw_loop(leading_count, 4),
                isa.gld_a(
                    act_base,
                    allow_dma_overlap=_activation_dma_disjoint(
                        act_base, leading_count * first_k * 2)),
                isa.expert_dma(nt, 0, kt_total, slot, proj, WEIGHT_BANK),
                isa.gld_w(isa.Precision.INT1, WEIGHT_BANK * 4096,
                          group=0, expert_sel=slot, proj=proj),
                isa.gemm(imm=first_k, prec=isa.Precision.INT1,
                         expert_sel=slot, proj=proj),
            ])
            kt_start = leading_count * (first_k // 32)
            first_index = leading_count
            active_cfg_k = first_k
            if first_index < len(chunks) and chunks[first_index] != active_cfg_k:
                p.append(isa.set_cfg_k(chunks[first_index]))
                active_cfg_k = chunks[first_index]
        else:
            if not scale_preloaded:
                p.append(isa.nop_scale(nt))
            kt_start = 0
            first_index = 0

        for index in range(first_index, len(chunks)):
            chunk_k = chunks[index]
            if active_cfg_k != chunk_k:
                p.append(isa.set_cfg_k(chunk_k))
                active_cfg_k = chunk_k
            bank = WEIGHT_BANK ^ (index & 1)
            scale_group = kt_start // 4
            activation_addr = act_base + kt_start * 64
            p.append(isa.gld_a(
                activation_addr,
                allow_dma_overlap=_activation_dma_disjoint(
                    activation_addr, chunk_k * 2)))
            # Start this chunk's weight read only after its activation has
            # been dispatched. GLD.A is blocked by an active expert DMA; on
            # later unrolled chunks it can overlap the preceding GEMM instead
            # of waiting behind this chunk's weight transfer.
            p.append(isa.expert_dma(nt, kt_start, kt_total, slot, proj, bank))
            if not scale_preloaded and index == first_index:
                # Preserve the metadata/weight completion fence before GLD.W.
                p.append(isa.barrier())
            p.append(isa.gld_w(isa.Precision.INT1, bank * 4096,
                               group=scale_group, expert_sel=slot,
                               proj=proj))
            next_k = chunks[index + 1] if index + 1 < len(chunks) else None
            cfg_k_after_current = (
                next_k if next_k is not None and next_k != chunk_k else
                trailing_cfg_k if next_k is None else None)
            p.append(isa.gemm(
                acc=(index != 0), imm=chunk_k, prec=isa.Precision.INT1,
                skip_store=(index + 1 != len(chunks)), expert_sel=slot,
                proj=proj))
            if (cfg_k_after_current is not None and
                    cfg_k_after_current != active_cfg_k):
                # GEMM takes K from its immediate. Update CFG_K after it is
                # accepted so decode can overlap the next-width setup with
                # this compute; the following iteration does not issue its
                # GLD.A or weight DMA until the update has been accepted.
                p.append(isa.set_cfg_k(cfg_k_after_current))
                active_cfg_k = cfg_k_after_current
            kt_start += chunk_k // 32
    # AC2V is dependency-gated by RES_GEMM in ucode_decode. Avoid a global
    # barrier here: it would also wait for unrelated DMA/RF activity.
    if prefetch_scale_nt is not None:
        p.append(isa.nop_scale(prefetch_scale_nt))
    # The following VST is serialized by the same post-process resource used
    # by AC2V, so a second global barrier here only waits on unrelated units.
    p.append(isa.ac2v(0, row=0))
    return p


def emit_token(token, hidden_size=2048, intermediate_size=1408,
               current_token=False, output_base=OUTPUT,
               restore_cached_route=True,
               scale_metadata_n_invariant=False, configure_cfg_m=True,
               defer_output_fence=False):
    """Lower the complete routed branch for one cached token (no expert IDs).

    K tiles use the WRF/ARF ping-pong banks: while one GEMM is active, the
    following tile may be fetched and loaded into the inactive RF bank. The
    decoder's DMA/GLD/GEMM readiness checks provide producer-consumer ordering;
    no global barrier is used inside the K loop.
    No host intervention is needed between route selection and combination.
    Each selected expert retains its complete activated intermediate vector in
    a separate L1 scratch region until its Down projection has consumed it.
    INPUT and OUTPUT remain token-local regions reused by the outer loop.
    Set ``configure_cfg_m=False`` only when the caller establishes M=1 before
    entering a repeated body and no instruction in that body changes M.
    Set ``defer_output_fence=True`` only when the next stage consumes OUTPUT
    through PostProc/GEMM dependencies; the base CFG_K is restored before the
    final stores, which then retire in parallel with independent setup work.
    """
    if (hidden_size < 32 or hidden_size > 2048 or hidden_size % 32 or
            intermediate_size < 32 or intermediate_size > 1408 or intermediate_size % 32):
        raise ValueError("dimensions must be 32-lane tile multiples within L1 scratch capacity")
    h_tiles, i_tiles = hidden_size // 32, intermediate_size // 32
    scratch_bytes = intermediate_size * 2
    scratch_regions = [(GATE, scratch_bytes)] + [
        (base, scratch_bytes) for base in TOKEN_UP_BASES]
    output_region = (output_base, hidden_size * 2)
    for index, (base, size) in enumerate(scratch_regions):
        if base < 0 or base + size > 0x10000:
            raise ValueError("routed token scratch exceeds the 64-KiB L1 address space")
        if base < INPUT + hidden_size * 2 and INPUT < base + size:
            raise ValueError("routed token scratch overlaps the input activation")
        if base < output_region[0] + output_region[1] and output_region[0] < base + size:
            raise ValueError("routed token scratch overlaps the output activation")
        for other_base, other_size in scratch_regions[:index]:
            if base < other_base + other_size and other_base < base + size:
                raise ValueError("routed token scratch regions overlap")
    if output_base < 0 or output_region[0] + output_region[1] > 0x10000:
        raise ValueError("routed token output exceeds the 64-KiB L1 address space")
    input_region = (INPUT, hidden_size * 2)
    weight_region = (WEIGHT_BANK * 4096, (WEIGHT_BANK + 2) * 4096)
    if (output_region[0] < input_region[0] + input_region[1] and
            input_region[0] < output_region[0] + output_region[1]):
        raise ValueError("routed token output overlaps the input activation")
    if (output_region[0] < weight_region[1] and
            weight_region[0] < output_region[0] + output_region[1]):
        raise ValueError("routed token output overlaps the weight staging banks")
    # Gate/Up use uniform wide K loops. Down may use an explicit mixed-width
    # partition so a non-multiple tail does not force every GEMM to a narrower
    # K; the descriptor still selects each expert's actual weight precision.
    gate_up_gemm_k = select_expert_gemm_k(h_tiles)
    down_k_chunks = split_expert_k_tiles(i_tiles, 256)
    down_gemm_k = max(down_k_chunks)
    # This body processes one token at a time. Standalone callers explicitly
    # restore M=1 if a preceding grouped-expert body selected M=32.
    p = [isa.set_cfg_m(1)] if configure_cfg_m else []
    if restore_cached_route:
        # ROUTE itself waits for all prior work, and the next CFG_K/NOP_SCALE
        # command also waits for route completion. Avoid an extra global wait.
        p.append(isa.route_current_token() if current_token else isa.route(token))
    # Route weights are constant for the token. Reserve regs 2..4 across the
    # Gate/Up epilogue; reg 1 is its temporary and is loaded after that stage.
    for slot in range(2, 5):
        p.append(isa.rweight(TOKEN_ROUTE_WEIGHT_REGS[slot - 1], slot))
    if scale_metadata_n_invariant:
        # The descriptor loader fills scale/zero SRAM for all four selected
        # experts and all three projections. This reuse is valid only when the
        # caller proves scale metadata is identical across every N tile.
        p.append(isa.nop_scale(0))
    if gate_up_gemm_k != 32:
        p.append(isa.set_cfg_k(gate_up_gemm_k))
    active_cfg_k = gate_up_gemm_k
    # Gate/Up N tiles are the intermediate dimension, not the hidden output
    # dimension. Load scale metadata once per N tile, then use it for all four
    # selected experts before advancing to the next N tile.
    prefetched_scale_nt = None
    for nt in range(i_tiles):
        if not scale_metadata_n_invariant:
            if prefetched_scale_nt != nt:
                p.append(isa.nop_scale(nt))
            # The following expert DMA queues behind the scale loader on the
            # shared AXI read port; GLD.W waits for that DMA before consuming
            # the selected scale context. Keep the prior tile's PostProc work
            # independent so it can overlap the scale transfer.
            prefetched_scale_nt = None
        for slot in range(1, 5):
            p.extend(projection(
                nt, h_tiles, slot, 0, INPUT,
                gemm_k=gate_up_gemm_k, scale_preloaded=True,
                active_cfg_k=active_cfg_k))
            p.append(isa.vst(0, GATE + nt * 64))
            p.extend(projection(
                nt, h_tiles, slot, 1, INPUT,
                gemm_k=gate_up_gemm_k, scale_preloaded=True,
                active_cfg_k=active_cfg_k,
                prefetch_scale_nt=(nt + 1 if
                                   not scale_metadata_n_invariant and
                                   slot == 4 and nt + 1 < i_tiles else None)))
            if (not scale_metadata_n_invariant and slot == 4 and
                    nt + 1 < i_tiles):
                # All four experts consumed this N tile before projection()
                # starts the next tile's scale DMA ahead of the ACC drain.
                prefetched_scale_nt = nt + 1
            # Fuse SiLU(Gate) * Up into the ACC drain while preserving one
            # complete intermediate activation vector for each route slot.
            p.extend([
                isa.vld(1, GATE + nt * 64),
                isa.silu(1, 1),
                isa.mul(0, 0, 1),
                isa.vst(0, TOKEN_UP_BASES[slot - 1] + nt * 64),
            ])
    # Gate/Up used PRF register 1 as its SiLU temporary. RWEIGHT uses the same
    # post-process resource as the final activation store, so it naturally
    # waits for that store without a global barrier. Restore route slot 1 once
    # before Down; slots 2..4 have remained resident since token entry.
    p.append(isa.rweight(TOKEN_ROUTE_WEIGHT_REGS[0], 1))

    # Keep a block of hidden-output sums resident in PRF across all four
    # experts. This removes per-expert L1 read/modify/write traffic while
    # preserving the original Top-4 accumulation order.
    for block_start in range(0, h_tiles, TOKEN_ROUTE_SUM_REG_COUNT):
        block_end = min(h_tiles, block_start + TOKEN_ROUTE_SUM_REG_COUNT)
        for nt in range(block_start, block_end):
            if not scale_metadata_n_invariant:
                if prefetched_scale_nt != nt:
                    p.append(isa.nop_scale(nt))
                # Expert DMA/GLD.W provides the scale-consumer dependency;
                # do not globally fence the prior tile's output drain.
                prefetched_scale_nt = None
            # Down N tiles span hidden_size; its K dimension is the complete
            # intermediate activation retained for this expert in L1.
            for slot in range(1, 5):
                p.extend(projection(
                    nt, i_tiles, slot, 2, TOKEN_UP_BASES[slot - 1],
                    gemm_k=down_gemm_k,
                    scale_preloaded=True,
                    active_cfg_k=active_cfg_k,
                    trailing_cfg_k=down_k_chunks[0],
                    prefetch_scale_nt=(nt + 1 if
                                       not scale_metadata_n_invariant and
                                       slot == 4 and nt + 1 < h_tiles else None)))
                if (not scale_metadata_n_invariant and slot == 4 and
                        nt + 1 < h_tiles):
                    # Start Down's next N-tile metadata before draining ACC.
                    prefetched_scale_nt = nt + 1
                active_cfg_k = down_k_chunks[0]
                sum_reg = TOKEN_ROUTE_SUM_REG_BASE + nt - block_start
                weight_reg = TOKEN_ROUTE_WEIGHT_REGS[slot - 1]
                if slot == 1:
                    p.append(isa.mul(sum_reg, TOKEN_ROUTE_OUTPUT_REG,
                                     weight_reg))
                else:
                    p.extend([
                        isa.mul(TOKEN_ROUTE_OUTPUT_REG,
                                TOKEN_ROUTE_OUTPUT_REG, weight_reg),
                        isa.add(sum_reg, sum_reg, TOKEN_ROUTE_OUTPUT_REG),
                    ])
        for nt in range(block_start, block_end):
            sum_reg = TOKEN_ROUTE_SUM_REG_BASE + nt - block_start
            # Restore the shared-expert base K before the final L1 output
            # stores. A fused consumer may start independent scale/weight
            # transfers while those stores retire through PostProc.
            if nt == h_tiles - 1 and active_cfg_k != 32:
                p.append(isa.set_cfg_k(32))
                active_cfg_k = 32
            p.append(isa.vst(sum_reg, output_base + nt * 64))
        # VST and the next block's MUL/ADD share the post-process resource, so
        # resource arbitration retires these stores before PRF sums are reused.
        # The next block's scale DMA, and invariant-scale GEMMs, can overlap.
    # Standalone callers receive a fully retired output. The fused Layer-0
    # caller can defer this fence: downstream VLD/GEMM dependencies serialize
    # consumption of OUTPUT while unrelated scale/weight transfers overlap.
    if not defer_output_fence:
        p.append(isa.barrier())
    if len(p) + 1 > 262144:
        raise ValueError("routed token body exceeds the hardware instruction SRAM")
    return p


def emit_router_cache(tokens=32):
    """Run all token TopKs before the two router ACC sets are overwritten."""
    if not 1 <= tokens <= 32:
        raise ValueError("router cache supports 1..32 tokens")
    # TOPK is itself all_idle-gated in ucode_decode. The next TOPK therefore
    # waits for the preceding result, making an explicit global barrier after
    # each row redundant while preserving the per-row completion ordering.
    return [isa.topk(token) for token in range(tokens)]


def emit_route_group_projection(expert_id, tokens, nt, kt_total, proj,
                                output_base, act_base=GROUP_ACTIVATION, gemm_k=64,
                                output_tile_stride=None, select_group=True,
                                gather_activations=True, configure=True,
                                restore=True, weight_bank=GROUP_WEIGHT_BANK,
                                output_gemm_k=None, fuse_gate_base=None,
                                output_page=0, output_spill_base=None,
                                output_spill_start_tile=None,
                                output_spill_page=0,
                                output_spill2_base=None,
                                output_spill2_start_tile=None,
                                output_spill2_page=2, k_chunk_order=None,
                                resident_activation_chunk=None,
                                preloaded_scale_nt=None,
                                scale_metadata_n_invariant=False,
                                prefetched_first_dma=False,
                                prefetched_first_weight=False,
                                prefetch_next_nt=None,
                                output_gemm_chunks=None,
                                output_spill_gemm_chunks=None,
                                output_spill2_gemm_chunks=None,
                                activation_source="act",
                                activation_byte_offset=0):
    """Run one projection using the current expert's active route-row count.

    The route-group DMA gathers only active DDR activation rows into a compact
    blocked L1 layout. Empty groups stage one zero row to keep GEMM M legal.
    Results are drained in compact route-group row order; a later combine/scatter stage must apply the
    per-token route weights and restore original token positions. ACT DDR must
    contain the token-major pre-expert activations and all requested Top-K
    rows must already be cached. Supplying output_gemm_k stores output in
    [K-chunk][route-row][K value] order; the fused Up path can apply
    SiLU(Gate)*Up while draining accumulator rows without a scratch tile.
    ``preloaded_scale_nt`` is used only by Up when Gate has just loaded the
    same N tile's scale metadata for all three projections. Set
    ``scale_metadata_n_invariant`` only when the caller preloads scale data
    once and guarantees it is identical across every output N tile.
    ``k_chunk_order`` enables an explicit K traversal for the grouped path;
    each weight, scale group, and activation address is still selected by the
    same logical chunk index. Reversing the reduction order preserves chunk
    pairing but changes FP32 accumulation rounding; its effect on the grouped
    golden tolerance must be checked in simulation.
    Down's extended expert-DMA encoding carries the next chunk's K width so a
    128/256 boundary can prefetch without changing the active GEMM CFG_K.
    Gate may consume a first chunk prefetched by the preceding Up tile; Up may
    prefetch the following Gate tile's first chunk at its final K step.
    """
    if not 0 <= expert_id < 60 or not 1 <= tokens <= 32:
        raise ValueError("route group requires expert 0..59 and 1..32 cached tokens")
    if proj not in (0, 1) or not 0 <= nt < 256:
        raise ValueError("DDR route-group gather currently supports Gate/Up projection only")
    if (preloaded_scale_nt is not None and
            (proj != 1 or preloaded_scale_nt != nt or
             scale_metadata_n_invariant)):
        raise ValueError(
            "same-tile scale reuse applies only to Up without global scale reuse")
    if kt_total not in range(1, 256):
        raise ValueError("expert K dimension must contain 1..255 physical 32-K tiles")
    if gemm_k not in (32, 64, 128, 256) or kt_total % (gemm_k // 32):
        raise ValueError("expert K tiles must divide evenly into GEMM K chunks")
    if not 1 <= weight_bank <= 15:
        raise ValueError("expert DMA ping-pong must start in a non-zero L1 bank")
    if act_base < 0x1000 or act_base > 0xF000 or act_base % 0x1000:
        raise ValueError("group activation staging base must be a 4-KiB L1 bank")
    if act_base // 32 + kt_total * tokens * 2 > 2048:
        raise ValueError("group activation K-chunk staging would exceed L1 capacity")
    if output_tile_stride is None:
        output_tile_stride = tokens * 64
    if output_tile_stride not in (0, tokens * 64):
        raise ValueError("group output tile stride must match the padded route-row capacity")
    if output_spill_base is not None and (
            output_gemm_k is None or output_spill_start_tile is None or
            output_gemm_k not in (32, 64, 128, 256) or
            output_spill_start_tile % (output_gemm_k // 32) or
            output_spill_base % 2048 or output_spill_page not in (0, 1) or
            output_spill_page == output_page):
        raise ValueError("group output spill must start on a K-chunk boundary in the opposite L1 page")
    if output_spill2_base is not None and (
            output_gemm_k is None or output_spill2_start_tile is None or
            output_gemm_k not in (32, 64, 128, 256) or
            output_spill2_start_tile % (output_gemm_k // 32) or
            output_spill2_page not in (0, 1, 2) or
            output_spill2_page == output_page or
            (output_spill_base is not None and
             output_spill2_page == output_spill_page)):
        raise ValueError("second grouped output spill must use a distinct page and K-chunk boundary")
    mapped_base, mapped_nt, mapped_page = _route_group_output_location(
        output_base, nt, output_page, output_spill_base,
        output_spill_start_tile, output_spill_page,
        output_spill2_base, output_spill2_start_tile,
        output_spill2_page)
    if output_gemm_chunks is not None:
        if mapped_page == output_spill2_page and output_spill2_base is not None:
            layout_chunks = (output_spill2_gemm_chunks
                             if output_spill2_gemm_chunks is not None
                             else output_gemm_chunks)
        elif mapped_page == output_spill_page and output_spill_base is not None:
            layout_chunks = (output_spill_gemm_chunks
                             if output_spill_gemm_chunks is not None
                             else output_gemm_chunks)
        else:
            layout_chunks = output_gemm_chunks
        output_tile_base, output_end = _route_group_kchunk_location(
            mapped_base, mapped_nt, 0, layout_chunks, rows=tokens)
    elif output_gemm_k is not None:
        chunk_base = mapped_base + (mapped_nt // (output_gemm_k // 32)) * (tokens * output_gemm_k * 2)
        output_end = chunk_base + tokens * output_gemm_k * 2
        output_tile_base = chunk_base
    else:
        output_tile_base = mapped_base + mapped_nt * output_tile_stride
        output_end = output_tile_base + tokens * 64
    activation_end = act_base + kt_total * tokens * 64
    # Only the matrix base must be 2-KiB aligned. Per-token row offsets in a
    # K-chunk-major output tile are smaller strides and need not be aligned.
    if (mapped_page not in (0, 1, 2) or mapped_base < 0 or
            output_base % 2048 or
            output_end > (GROUP_INTERMEDIATE_EXTRA_BYTES
                          if mapped_page == 2 else 0x10000) or
            (mapped_page == 0 and output_tile_base < activation_end and
             output_end > act_base)):
        raise ValueError("group output base must leave a 2-KiB tile for this N-tile")

    gemm_count = kt_total // (gemm_k // 32)
    if 16 <= tokens < 32 and k_chunk_order is None:
        # The loop instruction has room for an exact compact row stride only
        # through 15 rows. Keep larger non-32 groups explicitly addressed.
        k_chunk_order = list(range(gemm_count))
    p = []
    if select_group:
        p.append(isa.route_group(expert_id, tokens))
    if configure:
        # SET_CFG_M remains all-idle gated; SET_CFG_K waits only for its active
        # consumers. The M update already fences the preceding route setup.
        p.extend([isa.set_cfg_m(0), isa.set_cfg_k(gemm_k)])
    if gather_activations:
        p.append(isa.dma_rd_route_group(
            activation_byte_offset, l1_bank=act_base >> 12, proj=proj,
            source=activation_source))
    scale_load_pending = (preloaded_scale_nt is None and
                          not scale_metadata_n_invariant)
    if scale_load_pending:
        p.append(isa.nop_scale(nt, current_expert_only=True))
        # GLD.W is decoder-gated by scale_loader_busy, including when a
        # prefetched weight DMA has already completed before this scale load.
    if k_chunk_order is None:
        # Keep the same descriptor-selected mixed-precision behavior as the
        # compact loop above; INT1 is not a forced weight format for experts.
        p.extend([isa.hw_loop(gemm_count, 4),
                  # Prime each activation before starting its weight read.
                  # Subsequent loop iterations then hide GLD.A behind the
                  # current GEMM without changing K accumulation order.
                  isa.gld_a_batch_transpose(
                      act_base, padded_rows=True,
                      row_stride_rows=(tokens if tokens < 16 else None)),
                  isa.expert_dma(nt, 0, kt_total, 1, proj, weight_bank),
                  isa.gld_w(isa.Precision.INT1, weight_bank * 4096,
                            group=0, expert_sel=1, proj=proj),
                  isa.gemm(imm=gemm_k, prec=isa.Precision.INT1,
                           expert_sel=1, proj=proj)])
    else:
        order = list(k_chunk_order)
        if sorted(order) != list(range(gemm_count)):
            raise ValueError("K-chunk order must be a permutation of the projection chunks")
        if prefetched_first_dma and (proj != 0 or order[0] != 0):
            raise ValueError("prefetched first DMA is supported only for Gate chunk 0")
        if prefetched_first_weight and (
                not prefetched_first_dma or not scale_metadata_n_invariant):
            raise ValueError("prefetched Gate weights require invariant scale metadata")
        if prefetch_next_nt is not None and (
                proj != 1 or prefetch_next_nt != nt + 1 or
                prefetch_next_nt >= 64):
            raise ValueError("only Up may prefetch a following Gate N tile")
        if (resident_activation_chunk is not None and
                resident_activation_chunk not in range(gemm_count)):
            raise ValueError("resident activation chunk is outside the projection K range")
        if (resident_activation_chunk is not None and
                resident_activation_chunk != order[0]):
            raise ValueError(
                "resident activation must match the first scheduled K chunk")
        if weight_bank % 2 != 1:
            raise ValueError("explicit K-chunk DMA must start in an odd ping-pong bank")
        tiles_per_chunk = gemm_k // 32
        chunk_bytes = tokens * gemm_k * 2
        chunk_specs = []
        for order_idx, chunk_idx in enumerate(order):
            kt_start = chunk_idx * tiles_per_chunk
            dma_bank = weight_bank ^ (order_idx & 1)
            scale_group = (kt_start * 32) // 128
            chunk_specs.append((chunk_idx, kt_start, dma_bank, scale_group))

        # Stage the activation before starting its weight DMA. GLD.A is
        # conservatively serialized behind an active expert DMA, so starting
        # the DMA first would make activation gather wait for the transfer.
        for order_idx, chunk_spec in enumerate(chunk_specs):
            current_idx, current_kt_start, current_dma_bank, current_scale_group = \
                chunk_spec
            if current_idx != resident_activation_chunk:
                p.append(isa.gld_a_batch_transpose(
                    act_base + current_idx * chunk_bytes,
                    padded_rows=True, page=0))
            if not (order_idx == 0 and prefetched_first_dma):
                p.append(isa.expert_dma(
                    nt, current_kt_start, kt_total, 1, proj,
                    current_dma_bank))
            if not (order_idx == 0 and prefetched_first_weight):
                # expert_sel=1 makes the active expert/projection descriptor
                # authoritative for bit width; this instruction's INT1 value
                # is only used by the non-expert fallback path.
                p.append(isa.gld_w(
                    isa.Precision.INT1, current_dma_bank * 4096,
                    group=current_scale_group, expert_sel=1, proj=proj))
            # Keep cross-N prefetch behind the current activation dispatch;
            # it must not block the operand setup for this K chunk.
            next_chunk = (chunk_specs[order_idx + 1]
                          if order_idx + 1 < len(chunk_specs) else None)
            prefetch_next_tile = next_chunk is None and prefetch_next_nt is not None
            same_bank_without_activation_read = (
                prefetch_next_tile and
                weight_bank == current_dma_bank and
                current_idx == resident_activation_chunk)
            if prefetch_next_tile and not same_bank_without_activation_read:
                p.append(isa.expert_dma(
                    prefetch_next_nt, 0, kt_total, 1, 0, weight_bank))
            p.append(isa.gemm(
                acc=(order_idx != 0), imm=gemm_k,
                prec=isa.Precision.INT1, skip_store=(order_idx + 1 != len(order)),
                expert_sel=1, proj=proj))
            if same_bank_without_activation_read:
                # Without GLD.A there is no GLD resource dependency to ensure
                # the current L1 weight read has completed before this bank is
                # overwritten. Defer this rare same-bank boundary DMA until
                # GEMM acceptance proves the WRF staging is complete.
                p.append(isa.expert_dma(
                    prefetch_next_nt, 0, kt_total, 1, 0, weight_bank))
            resident_activation_chunk = current_idx
            if (next_chunk is None and prefetch_next_nt is not None and
                  scale_metadata_n_invariant):
                p.append(isa.gld_w(
                    isa.Precision.INT1, weight_bank * 4096,
                    group=0, expert_sel=1, proj=0))
    # AC2V is dependency-gated on the final GEMM, while any next-tile expert
    # DMA may continue in the background. Per-row PostProc operations serialize
    # through the shared resource scoreboard.
    for row in range(tokens):
        row_body = []
        if fuse_gate_base is not None:
            layout_k = output_gemm_k or gemm_k
            if mapped_page == output_spill2_page and output_spill2_base is not None:
                out_chunks = (output_spill2_gemm_chunks
                              if output_spill2_gemm_chunks is not None
                              else output_gemm_chunks)
            elif mapped_page == output_spill_page and output_spill_base is not None:
                out_chunks = (output_spill_gemm_chunks
                              if output_spill_gemm_chunks is not None
                              else output_gemm_chunks)
            else:
                out_chunks = output_gemm_chunks
            out_addr = (_route_group_kchunk_location(
                mapped_base, mapped_nt, row, out_chunks, rows=tokens)[0]
                if out_chunks is not None else
                _route_group_kchunk_addr(
                    mapped_base, mapped_nt, row, layout_k, rows=tokens))
            gate_base, gate_nt, gate_page = _route_group_output_location(
                fuse_gate_base, nt, output_page, output_spill_base,
                output_spill_start_tile, output_spill_page,
                output_spill2_base, output_spill2_start_tile,
                output_spill2_page)
            if gate_page == output_spill2_page and output_spill2_base is not None:
                gate_chunks = (output_spill2_gemm_chunks
                               if output_spill2_gemm_chunks is not None
                               else output_gemm_chunks)
            elif gate_page == output_spill_page and output_spill_base is not None:
                gate_chunks = (output_spill_gemm_chunks
                               if output_spill_gemm_chunks is not None
                               else output_gemm_chunks)
            else:
                gate_chunks = output_gemm_chunks
            gate_addr = (_route_group_kchunk_location(
                gate_base, gate_nt, row, gate_chunks, rows=tokens)[0]
                if gate_chunks is not None else
                _route_group_kchunk_addr(
                    gate_base, gate_nt, row, layout_k, rows=tokens))
            row_body.extend([
                isa.ac2v(0, row=row), isa.vld_page(1, gate_addr, gate_page),
                isa.silu(1, 1), isa.mul(0, 0, 1),
                isa.vst_page(0, out_addr, mapped_page),
            ])
        else:
            if mapped_page == output_spill2_page and output_spill2_base is not None:
                chunks = (output_spill2_gemm_chunks
                          if output_spill2_gemm_chunks is not None
                          else output_gemm_chunks)
            elif mapped_page == output_spill_page and output_spill_base is not None:
                chunks = (output_spill_gemm_chunks
                          if output_spill_gemm_chunks is not None
                          else output_gemm_chunks)
            else:
                chunks = output_gemm_chunks
            if chunks is not None:
                store_addr = _route_group_kchunk_location(
                    mapped_base, mapped_nt, row, chunks, rows=tokens)[0]
            elif output_gemm_k is not None:
                store_addr = _route_group_kchunk_addr(
                    mapped_base, mapped_nt, row, output_gemm_k, rows=tokens)
            else:
                store_addr = output_tile_base + row * 64
            row_body.extend([isa.ac2v(0, row=row),
                             isa.vst_page(0, store_addr, mapped_page)])
        p.extend([isa.group_skip_if_row_inactive(row, len(row_body)),
                  *row_body])
    if restore:
        p.extend([isa.set_cfg_k(32), isa.set_cfg_m(1)])
    return p


def _route_group_kchunk_addr(base, n_tile, row, gemm_k, rows=32):
    """Address one 32-value output tile in [K-chunk][row][K] order."""
    tiles_per_chunk = gemm_k // 32
    chunk_bytes = rows * gemm_k * 2
    return (base + (n_tile // tiles_per_chunk) * chunk_bytes +
            row * gemm_k * 2 + (n_tile % tiles_per_chunk) * 64)


def _route_group_kchunk_location(base, n_tile, row, gemm_chunks, rows=32):
    """Address an output tile in a mixed-size K-chunk-major tensor."""
    tile_cursor = 0
    byte_cursor = 0
    for chunk_k in gemm_chunks:
        if chunk_k not in (32, 64, 128, 256):
            raise ValueError("output K chunks must use legal GEMM sizes")
        tiles = chunk_k // 32
        if tile_cursor <= n_tile < tile_cursor + tiles:
            local_tile = n_tile - tile_cursor
            address = (base + byte_cursor + row * chunk_k * 2 +
                       local_tile * 64)
            end = base + byte_cursor + rows * chunk_k * 2
            return address, end
        tile_cursor += tiles
        byte_cursor += rows * chunk_k * 2
    raise ValueError("output N tile falls outside the configured K-chunk layout")


def _route_group_output_location(base, n_tile, page, spill_base,
                                 spill_start_tile, spill_page,
                                 spill2_base=None, spill2_start_tile=None,
                                 spill2_page=2):
    """Map a logical intermediate N tile into primary or either spill page."""
    if (spill2_base is not None and spill2_start_tile is not None and
            n_tile >= spill2_start_tile):
        return spill2_base, n_tile - spill2_start_tile, spill2_page
    if spill_base is not None and spill_start_tile is not None and n_tile >= spill_start_tile:
        return spill_base, n_tile - spill_start_tile, spill_page
    return base, n_tile, page


def emit_route_group_ffn_raw(expert_id, tokens, hidden_size, intermediate_size,
                             output_base=GROUP_OUTPUT_ACCUM,
                             act_base=GROUP_ACTIVATION, gemm_k=None,
                             dynamic_expert=False,
                             scale_metadata_n_invariant=False,
                             token_start=0, activation_source="act",
                             activation_byte_offset=0):
    """Emit grouped Gate/Up/SwiGLU/Down with active-M and weighted rows.

    Gate and Up run one N tile at a time. Gate is retained in the final
    intermediate buffer and replaced in-place by SiLU(Gate)*Up while draining
    Up rows, so no separate Up scratch tile is required. The buffer is
    K-chunk-major, with contiguous K values per route row, for Down's
    transposed GLD.A; no repack/copy is needed.
    The weight staging pair is alternated across explicit K-chunk commands;
    each next chunk can be fetched while the current GEMM runs. Weights and
    gathered activations stay in page 0; token-major output accumulation uses
    the lower part of page 1. Intermediate results fill the remaining page-1
    space first and spill whole K chunks into unused page-0 space. Down's
    explicit GLD.A chunk loads switch pages at the spill boundary.
    Down outputs are route-weighted, then added directly to the token-major
    high-page accumulator using the route-group token map. The caller must
    clear that accumulator once before the first expert group in the batch.
    With ``scale_metadata_n_invariant``, this body preloads scale/zero data
    once after the current route-group descriptor is selected; the caller must
    guarantee per-row scale metadata is identical across all output N tiles.
    """
    if (hidden_size < 32 or hidden_size > 2048 or hidden_size % 32 or
            intermediate_size < 32 or intermediate_size > 8160 or
            intermediate_size % 32):
        raise ValueError("grouped FFN dimensions must be 32-aligned in [32,8160]")
    if (not dynamic_expert and not 0 <= expert_id < 60) or not 1 <= tokens <= 32:
        raise ValueError("route group requires expert 0..59 and 1..32 cached tokens")
    if not 0 <= token_start < 32 or token_start + tokens > 32:
        raise ValueError("route-group token range must fit the 32-token cache")
    if activation_source not in ("act", "out"):
        raise ValueError("route-group source must be 'act' or 'out'")
    if activation_source == "out" and hidden_size != 2048:
        raise ValueError("OUT route-group input requires 2048-wide 4-KiB token rows")
    if (activation_byte_offset < 0 or activation_byte_offset % 16 or
            activation_byte_offset + 32 * 4096 > (1 << 20)):
        raise ValueError("route-group activation window exceeds the encoded DDR range")
    hidden_tiles = hidden_size // 32
    intermediate_tiles = intermediate_size // 32
    if gemm_k is None:
        gemm_k = select_expert_gemm_k(hidden_tiles)
    down_gemm_chunks = split_expert_k_tiles(intermediate_tiles, 256)
    down_gemm_k = max(down_gemm_chunks)
    if gemm_k not in (32, 64, 128, 256):
        raise ValueError("expert GEMM K must be 32, 64, 128, or 256")
    activation_end = act_base + tokens * hidden_size * 2
    intermediate_bytes = tokens * intermediate_size * 2
    # Keep the output accumulator at the bottom of page 1. Place intermediate
    # chunks immediately above it, then spill whole chunks into page 0 after
    # route-group activation staging and into the dedicated extension page.
    output_bytes = tokens * hidden_size * 2
    intermediate_base = output_base + output_bytes
    page1_intermediate_bytes = 0x10000 - intermediate_base
    page0_spill_capacity = 0x10000 - GROUP_INTERMEDIATE_SPILL_BASE
    page2_spill_capacity = GROUP_INTERMEDIATE_EXTRA_BYTES
    primary_output_chunks = []
    spill_output_chunks = []
    spill2_output_chunks = []
    chunk_byte_cursor = 0
    page0_spill_bytes = 0
    page2_spill_bytes = 0
    spill_start_tile = None
    spill2_start_tile = None
    chunk_tile_cursor = 0
    for chunk_k in down_gemm_chunks:
        chunk_bytes = tokens * chunk_k * 2
        if chunk_byte_cursor + chunk_bytes <= page1_intermediate_bytes:
            primary_output_chunks.append(chunk_k)
        elif (GROUP_INTERMEDIATE_SPILL_BASE >= activation_end and
              page0_spill_bytes + chunk_bytes <= page0_spill_capacity):
            if spill_start_tile is None:
                spill_start_tile = chunk_tile_cursor
            spill_output_chunks.append(chunk_k)
            page0_spill_bytes += chunk_bytes
        elif page2_spill_bytes + chunk_bytes <= page2_spill_capacity:
            if spill2_start_tile is None:
                spill2_start_tile = chunk_tile_cursor
            spill2_output_chunks.append(chunk_k)
            page2_spill_bytes += chunk_bytes
        else:
            raise ValueError(
                "grouped intermediate exceeds primary, page-0 spill, and extension-page capacity")
        chunk_byte_cursor += chunk_bytes
        chunk_tile_cursor += chunk_k // 32
    if chunk_byte_cursor != intermediate_bytes:
        raise ValueError("Down K chunks do not cover the intermediate tensor")
    spill_bytes = page0_spill_bytes
    intermediate_spill_base = (GROUP_INTERMEDIATE_SPILL_BASE
                                if spill_bytes else None)
    intermediate_spill2_base = 0 if page2_spill_bytes else None
    if output_base < 0 or output_base % 64:
        raise ValueError("group output accumulator base must be non-negative and 64-byte aligned")
    if not (GROUP_WEIGHT_BANK & 1):
        raise ValueError("group weight bank must be odd so XOR-1 selects its adjacent pair")
    group_weight_banks = (GROUP_WEIGHT_BANK, GROUP_WEIGHT_BANK ^ 1)
    weight_l1_base = min(group_weight_banks) * 4096
    weight_l1_end = (max(group_weight_banks) + 1) * 4096
    if (act_base < weight_l1_end or act_base % 0x1000 or
            activation_end > 0x10000 or
            intermediate_base < output_base + output_bytes or
            intermediate_base % 2048 or
            intermediate_base < weight_l1_end or
            output_base < 0 or output_base % 64 or
            output_base + output_bytes > 0x10000 or
            (spill_bytes and
             (intermediate_spill_base % 2048 or
              # Gate/Up outputs may spill before their full-K activation
              # gather is dead; never alias this live staging interval.
              intermediate_spill_base < activation_end or
              intermediate_spill_base + spill_bytes > 0x10000)) or
            page2_spill_bytes > GROUP_INTERMEDIATE_EXTRA_BYTES):
        raise ValueError("grouped FFN buffers exceed the primary, spill, or extension L1 regions")
    if hidden_tiles % (gemm_k // 32):
        raise ValueError("Gate/Up dimensions must divide evenly into GEMM K chunks")

    route_cmd = (isa.route_group_current_expert(tokens, token_start)
                 if dynamic_expert else
                 isa.route_group(expert_id, tokens, token_start))
    # The projection emitters use expert_id only for argument validation;
    # runtime descriptor selection follows the route-group command latch.
    expert_id = 0 if dynamic_expert else expert_id
    # ROUTE is all-idle gated. The first operation in the body is either
    # NOP_SCALE or SET_CFG_M, both of which provide the required completion
    # boundary before using the newly selected expert descriptor.
    program = [route_cmd]
    body = []
    if scale_metadata_n_invariant:
        # This must remain inside the per-expert body: route-group descriptor
        # selection maps slot 1 to the current expert, not topk_idx_arr[0].
        body.append(isa.nop_scale(0, current_expert_only=True))
    # Gate and Up reuse the complete gathered activation matrix throughout both
    # projections; its live interval constrains the page-0 intermediate spill.
    # SET_CFG_M and grouped gather are all-idle gated, so barriers between
    # them only repeat the same global wait. SET_CFG_K has targeted hazards.
    body.extend([isa.set_cfg_m(0), isa.set_cfg_k(gemm_k),
                 isa.dma_rd_route_group(
                     activation_byte_offset, l1_bank=act_base >> 12, proj=0,
                     source=activation_source)])
    activation_resident_chunk = None
    gate_chunk_order = list(range(hidden_tiles // (gemm_k // 32)))
    up_chunk_order = list(reversed(gate_chunk_order))
    # Keep the active ARF endpoint resident across Gate/Up and adjacent N
    # tiles. Up's reverse traversal ends at the next Gate tile's first K block,
    # allowing that Gate weight block to be prefetched during Up's last GEMM.
    for nt in range(intermediate_tiles):
        body.extend(emit_route_group_projection(
            expert_id, tokens, nt, hidden_tiles, 0, intermediate_base,
            act_base=act_base, gemm_k=gemm_k, select_group=False,
            gather_activations=False, configure=False, restore=False,
            weight_bank=GROUP_WEIGHT_BANK, output_gemm_k=down_gemm_k,
            output_page=1, output_spill_base=intermediate_spill_base,
            output_spill_start_tile=spill_start_tile if spill_bytes else None,
            output_spill_page=0, k_chunk_order=None,
            output_spill2_base=intermediate_spill2_base,
            output_spill2_start_tile=spill2_start_tile,
            output_spill2_page=2,
            output_gemm_chunks=primary_output_chunks,
            output_spill_gemm_chunks=spill_output_chunks,
            output_spill2_gemm_chunks=spill2_output_chunks,
            scale_metadata_n_invariant=scale_metadata_n_invariant))
        activation_resident_chunk = gate_chunk_order[-1]
        body.extend(emit_route_group_projection(
            expert_id, tokens, nt, hidden_tiles, 1, intermediate_base,
            act_base=act_base, gemm_k=gemm_k,
            select_group=False, gather_activations=False,
            configure=False, restore=False, weight_bank=GROUP_WEIGHT_BANK,
            output_gemm_k=down_gemm_k, fuse_gate_base=intermediate_base,
            output_page=1, output_spill_base=intermediate_spill_base,
            output_spill_start_tile=spill_start_tile if spill_bytes else None,
            output_spill_page=0, k_chunk_order=up_chunk_order,
            output_spill2_base=intermediate_spill2_base,
            output_spill2_start_tile=spill2_start_tile,
            output_spill2_page=2,
            output_gemm_chunks=primary_output_chunks,
            output_spill_gemm_chunks=spill_output_chunks,
            output_spill2_gemm_chunks=spill2_output_chunks,
            resident_activation_chunk=activation_resident_chunk,
            preloaded_scale_nt=(None if scale_metadata_n_invariant else nt),
            scale_metadata_n_invariant=scale_metadata_n_invariant,
            prefetch_next_nt=None))
        activation_resident_chunk = up_chunk_order[-1]

    # K-chunk-major storage follows Down's own K block, which may differ from
    # Gate/Up while retaining contiguous activation rows and safe page spills.
    down_gemm_chunks = split_expert_k_tiles(intermediate_tiles, 256)
    down_chunk_tile_starts = []
    down_chunk_byte_starts = []
    down_k_tile_cursor = 0
    down_byte_cursor = 0
    for chunk_k in down_gemm_chunks:
        down_chunk_tile_starts.append(down_k_tile_cursor)
        down_chunk_byte_starts.append(down_byte_cursor)
        down_k_tile_cursor += chunk_k // 32
        down_byte_cursor += tokens * chunk_k * 2
    gemm_count = len(down_gemm_chunks)
    activation_resident_chunk = None
    prefetched_down_chunk = None
    prefetched_down_weight_staged = False
    active_gld_k = gemm_k
    body.extend([isa.set_cfg_m(0), isa.set_cfg_k(down_gemm_k)])
    active_gld_k = down_gemm_k
    for nt in range(hidden_tiles):
        down_order = (list(range(gemm_count)) if nt % 2 == 0
                      else list(reversed(range(gemm_count))))
        if (activation_resident_chunk is not None and
                activation_resident_chunk != down_order[0]):
            raise ValueError(
                "resident Down activation must match the first scheduled K chunk")
        consume_prefetched_down_chunk = (
            prefetched_down_chunk == (nt, down_order[0]))
        consume_prefetched_down_weight = (
            consume_prefetched_down_chunk and prefetched_down_weight_staged)
        if (prefetched_down_chunk is not None and
                not consume_prefetched_down_chunk):
            raise ValueError("prefetched Down weight does not match the next N tile")
        if not scale_metadata_n_invariant:
            # NOP_SCALE covers the complete descriptor K range, so request
            # the widest scale-load count before issuing mixed-width chunks.
            if active_gld_k != down_gemm_k:
                body.append(isa.set_cfg_k(down_gemm_k))
                active_gld_k = down_gemm_k
            body.append(isa.nop_scale(nt, current_expert_only=True))
        down_chunk_specs = []
        # Keep the first chunk of each N tile in the bank that was filled by
        # the previous tile's cross-N prefetch. When the number of K chunks is
        # odd, the final chunk occupies the opposite bank from the fixed
        # origin, so the next tile must rotate its origin to avoid overwriting
        # a bank still being consumed by the final GEMM.
        for order_idx, chunk_idx in enumerate(down_order):
            chunk_k = down_gemm_chunks[chunk_idx]
            kt_start = down_chunk_tile_starts[chunk_idx]
            dma_bank = group_down_weight_bank(nt, order_idx, gemm_count)
            scale_group = (kt_start * 32) // 128
            act_base, act_nt, act_page = _route_group_output_location(
                intermediate_base, kt_start, 1, intermediate_spill_base,
                spill_start_tile if spill_bytes else None, 0,
                intermediate_spill2_base, spill2_start_tile, 2)
            if act_page == 2 and intermediate_spill2_base is not None:
                act_chunks = spill2_output_chunks
            elif act_page == 0 and intermediate_spill_base is not None:
                act_chunks = spill_output_chunks
            else:
                act_chunks = primary_output_chunks
            act_addr = _route_group_kchunk_location(
                act_base, act_nt, 0, act_chunks, rows=tokens)[0]
            down_chunk_specs.append((chunk_idx, chunk_k, kt_start, dma_bank,
                                     scale_group, act_addr, act_page))

        (first_idx, first_k, first_kt_start, first_dma_bank, first_scale_group,
         first_act_addr, first_act_page) = down_chunk_specs[0]
        if active_gld_k != first_k:
            body.append(isa.set_cfg_k(first_k))
            active_gld_k = first_k
        # Load the activation before launching its weight read. GLD.A stalls
        # behind DMA_RD, so issue activation first to let both transfers
        # overlap once the expert DMA is accepted.
        if first_idx != activation_resident_chunk:
            body.append(isa.gld_a_batch_transpose(
                first_act_addr, padded_rows=True, page=first_act_page))
        activation_resident_chunk = first_idx
        if not consume_prefetched_down_weight:
            if consume_prefetched_down_chunk:
                body.append(isa.gld_w(
                    isa.Precision.INT1, first_dma_bank * 4096,
                    group=first_scale_group, expert_sel=1, proj=2))
            else:
                body.append(isa.expert_dma(nt, first_kt_start,
                                           intermediate_tiles, 1, 2,
                                           first_dma_bank))
                body.append(isa.gld_w(isa.Precision.INT1,
                                      first_dma_bank * 4096,
                                      group=first_scale_group,
                                      expert_sel=1, proj=2))
        prefetched_down_chunk = None
        prefetched_down_weight_staged = False

        for order_idx, (chunk_idx, chunk_k, _, _, _, _, _) in enumerate(down_chunk_specs):
            next_chunk = (down_chunk_specs[order_idx + 1]
                          if order_idx + 1 < len(down_chunk_specs) else None)
            body.append(isa.gemm(
                acc=(order_idx != 0), imm=chunk_k,
                prec=isa.Precision.INT1,
                skip_store=(order_idx + 1 != len(down_order)),
                expert_sel=1, proj=2))
            if next_chunk is not None:
                (next_idx, next_k, next_kt_start, next_dma_bank, next_scale_group,
                 next_act_addr, next_act_page) = next_chunk
                if next_k != chunk_k:
                    body.append(isa.set_cfg_k(next_k))
                    active_gld_k = next_k
                if next_idx != activation_resident_chunk:
                    body.append(isa.gld_a_batch_transpose(
                        next_act_addr, padded_rows=True, page=next_act_page))
                body.append(isa.expert_dma(
                    nt, next_kt_start, intermediate_tiles, 1, 2,
                    next_dma_bank))
                body.append(isa.gld_w(
                    isa.Precision.INT1, next_dma_bank * 4096,
                    group=next_scale_group, expert_sel=1, proj=2))
                activation_resident_chunk = next_idx
            else:
                activation_resident_chunk = chunk_idx
                if nt + 1 < hidden_tiles:
                    # Adjacent N tiles traverse K in opposite directions, so
                    # the current resident endpoint is the next tile's first
                    # chunk. Start this read after GEMM acceptance so it
                    # overlaps compute without blocking current GLD.A.
                    next_first_chunk = (gemm_count - 1
                                        if (nt + 1) % 2 else 0)
                    next_first_kt_start = down_chunk_tile_starts[next_first_chunk]
                    next_first_k = down_gemm_chunks[next_first_chunk]
                    next_first_dma_bank = group_down_weight_bank(
                        nt + 1, 0, gemm_count)
                    if active_gld_k != next_first_k:
                        raise ValueError(
                            "cross-N Down prefetch K must match the active CFG_K")
                    body.append(isa.expert_dma(
                        nt + 1, next_first_kt_start, intermediate_tiles,
                        1, 2, next_first_dma_bank))
                    prefetched_down_chunk = (nt + 1, next_first_chunk)
                    if scale_metadata_n_invariant:
                        next_first_scale_group = (next_first_kt_start * 32) // 128
                        body.append(isa.gld_w(
                            isa.Precision.INT1, next_first_dma_bank * 4096,
                            group=next_first_scale_group,
                            expert_sel=1, proj=2))
                        prefetched_down_weight_staged = True
        # The row drains wait for the final GEMM; the cross-N weight DMA can
        # remain in flight while AC2V and output accumulation use PostProc.
        for row in range(tokens):
            row_body = [
                isa.ac2v(0, row=row), isa.group_weight(2, row),
                isa.mul(0, 0, 2),
                isa.vld_group_token(1, output_base + nt * 64,
                                    hidden_size * 2),
                isa.add(0, 0, 1),
                isa.vst_group_token(0, output_base + nt * 64,
                                    hidden_size * 2),
            ]
            body.extend([isa.group_skip_if_row_inactive(row, len(row_body)),
                         *row_body])
    if len(body) > 0xFFFF:
        raise ValueError("grouped FFN body exceeds conditional-skip encoding")
    program.extend([isa.group_skip_if_empty(len(body)), *body])
    program.extend([isa.set_cfg_k(32), isa.set_cfg_m(1)])
    return program


def emit_all_experts_grouped(tokens=32, hidden_size=512,
                             intermediate_size=512, num_experts=60,
                             cache_routes=True,
                             scale_metadata_n_invariant=False,
                             output_window=0, token_start=0,
                             activation_source="act",
                             activation_byte_offset=0):
    """Run all routed experts with one grouped body and hardware expert loop.

    When cache_routes is true, Top-4 results are generated and cached before
    the accumulator is cleared. Each loop iteration gathers the active tokens
    for one expert, computes the grouped FFN, and scatters weighted rows into
    the original token positions.
    When scale metadata is N-invariant, it is loaded once inside each
    non-empty expert body after that expert's descriptor becomes active, then
    retained while that expert processes every output tile.
    """
    if num_experts != 60:
        raise ValueError("grouped routing must scan all 60 experts to preserve Top-4 coverage")
    if not 0 <= token_start < 32 or token_start + tokens > 32:
        raise ValueError("route-group token range must fit the 32-token cache")
    output_bytes = tokens * hidden_size * 2
    body = emit_route_group_ffn_raw(
        expert_id=0, tokens=tokens, hidden_size=hidden_size,
        intermediate_size=intermediate_size, dynamic_expert=True,
        scale_metadata_n_invariant=scale_metadata_n_invariant,
        token_start=token_start, activation_source=activation_source,
        activation_byte_offset=activation_byte_offset)
    program = [*emit_router_cache(tokens)] if cache_routes else []
    program.extend([
        isa.l1_clear_group_accumulator(output_bytes),
        # EXPERT_LOOP_BEGIN is all_idle-gated and also waits for L1 clear.
        isa.expert_loop_begin(num_experts), *body, isa.expert_loop_end(),
        # EXPERT_LOOP_END is all_idle-gated before the output DMA.
        *isa.dma_wr_group_tokens(GROUP_OUTPUT_ACCUM, hidden_size, tokens,
                                 output_window=output_window),
    ])
    return program


def emit_all_experts_grouped_batches(tokens=32, rows_per_group=None,
                                    hidden_size=2048,
                                    intermediate_size=1408,
                                    num_experts=60, cache_routes=True,
                                    scale_metadata_n_invariant=False,
                                    activation_source="act",
                                    activation_byte_offset=0):
    """Run cached tokens in compact groups and drain each group to its own window.

    Each group retains the descriptor-driven expert loop: the active expert's
    per-projection precision and scale/zero metadata continue to control weight
    loading, hardware dequantization, and GEMM execution. If no group size is
    specified, choose the largest size whose activation and spill layout fits
    L1 for every group in the requested token range.
    """
    if not 1 <= tokens <= 32:
        raise ValueError("grouped batches require 1..32 tokens")
    if activation_source not in ("act", "out"):
        raise ValueError("route-group source must be 'act' or 'out'")
    if activation_source == "out" and hidden_size != 2048:
        raise ValueError("OUT route-group input requires 2048-wide 4-KiB token rows")
    if rows_per_group is None:
        min_rows = max(1, (tokens + 15) // 16)
        rows_per_group = next((candidate for candidate in
                               range(min(tokens, 32), min_rows - 1, -1)
                               if all(route_group_l1_fits(
                                   min(candidate, tokens - start), hidden_size,
                                   intermediate_size)
                                   for start in range(0, tokens, candidate))),
                              None)
        if rows_per_group is None:
            raise ValueError("no route-group size fits the L1 layout and output-window limit")
    if not 1 <= rows_per_group <= 32:
        raise ValueError("grouped batches require 1..32 rows per group")
    if (tokens + rows_per_group - 1) // rows_per_group > 16:
        raise ValueError("grouped batch output exceeds the 16-window DMA selector")
    if not all(route_group_l1_fits(
            min(rows_per_group, tokens - start), hidden_size,
            intermediate_size)
            for start in range(0, tokens, rows_per_group)):
        raise ValueError("requested route-group rows do not fit the L1 layout")
    program = emit_router_cache(tokens) if cache_routes else []
    for token_start in range(0, tokens, rows_per_group):
        group_tokens = min(rows_per_group, tokens - token_start)
        program.extend(emit_all_experts_grouped(
            tokens=group_tokens, hidden_size=hidden_size,
            intermediate_size=intermediate_size, num_experts=num_experts,
            cache_routes=False,
            scale_metadata_n_invariant=scale_metadata_n_invariant,
            output_window=token_start // rows_per_group,
            token_start=token_start,
            activation_source=activation_source,
            activation_byte_offset=activation_byte_offset))
    return program


def emit_all_tokens(tokens=32, hidden_size=2048, intermediate_size=1408,
                    output_base=OUTPUT):
    """Emit one body and repeat it in hardware with the current token index."""
    # Bank 4 maps to byte address 0x4000. Token-strided DDR addresses remain
    # outside the large instruction body, so one body serves all 32 tokens.
    # SET_CFG_M/ROUTE in the body serialize behind this transfer via all_idle.
    body = [isa.dma_rd_token(0, hidden_size, l1_bank=4)]
    body.extend(emit_token(0, hidden_size, intermediate_size,
                           current_token=True, output_base=output_base))
    # DMA_WR reads the L1 output region through a separate resource from VST;
    # keep this barrier so it cannot capture the previous tile before the
    # final PostProc store has committed. TOKEN_END then waits for the DMA.
    body.extend([isa.barrier(), isa.dma_wr_token(output_base, hidden_size)])
    return [isa.token_loop_begin(tokens), *body, isa.token_loop_end()]
