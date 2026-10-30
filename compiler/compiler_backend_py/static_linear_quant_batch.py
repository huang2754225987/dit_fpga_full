"""Batched static quantized projections with token-strided activations."""
try:
 from . import isa,static_linear_quant_ops as ops
 from .static_linear_fp16_batch_ops import drain
except ImportError:
 import isa,static_linear_quant_ops as ops
 from static_linear_fp16_batch_ops import drain


def emit_quantized_batch(region, activation_base, output_base, tokens,
                         activation_page=0, output_page=0, bias_base=None,
                         bias_page=2, activation_row_stride_bytes=4096,
                         configure_cfg=True, restore_cfg=True,
                         wait_after_store=True, silu_base=None,
                         silu_page=None, configure_cfg_m=None,
                         restore_cfg_m=None, configure_cfg_k=None,
                         restore_cfg_k=None,
                         wait_postproc_before_gld_a=False,
                         allow_silu_output_alias=False):
 """Emit one quantized linear projection over several token rows.

 The static weights/scales are fetched once per output/K group and reused
 across all rows. Activations remain token-major; GLD.A performs the
 row-strided transpose directly into the ARF.
 """
 if region.nbits not in (1, 2, 3, 4) or region.scale_zero is None:
  raise ValueError("batched quantized projection requires 1..4-bit weights")
 if region.groups != (region.k_tiles + 3) // 4:
  raise ValueError("quantized scale groups must cover each four-tile K block")
 if not 1 <= tokens <= 32:
  raise ValueError("batched quantized projection supports 1..32 rows")
 if activation_page not in (0, 1) or output_page not in (0, 1, 2):
  raise ValueError("batched quantized projection page is unsupported")
 if (activation_row_stride_bytes < region.cols * 2 or
     activation_row_stride_bytes % 32):
  raise ValueError("activation row stride must cover and align the input row")
 activation_end = (activation_base + (tokens - 1) *
                   activation_row_stride_bytes + region.cols * 2)
 activation_limit = 16384 if activation_page == 2 else 65536
 output_end = output_base + tokens * region.rows * 2
 output_limit = 16384 if output_page == 2 else 65536
 if (activation_base < 0 or activation_base % 32 or
     activation_end > activation_limit or output_base < 0 or
     output_base % 64 or output_end > output_limit):
  raise ValueError("batched quantized projection exceeds its L1 page")
 if activation_page == output_page and activation_base < output_end and output_base < activation_end:
  raise ValueError("batched quantized projection activation/output aliases")
 if bias_base is not None:
  bias_limit = 16384 if bias_page == 2 else 65536
  if (bias_page not in (0, 1, 2) or bias_base < 0 or bias_base % 64 or
      bias_base + region.rows * 2 > bias_limit):
   raise ValueError("batched quantized bias exceeds its L1 page")
 if silu_base is not None:
  if silu_page is None:
   silu_page=output_page
  silu_limit=16384 if silu_page==2 else 65536
  silu_end=silu_base+tokens*region.rows*2
  if (silu_page not in (0,1,2) or silu_base<0 or silu_base%64 or
      silu_end>silu_limit):
   raise ValueError("batched quantized SiLU source exceeds its L1 page")
  silu_overlap=(silu_page==output_page and silu_base<output_end and
                output_base<silu_end)
  exact_silu_alias=(silu_page==output_page and silu_base==output_base and
                    silu_end==output_end)
  if silu_overlap and not (allow_silu_output_alias and exact_silu_alias):
   raise ValueError("SiLU source aliases batched quantized output")
 if not configure_cfg and restore_cfg:
  raise ValueError("CFG restore requires the batch emitter to configure CFG")
 wide_group_bytes = ops.k128_weight_group_bytes(region)
 wide_gemm = (wide_group_bytes != 0 and region.k_tiles % 4 == 0 and
              region.weight % wide_group_bytes == 0)
 if configure_cfg_m is None:
  configure_cfg_m=configure_cfg
 if restore_cfg_m is None:
  restore_cfg_m=restore_cfg
 if configure_cfg_k is None:
  configure_cfg_k=configure_cfg or wide_gemm
 if restore_cfg_k is None:
  restore_cfg_k=restore_cfg or (wide_gemm and not configure_cfg)
 if not configure_cfg_m and restore_cfg_m:
  raise ValueError("CFG_M restore requires the batch emitter to configure CFG_M")
 if not configure_cfg_k and restore_cfg_k:
  raise ValueError("CFG_K restore requires the batch emitter to configure CFG_K")

 program = []
 if configure_cfg_m:
  program.append(isa.set_cfg_m(tokens))
 if configure_cfg_k:
  program.append(isa.set_cfg_k(128 if wide_gemm else 32))
 prefetched_scale_nt = None
 prefetched_weight_nt = None
 prefetched_weight_page = None
 for n_tile in range(region.n_tiles):
  weight_page = None
  prefetched_weight_group = None
  if prefetched_weight_nt == n_tile:
   weight_page = prefetched_weight_page
   prefetched_weight_group = 0
   prefetched_weight_nt = None
   prefetched_weight_page = None
  loaded_scale_chunk = None
  group = 0
  while group < region.groups:
   scale_chunk = (group // 16) * 16
   if loaded_scale_chunk != scale_chunk:
    if scale_chunk == 0 and prefetched_scale_nt == n_tile:
     prefetched_scale_nt = None
    else:
     scale_addr = region.scale_zero + (
         (n_tile * region.groups + scale_chunk) << 7)
     scale_count = min(16, region.groups - scale_chunk)
     program.extend(isa.static_scale_at(
         scale_addr, group_count=scale_count))
    loaded_scale_chunk = scale_chunk
   if (wide_gemm and region.groups - group > 1 and
       not (group == 0 and prefetched_weight_group == 0)):
    group_addr = region.weight + (
        n_tile * region.k_tiles + group * 4) * region.tile_bytes
    groups_to_page = (0x10000 - (group_addr & 0xFFFF)) // wide_group_bytes
    loop_count = min(region.groups - group,
                     16 - (group - scale_chunk), groups_to_page)
    if loop_count > 1:
     group_code, weight_page = ops.static_wide_k_loop(
         region, activation_base, n_tile, group, loop_count, tokens,
         activation_row_stride_bytes, activation_page,
         scale_group=group - scale_chunk, weight_page=weight_page,
         acc=group != 0, skip_store=group + loop_count != region.groups,
         wait_postproc_before_gld_a=(
             wait_postproc_before_gld_a and n_tile == 0 and group == 0))
     program.extend(group_code)
     group += loop_count
     prefetched_weight_group = None
     continue
   weight_addr = region.weight + (
       n_tile * region.k_tiles + group * 4) * region.tile_bytes
   weight_bank = 2 ^ (group & 1)
   current_k_tiles = min(4, region.k_tiles - group * 4)
   if current_k_tiles == 4:
    prefetch_next = None
    if (group + 1 < region.groups and
        not (wide_gemm and prefetched_weight_group == 0)):
     next_addr = region.weight + (
         n_tile * region.k_tiles + (group + 1) * 4) * region.tile_bytes
     next_page = next_addr >> 16
     next_scale_chunk = ((group + 1) // 16) * 16
     next_k_tiles = min(4, region.k_tiles - (group + 1) * 4)
     if (next_k_tiles == 4 and next_page == weight_addr >> 16 and
         next_scale_chunk == scale_chunk):
      next_bytes = 4 * region.tile_bytes
      prefetch_next = (
          next_addr, 2 ^ ((group + 1) & 1), next_bytes, next_page)
    group_code, weight_page = ops.batch_group_burst(
        region, activation_base, n_tile, group, tokens,
        activation_row_stride_bytes, activation_page,
        scale_group=group - scale_chunk,
        weight_bank=weight_bank, weight_page=weight_page,
        prefetch_next=prefetch_next,
        current_prefetched=(prefetched_weight_group == group),
        wide_gemm=wide_gemm,
        wait_postproc_before_gld_a=(
            wait_postproc_before_gld_a and n_tile == 0 and group == 0))
    prefetched_weight_group = (
        group + 1 if prefetch_next is not None else None)
   else:
    # A partial final scale group keeps the exact-length transfer; do not
    # overread into the next matrix merely to preserve burst alignment.
    group_code, weight_page = ops.group(
        region, activation_base, n_tile, group,
        acc=group != 0, skip_store=group + 1 != region.groups,
        scale_group=group - scale_chunk,
        preload_scales=True, weight_page=weight_page,
        barrier_after=False, batch_rows=tokens,
        activation_row_stride_bytes=activation_row_stride_bytes,
        activation_page=activation_page,
        wait_postproc_before_gld_a=(
            wait_postproc_before_gld_a and n_tile == 0 and group == 0))
    prefetched_weight_group = None
   program.extend(group_code)
   group += 1
  if n_tile + 1 < region.n_tiles:
   # The RTL keeps an independent scale snapshot for the active GEMM. Fetch
   # the next output tile's scale/zero groups before draining this tile so the
   # metadata transfer overlaps GEMM/AC2V without changing K accumulation.
   next_scale_addr = region.scale_zero + ((n_tile + 1) * region.groups << 7)
   program.extend(isa.static_scale_at(
       next_scale_addr, group_count=min(16, region.groups)))
   prefetched_scale_nt = n_tile + 1
   if region.k_tiles >= 4:
    prefetch_code, prefetched_weight_page = ops.prefetch_next_n_tile_group(
        region, activation_base, n_tile + 1, tokens,
        activation_row_stride_bytes, activation_page,
        active_weight_page=weight_page)
    program.extend(prefetch_code)
    prefetched_weight_nt = n_tile + 1
  program.extend(drain(
      output_base, tokens, region.rows, n_tile, output_page,
      bias_base=bias_base, bias_page=bias_page,
      silu_base=silu_base, silu_page=silu_page,
      wait_after_store=(wait_after_store and n_tile == region.n_tiles - 1)))
 if restore_cfg_m:
  program.append(isa.set_cfg_m(1))
 if restore_cfg_k:
  program.append(isa.set_cfg_k(32))
 return program
