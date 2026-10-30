"""Compact aligned FP16 K=64 traversals with the hardware loop unit."""
try:
 from . import isa
except ImportError:
 import isa


def emit(region, activation_base, rows, n_tile, activation_page,
         previous_weight_page, batch=False, k_tiles_per_gemm=1,
         wait_postproc_before_gld_a=False,
         activation_row_stride_bytes=None, prefetched_first=False):
 """Return a loop-compressed K traversal, or None when alignment is unsafe."""
 if not supports_prefetched_first(region, n_tile, k_tiles_per_gemm):
  return None

 weight_base = region.weight + n_tile * region.k_tiles * region.tile_bytes
 first_k_tile = 2 if prefetched_first else 0

 # A loop's 4-KiB static-DMA stride is represented in the 16-bit immediate.
 # Split at each 64-KiB weight-page boundary before that immediate can wrap.
 plans = []
 addr = weight_base + first_k_tile * region.tile_bytes
 k_tile = first_k_tile
 remaining = (region.k_tiles - first_k_tile) // k_tiles_per_gemm
 while remaining:
  page = addr >> 16
  page_offset = addr & 0xFFFF
  count = min(remaining, (0x10000 - page_offset) // 4096)
  if count < (1 if prefetched_first else 2):
   return None
  plans.append((addr, page, k_tile, count))
  addr += count * 4096
  k_tile += count * k_tiles_per_gemm
  remaining -= count

 program = []
 weight_page = previous_weight_page
 if prefetched_first:
  activation_addr = activation_base
  if batch:
   if activation_row_stride_bytes is None:
    activation_row_stride_bytes=region.cols*2
   activation_tile_bytes=32*k_tiles_per_gemm*2
   gld_a=(isa.gld_a_batch_transpose(
              activation_addr,page=activation_page,
              wait_postproc=wait_postproc_before_gld_a)
          if activation_row_stride_bytes==activation_tile_bytes else
          isa.gld_a_batch_transpose_strided(
              activation_addr,activation_row_stride_bytes,page=activation_page,
              wait_postproc=wait_postproc_before_gld_a))
  else:
   gld_a=isa.gld_a(
       activation_addr,wait_postproc=wait_postproc_before_gld_a,
       page=activation_page)
  program.extend([
      isa.gld_w(isa.Precision.FP16,2*4096),gld_a,
      isa.gemm(acc=False,skip_store=True,imm=32,
               prec=isa.Precision.FP16)])
 for plan_idx, (addr, page, first_k_tile, count) in enumerate(plans):
  if page != weight_page:
   program.append(isa.set_weight_page(page))
   weight_page = page
  activation_addr = activation_base + first_k_tile * (
      64 if batch else rows * 64)
  if batch:
   if activation_row_stride_bytes is None:
    activation_row_stride_bytes=region.cols*2
   activation_tile_bytes=32*k_tiles_per_gemm*2
   wait_postproc=(wait_postproc_before_gld_a and plan_idx == 0)
   gld_a=(isa.gld_a_batch_transpose(
              activation_addr,page=activation_page,
              wait_postproc=wait_postproc)
          if activation_row_stride_bytes==activation_tile_bytes else
          isa.gld_a_batch_transpose_strided(
              activation_addr,activation_row_stride_bytes,page=activation_page,
              wait_postproc=wait_postproc))
  else:
   gld_a = isa.gld_a(
       activation_addr,
       wait_postproc=(wait_postproc_before_gld_a and plan_idx == 0),
       page=activation_page)
  if prefetched_first and count == 1:
   program.extend([
       isa.static_dma(addr,l1_bank=2,fp16=True)[1],
       isa.gld_w(isa.Precision.FP16,2*4096),gld_a,
       isa.gemm(acc=True,skip_store=(plan_idx+1<len(plans)),imm=32,
                prec=isa.Precision.FP16)])
   continue
  # Each page segment starts a new hardware-loop command. Preserve the
  # systolic accumulator across those command boundaries; only the first
  # segment of a non-prefetched output tile may clear it.
  accumulate_segment=prefetched_first or plan_idx>0
  managed_gemm=isa.gemm(
      acc=accumulate_segment,
      skip_store=(accumulate_segment and plan_idx+1<len(plans)),
      imm=32,prec=isa.Precision.FP16,loop_managed=True)
  program.extend([
      isa.hw_loop(count, body_len=4, stride_code=0),
      isa.static_dma(addr, l1_bank=2, fp16=True)[1],
      isa.gld_w(isa.Precision.FP16, 2 * 4096),
      gld_a,
      managed_gemm,
  ])
 return program, weight_page


def supports_prefetched_first(region,n_tile,k_tiles_per_gemm):
 """Whether an aligned K=64 loop can consume its first group from L1."""
 if (k_tiles_per_gemm != 2 or region.tile_bytes != 2048 or
     region.k_tiles < 4 or region.k_tiles % 2):
  return False
 weight_base=region.weight+n_tile*region.k_tiles*region.tile_bytes
 return weight_base % 4096 == 0
