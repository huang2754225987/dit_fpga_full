"""Batched FP16 projection on compact K-tile activation tiles."""
try:
 from . import (isa,static_linear_fp16_batch_tile as tile,
                static_linear_fp16_kloop as kloop)
 from .static_linear_fp16_batch_validate import validate,tiles_per_gemm
except ImportError:
 import isa,static_linear_fp16_batch_tile as tile
 import static_linear_fp16_kloop as kloop
 from static_linear_fp16_batch_validate import validate,tiles_per_gemm
def emit_fp16_batch(region,activation_base,output_base,tokens,activation_page=0,
                    output_page=0,bias_base=None,bias_page=2,
                    configure_cfg=True,restore_cfg=True,
                    wait_after_store=True,silu_base=None,silu_page=None,
                    activation_row_stride_bytes=None,
                    output_row_stride_bytes=None,acc_only=False,
                    configure_cfg_m=None,restore_cfg_m=None,
                    configure_cfg_k=None,restore_cfg_k=None,
                    wait_postproc_before_gld_a=False,
                    allow_silu_output_alias=False):
 """Emit a batched projection, optionally leaving results in the ACC banks.

 ACC-only mode is for a following hardware consumer such as Top-K; it skips
 vector conversion and L1 writeback, which do not modify the accumulator.
 """
 if activation_row_stride_bytes is None:
  activation_row_stride_bytes=region.cols*2
 validate(region,activation_base,output_base,tokens,activation_page,output_page,
          activation_row_stride_bytes,output_row_stride_bytes)
 if not configure_cfg and restore_cfg:
  raise ValueError("CFG restore requires the batch emitter to configure CFG")
 if configure_cfg_m is None:
  configure_cfg_m=configure_cfg
 if restore_cfg_m is None:
  restore_cfg_m=restore_cfg
 if configure_cfg_k is None:
  configure_cfg_k=configure_cfg
 if restore_cfg_k is None:
  restore_cfg_k=restore_cfg
 if not configure_cfg_m and restore_cfg_m:
  raise ValueError("CFG_M restore requires the batch emitter to configure CFG_M")
 if not configure_cfg_k and restore_cfg_k:
  raise ValueError("CFG_K restore requires the batch emitter to configure CFG_K")
 if bias_base is not None:
  bias_limit=16384 if bias_page==2 else 65536
  if (bias_page not in (0,1,2) or bias_base<0 or bias_base%64 or
      bias_base+region.rows*2>bias_limit):
   raise ValueError("invalid FP16 bias vector location")
 if silu_base is not None:
  if silu_page is None:
   silu_page=output_page
  silu_limit=16384 if silu_page==2 else 65536
  silu_end=silu_base+tokens*region.rows*2
  output_end=output_base+tokens*region.rows*2
  if (silu_page not in (0,1,2) or silu_base<0 or silu_base%64 or
      silu_end>silu_limit):
   raise ValueError("invalid FP16 SiLU source location")
  silu_overlap=(silu_page==output_page and silu_base<output_end and
                output_base<silu_end)
  exact_silu_alias=(silu_page==output_page and silu_base==output_base and
                    silu_end==output_end)
  if silu_overlap and not (allow_silu_output_alias and exact_silu_alias):
   raise ValueError("SiLU source aliases batched projection output")
 if acc_only and (bias_base is not None or silu_base is not None):
  raise ValueError("ACC-only output cannot apply bias or SiLU")
 k_tiles_per_gemm=tiles_per_gemm(region)
 p=[]
 if configure_cfg_m:
  p.append(isa.set_cfg_m(tokens))
 if configure_cfg_k:
  p.append(isa.set_cfg_k(32*k_tiles_per_gemm))
 page=None
 prefetched_first=False
 for n in range(region.n_tiles):
  prefetch_next=(
      not acc_only and n+1<region.n_tiles and
      kloop.supports_prefetched_first(
          region,n+1,k_tiles_per_gemm))
  q,page=tile.emit(region,activation_base,output_base,tokens,n,activation_page,
                   page,output_page,bias_base,bias_page,k_tiles_per_gemm,
                   wait_after_store,silu_base,silu_page,
                   activation_row_stride_bytes,
                   output_row_stride_bytes,acc_only,
                   wait_postproc_before_gld_a,
                   prefetched_first=prefetched_first,
                   prefetch_next=prefetch_next)
  p+=q
  prefetched_first=prefetch_next
 if restore_cfg_m:
  p.append(isa.set_cfg_m(1))
 if restore_cfg_k:
  p.append(isa.set_cfg_k(32))
 return p
