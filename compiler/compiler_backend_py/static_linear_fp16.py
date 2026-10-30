try: from . import (isa,static_linear_fp16_tile as tile,
                   static_linear_fp16_kloop as kloop)
except ImportError:
 import isa,static_linear_fp16_tile as tile
 import static_linear_fp16_kloop as kloop
try:
 from .static_linear_fp16_batch_validate import tiles_per_gemm
except ImportError:
 from static_linear_fp16_batch_validate import tiles_per_gemm
def emit_fp16(region,act_base,out_base,n_tile_start=0,n_tile_count=None,
              bias_base=None,residual_base=None,
              rmsnorm_accumulate=False,route_base=None,scale_reg=None,
              route_first=False,silu_base=None,configure_cfg_k=True,
              restore_cfg_k=True,wait_postproc_before_gld_a=False,
              output_page=0,input_page=0,acc_only=False,wait_after_store=True,
              prefetch_weights=False):
 r=region;a=act_base;o=out_base
 if r.nbits!=16: raise ValueError("FP16 lowering requires an FP16 region")
 ns=n_tile_start;c=r.n_tiles-ns if n_tile_count is None else n_tile_count
 if ns<0 or c<1 or ns+c>r.n_tiles: raise ValueError("invalid output tile range")
 if rmsnorm_accumulate and (ns!=0 or c!=r.n_tiles):
  raise ValueError("fused RMSNorm accumulation requires the complete output vector")
 if ((route_base is not None and scale_reg is None) or
     (route_first and route_base is None)):
  raise ValueError("fused route epilogue requires a route vector and scale register")
 if scale_reg is not None and not 1 <= scale_reg <= 15:
  raise ValueError("fused route scale register must be in [1,15]")
 if acc_only and (bias_base is not None or residual_base is not None or
                  rmsnorm_accumulate or route_base is not None or
                  silu_base is not None):
  raise ValueError("ACC-only output cannot apply a vector epilogue")
 if output_page not in (0,1,2):
  raise ValueError("FP16 output page must be 0, 1, or 2")
 if input_page not in (0,1):
  raise ValueError("FP16 activation page must be 0 or 1")
 e=o+c*64
 output_limit=16384 if output_page==2 else 65536
 if o<0 or e<o or e>output_limit:
  raise ValueError("FP16 output vector exceeds its L1 page")
 ae=a+r.k_tiles*64
 safe=(output_page!=0 or
       (o>=0 and e>=o and not(o<0x4000 and 0x2000<e) and
        not(o<ae and a<e)))
 k_tiles_per_gemm=tiles_per_gemm(r)
 p=([isa.set_cfg_k(32*k_tiles_per_gemm)]
    if configure_cfg_k and k_tiles_per_gemm>1 else [])
 page=None
 prefetched_first=False
 for n in range(ns,ns+c):
  prefetch_next=(
      (not acc_only or prefetch_weights) and n+1<ns+c and
      kloop.supports_prefetched_first(
          r,n+1,k_tiles_per_gemm))
  wait=(not(safe and n!=ns+c-1) or
        (route_base is not None and n==ns+c-1))
  if n==ns+c-1 and not wait_after_store:
   wait=False
  code,page=tile.emit_tile(r,a,o,n,ns,page,wait,k_tiles_per_gemm,
                           bias_base,residual_base,rmsnorm_accumulate,
                           route_base,scale_reg,route_first,silu_base,
                           wait_postproc_before_gld_a=(
                               wait_postproc_before_gld_a and n == ns),
                           output_page=output_page,input_page=input_page,
                           acc_only=acc_only,
                           prefetched_first=prefetched_first,
                           prefetch_next=prefetch_next);p.extend(code)
  prefetched_first=prefetch_next
 if restore_cfg_k and k_tiles_per_gemm>1:
  p.append(isa.set_cfg_k(32))
 return p
