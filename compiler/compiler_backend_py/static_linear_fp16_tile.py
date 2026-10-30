try: from . import static_linear_fp16_ops as ops,static_linear_fp16_kloop as kloop
except ImportError:
 import static_linear_fp16_ops as ops
 import static_linear_fp16_kloop as kloop
def emit_tile(r,a,o,n,start,page=None,wait_after_store=True,
              k_tiles_per_gemm=1,bias_base=None,residual_base=None,
              rmsnorm_accumulate=False,route_base=None,scale_reg=None,
              route_first=False,silu_base=None,
              wait_postproc_before_gld_a=False,output_page=0,
              input_page=0,acc_only=False,prefetched_first=False,prefetch_next=False):
 loop_program=kloop.emit(r,a,1,n,input_page,page,batch=False,
                         k_tiles_per_gemm=k_tiles_per_gemm,
                         prefetched_first=prefetched_first,
                         wait_postproc_before_gld_a=wait_postproc_before_gld_a)
 if loop_program is not None:
  p,page=loop_program
  if acc_only:
   if prefetch_next:
    next_addr=r.weight+(n+1)*r.k_tiles*r.tile_bytes
    q,page=ops.weight_dma(next_addr,2,page);p.extend(q)
   return p,page
  if prefetch_next:
   next_addr=r.weight+(n+1)*r.k_tiles*r.tile_bytes
   x,page=ops.weight_dma(next_addr,2,page);p.extend(x)
  p.extend(ops.drain(o+(n-start)*64,wait_after_store,n,
                     bias_base,residual_base,rmsnorm_accumulate,
                     route_base,scale_reg,route_first,silu_base,output_page))
  return p,page
 if prefetched_first:
  raise ValueError("prefetched FP16 K group requires the aligned loop path")
 i=ops.isa;p,page=ops.weight_dma(r.weight+n*r.k_tiles*r.tile_bytes,2,page)
 p.extend([i.gld_w(i.Precision.FP16,2*4096),
           i.gld_a(a, wait_postproc=wait_postproc_before_gld_a,
                   page=input_page)])
 for k in range(0,r.k_tiles,k_tiles_per_gemm):
  next_k=k+k_tiles_per_gemm
  p.append(i.gemm(acc=k!=0,skip_store=next_k<r.k_tiles,imm=32,
                  prec=i.Precision.FP16))
  if next_k<r.k_tiles:
   group=next_k//k_tiles_per_gemm
   b=2+(group&1);w=r.weight+(n*r.k_tiles+next_k)*r.tile_bytes
   x,page=ops.weight_dma(w,b,page);p.extend(x)
   p.extend([i.gld_w(i.Precision.FP16,b*4096),
             i.gld_a(a+next_k*64,page=input_page)])
 if acc_only:
  if prefetch_next:
   next_addr=r.weight+(n+1)*r.k_tiles*r.tile_bytes
   q,page=ops.weight_dma(next_addr,2,page);p.extend(q)
  return p,page
 if prefetch_next:
  next_addr=r.weight+(n+1)*r.k_tiles*r.tile_bytes
  x,page=ops.weight_dma(next_addr,2,page);p.extend(x)
 p.extend(ops.drain(o+(n-start)*64,wait_after_store,n,
                    bias_base,residual_base,rmsnorm_accumulate,
                    route_base,scale_reg,route_first,silu_base,
                    output_page));return p,page
