"""Emit one output tile for an M-row FP16 projection."""
try:
 from . import (isa,static_linear_fp16_batch_load as load_ops,
                static_linear_fp16_batch_ops as batch_ops,
                static_linear_fp16_ops as weight_ops,
                static_linear_fp16_kloop as kloop)
except ImportError:
 import isa,static_linear_fp16_batch_load as load_ops
 import static_linear_fp16_batch_ops as batch_ops
 import static_linear_fp16_ops as weight_ops
 import static_linear_fp16_kloop as kloop
def emit(r,a,o,m,n,ap,page,op=0,bias_base=None,bias_page=2,
         k_tiles_per_gemm=1,wait_after_store=True,
         silu_base=None,silu_page=None,
         activation_row_stride_bytes=None,output_row_stride_bytes=None,
         acc_only=False,wait_postproc_before_gld_a=False,
         prefetched_first=False,prefetch_next=False):
 loop_program=kloop.emit(r,a,m,n,ap,page,batch=True,
                         k_tiles_per_gemm=k_tiles_per_gemm,
                         prefetched_first=prefetched_first,
                         wait_postproc_before_gld_a=(
                             wait_postproc_before_gld_a and n==0),
                         activation_row_stride_bytes=activation_row_stride_bytes)
 if loop_program is not None:
  p,page=loop_program
  if acc_only:
   return p,page
  if prefetch_next:
   next_addr=r.weight+(n+1)*r.k_tiles*r.tile_bytes
   q,page=weight_ops.weight_dma(next_addr,2,page);p+=q
  return p+batch_ops.drain(o,m,r.rows,n,op,bias_base,bias_page,
                           wait_after_store=(wait_after_store and
                                             n==r.n_tiles-1),
                           silu_base=silu_base,silu_page=silu_page,
                           output_row_stride_bytes=output_row_stride_bytes),page
 if prefetched_first:
  raise ValueError("prefetched FP16 K group requires the aligned loop path")
 p,page=load_ops.load(r,a,m,n,0,ap,page,k_tiles_per_gemm,
                      activation_row_stride_bytes,
                      wait_postproc_before_gld_a=(
                          wait_postproc_before_gld_a and n==0))
 for k in range(0,r.k_tiles,k_tiles_per_gemm):
  next_k=k+k_tiles_per_gemm
  p.append(isa.gemm(acc=k!=0,skip_store=next_k<r.k_tiles,imm=32,
                    prec=isa.Precision.FP16))
  if next_k<r.k_tiles:
   q,page=load_ops.load(r,a,m,n,next_k,ap,page,k_tiles_per_gemm,
                        activation_row_stride_bytes);p+=q
 if acc_only:
  return p,page
 if prefetch_next:
  next_addr=r.weight+(n+1)*r.k_tiles*r.tile_bytes
  q,page=weight_ops.weight_dma(next_addr,2,page);p+=q
 return p+batch_ops.drain(o,m,r.rows,n,op,bias_base,bias_page,
                          wait_after_store=(wait_after_store and
                                            n==r.n_tiles-1),
                          silu_base=silu_base,silu_page=silu_page,
                          output_row_stride_bytes=output_row_stride_bytes),page
