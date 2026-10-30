"""Load one static weight and compact activation K tile."""
try: from . import isa,static_linear_fp16_ops as ops
except ImportError: import isa,static_linear_fp16_ops as ops
def load(r,a,m,n,k,ap,page,k_tiles_per_gemm=1,
         activation_row_stride_bytes=None,
         wait_postproc_before_gld_a=False):
 group=k//k_tiles_per_gemm
 b=2 if group==0 else 2+(group&1)
 addr=r.weight+(n*r.k_tiles+k)*r.tile_bytes
 compact_bytes=(2048 if r.tile_bytes==2048 and k_tiles_per_gemm==1
                else None)
 p,page=ops.weight_dma(addr,b,page,transfer_bytes=compact_bytes)
 if activation_row_stride_bytes is None:
  activation_row_stride_bytes=r.cols*2
 activation_tile_bytes=32*k_tiles_per_gemm*2
 wait_postproc=wait_postproc_before_gld_a and k==0
 activation_load=(isa.gld_a_batch_transpose(
                      a+k*64,page=ap,wait_postproc=wait_postproc)
                  if activation_row_stride_bytes==activation_tile_bytes else
                  isa.gld_a_batch_transpose_strided(
                      a+k*64,activation_row_stride_bytes,page=ap,
                      wait_postproc=wait_postproc))
 p += [isa.gld_w(isa.Precision.FP16,b*4096),activation_load]
 return p,page
