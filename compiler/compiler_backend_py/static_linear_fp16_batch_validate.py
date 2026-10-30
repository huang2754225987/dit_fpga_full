"""Validate batch tile geometry and prevent L1 live-range aliasing."""
def tiles_per_gemm(r):
 """Use one 4-KiB DMA per FP16 weight tile when adjacent K tiles fit."""
 if r.tile_bytes == 2048 and r.k_tiles >= 2 and r.k_tiles % 2 == 0:
  return 2
 return 1

def validate(r,a,o,m,activation_page,output_page,
             activation_row_stride_bytes=None,
             output_row_stride_bytes=None):
 if (r.nbits!=16 or r.rows<1 or r.cols%32 or not 1<=m<=32 or
     activation_page not in (0,1,2) or output_page not in (0,1,2)):
  raise ValueError("invalid batch geometry")
 if activation_row_stride_bytes is None:
  activation_row_stride_bytes=r.cols*2
 if (activation_row_stride_bytes<r.cols*2 or
     activation_row_stride_bytes%32):
  raise ValueError("activation row stride must cover and align the input row")
 if output_row_stride_bytes is None:
  output_row_stride_bytes=r.n_tiles*64
 if (output_row_stride_bytes<r.n_tiles*64 or output_row_stride_bytes%64):
  raise ValueError("output row stride must cover and align the output row")
 ae=a+(m-1)*activation_row_stride_bytes+r.cols*2
 oe=o+(m-1)*output_row_stride_bytes+r.n_tiles*64
 a_limit=16384 if activation_page==2 else 65536
 o_limit=16384 if output_page==2 else 65536
 if a<0 or o<0 or a%32 or o%64 or ae>a_limit or oe>o_limit: raise ValueError("invalid L1 range")
 if activation_page==0 and a<0x4000 and 0x1000<ae: raise ValueError("activation aliases weight staging")
 if output_page==0 and o<0x4000 and 0x1000<oe: raise ValueError("output aliases weight staging")
 if activation_page==output_page and a<oe and o<ae: raise ValueError("activation/output alias")
