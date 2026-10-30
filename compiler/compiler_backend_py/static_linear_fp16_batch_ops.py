"""ACC row drain for batched FP16 projections."""
try: from . import isa
except ImportError: import isa
def drain(output,rows,region_rows,n_tile,output_page=0,
          bias_base=None,bias_page=2,wait_after_store=False,
          silu_base=None,silu_page=None,output_row_stride_bytes=None):
 p=[]
 if output_row_stride_bytes is None:
  output_row_stride_bytes=((region_rows+31)//32)*64
 for row in range(rows):
  # AC2V waits for GEMM; subsequent PostProc operations and AC2V drains are
  # serialized by the shared PostProc scoreboard.
  p.append(isa.ac2v(0,row=row))
  if bias_base is not None:
   p += [isa.vld_page(1,bias_base+n_tile*64,bias_page),isa.add(0,0,1)]
  if silu_base is not None:
   page=output_page if silu_page is None else silu_page
   p += [isa.vld_page(1,silu_base+row*region_rows*2+n_tile*64,page),
         isa.silu(1,1),isa.mul(0,0,1)]
  p += [isa.vst_page(0,output+row*output_row_stride_bytes+n_tile*64,
                     output_page)]
 # Intermediate N tiles may overlap their L1 writeback with the next weight
 # load. The following GEMM waits for PostProc to retire before touching ACC.
 # Keep a final fence where the caller may immediately DMA-read this output.
 if wait_after_store:
  p.append(isa.barrier())
 return p
