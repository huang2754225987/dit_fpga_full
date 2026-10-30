try: from . import isa
except ImportError: import isa
def weight_dma(addr,bank,page,transfer_bytes=None):
 n=addr>>16;p=[isa.set_weight_page(n)] if n!=page else []
 p.append(isa.static_dma(addr,l1_bank=bank,fp16=True,
                         transfer_bytes=transfer_bytes)[1])
 return p,n
def drain(dst,wait_after_store=True,n_tile=0,bias_base=None,
          residual_base=None,rmsnorm_accumulate=False,route_base=None,
          scale_reg=None,route_first=False,silu_base=None,output_page=0):
 # AC2V waits for GEMM in the decoder; following vector operations wait on
 # the shared PostProc resource, so a global barrier is unnecessary here.
 p=[isa.ac2v(0,row=0)]
 if bias_base is not None:
  p += [isa.vld(1,bias_base+n_tile*64),isa.add(0,0,1)]
 if silu_base is not None:
  p += [isa.vld(1,silu_base+n_tile*64),isa.silu(1,1),isa.mul(0,0,1)]
 if scale_reg is not None:
  p.append(isa.mul(0,0,scale_reg))
 if route_base is not None:
  p.append(isa.vld(1,route_base+n_tile*64))
  p.append(isa.add(0,1,0) if route_first else isa.add(0,0,1))
 if residual_base is not None:
  p += [isa.vld(1,residual_base+n_tile*64),isa.add(0,0,1)]
 if rmsnorm_accumulate:
  p.append(isa.rmsnorm_accumulate(0,reset=n_tile==0))
 p.append(isa.vst(0,dst) if output_page==0 else
          isa.vst_page(0,dst,output_page))
 if wait_after_store: p.append(isa.barrier())
 return p
