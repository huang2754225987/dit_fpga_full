try:
 from . import isa,static_linear_quant_ops as ops
except ImportError:
 import isa,static_linear_quant_ops as ops
def emit_quantized(region,act_base,out_base,n_tile_start=0,n_tile_count=None,
                   silu_base=None,route_base=None,scale_reg=None,
                   route_first=False,residual_base=None,output_page=0,
                   configure_cfg_k=True,restore_cfg_k=True,
                   wait_after_store=True,
                   wait_postproc_before_gld_a=False,input_page=0):
 r=region
 ns=n_tile_start;c=r.n_tiles-ns if n_tile_count is None else n_tile_count
 if r.nbits not in (1,2,3,4) or r.scale_zero is None:
  raise ValueError("quantized linear requires 1..4-bit weights and scale metadata")
 expected_groups=(r.k_tiles+3)//4
 if r.groups!=expected_groups:
  raise ValueError("quantized scale groups must cover each four-tile K block")
 if ns<0 or c<1 or ns+c>r.n_tiles:
  raise ValueError("invalid output tile range")
 if output_page not in (0,1,2):
  raise ValueError("quantized output page must be 0, 1, or 2")
 if input_page not in (0,1):
  raise ValueError("quantized activation page must be 0 or 1")
 output_limit=16384 if output_page==2 else 65536
 if out_base<0 or out_base%32 or out_base+c*64>output_limit:
  raise ValueError("quantized output vector exceeds its L1 page")
 if output_page!=0 and any(value is not None for value in
                           (silu_base,route_base,residual_base)):
  raise ValueError("fused quantized epilogues currently require page 0")
 if ((route_base is not None and scale_reg is None) or
     (route_first and route_base is None) or
     ((route_base is None) != (residual_base is None))):
  raise ValueError("fused route epilogue requires scale, route, and residual")
 if scale_reg is not None and not 1 <= scale_reg <= 15:
  raise ValueError("fused route scale register must be in [1,15]")
 p=[]
 wide_group_bytes=ops.k128_weight_group_bytes(r)
 wide_gemm=(wide_group_bytes!=0 and r.k_tiles%4==0 and
            r.weight%wide_group_bytes==0)
 if wide_gemm and configure_cfg_k:
  p.append(isa.set_cfg_k(128))
 prefetched_scale_nt=None
 prefetched_weight_nt=None
 prefetched_weight_page=None
 for nt in range(ns,ns+c):
  weight_page=None
  prefetched_weight_group=None
  if prefetched_weight_nt==nt:
   weight_page=prefetched_weight_page
   prefetched_weight_group=0
   prefetched_weight_nt=None
   prefetched_weight_page=None
  loaded_scale_chunk=None
  g=0
  while g<r.groups:
   scale_chunk=(g//16)*16
   if loaded_scale_chunk!=scale_chunk:
    if scale_chunk==0 and prefetched_scale_nt==nt:
     prefetched_scale_nt=None
    else:
     scale_addr=r.scale_zero+((nt*r.groups+scale_chunk)<<7)
     scale_count=min(16,r.groups-scale_chunk)
     p.extend(isa.static_scale_at(scale_addr,group_count=scale_count))
    loaded_scale_chunk=scale_chunk
   if (wide_gemm and r.groups-g>1 and
       not (g==0 and prefetched_weight_group==0)):
    weight_addr=r.weight+(nt*r.k_tiles+g*4)*r.tile_bytes
    groups_to_page=(0x10000-(weight_addr&0xFFFF))//wide_group_bytes
    loop_count=min(r.groups-g,16-(g-scale_chunk),groups_to_page)
    if loop_count>1:
     code,weight_page=ops.static_wide_k_loop(
        r,act_base,nt,g,loop_count,1,64,input_page,
         scale_group=g-scale_chunk,weight_page=weight_page,
         acc=g!=0,skip_store=g+loop_count!=r.groups,
         wait_postproc_before_gld_a=(
             wait_postproc_before_gld_a and nt==ns and g==0))
     p.extend(code)
     g+=loop_count
     prefetched_weight_group=None
     continue
   if wide_gemm:
    weight_addr=r.weight+(nt*r.k_tiles+g*4)*r.tile_bytes
    prefetch_next=None
    if (g+1<r.groups and
        not (wide_gemm and prefetched_weight_group==0)):
     next_addr=r.weight+(nt*r.k_tiles+(g+1)*4)*r.tile_bytes
     next_page=next_addr>>16
     next_scale_chunk=((g+1)//16)*16
     next_k_tiles=min(4,r.k_tiles-(g+1)*4)
     if (next_k_tiles==4 and next_page==weight_addr>>16 and
         next_scale_chunk==scale_chunk):
      prefetch_next=(next_addr,2^((g+1)&1),
                     4*r.tile_bytes,next_page)
    code,weight_page=ops.batch_group_burst(
        r,act_base,nt,g,1,64,input_page,g-scale_chunk,
        weight_bank=2^(g&1),weight_page=weight_page,
        prefetch_next=prefetch_next,
        current_prefetched=(prefetched_weight_group==g),wide_gemm=True,
        wait_postproc_before_gld_a=(
            wait_postproc_before_gld_a and nt == ns and g == 0))
    prefetched_weight_group=(g+1 if prefetch_next is not None else None)
   else:
    code,weight_page=ops.group(
        r,act_base,nt,g,acc=g!=0,skip_store=g+1!=r.groups,
        scale_group=g-scale_chunk,
        preload_scales=True,weight_page=weight_page,
        barrier_after=False,
        wait_postproc_before_gld_a=(
            wait_postproc_before_gld_a and nt == ns and g == 0),
        activation_page=input_page)
    prefetched_weight_group=None
   p.extend(code)
   g+=1
  if nt+1<ns+c:
   # SCALE has banked snapshots for the active GEMM, so the next N-tile's
   # metadata can be fetched after the final K GEMM is issued and while that
   # GEMM/its accumulator drain is still in flight. Preserve the K reduction
   # order; only move this independent metadata read across the output drain.
   next_scale_addr=r.scale_zero+(((nt+1)*r.groups)<<7)
   p.extend(isa.static_scale_at(
       next_scale_addr,group_count=min(16,r.groups)))
   prefetched_scale_nt=nt+1
   if r.k_tiles>=4:
    prefetch_code,prefetched_weight_page=ops.prefetch_next_n_tile_group(
        r,act_base,nt+1,1,64,input_page,active_weight_page=weight_page)
    p.extend(prefetch_code)
    prefetched_weight_nt=nt+1
  # AC2V is dependency-gated by RES_GEMM in ucode_decode. Do not insert a
  # global fence between the final accumulation GEMM and its accumulator drain.
  p.extend(ops.drain(out_base+(nt-ns)*64,silu_base,nt,route_base,
                     scale_reg,route_first,residual_base,
                     wait_after_store=(wait_after_store and nt==ns+c-1),
                     output_page=output_page))
 if wide_gemm and restore_cfg_k:
  p.append(isa.set_cfg_k(32))
 return p
