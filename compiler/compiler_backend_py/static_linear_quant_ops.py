try: from . import isa
except ImportError: import isa
L1_BANK_BYTES=4096


def k128_weight_group_bytes(region):
 """Return storage bytes for one four-K-tile static K=128 group."""
 compact_bytes={1:128,2:256,3:512}.get(region.nbits)
 if compact_bytes is not None and region.tile_bytes==compact_bytes:
  return 4*compact_bytes
 # Padded static matrices retain the K=128 path at every quantized precision.
 # Compact INT1/2/3 groups use their packed four-tile footprint above.
 if region.tile_bytes==1024:
  return 4*region.tile_bytes
 return 0


def _activation_overlaps_weight_staging(r, act, g, k, batch_rows,
                                        activation_row_stride_bytes,
                                        activation_page,
                                        dma_banks=(2,3),
                                        toggle_loop_addresses=True,
                                        weight_bytes=None):
 if activation_page == 1:
  return False
 if activation_page != 0:
  return True
 # Static quantized DMA starts in bank 2 and alternates to bank 3 on odd K
 # iterations; GLD.W applies the matching source-bank toggle.
 if weight_bytes is None:
  weight_bytes=r.tile_bytes
 weight_ranges=tuple((bank*L1_BANK_BYTES,
                      bank*L1_BANK_BYTES+weight_bytes)
                     for bank in dma_banks)
 row_stride=(64 if batch_rows==1 else activation_row_stride_bytes)
 for k_iter in range(k):
  instruction_base=act+g*256
  if toggle_loop_addresses and k_iter & 1:
   # LOOP replay toggles GLD.A imm[12] before decode adds its K stride.
   instruction_base ^= L1_BANK_BYTES
  # Scalar GLD.A and the emitted strided transpose form both advance one
  # 32-element K tile (64 bytes), independent of the number of batch rows.
  k_base=instruction_base+k_iter*64
  for row in range(batch_rows):
   row_start=k_base+row*row_stride
   for weight_start,weight_end in weight_ranges:
    if row_start < weight_end and weight_start < row_start+64:
     return True
 return False


def prefetch_next_n_tile_group(r, act, next_n_tile, tokens,
                               activation_row_stride_bytes, activation_page,
                               active_weight_page=None):
 """Prefetch the next output tile's first K group into static L1 bank 2.

 The current output tile has already issued its final GEMM when this helper is
 used. Its WRF contents are therefore independent of the L1 staging bank, so
 the first weight burst for the next tile can overlap the current GEMM/drain.
 """
 if (not 0 <= next_n_tile < r.n_tiles or r.k_tiles < 4 or
     tokens < 1 or tokens > 32):
  raise ValueError("next-N-tile prefetch requires a complete four-tile group")
 weight_addr=r.weight+next_n_tile*r.k_tiles*r.tile_bytes
 transfer_bytes=4*r.tile_bytes
 if transfer_bytes > L1_BANK_BYTES:
  raise ValueError("next-N-tile weight group exceeds its L1 staging bank")
 if _activation_overlaps_weight_staging(
     r,act,0,4,tokens,activation_row_stride_bytes,activation_page,
     dma_banks=(2,),toggle_loop_addresses=False,
     weight_bytes=transfer_bytes):
  raise ValueError("next-N-tile DMA aliases the activation source")
 page=weight_addr >> 16
 dma=isa.static_dma(weight_addr,l1_bank=2,transfer_bytes=transfer_bytes)
 code=[]
 if active_weight_page != page:
  code.append(dma[0])
 code.append(dma[1])
 return code,page


def drain(dst,silu_base=None,n_tile=0,route_base=None,scale_reg=None,
          route_first=False,residual_base=None,wait_after_store=True,
          output_page=0):
 # AC2V waits for GEMM in the decoder; following vector operations wait on
 # the shared PostProc resource, so a global barrier is unnecessary here.
 p=[isa.ac2v(0,row=0)]
 if silu_base is not None:
  p += [isa.vld(1,silu_base+n_tile*64),isa.silu(1,1),isa.mul(0,0,1)]
 if scale_reg is not None:
  p.append(isa.mul(0,0,scale_reg))
 if route_base is not None:
  p.append(isa.vld(1,route_base+n_tile*64))
  p.append(isa.add(0,1,0) if route_first else isa.add(0,0,1))
 if residual_base is not None:
  p += [isa.vld(1,residual_base+n_tile*64),isa.add(0,0,1)]
 p.append(isa.vst(0,dst) if output_page==0 else
          isa.vst_page(0,dst,output_page))
 if wait_after_store:p.append(isa.barrier())
 return p
def group(r,act,nt,g,acc=False,skip_store=False,scale_group=0,
          preload_scales=False,weight_page=None,barrier_after=True,
          batch_rows=1,activation_row_stride_bytes=None,activation_page=0,
          wait_postproc_before_gld_a=False):
 s=r.scale_zero+((nt*r.groups+g)<<7)
 w=r.weight+(nt*r.k_tiles+g*4)*r.tile_bytes
 k=min(4,r.k_tiles-g*4)
 # Marked GLD.A loops keep the compiler-proven contiguous source addresses;
 # legacy loops retain their historical address ping-pong. Validate the
 # address sequence that will actually be emitted before allowing overlap.
 allow_dma_overlap=(
     activation_page == 1 and batch_rows > 1 or
     activation_page == 0 and not _activation_overlaps_weight_staging(
         r,act,g,k,batch_rows,activation_row_stride_bytes,activation_page,
         dma_banks=(0,1,2,3),toggle_loop_addresses=False))
 if (not allow_dma_overlap and
     _activation_overlaps_weight_staging(
         r,act,g,k,batch_rows,activation_row_stride_bytes,activation_page,
         dma_banks=(2,3),toggle_loop_addresses=True)):
  raise ValueError("activation tile aliases the static weight DMA staging range")
 d=isa.static_dma(w,2,loop_stride_bytes=r.tile_bytes)
 page=w>>16
 p=[]
 if not preload_scales:
  # Weight DMA is queued behind the scale loader on the shared AXI read port;
  # GLD.W is resource-gated behind that DMA before consuming scale metadata.
  p.extend(isa.static_scale_at(s))
 if not preload_scales or page!=weight_page:p.append(d[0])
 # The scalar helper has no page selector, so page 1 is a safe opt-in only
 # for the batch-transpose form that actually encodes that page.
 activation_load=(isa.gld_a(act+g*256,page=activation_page,
                             wait_postproc=wait_postproc_before_gld_a,
                             allow_dma_overlap=allow_dma_overlap)
                  if batch_rows==1 else
                  isa.gld_a_batch_transpose_strided(
                      act+g*256,activation_row_stride_bytes,
                      page=activation_page,
                      allow_dma_overlap=allow_dma_overlap,
                      wait_postproc=wait_postproc_before_gld_a))
 # The alias-checked opt-in lets the next activation load overlap the active
 # weight DMA. GEMM remains dependency-gated behind both completed producers.
 p.append(isa.hw_loop(k,4))
 if wait_postproc_before_gld_a and allow_dma_overlap:
  # The marked activation read waits for a prior PostProc producer while the
  # disjoint weight DMA proceeds independently.
  p += [d[1],activation_load]
 else:
  p += [activation_load,d[1]]
 p += [
       isa.static_gld_w(r.nbits,8192,group=scale_group),
       isa.static_gemm(r.nbits,acc=acc,skip_store=skip_store,
                       imm=32,loop_managed=True)]
 if barrier_after:p.append(isa.barrier())
 return p,page


def batch_group_burst(r, act, n_tile, group_idx, tokens,
                      activation_row_stride_bytes, activation_page,
                      scale_group, weight_bank, weight_page,
                      prefetch_next=None, current_prefetched=False,
                      wide_gemm=False,
                      wait_postproc_before_gld_a=False):
 """Load one four-K-tile static group and issue its K tiles in order.

 The four adjacent weight tiles share scale metadata and fit in one 4-KiB
 L1 bank. A single DMA fills that bank. The wide-GEMM path consumes all four
 tiles with one K=128 GLD.W/GEMM for either padded or compact tile storage.
 The optional next-group DMA targets the other ping-pong bank after the first
 weight load has been accepted.
 """
 if not 1 <= tokens <= 32 or weight_bank not in (2, 3):
  raise ValueError("batched weight bursts require 1..32 rows and bank 2/3")
 k_tiles=min(4,r.k_tiles-group_idx*4)
 if k_tiles != 4:
  raise ValueError("batched weight bursts require four complete K tiles")
 weight_addr=r.weight+(n_tile*r.k_tiles+group_idx*4)*r.tile_bytes
 burst_bytes=k_tiles*r.tile_bytes
 if burst_bytes > L1_BANK_BYTES:
  raise ValueError("static quantized K group exceeds its 4-KiB L1 bank")

 aliases_weights=_activation_overlaps_weight_staging(
     r,act,group_idx,k_tiles,tokens,activation_row_stride_bytes,
     activation_page,dma_banks=(2,3),toggle_loop_addresses=False,
     weight_bytes=L1_BANK_BYTES)
 if aliases_weights:
  raise ValueError("activation tile aliases the static weight DMA staging range")
 allow_dma_overlap=(
     activation_page == 1 or
     activation_page == 0 and not _activation_overlaps_weight_staging(
         r,act,group_idx,k_tiles,tokens,activation_row_stride_bytes,
         activation_page,dma_banks=(0,1,2,3),
         toggle_loop_addresses=False,weight_bytes=L1_BANK_BYTES))

 page=weight_addr>>16
 p=[]
 activation_loads=[]
 for tile_idx in range(k_tiles):
  act_addr=act+group_idx*256+tile_idx*64
  if tokens>1:
   activation_loads.append(isa.gld_a_batch_transpose_strided(
       act_addr,activation_row_stride_bytes,page=activation_page,
       allow_dma_overlap=allow_dma_overlap,
       wait_postproc=(wait_postproc_before_gld_a and tile_idx == 0)))
  else:
   activation_loads.append(isa.gld_a(
       act_addr,
       wait_postproc=(wait_postproc_before_gld_a and tile_idx == 0),
       allow_dma_overlap=allow_dma_overlap,page=activation_page))

 # Without the hardware overlap assertion, issue the independent activation
 # read before DMA starts. This preserves first-tile latency hiding for legal
 # page-0 inputs in banks 0/1 while still forbidding aliases with banks 2/3.
 activation_issued=False
 if not current_prefetched and not allow_dma_overlap:
  p.append(activation_loads[0])
  activation_issued=True
 if current_prefetched:
  if weight_page!=page:
   raise ValueError("prefetched K group changed the active static weight page")
 else:
  dma=isa.static_dma(weight_addr,l1_bank=weight_bank,
                     transfer_bytes=burst_bytes)
  if weight_page!=page:
   p.append(dma[0])
  p.append(dma[1])

 if prefetch_next is not None:
  next_addr,next_bank,next_bytes,next_page=prefetch_next
  if next_page!=page or next_bank==weight_bank or next_bytes > L1_BANK_BYTES:
   raise ValueError("prefetched K group must fit the other bank on the same page")
  next_dma=isa.static_dma(next_addr,l1_bank=next_bank,
                          transfer_bytes=next_bytes)

 if wide_gemm:
  expected_burst=k128_weight_group_bytes(r)
  if (r.nbits not in (1,2,3,4) or expected_burst == 0 or
      burst_bytes != expected_burst):
   raise ValueError("wide static GEMM requires a complete four-tile weight group")
  if not activation_issued:
   p.append(activation_loads[0])
  p.append(isa.static_gld_w(
      r.nbits,weight_bank*L1_BANK_BYTES,group=scale_group))
  if prefetch_next is not None:
   # Capture the current weight tile before reusing the other bank for the
   # next group; the next DMA can overlap this 128-K GEMM.
   p.append(next_dma[1])
  p.append(isa.static_gemm(
      r.nbits,acc=(group_idx!=0),
      skip_store=(group_idx+1!=r.groups),imm=128,loop_managed=False))
  return p,page

 for tile_idx in range(k_tiles):
  if not (tile_idx==0 and activation_issued):
   p.append(activation_loads[tile_idx])
  weight_l1_addr=weight_bank*L1_BANK_BYTES+tile_idx*r.tile_bytes
  p.append(isa.static_gld_w(
      r.nbits,weight_l1_addr,group=scale_group))
  if tile_idx==0 and prefetch_next is not None:
   # GLD.W waits for the current-bank fill. Starting the next-bank read after
   # this accepted command lets it overlap the current group's GEMM reduction.
   p.append(next_dma[1])
  global_k=group_idx*4+tile_idx
  is_last=(global_k==r.k_tiles-1)
  p.append(isa.static_gemm(
      r.nbits,acc=(global_k!=0),skip_store=not is_last,
      imm=32,loop_managed=False))
 return p,page


def static_wide_k_loop(r, act, n_tile, group_idx, loop_count, tokens,
                       activation_row_stride_bytes, activation_page,
                       scale_group, weight_page, acc, skip_store,
                       wait_postproc_before_gld_a=False):
 """Replay padded or compact K=128 groups with strided address advances.

 Each iteration transfers the four K tiles at their stored tile width, loads
 the corresponding 128-value activation block, and issues one wide GEMM. The
 loop is split at scale-selector and DDR-page boundaries.
 """
 group_bytes=k128_weight_group_bytes(r)
 if (not 1 <= tokens <= 32 or not 1 <= loop_count <= 16 or
     group_bytes not in (512,1024,2048,4096) or r.k_tiles % 4 or
     activation_page not in (0, 1) or
     not 0 <= n_tile < r.n_tiles or
     not 0 <= group_idx < r.groups or group_idx + loop_count > r.groups or
     not 0 <= scale_group < 16 or scale_group + loop_count > 16):
  raise ValueError("wide batch K loop has incompatible precision or geometry")
 weight_addr=r.weight+(n_tile*r.k_tiles+group_idx*4)*r.tile_bytes
 page=weight_addr >> 16
 if weight_addr % group_bytes:
  raise ValueError("wide batch K loop must align to the packed weight group")
 groups_to_page=(0x10000-(weight_addr & 0xFFFF)) // group_bytes
 if loop_count > groups_to_page:
  raise ValueError("wide batch K loop crosses a static weight page")
 aliases_weights=_activation_overlaps_weight_staging(
     r,act,group_idx*4,loop_count*4,tokens,activation_row_stride_bytes,
     activation_page,dma_banks=(2,3),toggle_loop_addresses=False,
     weight_bytes=group_bytes)
 if aliases_weights:
  raise ValueError("activation tile aliases the static weight DMA staging range")
 allow_dma_overlap=(
     activation_page == 1 or
     activation_page == 0 and not _activation_overlaps_weight_staging(
         r,act,group_idx*4,loop_count*4,tokens,activation_row_stride_bytes,
         activation_page,dma_banks=(0,1,2,3),
         toggle_loop_addresses=False,weight_bytes=group_bytes))
 activation_addr=act+group_idx*256
 if tokens>1:
  activation_load=isa.gld_a_batch_transpose_strided(
      activation_addr,activation_row_stride_bytes,page=activation_page,
      allow_dma_overlap=allow_dma_overlap,
      wait_postproc=wait_postproc_before_gld_a)
 else:
  activation_load=isa.gld_a(
      activation_addr,allow_dma_overlap=allow_dma_overlap,
      wait_postproc=wait_postproc_before_gld_a,page=activation_page)
 dma=isa.static_dma(weight_addr,l1_bank=2,transfer_bytes=group_bytes)
 p=[]
 if weight_page != page:
  p.append(dma[0])
 p.append(isa.static_cached_wide_loop(loop_count,body_len=4))
 body=[dma[1],activation_load,
       isa.static_gld_w(r.nbits,2*L1_BANK_BYTES,group=scale_group),
       isa.static_gemm(r.nbits,acc=acc,skip_store=skip_store,
                       imm=128,loop_managed=False)]
 if allow_dma_overlap:
  p.extend(body)
 else:
  p.extend([body[1],body[0],body[2],body[3]])
 return p,page
