"""FP16 static linear instruction emitter for Supra2-IMG."""
try:
 from .static_linear_fp16 import emit_fp16 as _fp16
except ImportError:
 from static_linear_fp16 import emit_fp16 as _fp16

def emit_fp16(region,act_base,out_base,n_tile_start=0,n_tile_count=None,
              bias_base=None,residual_base=None,
              rmsnorm_accumulate=False,route_base=None,scale_reg=None,
              route_first=False,silu_base=None,configure_cfg_k=True,
              restore_cfg_k=True,wait_postproc_before_gld_a=False,
              output_page=0,input_page=0,acc_only=False,wait_after_store=True):
 return _fp16(region,act_base,out_base,n_tile_start,n_tile_count,
               bias_base,residual_base,rmsnorm_accumulate,
               route_base,scale_reg,route_first,silu_base,
               configure_cfg_k,restore_cfg_k,
               wait_postproc_before_gld_a,output_page,
               input_page=input_page,acc_only=acc_only,
               wait_after_store=wait_after_store)
