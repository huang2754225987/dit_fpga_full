"""FP16 static linear API used by Supra2-IMG projection lowering."""
try:
 from . import isa
 from .static_linear_region import LinearRegion,TILE,WEIGHT_BANK,WEIGHT_L1
 from .static_linear_emit import emit_fp16
 from .static_linear_fp16_batch import emit_fp16_batch
except ImportError:
 import isa
 from static_linear_region import LinearRegion,TILE,WEIGHT_BANK,WEIGHT_L1
 from static_linear_emit import emit_fp16
 from static_linear_fp16_batch import emit_fp16_batch
