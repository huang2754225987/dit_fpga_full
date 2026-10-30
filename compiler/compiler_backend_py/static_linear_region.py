"""Static linear tensor and tiling metadata."""
from dataclasses import dataclass
TILE=32
WEIGHT_BANK=2
WEIGHT_L1=WEIGHT_BANK*4096
@dataclass(frozen=True)
class LinearRegion:
 weight:int
 scale_zero:int|None
 rows:int
 cols:int
 nbits:int
 tile_bytes:int
 groups:int
 bias:int|None=None
 n_tiles=property(lambda s:(s.rows+TILE-1)//TILE)
 k_tiles=property(lambda s:(s.cols+TILE-1)//TILE)
