# GEMM tile pipeline architecture exploration

## Scope and reference

This model explores the seven INT8 QLinear phases of the existing block-13, non-`PROGRAMMED_TILES` controller. It does not model attention QK/PV, DMA, inter-stage overlap, or numerical arithmetic. Shapes are taken from `supra2_block_top.sv`: QKV 256x576x1728; SPROJ/CQ/CPROJ 256x576x576; CKV 32x576x1152; MLP0 256x576x2304; MLP2 256x2304x576, expressed as MxKxN. Reference stage counters are from `block13_reference.json`; they include controller and weight-related time beyond the local tasks modeled here.

## Schedule and assumptions

The baseline completes the entire activation frontend before GEMM, as the existing controller does. For each 32-row M tile it then processes all N tiles in order. A tile occupies the array for one clear cycle, `K+32+32+3` feed cycles, and a hypothetical one-cycle accumulator wait/snapshot. The current RTL waits for `acc_valid`; the actual wait has not been independently counted and the one-cycle value is an explicit assumption. A serial baseline then reads 32 output rows at the assumed epilogue initiation interval.

Four designs are compared with identical hypothetical stage costs:

1. `serial`: frontend first, then array and drain serially.
2. `frontend_pingpong`: frontend prepares the next M tile while the array consumes the current M tile. A two-slot activation buffer cannot be overwritten until all N tiles using its previous M tile have finished their array accesses.
3. `drain_pingpong`: the array runs the next tile while a separate epilogue drains the prior snapshot. Two 4 KiB snapshots are modeled; to avoid overwrite, the array waits for its assigned slot to be released before starting a tile.
4. `both_pingpong`: combines both types of overlap. The model retains startup, final drain, resource dependencies and the finite buffer constraints.

Frontend cost is a sensitivity input, not a measurement: 16, 64 or 256 cycles per row at K=576, scaled linearly with K for this experiment. Real Hadamard and quantization timing depends on block size, handshakes and implementation; in particular, linear scaling for K=2304 is only a hypothesis. Epilogue initiation intervals of 1, 2 or 4 cycles per output row represent hypothetical throughput. The snapshot is assumed to capture an entire 32x32 INT32 grid within the assumed accumulator-wait period, and independent memory ports are assumed. Snapshot timing, FPGA storage mapping, routing, Fmax and bank conflicts are not established.

The current engine already has a full activation buffer and one 4 KiB accumulator snapshot. The CSV reports total activation payload storage under each design, not an incremental FPGA resource estimate: a two-M-tile buffer is `2*32*K` INT8 bytes versus `M*K` bytes for the full input activation payload. Scale/state/metadata, concurrent access ports and allocation overhead are excluded. For CKV, M=32: the nominal two-slot design overprovisions the single group and has no inter-group frontend overlap.

## Illustrative results

With the hypothetical frontend cost of 64 cycles/row at K=576, linear K scaling, epilogue initiation interval 1, and wait/snapshot one cycle:

| Stage | Serial local cycles | Frontend overlap | Drain overlap | Both overlap | Existing full stage counter |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV | 308,848 | 294,512 | 295,056 | 280,720 | 531,053 |
| MLP0 | 406,336 | 392,000 | 387,936 | 373,600 | 652,317 |
| MLP2 | 411,856 | 354,512 | 407,280 | 349,936 | 808,012 |
| CKV | 26,420 | 26,420 | 25,300 | 25,300 | 96,691 |

For the seven shapes together, this scenario yields 129,024 fewer local cycles for frontend overlap, 51,616 for drain overlap, and 180,640 for both. These are hypothetical local schedule deltas, not a predicted block speedup: the model has not reproduced each measured baseline and assumes restructuring does not change stage-control or memory costs. For frontend costs 16 and 256, the both-overlap local deltas are 83,872 and 567,712 cycles respectively. The magnitude depends strongly on an unmeasured input and these numbers are not confidence bounds.

The CSV flags configurations whose modeled baseline already exceeds the corresponding complete reference stage. Passing that check does not calibrate the model: a baseline below the reference merely leaves unexplained time for weights, control, actual frontend, memory and wait costs.

## Validation and next steps

Run `gemm_architecture_model --self-test`, then `gemm_architecture_model <report.csv>`. The tests check serial accounting, drain overlap, double-buffer reuse behavior through the scheduler, slow-consumer backpressure, and the single-tile case. `scripts/run_cpp_latency_model.ps1` also generates the CSV.

Measure per-stage `S_FE_START`, `S_FE_FEED`, `S_FE_DRAIN`, `S_TILE_CLEAR`, `S_FEED`, `S_WAIT_ACC`, `S_DRAIN`, and weight-wait cycles in the same block-13 RTL/MIG run; include array enable, snapshot and memory port handshakes. Fit frontend and wait values from those observations and compare baseline event order before using optimized deltas. Then implement double-buffer ownership, output metadata and valid/ready handling in RTL, check numerical outputs against existing fixtures, and evaluate synthesis resources and timing. No new RTL implementation, simulation, synthesis or board test is claimed.
