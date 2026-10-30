# DMA, on-chip buffer, and compute timing model

## Scope

`NpuPipelineModel` combines DMA descriptors, on-chip tile availability,
compute-resource occupancy, and the `MigAxiModel`/`Ddr4Device` timing path on
one MIG UI clock timeline. The model consumes a dependency graph of
`PipelineOperation` entries. It can overlap a DMA descriptor with one or more
compute operations when their data and hardware resources permit it.

The implementation follows the current `supra2_axi_read_dma.sv` control
contract in these respects: one active DMA descriptor and one outstanding AXI
burst, full 512-bit beats, at most 64 beats per burst, no crossing a 4-KiB
boundary, and a downstream beat that advances only when the tile buffer can
accept it. The read path uses `MigAxiModel::read_beat_events()` to obtain each
beat's earliest data-ready UI cycle. Burst order is preserved at the DMA
interface even if DDR commands were scheduled across banks.

## Operation types

| Kind | Starts when | Completes when |
| --- | --- | --- |
| `DmaRead` | Dependencies done, DMA idle, destination has capacity | All AXI bursts complete and every read beat is accepted by the destination buffer |
| `DmaWrite` | Dependencies done, DMA idle, source tile contains all bytes | All write bursts receive MIG completion |
| `Compute` | Dependencies done, inputs resident, resource slot free, output capacity reservable | `compute_cycles` have elapsed and the output tile is published |

The compute duration must represent the unit's internal execution without
the external DMA wait that this pipeline model already generates. For
example, a measured QLinear stage duration that includes weight-fetch waits
cannot be used directly alongside a separate weight-DMA operation: that would
double count the weight wait. The `compute_units.h` 32x32 array estimate is
an analytical feed lower bound; use separately calibrated pure execution
cycles for quantitative block-13 predictions.

`VectorUnitModel` estimates a continuous 32-lane stream as
`ceil(elements / 32) + pipeline_overhead`. `SoftmaxTileModel` uses the
measured 2,401-cycle profile for one 32-row, 256-key tile. The block-13
profile table now records weight-wait cycles numerically;
`block13_cycles_excluding_weight_wait()` subtracts those recorded waits but
keeps the stage's internal control and on-chip access time. Attention QK and
PV use the block top's shared 32x32 array; represent them as ARRAY tasks,
with softmax represented as a separate task/resource.

## Buffer model

Each `OnChipTileBuffer` has a byte capacity, bank count, line size, and number
of write ports per bank. A DMA read beat is mapped to a bank by its local
byte offset. If the bank has no remaining write port in the current UI cycle
or the buffer has no free capacity, the DMA holds that beat and tries again
on the next cycle. Compute output capacity is reserved at task start and
published at finish. DMA destination capacity is reserved at descriptor start.
Compute inputs remain pinned until execution ends; `release_after_read=true`
then releases their storage, while false retains them for reuse. Retained
readers can share a tile; a consuming reader or DMA store requires exclusive
access. A tile cannot be refilled while compute still reads it.

Compute tasks keep resident input storage for their full duration. Their
`compute_cycles` must include local SRAM feed and result-write
timing. The buffer model does not independently add SRAM read cycles or
model a full cache hierarchy. Named tile buffers describe scratchpads, not
GPU-style automatic caches.

## API and example

1. Declare buffer capacities and compute resource instance counts.
2. Create `PipelineOperation` records with unique IDs and dependencies.
3. Use `DmaRead` to load each activation/weight tile, `Compute` for ARRAY,
   LN, SOFTMAX, GELU, RESIDUAL, or other units, and `DmaWrite` for output.
4. Call `NpuPipelineModel::run()` and inspect operation start/finish UI cycles
   and MIG traffic statistics. `DmaStats` reports burst count, read beats
   accepted by the tile buffer, write beats staged, AXI admission wait,
   tile-buffer backpressure, and memory wait. `resource_busy_ui` reports
   summed busy UI cycles for each compute resource.

`src/npu_pipeline_demo.cpp` shows one illustrative 32x32 GEMM tile followed
by GELU and an output DMA. Its constants are not a block-13 calibration.

```powershell
cmake -S simulator/cpp_latency -B build/cpp_latency
cmake --build build/cpp_latency --config Release --target npu_pipeline_demo
build/cpp_latency/Release/npu_pipeline_demo.exe
```

Single-configuration generators place the executable directly in
`build/cpp_latency/`.

## Limits and calibration

The MIG layer exposes read-beat ready events, but currently has an unbounded
internal event list. A stalled tile-buffer consumer delays DMA completion and
the next burst; it does not push a finite `RREADY` response FIFO limit back
into DDR scheduling. The write path stages a full burst from the source tile
at AXI submission rather than modeling the AW/W/B channel handshakes beat by
beat. Per-bank compute read ports, crossbar contention, queue depths inside
MIG, DDR refresh insertion, and independent command/DQS timing calibration
remain open. Configured CL/CWL are 18/12 CK; MIG UI response delays remain
uncalibrated.

The existing connected block-13 RTL/MIG run recorded 178,086 UI cycles for
input DMA, 4,442,603 for block execution, 43,145 for output DMA, and
4,663,834 total. The 385,969 weight-wait cycles are inside the block count.
Use these numbers as calibration targets after extracting actual tile
descriptors and compute-only durations. The demonstration is a software
timing model, not RTL or FPGA measurement.
