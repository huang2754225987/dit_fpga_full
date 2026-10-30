# C++ latency model execution and accuracy assessment

## Executed scope

The current C++17 sources were compiled with MSYS2 GCC 14.2.0 in Debug mode
on 2026-10-29. `scripts/run_cpp_latency_model.ps1` selects this compiler
explicitly, runs CTest and the examples, and writes results under
`build/cpp_latency_msys2/`. The older GCC 8.1 earlier on PATH fails to launch;
this is no longer a blocker because the MSYS2 compiler works.

All three CTest targets passed: the existing task/burst/cache tests, the DDR4
device tests, and the new integrated pipeline tests. The latter cover burst
splitting across 4 KiB, dependency order, shared-resource exclusion,
independent-resource overlap, tile lifetime and refill exclusion, shared
read-only weight retention, idle TDM phase, weight-wait subtraction, and
DDR read/write turnaround boundaries. These are software-model checks.

The final small pipeline example completed in 232 UI cycles: activation
load 0..38, weight load 38..62, illustrative GEMM 62..160, GELU 160..194,
and output DMA 194..232. It read 2,048 bytes and wrote 2,048 bytes in three
AXI bursts. The example pin replay accepted ACT=2, RD=3, WR=1, PRE=2, REF=1
with zero modeled timing rejections. It is a synthetic example, not a
captured MIG trace.

The measured phase ledger also returned 4,663,834 UI cycles. That result is
an arithmetic replay of durations supplied in its input, not independent
prediction or calibration evidence.

No new RTL/MIG simulation, synthesis, or physical FPGA measurement ran in
this assessment. The reference is the already completed behavioral
Questa/MIG run `block13_xread_guard_20261028_145709/system.log`, with a
3,332 ps UI period and host preload/readback excluded from the timed phases.

## Independent traffic characterization

`model_characterization` constructs traffic from the checked block-13 layout
snapshot and predicts it with the current generic DMA/MIG path. Unlike the
phase ledger, it does not use the reference phase duration to generate the
prediction. The compared numbers are not complete block inference results.

| Scope | Current prediction (UI) | Existing RTL/MIG reference (UI) | Signed error |
| --- | ---: | ---: | ---: |
| Input DMA, components 0..9, 340,992 bytes | 6,384 | 178,086 | -96.4152% |
| Output DMA, 294,912 bytes | 5,086 | 43,145 | -88.2118% |
| QKV weight, scale and bias load | 23,853 | 71,672 | -66.7192% |

The QKV case contains 972 separate 1,024-byte weight tiles plus two
6,912-byte vectors. Its comparator is QKV's recorded weight-wait counter,
so the reference includes descriptor/acknowledgement control absent from
the generic path. The output comparator excludes the synthetic one-cycle
producer used only to seed an already-resident output tile.

Machine-readable results are in
`build/cpp_latency_msys2/characterization.json`. The characterization source
documents its layout constants; regenerate that snapshot if the DDR layout
changes. Its `calibrated` flag remains false.

## Corrections made while executing

- Added the missing `task_graph.h` include to `axi_ddr_model.h`; the original
  test translation unit could not compile because `Cycle` was undefined.
- Protected compute inputs until execution finishes. Input storage is no
  longer freed at task start, so a DMA cannot overwrite a tile while it
  feeds compute. Retained weights support shared readers. DMA destinations
  reserve capacity at descriptor start, preventing capacity overcommit.
- Fixed optional TDM phase advancement during idle UI cycles.
- Recovered the actual board parameters from the XCI user fields and the
  generated IP. CL=18 and CWL=12; the separate MODELPARAM CL/CWL=0 entries
  must not be used as the physical latency.
- Corrected the arbitration provenance. The actual XCI user parameter is
  `C0.DDR4_AxiArbitrationScheme=RD_PRI_REG`, confirmed by the generated
  `ddr4_pl_axi_ddr4.sv` parameter and its unoverridden top-level instance.
  BUSIF metadata saying TDM does not override it. The current model now
  defaults to a read-priority approximation; RD_PRI_REG hold/QoS/starvation
  state is still unmodeled.
- Added configured tRTW=14 CK and tWTR_S/L=4/10 CK checks. The same-BG write
  history is retained even across commands to other bank groups. The
  synthetic pin example was updated to satisfy these configured timings.

## Largest accuracy gaps, in implementation order

### 1. Reproduce the connected block DMA's local states

The passing block-13 baseline uses `supra2_block_dma.sv`, not just the
streaming `supra2_axi_read_dma.sv`. The current C++ DMA approximates the
latter and omits the former's local unpack, transpose, and pack states.

| RTL path | Local behavior at AXI512 | Necessary C++ state |
| --- | --- | --- |
| FP16 input | One R beat, then 32 `S_FP_EMIT` cycles with RREADY low | RECEIVE -> EMIT(32) -> RECEIVE |
| INT8 weights | Receive one 16-beat, 1-KiB tile, then 32 transpose-emission cycles | TILE_RECEIVE -> TILE_EMIT(32) -> NEXT_TILE |
| FP32 scale/bias | One R beat, then 16 scalar emission cycles | RECEIVE -> EMIT(16) -> RECEIVE |
| Result output | Four 128-bit segments, each READ then PACK, followed by W handshake | READ/PACK repeated 4 times -> W -> next beat/B |

Ignoring all external DDR and setup stalls, these local states alone imply:

- Input: `(340992 / 64) * (1 + 32) = 175824` UI cycles, already 98.73% of
  the 178,086-cycle reference.
- Output: `(294912 / 64) * (4 * 2 + 1) = 41472` UI cycles, 96.12% of
  the 43,145-cycle reference.
- QKV load: `972 * (16 + 32) + (2 * 6912 / 64) * (1 + 16) = 50328`
  local service cycles before address, control, and DDR waiting.

These are component-cycle lower bounds, not calibrated predictions.
Implement the explicit states rather than fitting a single bandwidth or
applying a correction multiplier. Keep a selectable generic streaming-DMA
profile for architectural exploration.

### 2. Model compute control around the shared array

`supra2_gemm_engine.sv` has frontend start/feed/drain, tile clear, array feed,
accumulator wait, 32-row output drain, and done states. Its `feed_cycles` is
`K + 32 + 32 + 3`, already different from the current analytical `K + 66`.
The tile clear, accumulator wait, output drain, Hadamard/quantization, and
FP32 epilogue must also be accounted for at their real loop boundaries.

Preserve the current block controller's stage ordering and weight-load
request/acknowledgement boundary. QK and PV share the same physical 32x32
array as GEMMs. Softmax, LN, GELU, and residual resources should follow the
actually instantiated units and handshakes. Do not calibrate the existing
connected baseline to the compiler's proposed overlapped tile schedule;
that schedule has not replaced the arithmetic RTL loops.

Subtracting the 385,969 recorded weight-wait cycles from the 4,442,603-cycle
block interval leaves 4,056,634 cycles containing compute, SRAM and control
work. This is a diagnostic reference, not a shape-derived compute model.
Derive control timing and validate the residual by stage rather than
distributing aggregate measured cycles uniformly across tiles.

### 3. Connect finite AXI queues and exact handshake completion

The current ready-event list is unbounded. A slow consumer delays the C++
DMA but does not fill a finite MIG R-response queue and stall further DDR
reads. Add queue capacity, occupancy, RVALID stability and RREADY acceptance.
Model AW and W independently: a write response cannot precede accepted WLAST
and the controller's response rule. Preserve AXI ID/order and same-address
read/write hazards; the current request scheduler does not enforce the
complete MIG retirement rules.

Distinguish three events: data has arrived from DDR, data has been accepted
by DMA, and data has been committed to local storage. Their timestamps
cannot be collapsed when backpressure or scalar unpacking is active.

### 4. Refine the configured MIG controller and DDR rank

Use actual RD_PRI_REG direction hold, wait and starvation behavior; then
model finite controller group queues, row/column arbitration, Group FSM
issue restrictions, and the configured precharge policy. The current
per-beat round-robin/open-page controller is an approximation. Row-locality
statistics also need unambiguous definitions: current MIG `row_hits` counts
accepted column commands, while the device's `row_reuses` counts later
accesses within one open-row episode.

Add periodic refresh scheduling/debt and tRFC occupancy over a 4.66M-UI-cycle
interval. Check same-BG tCCD/tRRD histories, all turnaround rules, and
auto-precharge behavior using actual commands. Keep physical timing, MIG
control latency, AXI adapter latency and local DMA latency as separate
parameter sources. A configured CL/CWL is not a calibrated end-to-end
AXI first-response latency.

### 5. Replace whole-task buffer assumptions with actual SRAM accesses

The current lifetime fix protects data until compute ends but does not
simulate its individual reads. Add the actual memory bank/address map,
read latency, port width, read/write collisions, accumulator accesses, and
ping-pong ownership. Output reservations currently require separate space;
in-place storage needs an explicit alias/lifetime policy. A global byte
capacity or LRU cache cannot represent these constraints accurately.

## Required traces and calibration gates

Use one common UI origin with DDR CK timestamps related by the configured
4:1 ratio. For the known passing baseline, capture:

- AR/AW handshakes: address, LEN, SIZE, ID, phase, descriptor/tile ID.
- Every R/W/B handshake, RLAST/WLAST, response and ready/valid stall cycles.
- DMA state transitions, scalar/lane emit events, weight tile completion,
  `wload_req/wload_ack`, result-RAM reads, and local buffer writes.
- Compute start/done, stage/substate, array enable/clear and accumulator
  valid, frontend valid/ready, and streaming-unit handshakes.
- ACT/RD/WR/PRE/REF with BG/BA/row/column, mode-register CL/CWL and DQS
  data-window timing. Existing `+DDR_COMMAND_TRACE` only records commands
  after calibration; it does not capture initialization MRS or DQ windows.

Fit interface/queue constants from short microbenchmarks with repeated
same-row reads, row conflicts, alternating BGs, mixed directions, queue
pressure, and downstream pauses. Validate on held-out addresses, strides,
4-KiB boundaries and tile shapes including K=576/2304. Compare the first
divergent event, not only the final cycle count.

Suggested acceptance targets, not yet achieved: exact byte/beat/descriptor
counts and dependency/ownership order; no negative queues or overwritten
live tiles; per-burst and per-unit residual distributions reported with
maximum and percentile errors; each DMA phase within 5% and the complete
single-block interval within 3% on held-out runs. Report input/block/output
and weight-wait errors separately so opposite-signed errors cannot cancel.

Use `178086 + 4442603 + 43145 = 4663834` as the same-scope baseline. Weight
waiting is a subset of block execution and is never added again. Continue
to label comparisons as RTL/MIG behavioral simulation calibration, with
host setup, FPGA measurement and 50-step inference outside that claim.
