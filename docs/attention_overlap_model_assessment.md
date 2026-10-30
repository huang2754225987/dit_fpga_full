# Single-array attention overlap model

## Scope and reference

This is a sensitivity study for self-attention in block 13, with 256 tokens and one 32 x 32 systolic array shared by QK and PV. The reference is the existing behavioral RTL/MIG simulation, not an FPGA board measurement or a complete 50-step inference. The UI clock period is 3,332 ps.

| Reference stage | UI cycles |
| --- | ---: |
| Input DMA | 178,086 |
| Block | 4,442,603 |
| Self-attention inside block | 1,131,363 |
| Output DMA | 43,145 |
| Input + block + output | 4,663,834 |

The weight wait is already included in block cycles and must not be added again. The model does not predict DDR traffic or DMA improvements for this experiment.

## Baseline decomposition

There are 9 heads, 8 query tiles per head and 8 key tiles per query tile, giving 576 QK/PV tile pairs. Each QK tile has M=32, N=32, K=64. The following are derived from the current state-machine structure and are not separate measured stage counters:

| Component | Cycles |
| --- | ---: |
| V loading and output streaming, held unchanged | 294,912 total |
| Q/K preparation | 128 per tile |
| QK array clear and feed | 132 per tile |
| QK score drain | 32 per tile |
| Two PV array passes | 206 per tile |
| Two PV merge passes | 64 per tile |
| Known per-tile total | 562 per tile; 323,712 total |
| Remaining unmeasured baseline time | 512,739 total |

The residual is 1,131,363 - 294,912 - 323,712. It includes softmax and other state transitions, handshakes, waiting and finalization. Its internal breakdown is not measured. The baseline total is supplied to the model, so matching it is not independent validation.

## Modeled hardware schedule

The model constructs a task graph for each query group of eight key tiles. QK preparation precedes QK array execution, then score drain and online softmax. PV runs on the same array after softmax, followed by PV merge. The online softmax state is ordered across key tiles. QK and PV cannot occupy the array at the same time; other stages can overlap only under the independent-port and separate-control assumptions below. PV receives priority when both array tasks become eligible together.

The graph reserves two score slots and two softmax result slots. A score slot is not reused until its consumer has finished; a softmax result slot is not reused until PV merge has consumed it. QK snapshot ownership and input preparation dependencies prevent the next tile from overwriting in-flight data. PV operations are ordered conservatively through the preceding merge. The remaining residual is added serially, and V loading and output streaming remain unchanged. No cross-head or cross-query-group overlap is modeled.

The model assumes one UI cycle to capture each full 32 x 32 INT32 array snapshot. Two PV half snapshots need about 8 KiB; adding a second score tile and second softmax result tile requires about another 4 KiB, for roughly 12 KiB of additional logical storage before metadata and other state. The 32,768-bit snapshot path, RAM port availability, fanout and timing are unverified. The current RTL requires controller, buffer and arbitration changes to realize this schedule; the model does not establish that those changes will meet timing.

For the 50% scenario, the first group trace illustrates overlap: tile 0 softmax runs from cycle 293 to 738 while tile 1 uses the array from 389 to 522. Tile 0 PV occupies the array from 783 to 991, followed by tile 3 QK from 991 to 1124 while tile 0 merge proceeds. These are model cycles, not RTL waveforms. The trace excludes fixed V/output time and the serial residual.

## Sensitivity, not a calibrated prediction

The scenarios assume that the stated fraction of the unmeasured 512,739-cycle residual can be modeled as an independent online-softmax stage; the rest is kept serial. These fractions are hypothetical, not measured softmax shares. Reported block and transaction totals replace only self-attention and hold all other reference stages constant.

| Assumed softmax share | Self-attention UI cycles | Self reduction | Block UI cycles | Transaction UI cycles |
| ---: | ---: | ---: | ---: | ---: |
| 25% | 903,843 | 20.110% | 4,215,083 | 4,436,314 |
| 50% | 848,331 | 25.017% | 4,159,571 | 4,380,802 |
| 75% | 848,331 | 25.017% | 4,159,571 | 4,380,802 |
| 90% | 848,331 | 25.017% | 4,159,571 | 4,380,802 |

The identical results at 50%, 75% and 90% are a consequence of this particular task graph and residual allocation. They do not prove the real hardware has reached a performance limit. The scenarios are not confidence intervals or guaranteed bounds.

The graph reserves 196,416 array cycles, including three assumed snapshot cycles per tile. This reservation is not useful-MAC utilization: array feed, clear, snapshot and inactive lanes have different behavior. Existing inferred QK/PV clear-and-feed time is 194,688 cycles, about 17.2% of the reference self-attention time, but neither figure measures actual PE utilization.

## Reproduction and next calibration

Build and run with `scripts/run_cpp_latency_model.ps1`, or run `attention_overlap_model.exe --report <json> --trace <csv>` from the CMake build. The trace contains the first query group for the 50% scenario. `attention_overlap_tests` checks graph dependency and resource exclusivity; it does not validate the assumed hardware timing.

To calibrate, instrument the same block-13 RTL/MIG test with per-state and handshake counters for QK preparation, array enable, score drain, softmax, PV feed and merge, V/output streaming, and buffer/port stalls. Capture a common UI-clock timeline and separate overlapping time from serial time before fitting the residual. Compare the C++ trace with matched RTL traces, then validate on other heads or workloads not used for fitting. Implement and numerically test online-softmax state and snapshot ownership in RTL, and check synthesis resources and timing before claiming an achievable speedup.
