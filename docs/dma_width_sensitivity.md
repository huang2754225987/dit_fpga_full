# DMA width and FIFO sensitivity model

## Scope

This experiment models the block-13 outer input and output DMA phases. It does not change RTL or model the weight transfers inside the block. The reference behavioral RTL/MIG counts are input 178,086 UI cycles, block 4,442,603, output 43,145, and transaction 4,663,834. The UI period is 3,332 ps. These are not board measurements.

The input transfers ten components totaling 340,992 bytes, or 5,328 AXI512 beats. An input FIFO stores complete beats, and one active beat is unpacked into 4 or 8 FP16 lanes per cycle. Every component begins with an empty FIFO and the next component starts after its predecessor completes. The FIFO can accept a new beat in the same cycle an older beat begins unpacking. A newly accepted beat can only be consumed on the next cycle. FIFO depth is four beats, in addition to the active beat register.

The output transfers 294,912 bytes, or 4,608 beats. This experiment separately tests 128-, 256-, and 512-bit local reads. A one-cycle RAM read is assumed fully pipelined at one segment per cycle; assembled 512-bit beats enter a four-entry output FIFO and then the AXI W channel. The output packer can retain one completed beat outside the FIFO. The model includes finite queues and backpressure, but does not model actual data values, AXI burst address/B response timing, bank conflicts, descriptor overhead, clock crossing, or MIG command scheduling.

The local read-width cases are assumptions about a redesigned result memory interface, not claims about its present ports. In particular, 128-bit pipelined reads already improve on the current READ/PACK alternating controller; 256/512 bits require correspondingly wider or banked storage. Four or eight independent FP16 input writes require suitable storage ports and conflict-free address mapping.

## Results

AXI period is a synthetic availability interval: period 1 permits one beat each UI cycle, period 4 permits one every four UI cycles, with instantaneous readiness at each opportunity. It is not a calibrated MIG traffic trace. Each input component starts at a fresh availability phase. The result assumes the input and output DMA phases are sequential around an unchanged reference block; it combines idealized DMA phases with the measured block and is not an end-to-end RTL prediction.

| Input lanes | Output local read width | AXI period | Input UI | Output UI | Combined UI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 128 bit | 1 | 42,634 | 18,434 | 4,503,671 |
| 4 | 256 bit | 1 | 42,634 | 9,218 | 4,494,455 |
| 4 | 512 bit | 1 | 42,634 | 4,610 | 4,489,847 |
| 8 | 128 bit | 1 | 21,322 | 18,434 | 4,482,359 |
| 8 | 256 bit | 1 | 21,322 | 9,218 | 4,473,143 |
| 8 | 512 bit | 1 | 21,322 | 4,610 | 4,468,535 |
| 4 | 256 bit | 4 | 42,634 | 18,433 | 4,503,670 |
| 8 | 256 bit | 4 | 21,322 | 18,433 | 4,482,358 |

When AXI accepts one beat per four cycles, widening the output read from 256 to 512 bits cannot raise sustained throughput in this example. Input backpressure and output FIFO stalls are reported in the generated CSV as counts of modeled unavailable admission opportunities and cycles with a completed pack waiting, respectively; neither is a measured RTL counter.

These idealized results omit setup, address handshakes and response latency. Do not interpret the difference from measured reference as a guaranteed saving or as a range of actual performance. The prior structural baseline of 175,824 input and 41,472 output cycles describes the current RTL state sequence without external stalls; the new cases describe a different, pipelined architecture.

## Reproduction and calibration

Build `dma_width_model` with CMake, run `dma_width_model --self-test`, then run `dma_width_model <report.csv>`. `scripts/run_cpp_latency_model.ps1` also generates `build/cpp_latency_msys2/dma_width_result.csv`. The small tests exercise transfer completion, FIFO depth and both forms of backpressure.

For realistic estimates, capture AXI AR/R and AW/W/B handshakes, burst boundaries, local input writes, output reads, and FIFO occupancy from the matching RTL/MIG configuration. Implement the widened/banked memory ports and FIFOs in RTL, verify bit ordering, address mapping, resets and backpressure with a scoreboard, and compare end-to-end simulation counts. Synthesis must establish memory resources, port feasibility and clock timing. No such RTL implementation, RTL simulation, synthesis or board measurement is claimed here.
