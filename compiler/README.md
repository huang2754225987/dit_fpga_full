# Supra2-IMG Compiler Bring-Up

## Connected block-13 stage program

`compiler/supra2/compile_block_program.py` emits a 17-word, 32-bit program
for the existing 16-stage connected block datapath:

```powershell
python -m compiler.supra2.compile_block_program
```

The output is `build/supra2_block_program/block13_stage_program.hex`, with
an annotated listing and JSON manifest beside it. Stage ISA v1 encodes
`opcode[31:28]`, `stage_id[27:23]`, and zero reserved bits `[22:0]`.
`RUN_STAGE=1` launches the named block stage when the previous stage has
completed; `END=15` closes the block after RES2. The RTL sequencer reads the
compiler's ROM, rejects invalid order/encoding, and feeds the stage command
port on `supra2_block_top`. The block's arithmetic, tile loops, online
attention, RAM arbitration, and descriptor-driven weight requests are still
performed by their existing RTL state machines. The stage program controls
the actual stage boundaries; this is not yet a full operator/tile ISA.

The board-MIG simulation runner regenerates and copies the program into its
run directory before compiling the new sequencer. The board bench sets
`USE_STAGE_PROGRAM=1` and separately counts `PROGRAM_WAIT` cycles. Run:

```powershell
scripts/run_supra2_block_ddr4_axi512_questa.ps1
```

Compare its measured `input_cycles + block_cycles + output_cycles` and
per-stage cycles with the fixed-FSM baseline recorded in `HANDOFF.md`.
The one-block timer excludes host DDR preload and readback; weight waits
are already included in the block phase. Check the final PASS marker and
golden error before interpreting any cycle delta. This measures a full
AXI512/MIG/DDR behavioral simulation at the generated 300.12 MHz UI clock,
not routed 400 MHz timing or board execution.

## 32x32 tile compiler and bank-aware schedule prototype

`compiler/supra2/compile_block13_tile_program.py` lowers the block-13 QLinear
shapes to 32-row macros with explicit N/K tile counts, emits one attention
macro per query tile/head, and constructs a dependency/resource schedule.
It writes a 128-bit issue image with two source banks, two destination banks,
stage descriptors and dependency/completion token IDs, annotated JSON, a bank allocation/lifetime
plan, and a resource/bank schedule table:

```powershell
python -m compiler.supra2.compile_block13_tile_program
scripts/run_supra2_compiled_tile_schedule.ps1
```

The current plan uses 32x576 FP16 row tiles, 32x32 INT8 array tiles, 18 K
tiles for K=576 and 72 for K=2304. The memory plan assigns short feature,
projection, and AdaLN tiles to BRAM; persistent self-Q/K/V and MLP ping-pong
storage to URAM; score scratch to distributed RAM; and the current/next
1-KiB weight tiles plus scale/bias to LUTRAM. The planned allocation is
1,703,424 bytes total (488,448 BRAM, 1,179,648 URAM, 32,768 distributed,
2,560 LUTRAM) under a configurable 1,835,008-byte planning budget. These are
compiler targets; Vivado has not confirmed the physical mapping.

The scheduler is anchored to the passing block-13 MIG simulation's measured
stage cycles and weight-wait subsets. It currently emits 258 scheduled macro
tasks and a 259-word image including END. XSim RTL replay of the instruction
fields and the 1,160 resource/bank reservations passes with zero dependency
or exclusivity errors; its modeled makespan is 4,644,728 cycles. That value
is a compiler schedule estimate, not measured datapath latency. The replay
checks the generated schedule and instruction encoding; it does not execute
the QLinear/attention arithmetic. The existing connected top still runs the
stage-program FSM, so applying this overlap schedule to actual hardware
requires replacing its internal stage loops with ready/done tile commands,
dependency-token handling and the planned bank buffers. Keep the full DDR golden run as
the data-correctness gate and the schedule replay as the compiler/control
feasibility gate.

For a full compiler-controlled datapath, the next ISA revision needs typed
descriptors for DDR-to-bank transfers, QLinear tile execution, attention
query/key tile loops with online softmax state, residual and normalization
operations, output transfer, and barriers. The Python backend should lower
the model graph into 256-token blocks, assign 32-row and 32-column tiles,
allocate the packed on-chip banks, attach weight/scale/bias descriptors,
schedule dependencies and DMA overlap, then emit a versioned instruction
image plus a descriptor table. The RTL must expose ready/done and bank
ownership for each operator before those finer commands can replace the
current internal FSM. Extend the same connected DDR/golden bench to each
incremental replacement so compiler overhead and any overlap are measured
from actual handshakes rather than summed operator estimates.

## Supra2 FlashAttention instruction emission

`compiler/supra2/flash_attention.py` specializes the existing microcode
FlashAttention emitter for this model's non-causal 256-token, 9-head,
64-feature workload. It emits each head as eight 32-query blocks. Each block
streams eight 32-key tiles through QK GEMM, online-softmax state updates, and
PV GEMM/rescaling; K/V tile loader instructions are supplied by the caller.
The current emitter assumes Q/K/V and output storage are already assigned in
the documented L1 pages; it does not yet generate a complete Supra2 DDR-to-L1
loading program or execute the instruction stream on a connected Supra2 top.

Compiler contract check:

```powershell
python -m unittest tests.test_supra2_flash_attention_compiler -v
```

The RTL numerical regression for the online-softmax/PV datapath is separate:
`scripts/run_attention_online_pv_batch.ps1`.

The active source of truth is `supra2-img-hw-export.tar.gz`, extracted under
`build/supra2_hw_export/export/`. Its manifest describes 100 INT8 QLinear
layers, 37 FP16 passthrough tensors, unit/block/step goldens, and a block-skip
trace. The package's target operation is
`acc_int32 * s_t * weight_scale + bias`; it is not the earlier FP16 checkpoint
projection prototype.

## Package weight-image export

Pack the archive tensors into byte-addressable, zero-padded 32x32 tiles:

```powershell
python -m compiler.supra2.export_quantized_weights
python -m compiler.supra2.export_quantized_weights --verify-only
```

The output is under `build/supra2_quant_export/`:

- `weights_int8_tiled.bin`: signed INT8 `[K,N]` row-major tile bytes, ordered
  by output tile then K tile; each 32x32 tile is 1 KiB and projection starts
  are 4-KiB aligned.
- `weight_scales_fp32.bin` and `weight_biases_fp32.bin`: per-output-channel
  FP32 vectors, padded to a 32-lane boundary and 64-byte aligned.
- `passthrough_fp16.bin`: the package's 37 non-quantized tensors, stored as
  FP16 as specified by its README.
- `weight_image_manifest.json`: dimensions, padding, offsets, source mapping,
  file sizes, and SHA256 hashes.

The packer verifies every INT8 matrix, scale, bias, and FP16 tensor by
round-tripping against the NPZ files. This generic weight image is not yet
consumed by the full-model top; the existing block-13 system uses a separate
seven-stage payload layout.

## Full-model DDR image and block descriptors

Build the runtime image for the 14-block data path after exporting the weight
image:

```powershell
python -m compiler.supra2.prepare_dit_ddr_image
python -m compiler.supra2.prepare_dit_ddr_image --verify-only
```

The default output is `build/supra2_dit_ddr_image/`. The 64-byte-aligned
payload contains all 100 tiled INT8 QLinear matrices, their FP32 scales and
biases, all 37 FP16 passthrough tensors, and runtime `z0`, conditional and
unconditional text contexts, masks, the 50-step timestep schedule, and the
50x14 skip trace. Weight components start on 1-KiB boundaries so a DMA can
fetch each 32x32 INT8 tile directly. The manifest records byte offsets and
SHA256 hashes; `supra2_dit_ddr_layout.sv` emits the same offsets plus the
14x7 block-stage-to-QLinear index table, per-stage weight/scale/bias offsets
and matrix/tile dimensions, N/K tile byte strides, explicit weight/scale/bias
byte counts and 512-bit beat counts, per-block AdaLN offsets,
named IDs for timestep, position, and final-head passthrough tensors, model
geometry constants, and `BLOCK_SKIP_MASK[0:49]`. Each 1-KiB weight tile maps to
16 512-bit beats; an exhaustive descriptor check confirms every tile remains
inside one 4-KiB page. In each 14-bit mask, bit 1 means skip and bit position
is the block id, so the package skip schedule can drive a step/block controller
without reinterpreting the byte-oriented DDR trace. Offsets are relative to
DDR base `0x00100000`.

`rtl/supra2/supra2_dit_runtime_controller.sv` consumes the generated package
and sequences one shared context projection followed by all 50 step-prep,
skip-aware block, final-head, CFG-combine, and Euler handshakes. It exposes the
active block-stage byte addresses, transfer lengths, beat counts, tile strides,
and AdaLN addresses. Run its isolated Questa schedule/descriptor check with
`scripts/run_supra2_dit_runtime_controller_questa.ps1`; the self-check expects
50 steps and 414 executed block events. Its engine handshakes are still abstract
and are not wired to the existing block-13 DMA/top or real compute datapath, so
full-model inference, whole-top synthesis, P&R, and bitstream validation remain
integration work.

The standalone `rtl/supra2/supra2_axi_read_dma.sv` is a single-outstanding
512-bit AXI4 read engine with stream backpressure, 64-beat burst limiting,
4-KiB boundary splitting, response/ID/`RLAST` checks, and descriptor validation.
Run its isolated Questa test with
`scripts/run_supra2_axi_read_dma_questa.ps1`. The test checks a 1+3-beat
boundary split, a 16-beat/1-KiB weight-tile read, stable output under stalls,
and rejection of an unaligned descriptor. This reader is not yet instantiated
under the runtime controller or connected to MIG or a compute sink.

For a module-only hardware check, run
`vivado -mode batch -source scripts/synth_supra2_axi_read_dma_ooc.tcl`. The
script targets ZCU106 `xczu7ev-ffvc1156-2-e`, synthesizes and routes this reader
out of context, and writes reports/checkpoint under
`build/supra2_axi_read_dma/vivado_ooc/`. OOC timing omits top-level clock
insertion and external I/O constraints; it is not a board or full-system
frequency result.

## Golden-driven tests

Run the archive/quantization contracts and earlier projection contracts:

```powershell
python -m unittest tests.test_supra2_archive_contract -v
python -m unittest tests.test_fpga_projection_contract -v
python -m unittest tests.test_supra2_activation_quant -v
python -m unittest tests.test_supra2_qlinear_epilogue -v
python -m unittest tests.test_supra2_activation_quantizer -v
```

`compiler.supra2.activation_quant` implements the reference block-diagonal
Sylvester Hadamard transform and per-token symmetric INT8 quantizer. Its
synthetic contracts test butterfly ordering, normalized output, ties-to-even,
and zero rows. The archive contract checks Hadamard output against the stored
FP16 tensor within its storage-rounding error.

Important golden limitation: `x_raw` and `x_rot` are stored as FP16, while
`x_int8`/`s_t` were generated before those tensors were rounded for NPZ storage.
The RTL Hadamard widens FP16 `x_raw` to FP32, performs the butterfly and power-
of-two normalization in FP32, and rounds the result to FP16 before dynamic
quantization. The full 256-row QKV test matches all rotated FP16 and INT8
values generated from the stored `x_raw`, then runs all array tiles and the
epilogue: 322,784 testbench cycles, 0 software-chain mismatches, and 0.3872%
maximum relative error against package `y_out` (0.45% gate). It is not bitwise
identical to the package's pre-storage `x_int8`/`y_out`, which were generated
before `x_raw` was rounded into the NPZ archive.

Generate one complete-K golden tile for the independent systolic-array RTL
test:

```powershell
python -m compiler.supra2.prepare_int8_golden_tile
```

The reproducible source list and ModelSim commands are in
`rtl/supra2/README.md`. A combined 32x32 QLinear tile now streams complete-K
INT8 activations/weights through the systolic array, checks every INT32 result
exactly, and feeds that array output into the FP32 scale/bias-to-FP16 epilogue.
Both QKV (K=576) and MLP.2 (K=2304) pass at initial and offset row/channel
tiles. This still bypasses RTL Hadamard/activation quantization and DMA/RF
loading; it is not a full projection, block, or denoiser test.

For a full 256-row QKV projection, create a complete archive fixture and use
the dedicated 32x32 array/epilogue testbench:

```powershell
python -m compiler.supra2.prepare_qlinear_projection
python -m unittest tests.test_supra2_qlinear_projection -v
```

The fixture builder verifies all 256x1728 INT32 golden results against the
archive weights before writing simulator inputs. RTL commands and limitations
are documented in `rtl/supra2/README.md`; this projection test consumes the
package's precomputed INT8 activations and does not include the transform or
memory subsystem.

The token-level integrated RTL fixture starts from archive `x_raw` (FP16),
applies the FP32-stage/final-FP16 Hadamard emulator used by the current RTL,
then derives INT8, INT32, and FP16 expected values. The RTL file list now includes
`hadamard_transform.sv`; four QKV/MLP.2 row/channel cases pass through the
Hadamard, quantizer, array, and epilogue. These are selected tiles only. Run
the software fixture contracts with:

```powershell
python -m unittest tests.test_supra2_hadamard tests.test_supra2_qlinear_token -v
```

The archive also supplies `z0`, all 50 `v_per_step` tensors, and `z_final`.
`prepare_euler_replay.py` exports a golden-velocity fixture without executing
the denoiser. `rtl/supra2/euler_update.sv` replays the 50 Euler updates in
6,400 lane-beat cycles, matches the FP16 software replay bit-for-bit, and has
0.1137% relL2 against package `z_final`. This checks the sampler update
datapath only; it does not establish FPGA denoiser throughput or complete
model inference. Run the RTL replay using `scripts/run_euler_replay.ps1`.

## Previous FP16 baseline

`export_fp16_weights.py`, `prepare_projection_smoke.py`, and
`verify_projection_smoke.py` belong to the initial local-checkpoint bring-up.
The `x_embed` smoke used real FP16 checkpoint weights but deterministic
synthetic activations. It remains useful for generic DMA/GEMM checks, but it is
not the archive's INT8 model path and must not be counted as end-to-end Supra2
validation.

### 512-bit DDR read path (control-plane integration)

`rtl/supra2/supra2_axi_read_dma.sv` transfers 64 bytes per AXI beat, with
4-KiB-safe bursts and consumer backpressure. `supra2_dit_runtime_dma.sv` joins
it to the generated per-stage descriptors and the 50-step skip-aware runtime
controller. The controller time-multiplexes the 256-row compute engine over
both CFG branches, then issues one CFG combine and Euler handshake per step.
Weight tile requests select one 1-KiB tile (16 beats); scale/bias
tile requests select 128 bytes (2 beats). Re-run the shell integration test
with `scripts/run_supra2_dit_runtime_dma_questa.ps1` after generating the DDR
layout package.

The Questa test completes both CFG branches over the 50-step schedule (828
block starts) and checks one descriptor-derived weight-tile transfer against
an AXI memory model.
The compute engines are abstract handshakes and the memory model returns
synthetic data: this is control/DMA interface validation, not full inference,
MIG validation, DDR bandwidth measurement, or board timing. The standalone
reader's OOC synthesis and timing results are recorded in `HANDOFF.md`.

`rtl/supra2/supra2_dit_compute_bridge.sv` now wires that runtime controller
and DMA to `supra2_block_top`, including the stage loader's weight/scalar
write ports and block completion handshake. Each active block has a
`block_prepare_start` / `block_prepare_done` window before the compute launch;
the external preparation engine supplies x/context/AdaLN data through the
load port. `scripts/run_supra2_dit_compute_bridge_smoke_questa.ps1` compiles
and elaborates this full module graph. The smoke test does not launch the
compute block or validate its data. The bridge currently runs one 256-row CFG
branch at a time; external data preparation, final head, CFG arithmetic, and
Euler update units are still required for a full sample.

`rtl/supra2/supra2_dit_weight_stage_loader.sv` implements the next boundary:
it walks every N/K weight tile for a requested stage, turns each 64-byte beat
into two 32-lane engine writes, then streams each 128-byte scale/bias tile into
the GEMM scalar ports. Its block-0 stage-1 regression checks all 18x18 weight
tiles, scalar vectors, descriptor ordering, and 11,520 engine writes with
synthetic beat contents. Run it with
`scripts/run_supra2_dit_weight_stage_loader_questa.ps1`. This validates the
loader-to-engine protocol. The loader is now instantiated in
`supra2_dit_compute_bridge`; only module-graph elaboration has been checked at
that integration level so far.

`supra2_block_top` now emits a one-cycle completion and returns from `ST_FIN`
to `ST_IDLE`, allowing one physical block engine to be reused by the outer
block/step controller. `scripts/run_supra2_block_top_rearm_questa.ps1` runs the
residual-stage FSM twice and passes the re-arm check. It validates controller
reuse only, not repeated-block numerical results.

## Remaining compiler/runtime work

The compiler does not yet emit or run the complete denoiser. Block-0
self-attention now has a full 256-query RTL chain through LayerNorm/AdaLN,
QKV, QK, 32-query online softmax/PV, INT8 output projection, and gated
residual. The batched PV result matches the single-row fused implementation
bit-for-bit while reducing its ModelSim busy cycles from 4,603,401 to 922,824.
The archive conditional text context also passes the 128×768 `ctx_proj` RTL
projection in 94,672 cycles and agrees with the block context golden within
0.4941% relative-to-peak error (`scripts/run_context_projection.ps1`).
Block-0 now has a complete module-bridged RTL path through self-attention,
masked cross-attention, and MLP/GELU to its final residual. The archive context
mask has 15 valid tokens; packing to one 32-key tile reduces QK/PV to 9,724 /
254,520 cycles. The 147,456-element RTL block output passes against archived
`b0.y` with 0.521% relative-to-peak error and 0.294% relative L2
(`scripts/run_cross_attention_regression.ps1`).
Stage boundaries still use files written by RTL testbenches; the timestep
AdaLN projection remains CPU-precomputed, and operators are not connected
through a Supra2 command/RF/DMA top. The AdaLN input projection is still
CPU-precomputed; no full denoiser comparison has passed. Remaining work
includes CFG batch
orchestration, cross-KV caching, the supplied block-skip trace, and connection
of FPGA-generated velocities to the verified Euler update. T5 context
preparation and final VAE decoding may follow the package's host partition; no
Transformer forward or GPU run is needed for the RTL golden tests.
