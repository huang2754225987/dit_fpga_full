# Supra2-IMG FPGA Bring-Up

This directory is an isolated bring-up area for `SupraLabs/Supra2-IMG`.
It does not copy model checkpoints into the repository. Place downloaded
artifacts under `supra2_img_fpga/model/` locally; that path is intentionally
kept out of source control.

See [`docs/README.md`](docs/README.md) for the project directory map and where
generated diagrams and research notes are stored.

The supplied FPGA data package `supra2-img-hw-export.tar.gz` is extracted
under `build/supra2_hw_export/export/`. Its README and manifest are the active
hardware contract: 100 INT8 QLinear layers, 37 FP16 passthrough tensors, unit,
block, and step goldens, plus a block-skip trace. FPGA golden validation does
not require a Transformer forward or GPU run.

## Model facts

Supra2-IMG is a 256x256 text-to-image rectified-flow DiT, not an image
classifier and not an LLM/MoE model.

| Item | Value |
| --- | --- |
| Parameters | about 104.1M |
| Latent | 4 x 32 x 32 |
| Patch | 2 x 2, 256 sequence tokens |
| DiT width/depth | 576 / 14 |
| Attention | 9 heads x 64 |
| Text context | Flan-T5-Base, 128 x 768 |
| VAE | `stabilityai/sd-vae-ft-mse` |
| Sampling | 50 Euler-flow steps, CFG=3.0 in the supplied golden pack |

The supplied pack includes `ctx_all`, the initial latent, per-step velocity
goldens, final latent, and host-side VAE image. Its execution recipe batches
conditional/unconditional rows as M=512, caches cross-attention K/V per
prompt, and skips blocks according to a 50x14 trace (40.857% skip ratio).
T5 context preparation and VAE decode are outside the FPGA DiT path.

## Golden-driven FPGA validation

Use the supplied NPZ goldens rather than running a Transformer/model forward.
The package's host partition provides text context and the final VAE image;
FPGA validation starts at `z0`/`ctx_all` and compares block and per-step
outputs against the supplied tensors.

## Current FPGA assessment

The isolated RTL tree has been pruned of the generic Qwen/MoE top and
model-inapplicable Top-K/expert, low-bit dequantization, RoPE, and RMSNorm
datapaths. It retains the 32x32 INT8/FP16 array, model operators, SiLU for the
timestep MLP, and reusable DMA/memory/control primitives. A dedicated Supra2 system top is not
implemented yet. Tested model-specific RTL slices connect the standalone
activation quantizer, signed INT8 array, and FP32 scale/bias epilogue at
token/tile level; the full control/RF/DMA path is still not connected.

The package-driven compiler tiles and round-trips all 100 INT8 weight
matrices, scales/biases, and 37 FP16 passthrough tensors. The FPGA RTL has a
tested token-level FP16 block-Hadamard -> dynamic INT8 quantizer -> signed
INT8 array -> FP32 scale/bias -> FP16 epilogue slice, plus a full 256-row QKV
array/epilogue test using archive activations. The 256-row raw-input frontend
now matches all generated FP16 Hadamard and INT8 activation values; the full
QKV test passes its 0.45% package-output relative-error gate. Remaining work
now includes a 256-row LayerNorm/AdaLN modulation RTL unit whose output feeds
the full QKV projection regression at 0.4425% maximum package-output relative
error. The AdaLN timestep linear remains host-precomputed in this test; the
deployment schedule must compute/broadcast its parameters. Block-0 QKV→QK
array→online softmax→PV array→self-attention projection→gated residual
components now form a fixture-bridged numerical chain. A dedicated online
softmax→PV wrapper consumes each 32-key score tile directly, caches V once per
head, and does not materialize the 256×256 probability matrix. Its 32-query
microbatch variant uses all rows of the 32×32 PV array; full 256-query,
9-head RTL simulation passes at 922,824 busy cycles and 0.242%
relative-to-peak error. Its RTL output feeds the 256×576 INT8
self-attention projection, which passes exact activation/accumulator/software
chain checks. Block-0 now continues through archive-masked cross-attention,
MLP.0, GELU(tanh), MLP.2, and final gated residual. The latest XSim rerun
starts from archive `z0 -> x_embed -> position-add` RTL output and passes all
block-0 stages; its final 256x576 tensor compares against archive `b0.y` at
0.4627% relative-to-peak, RMSE=0.03462, and 0.3707% relative-L2 (2% gate).
Summing the independent block-0 operator counters gives 4,902,792 cycles, or
about 24.51 ms at an assumed 200 MHz; this is component-cycle accounting, not
a connected-top latency measurement. Stage tensors still cross through files,
timestep AdaLN parameters remain host-precomputed, and there is no connected
DMA/RF top. CFG=512, dynamic skip scheduling, and a 50-step run driven by
RTL-generated velocities remain unverified; no MIG/board timing is claimed.

## Remaining implementation order

```text
package weight image (100 QLinear + passthrough tensors)
    -> INT8 Hadamard/activation quantizer + command/RF/DMA integration
    -> QLinear scale/bias epilogue and golden unit-layer comparisons
    -> all 14 block numerical tests and dynamic 50x14 scheduling
    -> CFG=3 batch-512 orchestration, K/V cache, skip-trace scheduler
    -> connect generated velocities to the verified Euler update
    -> 50-step denoiser inference from z0 to z_final and FPGA cycle profile
```

For the supplied quantized package, serialize and verify all hardware weights
without running model inference:

```powershell
python -m compiler.supra2.export_quantized_weights
python -m compiler.supra2.export_quantized_weights --verify-only
```

The legacy `export_fp16_weights.py` and projection planner remain for the
earlier standalone checkpoint experiment; they are not the active quantized
package path. Hardware and compiler sources are under `rtl/`, `compiler/`,
and `include/`; the active compiler workflow is in `compiler/README.md`. For
the model-specific source boundary and 256-token optimization/compute-roofline
analysis, see [rtl/supra2/README.md](rtl/supra2/README.md).

## Current status

- Archive extracted; package contract checks 100 INT8 and 37 FP16 tensors.
- Packed image round-trip: all 100 QLinear matrices, scales/biases, and 37
  passthrough tensors pass verification.
- Python regression suite: 59 tests pass, including archive contracts,
  full-projection fixture contracts, and final-head unpatchify verification.
  Several suite modules (`test_reference`, `test_supra2_final_head`,
  `test_supra2_gelu_tanh`, `test_supra2_layernorm_fixture`,
  `test_supra2_position_add`, and two cases in
  `test_fpga_projection_contract`) import torch; run the suite with a Python
  that has torch installed. In a torch-less interpreter only 44 tests load
  and those 7 report `ModuleNotFoundError` import errors rather than running.
- INT8 systolic RTL golden: K=576 and K=2304 exact accumulator comparisons
  pass with zero simulator errors/warnings.
- Token-level and full 256-row Hadamard -> quantizer paths match generated
  expectations exactly. Full QKV M=256,N=1728,K=576 RTL projection passes in
  322,784 testbench cycles: 0 Hadamard mismatches, 0 quantizer mismatches,
  0 software-chain mismatches, and 0.3872% max relative error against package
  y_out (gate: 0.45%). Full MLP.2 M=256,N=576,K=2304,H=256 also passes in
  545,696 testbench cycles with 0 Hadamard/quantizer/software-chain mismatches
  and 0.3906% max relative error. QKV has 429,815 and MLP.2 has 142,197 FP16
  bit-pattern differences from package y_out, reflecting its pre-storage
  precision; both are tolerance passes, not bitwise equality. Neither bench
  includes DMA/DDR.
- Quantized x_embed now handles the model's partial K=16/H=16 input by
  zero-padding to the 32-lane beat. From archive z0 patchify through all 256
  rows and 576 outputs, the RTL projection passes with zero activation,
  accumulator, and output mismatches in 15,264 testbench cycles. This is a
  generated software golden from archived z0/weights, not an archived
  x_embed-layer golden.
- A dedicated streaming FP16 position-embedding adder, fixture compiler, and
  full-tensor testbench now define the missing `x_embed -> pos_embed` hardware
  boundary. Its full 256x576 XSim standalone test now passes bit-exactly in
  147,456 cycles, including subnormal position values; a cancellation vector
  also checks the shared FP16 normalization path. Reproduce the operator test
  with `scripts/run_position_add_xsim.ps1 -InputFile
  build/supra2_x_embed_qlinear/reference_output_fp16.hex`. This uses the
  compiler's x_embed reference tensor, not an RTL-produced x_embed result. The
  test uses a local GCC link workaround for the broken bundled compiler. An
  earlier generic 256-token QLinear projection testbench hit an internal
  `simulator-43-4100` elaboration error; the dedicated z0 frontend runner
  `scripts/run_z0_frontend_xsim.ps1` (see the 2026-10-19 XSim continuation
  section below) avoids it and produces the archive-driven RTL x_embed output
  bit-exactly, so the fresh block chain starts from archive `z0` through the
  RTL x_embed and position-add stages, not from archived `b0.x`.
  `scripts/run_z0_branch_chain.ps1` now provides the intended RTL-only serial
  run order from archive z0 through x_embed, position embedding, and blocks;
  as a single end-to-end script it remains unverified, since the recorded
  14-block chain ran stage by stage with separate invocations.
- Euler update RTL replays all 50 archived velocity tensors over the 4,096
  latent elements in 6,400 stream cycles. It matches the FP16 software replay
  bit-for-bit and reaches relL2=0.1137% against archived z_final (3,005 FP16
  bit differences); this validates the sampling update only, not FPGA-generated
  denoiser velocities.
- Full 256-query × 9-head × 256-key online softmax→PV simulation passes in
  922,824 busy cycles with 0.0048828 maximum absolute error and 0.2418%
  relative-to-peak error. This is 79.95% fewer cycles than the single-query
  fused baseline (4,603,401 cycles); both outputs are bit-identical. V is
  loaded once per head; normalized probabilities are never written to an
  intermediate tensor. The emitted 256×576 RTL output then feeds the INT8
  self-attention projection, which passes in 135,584 cycles with zero
  frontend/software-chain mismatches and zero RTL-vs-generated output
  mismatches. These remain separate ModelSim module runs bridged by actual
  QKV/QK/PV files, not an integrated DMA/DDR top.
- The 128×768 conditional T5 context passes through the full `ctx_proj` INT8
  RTL projection in 94,672 cycles with zero frontend, accumulator, or
  software-chain mismatches. Its output is within 0.4941% relative-to-peak
  error of archived `b0.ctx` (0.6% gate), validating the cross-attention
  context tensor boundary without running a model forward.
  Applying archive `mask_all[0]` shows only 15 of the 128 context tokens are
  valid. The cross path now packs those keys into one 32-key tile, projects
  M=32,N=1152 KV (28,648 cycles), and runs 256×32 QK in 9,724 cycles plus
  32-query online PV in 254,520 cycles. Against the previous unmasked padded
  run, QK falls 75.0% and PV falls 52.9%; PV relative-to-peak error is 0.487%.
  Cross Q (135,584 cycles), output projection (135,584), and both residuals
  also pass. The complete block-0 chain from actual RTL self-attention output
  through cross-attention, MLP.0, GELU(tanh), MLP.2, and final residual was
  originally measured at 0.521% relative-to-peak and 0.294% relative-L2.
  A later all-XSim rerun now compares all 147,456 elements against archive
  `b0.y` at max_abs=1.0, 0.4627% relative-to-peak, RMSE=0.03462, and
  relative-L2=0.3707%. This is
  an end-to-end block-0 component chain with RTL-generated file bridges, not a
  single DMA/DDR top or full denoiser run.
  The four selected token-chain runs take
  811 cycles for K=576 and 3,169 for K=2304. These are not block/image latency.
- Block 7 now runs the same separate-RTL-module chain from archived `b7.x`
  through self-attention, masked cross-attention, MLP.0, GELU(tanh), MLP.2,
  and final gated residual. The final 147,456-element tensor passes against
  `b7.y`: max_abs=1.25, relative-to-peak=0.3685%, RMSE=0.06806, and
  relative-L2=0.3448% (2% gate). Self QKV passes in 322,784 cycles; online
  self PV takes 922,824 cycles; masked cross Q/KV take 135,584/28,648 cycles,
  cross QK/PV take 9,724/254,520 cycles; MLP.0/GELU/MLP.2 take
  416,384/589,824/545,696 cycles. The QK checker uses a 0.1% relative-to-peak
  gate to accommodate block-dependent score magnitude; block-7 self/cross QK
  measured 0.0390%/0.0546%. The block-specific runner is
  `scripts/run_layernorm_qkv_regression.ps1 -Block 7` followed by
  `scripts/run_cross_attention_regression.ps1 -Block 7`. This proves a
  second block's operator chain, not CFG=512, inter-block execution, or a
  connected top/DDR schedule.
- A resumable RTL block-chain runner now connects actual output files
  sequentially. Blocks 1 through 7 have been executed from the preceding
  block's RTL output (0→1→2→3→4→5→6→7); each block's QLinear frontend, integer
  accumulators/software epilogues, attention, GELU, and residual boundaries
  pass their local checks. The archive contains no `b1.y` through `b6.y`,
  so blocks 1 through 6 lack package-level output goldens and are validated
  by per-operator references only. Blocks 4 through 6 each passed complete M=256
  component chains: self-QKV 322,784 cycles, self online-PV 922,824 cycles,
  MLP.0 416,384 cycles, and MLP.2 545,696 cycles (plus other separately
  simulated stages). This is still one CFG branch with file-based module
  bridges and host-computed timestep AdaLN—not the supplied M=512 CFG
  schedule or a single-top, 50-step inference. Importantly, block 7 was fed
  block 6's RTL output and then compared to archived `b7.y`: max_abs=3.5,
  rel-peak=1.032% (2% gate), RMSE=0.1589, rel-L2=0.805%. Blocks 8 and 9
  have since completed sequentially from actual predecessor RTL outputs
  (7→8→9); each final tensor is 884,736 bytes. The archive has no `b8.y` or
  `b9.y`, so their evidence is complete operator-chain checks, not package
  block-output comparisons. Blocks 10 and 11 have also completed sequentially
  from their preceding RTL outputs; each final tensor is 884,736 bytes, all
  operator checks passed, and the archive has no `b10.y` or `b11.y` golden.
  Block 12 has completed from block 11's RTL output; its 884,736-byte final
  tensor passed all operator checks, with no `b12.y` package golden. Block 13
  also completed from block 12's RTL output; its final 884,736-byte tensor
  passed the QLinear, attention, GELU, and residual module checks. There is no
  `b13.y` package golden. Thus all 14 blocks have now run sequentially as a
  one-branch M=256 file-bridged RTL chain, but this is not yet full inference:
  timestep AdaLN is host-computed, CFG=512 and skip scheduling are absent, and
  the chain is not a connected DMA/top or a 50-step Euler run using RTL
  velocities. The final-head fixture/compiler now consumes block 13's RTL
  tensor, applies final AdaLN and FP16 576→16 reference projection, and
  unpatchifies to 4×32×32. The FP16 systolic-array final projection RTL unit
  passes at 0.0417% relative-to-peak error (21 FP16 bit differences); two
  unpatchify unit tests pass. This is still a single conditional branch with
  host-computed final AdaLN and not a CFG-combined step velocity. CFG/step
  orchestration remains next.
- Earlier FP16 x_embed FakeDDR smoke passes for 256 synthetic tokens, but is
  only a generic-top baseline—not the archive model's quantized path.
- Dedicated Supra2 top with command/RF/DMA, complete denoiser execution,
  synthesis for the target FPGA part, and board timing: not yet completed.

## XSim continuation evidence (2026-10-19)

The standalone XSim runner now links every `xsim_*.win64.obj` emitted by
elaboration; larger testbenches split their generated entry point across more
than two objects. `scripts/run_z0_frontend_xsim.ps1` passes archive `z0`
through the RTL `x_embed` QLinear frontend, RTL FP16 position add, and RTL
LayerNorm/AdaLN modulation. The QLinear output is bit-exact across 256x576
elements (15,264 cycles); position add passes all 147,456 elements; LayerNorm
relative-to-peak error is 0.0183% (295,168 cycles). Timestep AdaLN linear
parameters remain host-precomputed.

The RTL-produced LayerNorm tensor then passed block-0 QKV in eight M=32 XSim
runs, covering the full M=256, N=1728, K=576 projection. All 442,368 FP16
outputs match the fixture reference bit-for-bit; Hadamard, quantizer, INT32
accumulator, and software-chain mismatch counts are zero. The sum of the
independent QKV batch counters is 322,784 cycles; this is not a connected-top
or board-latency measurement. The QLinear compiler accepts `--row-start` and
`--row-count` for resumable 32-row batches, with a test that checks sliced
fixtures against the full fixture.

The QK discrepancy was traced to the custom FP32 accumulator addition in
`mac_fp16.sv`; replacing it with the project's round-to-nearest-even helper
made the single-MAC probe pass. The full 256x9x256 XSim QK regression then
passed on RTL-produced QKV at 77,764 cycles, 0.0236% relative-to-peak, and zero
tolerance mismatches. The ensuing full online Softmax/PV run passed at 922,824
cycles and 0.2416% relative-to-peak. Self-attention projection, self residual,
masked cross-attention Q/KV/QK/PV/projection/residual, MLP LayerNorm/AdaLN,
MLP.0, GELU(tanh), MLP.2, and final residual were rerun with XSim using the
preceding RTL outputs as fixtures. MLP.0 and MLP.2 each used eight resumable
32-row QLinear batches; their merged 256-row tensors are bit-exact against
their generated references (416,384 and 545,696 summed cycles).

The 50-step Euler RTL replay also passes in 6,400 lane-beat cycles using the
archive's golden `v_per_step` tensors. RTL output is bit-exact to the FP16
software replay and has 0.1137% relative-L2 to archived `z_final`; this
validates the sampler datapath, not denoiser-generated velocities. The Python
regression suite passes 58 tests. The final head's final LayerNorm/AdaLN now
also runs in RTL (295,168 cycles, 0.0258% relative-to-peak). Its XSim 576->16
projection passes at 0.0417% relative-to-peak (45 FP16 bit differences from
the reference after RTL normalization); the unpatchified [4,32,32] latent
matches the corresponding reference layout at the same relative error. These
results still do not constitute a
connected 14-block/CFG=512/50-step hardware execution: block tensors use
fixture file bridges, timestep AdaLN is host-precomputed, and the Euler input
velocities come from the archive. A Supra2 command/DMA top, target-part
synthesis, MIG/board timing, and full inference from RTL-generated velocities
remain open.

The previously recorded 14-block serial-chain artifacts were generated before
the FP32 MAC addition fix. Block 0, block 1's complete 256-token operator chain,
and the final head have now been rerun with the corrected RTL. Blocks 2-13
still require revalidation before the old serial-chain artifacts can be treated
as current evidence. The package does not contain `golden_block.b1.y`; block 1's
final tensor is therefore checked against the operation-by-operation software
reference generated from the RTL-fed fixtures, not against a package block
output golden.

### Fresh block-1 serial RTL chain

Block 1 starts from the fresh RTL block-0 final tensor. Its self-attention
QKV -> QK -> online-softmax/PV -> output projection -> gated residual, then
cross-attention LayerNorm -> Q/KV -> masked QK -> online-softmax/PV -> output
projection -> ungated residual, and finally MLP LayerNorm/AdaLN -> MLP.0 ->
GELU(tanh) -> MLP.2 -> gated residual all consumed the preceding RTL output.
The shared `ctx_proj` also reran from archived `ctx_all[0]` and its block-1
context rows were packed to 15 valid keys in one 32-key tile.

Numerical results: QKV, self projection, cross Q/KV, cross projection, MLP.0,
and MLP.2 merged tensors are bit-exact against their generated QLinear
references. Self QK is 77,764 cycles / 0.0209% relative-to-peak; self online
PV is 922,824 cycles / 0.1679%; cross QK is 9,724 cycles / 0.0375%; cross
online PV is 254,520 cycles / 0.5488%. The two gated residuals and the
ungated cross residual pass at less than 3e-7 relative-to-peak. MLP LayerNorm
passes at 0.0128%; GELU(tanh) is 589,824 cycles / 0.1448%. The final 256x576
tensor differs from its generated block-operation reference by 2.77e-7
relative-to-peak, RMSE 1.15e-6, and relative-L2 1.21e-7 (81 FP16 bit
differences).

Summing these independent operator busy-cycle counters, including the one-time
shared context projection, gives 4,997,464 cycles (about 24.99 ms at an
assumed 200 MHz). This is component accounting, not a connected-top or board
latency: fixture files bridge the operators, timestep AdaLN is host-precomputed,
and CFG=512/skip scheduling are not part of this block-1 run.

`scripts/run_qlinear_xsim_batches.ps1` generalizes the resumable 32-row
QLinear flow to different model-layer K/N/H geometries, with per-batch cycle
logs and a merged full-tensor equality check. XSim workarounds include linking
all elaboration objects, optional top generic overrides, and omitting testbench
timescale directives that trigger the local XSim 43-4100 error.

### Fresh block-2 serial RTL chain

Block 2 was rerun after the FP32 MAC rounding correction, starting from the
fresh block-1 RTL final residual. The validated chain covers block-2 self
LayerNorm/AdaLN, QKV, QK, online softmax/PV, self projection and gated
residual; cross LayerNorm/AdaLN, Q, masked context KV, QK, online softmax/PV,
cross projection and ungated residual; then MLP LayerNorm/AdaLN, MLP.0,
GELU(tanh), MLP.2 and final gated residual. Every QLinear projection's merged
256-row output is bit-exact against its fixture reference. Both attention PV
stages pass their 1% relative-to-peak gates; self QK is 0.0676%, self PV is
0.3032%, cross QK is 0.0478%, and cross PV is 0.2641%. MLP LayerNorm is
0.0402%, GELU(tanh) is 0.0705%, and both residual RTL stages pass with
relative-to-peak error below 3e-7. The compiler fixture preparer now has a
`--branch mlp` option so MLP normalization uses the MLP AdaLN scale/shift; the
default remains `self_attn`.

Summing the independent block-2 operator cycle counters yields 4,902,792
cycles; including the already-validated one-time shared context projection
yields 4,997,464 cycles (24.99 ms at an assumed 200 MHz). This is a component
cycle ledger, not integrated-top or FPGA board latency. The package's
`golden_block.npz` contains block-level `b0.y` and `b7.y`, but no `b2.y`, so
block 2 is checked through per-operator generated references rather than a
package block-output tensor. Fixture files still bridge modules, timestep
AdaLN parameters remain host-precomputed, and CFG=512/50-step denoiser
execution is not yet part of this run.

### Fresh block-3 serial RTL chain

Block 3 was run from block 2's fresh RTL final residual through the complete
self-attention, masked cross-attention, and gated MLP chain. QLinear merged
outputs are bit-exact against their generated references. Self QK/PV pass at
0.0347%/0.2294% relative-to-peak; cross QK/PV pass at 0.0483%/0.2882%.
LayerNorm/AdaLN and GELU pass at 0.0447% and 0.0678%, respectively. Gated
residuals are below 2e-7 relative-to-peak, and the ungated cross residual
passes its elementwise RTL regression. The chain ends at
`build/supra2_rne_b3_final_residual/rtl_actual_fp16.hex`.

The block-3 component cycle sum is 4,902,792, or 4,997,464 including the
one-time shared context projection (24.99 ms at an assumed 200 MHz). As for
block 2, this is independent module busy-cycle accounting, not a connected
top or board timing result; block 3 has no package block-output golden and is
checked against per-operator references. Fresh sequential evidence now spans
blocks 0 through 3, all at M=256 for one conditioning branch. CFG=512,
skip-trace execution, RTL-derived velocities, and the complete 50-step sample
remain unverified.

### Fresh block-4 serial RTL chain

Block 4 was rerun from the fresh block-3 final residual through self-attention,
masked cross-attention, and the complete MLP. Its QKV, self projection,
cross-Q, cross-KV, cross projection, MLP.0, and MLP.2 RTL-produced tensors
were checked over all 256 rows using resumable 32-row XSim batches; every
merged tensor had zero FP16 bit mismatches against its generated software
reference. The online self PV took 922,824 cycles with 0.2107%
relative-to-peak error. Cross QK took 9,724 cycles with 0.0330% error; the
15-valid-key online cross PV took 254,520 cycles with 0.5581% error. LayerNorm,
GELU(tanh), both projections, and the final gated residual also passed their
full-tensor RTL checks. MLP.0/MLP.2 took 416,384/545,696 cycles; the final
residual error was 1.678e-7 relative to peak.

Summing block-4 operator busy-cycle counters gives 4,902,792 cycles; adding
the shared one-time context projection gives 4,997,464 cycles (24.99 ms at an
assumed 200 MHz). This is component accounting, not connected-top or board
latency. The chain is one conditioning branch at M=256, joined by fixture
files; timestep AdaLN remains host-precomputed, and there is no package
block-4 output golden, so validation is per operator. CFG=512, the dynamic
skip scheduler, and RTL-generated velocity feedback remain unverified.
Final block-4 tensor:
`build/supra2_rne_b4_mlp_residual/rtl_block4_output_fp16.hex`.
### Fresh block-5 serial RTL chain

Block 5 was run from the fresh block-4 final residual through self-attention,
masked cross-attention, and the complete gated MLP. QKV, self/cross projections,
MLP.0, and MLP.2 merged outputs are bit-exact against their generated
references. Self QK/PV pass at 0.0274%/0.2230% relative-to-peak; cross QK/PV
pass at 0.0484%/0.3021%. Cross QK has one diagnostic sample above its 0.005
absolute-error threshold, while remaining below its hard 0.1% relative-to-peak
gate. MLP LayerNorm passes at 0.0106%; GELU(tanh) at 0.0323%; residual
stages pass below 2e-7 relative-to-peak.

Block-5 operator busy-cycle sum is 4,902,792 cycles (about 24.51 ms at an assumed 200 MHz), excluding the one-time shared context projection. This is component accounting, not connected-top or board latency: operators are joined by fixture files, timestep AdaLN is host-precomputed, and the block has no package output golden. The chain remains one M=256 conditioning branch; CFG=512, skip scheduling, and RTL velocity feedback are not included.

For fixed 256-token hardware, the shape evidence points to these priorities:

- Keep the 32x32 array but stream the 8 row tiles without host/file boundaries; retain accumulators on-chip until the full output tile is complete.
- Treat MLP.0 (N=2304) as 72 output-channel tiles per row tile. Keep the input activation/quantization result resident and overlap weight-tile DMA with MAC and fused bias/scale/FP16 epilogue.
- For MLP.2 (K=2304), stream nine 256-wide K chunks per output tile. A K=2304 by N=32 INT8 tile is 72 KiB, so use ping-pong 256x32 (8 KiB) weight buffers and keep 32x32 partial sums on-chip across all nine K chunks.
- Exploit the fixed 15 valid context tokens by projecting/packing shared context once and reusing it across all 14 blocks and 50 denoising steps; use a 32-key padded tile only at the attention interface.
- These are architecture recommendations inferred from tensor shapes and operator counters, not measured area, frequency, or speedups. There is still no connected Supra2 system top or FPGA synthesis result.

Final block-5 tensor:
`build/supra2_rne_b5_mlp_residual/rtl_actual_fp16.hex`.

### Fresh block-6 serial RTL chain

Block 6 starts from the block-5 RTL final tensor and runs the complete
self-attention, masked cross-attention, and gated MLP chain at M=256. All QLinear
outputs (QKV, self/cross projections, cross-KV, MLP.0 and MLP.2) are bit-exact
to generated references. Self QK/PV pass at 0.0409%/0.1614%
relative-to-peak; cross QK/PV pass at 0.0131%/0.3369%. Self and cross LayerNorm
relative errors are 0.0162% and 0.0032%; MLP LayerNorm is 0.0478%, and
GELU(tanh) is 0.0763%. Both residual RTL operations pass (final gated residual
1.85e-7 relative-to-peak).

The block-6 operator busy-cycle sum is 4,902,792 cycles (24.51 ms at an
assumed 200 MHz). It is component accounting, not connected-top or FPGA board
latency. Fixture files bridge standalone operator simulations; timestep AdaLN
is host-precomputed and the package has no block-6 output golden. This adds a
fresh RTL-driven chain segment through blocks 0-6 for one M=256 conditioning
branch. CFG=512, skip-trace scheduling, RTL velocity feedback, and the full
50-step run are still outstanding.

Final block-6 tensor:
`build/supra2_rne_b6_mlp_residual/rtl_actual_fp16.hex`.

### Fresh block-7 serial RTL chain

Block 7 was completed from the fresh block-6 RTL final tensor through
cross-attention (query LayerNorm/AdaLN, Q, masked context KV, QK, online
softmax/PV, output projection, ungated residual) and the gated MLP
(MLP LayerNorm/AdaLN, MLP.0, GELU(tanh), MLP.2, final gated residual).
All stage-level checks pass on the corrected (RNE) RTL: GELU(tanh) passes in
589,824 cycles at 5.75e-4 relative-to-max; the eight resumable 32-row MLP.2
batches merge bit-exactly (0 FP16 mismatches, 545,696 summed cycles); the
final gated residual passes in 147,456 cycles.

The package-level comparison against archive `golden_block.b7.y` does **not**
meet the 2% relative-to-peak gate: max_abs=9.9375, relative-to-peak=2.9293%,
RMSE=0.64986, relative-L2=3.2923%. The drift is inherited, not introduced by
block 7: the fresh block-6 RTL final tensor already differs from archive
`b7.x` by max_abs=12.0, relative-to-peak=3.5982%, RMSE=0.57282,
relative-L2=2.8887%, and block 7's residual-preserving structure leaves the
error at the same magnitude. Fresh block 0 from archive `z0` still matches
`b0.y` at 0.4627% relative-to-peak, so the accumulation happens across
blocks 1-6, which have no package block-output goldens and were previously
validated only per operator. Per-stage attention errors (self online PV
0.16-0.31%, cross online PV 0.26-0.56% relative-to-peak) compound on the
residual stream at roughly 0.5% per block. Note the pre-fix serial chain had
measured 1.032% at the same boundary; why the corrected chain drifts faster
is an open question. This comparison was run with
`python -m compiler.supra2.verify_block0_output --block 7` (exit 1, gate
failure recorded above).

Final block-7 tensor:
`build/supra2_rne_b7_mlp_residual/rtl_actual_fp16.hex`.

### End-to-end performance estimate (component-cycle accounting)

All numbers below are sums of independent operator busy-cycle counters from
the RTL simulations above; they are **not** a connected-top measurement and
assume a 200 MHz clock only for time conversion. DMA, weight streaming,
control bubbles, and host AdaLN computation are not included.

Per-block operator-cycle sum (M=256, one conditioning branch): 4,902,792
cycles (three LayerNorm/AdaLN 3x295,168; QKV 322,784; self QK 77,764; self
online PV 922,824; self proj 135,584; self residual 147,456; cross Q 135,584;
cross KV 28,648; cross QK 9,724; cross online PV 254,520; cross proj 135,584;
cross residual 147,456; MLP.0 416,384; GELU 589,824; MLP.2 545,696; final
gated residual 147,456). One-time shared `ctx_proj`: 94,672 cycles.

Single-branch denoiser step (M=256, no CFG, no skipping): x_embed 15,264 +
position add 147,456 + 14 blocks (68,639,088) + final LN/AdaLN 295,168 +
final head projection ~5,181 + Euler update 6,400 = 69.11M cycles, about
345.5 ms at 200 MHz; 50 steps = 3.46G cycles, about 17.3 s per image.

CFG=3.0 step (M=512, both branches; row-scaling components double, the
cached context path cross KV 28,648 + cross QK 9,724 stays fixed): per-block
9,767,212 cycles. With the package's 50x14 skip trace (40.857% average skip,
8.28 blocks executed per step on average): per step 30,528 (x_embed) +
294,912 (pos add) + 8.28 x 9,767,212 (80,872,515) + 590,336 (final LN) +
~10,362 (final head) + ~12,800 (CFG combine + Euler) = 81.81M cycles, about
409 ms at 200 MHz; 50 steps = 4.09G cycles, about 20.5 s per image. Without
skipping it would be ~34.4 s, so the skip trace saves roughly 40%. A
connected top with real DMA/overlap would shift these numbers; report them
as component estimates only.

The isolated-copy Python suite currently passes 59 tests:
`python -m unittest discover -s tests`.

