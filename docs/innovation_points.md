# Thesis Innovation Points for the Supra2-IMG FPGA Project

## Recommended thesis framing

Frame the work as **workload-aware FPGA architecture and dataflow design for a quantized diffusion transformer**, rather than claiming a new diffusion model or a new quantization algorithm. The strongest evidence currently comes from RTL operator/block simulations and selected FPGA out-of-context implementation runs. A complete DDR-connected 50-step inference and board measurement have not been established; keep those boundaries explicit in the abstract, contribution list, and conclusions.

## Proposed contributions

### 1. A model-specialized rotation-quantization data path for a diffusion transformer

The design maps the supplied Supra2-IMG package's 100 INT8 QLinear matrices and 37 FP16 passthrough tensors onto a 32x32 signed INT8 systolic array. Its front end applies the package-defined block-diagonal Sylvester Hadamard transform and per-token dynamic INT8 quantization; the output path accumulates in INT32 and applies per-token/per-channel scales and bias in the FP32 epilogue. This gives a concrete hardware mapping of rotation-based post-training quantization to the model's 256-token, width-576, 14-block workload.

**Evidence to report:** compiler image round-trip for all packaged tensors; exact INT32 accumulator comparisons; full projection regressions for representative shapes; block-0 and block-7 package-output comparisons with their measured tolerances. State that these are simulations and identify which stages use archive goldens versus generated software references.

**Novelty boundary:** Hadamard rotation and dynamic quantization are established ideas. The defensible contribution is their model-specific integration, tile/stream implementation, and validation on this DiT workload, not invention of the underlying quantization method.

### 2. A streamed attention datapath that avoids materializing the probability matrix

For self-attention, the RTL consumes score tiles in an online-softmax/PV flow, reuses the value tile, and avoids writing the full 256x256 probability matrix to an intermediate tensor. For cross-attention, the design packs the 15 valid context tokens from the supplied mask into one 32-key tile and projects only the valid prefix. This uses the workload's fixed sequence and mask structure to reduce intermediate storage and unnecessary QK/PV work.

**Evidence to report:** full 256-query, 9-head online-softmax-to-PV simulation: 922,824 busy cycles and 0.2418% relative-to-peak error; the reported 79.95% cycle reduction is against the specified single-query fused baseline. Masked cross-attention measurements report 75.0% lower QK cycles and 52.9% lower PV cycles than the padded unmasked run. Make clear that these are module-level RTL simulations, not complete DDR traffic measurements.

### 3. A diffusion-step control and memory interface specialized for CFG, block skipping, and wide DDR transfers

The runtime control path combines a 50-step schedule, conditional/unconditional CFG branches, block-skip decisions, layer/tile descriptors, and a 512-bit read interface. The 512-bit stream is bridged to the ZU7EV MIG's 64-bit native interface. The control regression covers 50 steps and 828 non-skipped block launches; the MIG pin-model tests check data integrity across DMA and bridge transfers. This is a system-integration contribution: compiler-produced layer/tile metadata and model execution decisions are represented as hardware requests instead of requiring a fixed host-side sequence for each transfer.

**Evidence to report:** name each test separately. The 50-step/828-launch regression validates control sequencing; the MIG tests validate the tested transactions and data ordering. Neither proves a full denoising run through DDR. The 400 MHz placed-and-routed result applies only to the runtime-DMA/controller OOC hierarchy and its stated constraints, not the compute bridge or complete design. The array's OOC route is a separate result at its own clock constraint.

## Optional contribution: adaptive residual reuse (research extension)

The DiCache-inspired runtime hook probes a shallow block, accumulates per-branch error, chooses refresh or reuse, stores two residual anchors per CFG branch, and interpolates cached residuals through a 32-lane DCTA datapath. The policy can be enabled alongside the package skip-trace path. This is promising as an FPGA-oriented integration of adaptive caching with CFG scheduling, but current regressions use fixed probe error/gamma and reduced cache fixtures. Full 50-step quality, autonomous error estimation, full-size cache mapping, and measured speedup remain open.

**Recommendation:** include this as an exploratory extension or a secondary contribution with an explicit validation caveat. Do not present it as a demonstrated image-quality-preserving acceleration until full-schedule golden comparisons are run.

## Suggested contribution paragraph (thesis-ready)

This thesis investigates a workload-specialized FPGA architecture for the Supra2-IMG diffusion transformer. First, it maps the supplied rotation-based INT8 quantization recipe to a tiled 32x32 systolic datapath, including a streaming Hadamard and activation-quantization front end and an INT32-to-FP32 scale/bias epilogue. Second, it develops a streamed attention path that combines tiled online softmax with value accumulation and exploits the model's masked context prefix to reduce intermediate storage and redundant attention work. Third, it integrates CFG branch control, block-skip scheduling, layer/tile descriptors, and a 512-bit DMA interface with a MIG-compatible DDR bridge. RTL simulations and selected out-of-context FPGA implementation runs validate these components at their respective boundaries. The complete DDR-connected 50-step inference and board-level performance remain future validation targets.

## Claims and evidence checklist

| Claim | Current evidence | Safe wording |
|---|---|---|
| Rotation-quantized QLinear path is correct | Package/compiler checks and RTL projection/block regressions | “Validated by RTL simulation against the stated package or software references” |
| Attention reduces intermediate traffic | Online-softmax/PV and masked-prefix module regressions | “Avoids materializing the probability matrix in the tested datapath”; quantify only the measured cycle comparison |
| Runtime control supports the sampling schedule | 50-step, two-branch control regression with 828 launches | “Control-plane schedule regression passes”; do not say full inference passes |
| DMA works with MIG | Pin-model transaction tests | “Tested DMA/bridge transactions pass data-integrity checks”; do not extrapolate to full-model bandwidth |
| 400 MHz operation | Runtime-DMA/controller OOC placed-and-routed result | Limit the claim to that hierarchy and its constraints |
| DiCache improves quality/speed | No complete full-schedule quality/performance result | Do not claim yet; describe as implemented hook and ongoing evaluation |

## Experiments that would strengthen the thesis

1. Run a connected DDR-to-compute block test with real package tensors and compare the produced block output to a package golden.
2. Close the known block-chain numerical drift against archive references, then run both CFG branches through the complete skip schedule and all 50 Euler updates.
3. Evaluate DiCache on the same 50-step package inputs: final latent error, image/latent quality metric, refresh/reuse counts, bytes moved, and cycles against the skip-trace baseline.
4. Report synthesis and implementation for a clearly defined integrated hierarchy, with the target clock, constraints, resource utilization, timing slack, and any unconstrained paths. Keep OOC module results separate from full-top results.
5. For throughput claims, measure DDR traffic and stalls in the connected design; do not infer end-to-end latency by summing independent operator counters.

