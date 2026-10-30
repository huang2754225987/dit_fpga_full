# Supra2-IMG FPGA Work Area Instructions

## Scope

- This directory is the isolated work area for `SupraLabs/Supra2-IMG` FPGA deployment.
- Keep all model-specific source changes, generated fixtures, and notes inside this directory. Do not modify the parent NPU project unless the user explicitly requests it.
- Do not run a Transformer forward or GPU inference. Use the supplied hardware-export archive and its golden tensors as the numerical reference.
- Treat `HANDOFF.md` as the current task state and read it before continuing implementation. Read `README.md` and the relevant module README for detailed design context.
- Inspect only files relevant to the current operator or integration task. Avoid scanning checkpoints, generated `build/` contents, simulator logs, and simulator output unless needed for a specific verification.

## Engineering and verification

- Explain progress and results to the user in Chinese. Keep source code, identifiers, and technical documentation in English.
- Use `rg` for source search and `apply_patch` for hand edits. Preserve unrelated and untracked user changes.
- Distinguish software/compiler checks, standalone RTL simulation, integrated simulation, synthesis, and physical FPGA measurement. Never describe a standalone XSim result as full-top or board validation.
- Do not claim hardware latency from sums of independent operator counters. Label those sums as component-cycle estimates and state the assumed clock when converting them to time.
- Do not claim a full inference result until CFG orchestration, all required blocks and skip behavior, RTL-generated step velocities, and all 50 Euler updates are connected and checked against the supplied package references.
- When the user requests code-only changes, do not run simulations, tests, synthesis, weight generation, or documentation generation. Otherwise, run only verification relevant to the requested change and report exactly what ran.

## Active design assumptions

- 256x256 text-to-image rectified-flow DiT; latent 4x32x32; 2x2 patches; 256 tokens.
- Width/depth 576/14; 9 attention heads x 64; MLP ratio 4.
- Supplied quantized package: INT8 QLinear weights with FP16 passthrough tensors; 32x32 systolic array; FP16 input/Hadamard frontend, INT32 accumulators, FP32 QLinear epilogues, FP16 outputs.
- Sampling contract: CFG=3.0, conditional/unconditional batch of 512 rows, 50 Euler-flow steps; supplied skip trace is 50x14.
- Text encoder and VAE are outside the FPGA DiT path. Context K/V should be reused per prompt where the schedule permits.

