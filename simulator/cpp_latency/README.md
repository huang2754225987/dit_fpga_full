# Supra2 C++ Latency Model

This C++17 area contains a measured stage-ledger scheduler and newer
request-level MIG, DMA, tile-buffer, and compute-resource predictors. The
stage-ledger example uses measured block-13 counters as task durations;
the pipeline model instead advances AXI bursts and tile operations on one
UI-clock timeline. Neither mode is a calibrated cycle-exact MIG clone or a
complete 50-step inference model.

`include/onchip_memory_model.h` provides an LRU banked line-buffer model for
exploring activation/weight tile residency and bank contention. It is a
parameterized abstraction, not a cycle-accurate replica of each RTL RAM.
`include/compute_units.h` provides a 32x32 systolic feed lower-bound estimate
and the measured block-13 phase profile for array, LayerNorm, GELU, residual,
and control stages. The aggregate stage profile includes internal work and
weight waits as recorded in the RTL/MIG run; do not add those waits again.

The stage-ledger `EventSimulator` does not call the cache model to stall
compute tasks on misses. The newer `NpuPipelineModel` tracks named tile
buffers and DMA-produced tile availability. It still needs actual tile
descriptors and calibrated compute-only durations for block-13 predictions.

Build from the repository root:

On this workstation use `scripts/run_cpp_latency_model.ps1`; it selects
MSYS2 GCC 14.2 explicitly, builds Debug (assertions enabled), runs the C++
regressions/examples, and writes `build/cpp_latency_msys2/characterization.json`.
The older GCC 8.1 earlier on PATH fails to launch on this machine.
See [the execution and accuracy assessment](../../docs/cpp_latency_model_accuracy_assessment.md)
for current run results, configuration corrections, independent phase errors,
and the prioritized accuracy work.

```powershell
cmake -S simulator/cpp_latency -B build/cpp_latency
cmake --build build/cpp_latency --config Release
ctest --test-dir build/cpp_latency -C Release --output-on-failure
```

Run the measured phase-ledger example:

```powershell
build/cpp_latency/Release/supra2_latency_model.exe `
  --workload simulator/cpp_latency/examples/block13_reference.json `
  --config simulator/cpp_latency/examples/model_config.json `
  --output build/cpp_latency/block13_result.json
```

On single-configuration generators, the executable may be located directly in
`build/cpp_latency/`. Use `--help` for the command-line options.

The default DDR and on-chip memory response parameters in `model_config.json`
are placeholders, and their calibration flags are false. Calibrate them from
burst-level or RTL-bank measurements before using these abstractions to
estimate traffic. The measured
4,663,834-cycle transaction is a behavioral RTL/MIG simulation result, not a
physical board measurement.

`include/ddr4_device.h` and `src/ddr4_device.cpp` add a separate DDR4 pin
command/timing model. `ddr4_replay` checks post-calibration MIG pin traces;
it does not replace the AXI burst model. See [DDR4_DEVICE_MODEL.md](DDR4_DEVICE_MODEL.md)
for geometry, command rules, parameter provenance, replay format, and limits.

`include/mig_axi_model.h` and `src/mig_axi_model.cpp` add a request-level
AXI512-to-DDR64 MIG predictor with approximate read-priority address grants, PG150
`ROW_COLUMN_BANK` mapping, bank row-hit/conflict behavior, DDR command timing,
and response completion. Run the standalone request example with
`mig_axi_model --requests simulator/cpp_latency/examples/mig_axi_requests.csv`.
The instantiated board setting is RD_PRI_REG, CL=18, CWL=12. Optional TDM is
available for comparisons. Scope and limits are documented in [MIG_AXI_MODEL.md](MIG_AXI_MODEL.md).

`include/npu_pipeline_model.h` and `src/npu_pipeline_model.cpp` connect a
single-outstanding DMA, banked tile buffers, compute-resource slots, and MIG
read-beat ready events on one UI-clock timeline. The example executable is
`npu_pipeline_demo`; see [NPU_PIPELINE_MODEL.md](NPU_PIPELINE_MODEL.md).

The model contains no copied GPGPU-Sim source at this stage. The event and
burst abstractions are standalone implementations informed by the documented
queue behavior. If a later change copies source code, retain the applicable
copyright and BSD license notices and record the source path.
