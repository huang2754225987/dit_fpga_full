#include "compute_units.h"
#include "npu_pipeline_model.h"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace supra2::latency;

int main() {
  try {
    // One illustrative 32x32 GEMM tile. These operation sizes and GELU cycles
    // are examples, not the measured block-13 execution profile.
    const auto array_cycles = SystolicArrayModel{}.estimate_cycles({32, 32, 32});
    const auto gelu_cycles = VectorUnitModel{}.estimate_cycles(1024);
    const std::vector<BufferConfig> buffers = {
        {"ACT", 64 * 1024, 2, 64, 1},
        {"WEIGHT", 64 * 1024, 2, 64, 1},
        {"MID", 64 * 1024, 2, 64, 1},
        {"OUT", 64 * 1024, 2, 64, 1},
    };
    const std::vector<ComputeResourceConfig> resources = {
        {"ARRAY", 1}, {"GELU", 1},
    };
    const std::vector<PipelineOperation> operations = {
        {"load_activation", PipelineKind::DmaRead, {}, 0x0000, 1024, 0,
         {}, {"ACT", "x_tile", 1024, true}, "", 0},
        {"load_weight", PipelineKind::DmaRead, {}, 0x1000, 1024, 0,
         {}, {"WEIGHT", "w_tile", 1024, false}, "", 0},
        {"gemm_tile", PipelineKind::Compute,
         {"load_activation", "load_weight"}, 0, 0, 0,
         {{"ACT", "x_tile", 1024, true},
          {"WEIGHT", "w_tile", 1024, false}},
         {"MID", "gemm_result", 2048, true}, "ARRAY", array_cycles},
        {"gelu_tile", PipelineKind::Compute, {"gemm_tile"}, 0, 0, 0,
         {{"MID", "gemm_result", 2048, true}},
         {"OUT", "gelu_result", 2048, true}, "GELU", gelu_cycles},
        {"write_output", PipelineKind::DmaWrite, {"gelu_tile"},
         0x2000, 2048, 0,
         {{"OUT", "gelu_result", 2048, true}}, {}, "", 0},
    };

    NpuPipelineModel model({}, buffers, resources);
    const auto result = model.run(operations);
    std::cout << "makespan_ui=" << result.makespan_ui << '\n';
    for (const auto& op : result.operations)
      std::cout << op.id << " start_ui=" << op.start_ui
                << " finish_ui=" << op.finish_ui << '\n';
    std::cout << "ddr_read_bytes=" << result.mig_stats.read_bytes
              << " ddr_write_bytes=" << result.mig_stats.write_bytes
              << " dma_bursts=" << result.dma_stats.bursts
              << " dma_buffer_wait_ui=" << result.dma_stats.buffer_backpressure_ui
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
