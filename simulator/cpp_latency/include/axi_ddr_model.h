#pragma once

#include "model_config.h"
#include "task_graph.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace supra2::latency {

struct Burst {
  std::uint64_t address = 0;
  std::uint32_t beats = 0;
  std::uint64_t bytes = 0;
};

struct BurstTiming {
  Cycle queue_wait_cycles = 0;
  Cycle service_cycles = 0;
  Cycle response_cycle = 0;
};

// Generic AXI burst arithmetic only. It does not model DDR bank/row timing.
class AxiDdrModel {
 public:
  explicit AxiDdrModel(ModelConfig config) : config_(config) {
    config_.validate();
  }

  std::vector<Burst> split(std::uint64_t address, std::uint64_t bytes) const {
    if (bytes == 0) return {};
    const std::uint64_t beat_bytes = config_.axi_data_width_bits / 8;
    if (address % beat_bytes != 0)
      throw std::invalid_argument("AXI address must be beat aligned in v1 model");
    std::vector<Burst> bursts;
    while (bytes != 0) {
      const std::uint64_t page_bytes = 4096 - (address & 4095);
      const std::uint64_t max_bytes =
          static_cast<std::uint64_t>(config_.max_burst_beats) * beat_bytes;
      const std::uint64_t chunk = std::min(bytes, std::min(page_bytes, max_bytes));
      const std::uint64_t beats = (chunk + beat_bytes - 1) / beat_bytes;
      bursts.push_back({address, static_cast<std::uint32_t>(beats), chunk});
      const std::uint64_t transferred = beats * beat_bytes;
      address += transferred;
      bytes -= std::min(bytes, transferred);
    }
    return bursts;
  }

  Cycle service_cycles(std::uint32_t beats) const {
    if (beats == 0) return 0;
    return config_.first_response_latency_cycles +
           static_cast<Cycle>(beats) * config_.cycles_per_beat;
  }

  const ModelConfig& config() const { return config_; }

 private:
  ModelConfig config_;
};

}  // namespace supra2::latency
