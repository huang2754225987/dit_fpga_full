#pragma once

#include <cstdint>
#include <stdexcept>

namespace supra2::latency {

struct ModelConfig {
  std::uint64_t clock_period_ps = 3332;
  std::uint32_t axi_data_width_bits = 512;
  std::uint32_t max_burst_beats = 64;
  std::uint64_t first_response_latency_cycles = 0;
  std::uint64_t cycles_per_beat = 1;
  std::uint32_t outstanding_capacity = 1;
  bool read_write_share_channel = true;
  bool memory_parameters_calibrated = false;

  void validate() const {
    if (clock_period_ps == 0 || axi_data_width_bits == 0 ||
        axi_data_width_bits % 8 != 0 || max_burst_beats == 0 ||
        cycles_per_beat == 0 || outstanding_capacity == 0) {
      throw std::invalid_argument("invalid model configuration");
    }
  }
};

}  // namespace supra2::latency
