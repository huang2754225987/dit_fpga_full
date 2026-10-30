#pragma once

#include "task_graph.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace supra2::latency {

struct SystolicArrayConfig {
  std::uint32_t rows = 32;
  std::uint32_t columns = 32;
  Cycle pipeline_overhead_cycles = 66;
};

struct MatmulShape {
  std::uint32_t m = 0;
  std::uint32_t n = 0;
  std::uint32_t k = 0;
};

// Idealized 32x32 array feed model. It excludes measured frontend,
// quantization, weight-fetch stalls, and epilogue work unless calibrated data
// is supplied separately as a task duration.
class SystolicArrayModel {
 public:
  explicit SystolicArrayModel(SystolicArrayConfig config = {}) : config_(config) {
    if (config_.rows == 0 || config_.columns == 0)
      throw std::invalid_argument("systolic array dimensions must be positive");
  }

  Cycle estimate_cycles(MatmulShape shape) const {
    if (shape.m == 0 || shape.n == 0 || shape.k == 0)
      throw std::invalid_argument("matmul dimensions must be positive");
    const std::uint64_t m_tiles = (shape.m + config_.rows - 1) / config_.rows;
    const std::uint64_t n_tiles = (shape.n + config_.columns - 1) / config_.columns;
    const Cycle one_tile = shape.k + config_.pipeline_overhead_cycles;
    return m_tiles * n_tiles * one_tile;
  }

 private:
  SystolicArrayConfig config_;
};

struct VectorUnitConfig {
  std::uint32_t lanes = 32;
  Cycle pipeline_overhead_cycles = 2;
};

// 适用于每周期处理固定数量元素的 GELU/残差等流式单元。
// 输入断流造成的等待需由外部任务时间线表达。
class VectorUnitModel {
 public:
  explicit VectorUnitModel(VectorUnitConfig config = {}) : config_(config) {
    if (!config_.lanes) throw std::invalid_argument("vector lanes must be positive");
  }

  Cycle estimate_cycles(std::uint64_t elements) const {
    if (!elements) throw std::invalid_argument("vector elements must be positive");
    return (elements + config_.lanes - 1) / config_.lanes +
           config_.pipeline_overhead_cycles;
  }

 private:
  VectorUnitConfig config_;
};

// 当前 32-row、256-key 双遍 softmax RTL 的实测 tile 周期。
class SoftmaxTileModel {
 public:
  Cycle estimate_cycles(std::uint32_t rows, std::uint32_t keys) const {
    if (!rows || keys != 256)
      throw std::invalid_argument("softmax profile supports positive rows and 256 keys");
    return ((static_cast<Cycle>(rows) + 31) / 32) * 2401;
  }
};

struct UnitProfile {
  const char* name;
  const char* resource;
  Cycle measured_cycles;
  Cycle measured_weight_wait_cycles;
  const char* measurement_scope;
};

// Measured block-13 stage profiles from the connected AXI512/MIG simulation.
// Stage counters include the RTL's internal work; array-stage profiles include
// their weight waits where applicable. These are aggregate task durations.
inline constexpr UnitProfile kBlock13UnitProfiles[] = {
    {"LN0", "LN", 13900, 0, "connected block-13 stage"},
    {"QKV", "ARRAY", 531053, 71672, "connected block-13 stage"},
    {"ATTN_SELF", "ARRAY", 1131363, 0, "connected block-13 stage"},
    {"SPROJ", "ARRAY", 290212, 25807, "connected block-13 stage"},
    {"RES0", "RESIDUAL", 4610, 0, "connected block-13 stage"},
    {"LN1", "LN", 13900, 0, "connected block-13 stage"},
    {"CQ", "ARRAY", 288335, 23930, "connected block-13 stage"},
    {"CKV", "ARRAY", 96691, 51450, "connected block-13 stage"},
    {"ATTN_CROSS", "ARRAY", 280611, 0, "connected block-13 stage"},
    {"CPROJ", "ARRAY", 290044, 25639, "connected block-13 stage"},
    {"RES1", "RESIDUAL", 4610, 0, "connected block-13 stage"},
    {"LN2", "LN", 13900, 0, "connected block-13 stage"},
    {"MLP0", "ARRAY", 652317, 95448, "connected block-13 stage"},
    {"GELU", "GELU", 18434, 0, "connected block-13 stage"},
    {"MLP2", "ARRAY", 808012, 92023, "connected block-13 stage"},
    {"RES2", "RESIDUAL", 4610, 0, "connected block-13 stage"},
    {"FINAL_IDLE", "CONTROL", 1, 0, "connected block-13 stage"},
};

// 单独建模权重 DMA 时，只扣除明确测得的权重等待。
// 返回值仍包含阶段内部的控制和片上访问时间。
inline Cycle block13_cycles_excluding_weight_wait(const std::string& stage) {
  for (const auto& profile : kBlock13UnitProfiles)
    if (stage == profile.name)
      return profile.measured_cycles - profile.measured_weight_wait_cycles;
  throw std::invalid_argument("unknown block-13 stage: " + stage);
}

}  // namespace supra2::latency
