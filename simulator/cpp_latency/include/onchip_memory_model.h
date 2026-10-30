#pragma once

#include <algorithm>
#include <cstdint>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace supra2::latency {

struct OnChipMemoryConfig {
  std::uint64_t capacity_bytes = 64 * 1024;
  std::uint32_t bank_count = 1;
  std::uint32_t line_bytes = 64;
  std::uint64_t hit_latency_cycles = 1;
  std::uint64_t miss_latency_cycles = 1;
  std::uint32_t bytes_per_cycle_per_bank = 64;
};

struct MemoryAccessResult {
  bool hit = false;
  std::uint32_t bank = 0;
  std::uint64_t start_cycle = 0;
  std::uint64_t ready_cycle = 0;
};

struct OnChipMemoryStats {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t bank_wait_cycles = 0;
};

// Models a banked on-chip tile/weight buffer with LRU line retention.
// This is not a claim that the RTL implements a general-purpose CPU/GPU cache.
class OnChipMemoryModel {
 public:
  explicit OnChipMemoryModel(OnChipMemoryConfig config) : config_(config) {
    if (config_.capacity_bytes == 0 || config_.bank_count == 0 ||
        config_.line_bytes == 0 || config_.bytes_per_cycle_per_bank == 0)
      throw std::invalid_argument("invalid on-chip memory configuration");
    const auto total_lines = config_.capacity_bytes / config_.line_bytes;
    if (total_lines < config_.bank_count)
      throw std::invalid_argument("on-chip memory needs at least one line per bank");
    lines_per_bank_ = total_lines / config_.bank_count;
    banks_.resize(config_.bank_count);
  }

  MemoryAccessResult access(std::uint64_t address, std::uint64_t bytes,
                            std::uint64_t arrival_cycle) {
    if (bytes == 0) return {true, 0, arrival_cycle, arrival_cycle};
    if (address % config_.line_bytes + bytes > config_.line_bytes)
      throw std::invalid_argument("access() models one cache line; split crossing accesses first");
    const std::uint64_t line = address / config_.line_bytes;
    const std::uint32_t bank = static_cast<std::uint32_t>(line % config_.bank_count);
    auto& state = banks_[bank];
    auto found = state.resident.find(line);
    const bool hit = found != state.resident.end();
    const std::uint64_t start = std::max(arrival_cycle, state.available_cycle);
    stats_.bank_wait_cycles += start - arrival_cycle;
    const std::uint64_t transfer_cycles = hit ? 0 :
        (std::min<std::uint64_t>(bytes, config_.line_bytes) +
         config_.bytes_per_cycle_per_bank - 1) / config_.bytes_per_cycle_per_bank;
    const std::uint64_t ready = start +
        (hit ? config_.hit_latency_cycles : config_.miss_latency_cycles + transfer_cycles);
    state.available_cycle = ready;
    if (hit) {
      ++stats_.hits;
      state.lru.erase(found->second);
      state.lru.push_front(line);
      found->second = state.lru.begin();
    } else {
      ++stats_.misses;
      if (state.lru.size() == lines_per_bank_) {
        state.resident.erase(state.lru.back());
        state.lru.pop_back();
      }
      state.lru.push_front(line);
      state.resident[line] = state.lru.begin();
    }
    return {hit, bank, start, ready};
  }

  const OnChipMemoryStats& stats() const { return stats_; }

 private:
  struct BankState {
    std::uint64_t available_cycle = 0;
    std::list<std::uint64_t> lru;
    std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator> resident;
  };
  OnChipMemoryConfig config_;
  std::uint64_t lines_per_bank_ = 0;
  std::vector<BankState> banks_;
  OnChipMemoryStats stats_;
};

}  // namespace supra2::latency
