#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace supra2::latency {

using Cycle = std::uint64_t;

struct Task {
  std::string id;
  std::string stage;
  std::vector<std::string> dependencies;
  std::vector<std::string> resources;
  Cycle duration_cycles = 0;
  std::uint64_t bytes_read = 0;
  std::uint64_t bytes_written = 0;
};

struct ScheduledTask {
  Task task;
  Cycle start_cycle = 0;
  Cycle finish_cycle = 0;
};

struct Workload {
  unsigned version = 1;
  std::vector<Task> tasks;
};

}  // namespace supra2::latency
