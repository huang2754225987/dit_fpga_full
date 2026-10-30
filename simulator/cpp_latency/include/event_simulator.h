#pragma once

#include "task_graph.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace supra2::latency {

struct SimulationResult {
  std::vector<ScheduledTask> tasks;
  std::map<std::string, Cycle> resource_busy_cycles;
  std::map<std::string, Cycle> stage_cycles;
  Cycle makespan_cycles = 0;
  std::uint64_t bytes_read = 0;
  std::uint64_t bytes_written = 0;
};

class EventSimulator {
 public:
  SimulationResult run(const Workload& workload) const {
    std::unordered_map<std::string, const Task*> by_id;
    for (const auto& task : workload.tasks) {
      if (task.id.empty() || task.duration_cycles == 0 || task.resources.empty())
        throw std::invalid_argument("task requires id, positive duration, and resources");
      if (!by_id.emplace(task.id, &task).second)
        throw std::invalid_argument("duplicate task id: " + task.id);
    }
    for (const auto& task : workload.tasks) {
      for (const auto& dep : task.dependencies) {
        if (!by_id.count(dep)) throw std::invalid_argument("missing dependency: " + dep);
      }
    }

    SimulationResult result;
    std::unordered_map<std::string, Cycle> finish_by_id;
    std::unordered_map<std::string, Cycle> resource_available;
    std::set<std::string> scheduled;
    while (scheduled.size() < workload.tasks.size()) {
      const Task* chosen = nullptr;
      Cycle chosen_start = 0;
      for (const auto& task : workload.tasks) {
        if (scheduled.count(task.id)) continue;
        bool deps_ready = true;
        Cycle earliest = 0;
        for (const auto& dep : task.dependencies) {
          auto it = finish_by_id.find(dep);
          if (it == finish_by_id.end()) { deps_ready = false; break; }
          earliest = std::max(earliest, it->second);
        }
        if (!deps_ready) continue;
        for (const auto& resource : task.resources)
          earliest = std::max(earliest, resource_available[resource]);
        if (!chosen || earliest < chosen_start ||
            (earliest == chosen_start && task.id < chosen->id)) {
          chosen = &task;
          chosen_start = earliest;
        }
      }
      if (!chosen) throw std::invalid_argument("task dependency cycle detected");

      const Cycle finish = chosen_start + chosen->duration_cycles;
      result.tasks.push_back({*chosen, chosen_start, finish});
      result.bytes_read += chosen->bytes_read;
      result.bytes_written += chosen->bytes_written;
      finish_by_id[chosen->id] = finish;
      scheduled.insert(chosen->id);
      result.makespan_cycles = std::max(result.makespan_cycles, finish);
      result.stage_cycles[chosen->stage] += chosen->duration_cycles;
      for (const auto& resource : chosen->resources) {
        resource_available[resource] = finish;
        result.resource_busy_cycles[resource] += chosen->duration_cycles;
      }
    }
    std::stable_sort(result.tasks.begin(), result.tasks.end(),
                     [](const ScheduledTask& a, const ScheduledTask& b) {
                       if (a.start_cycle != b.start_cycle) return a.start_cycle < b.start_cycle;
                       return a.task.id < b.task.id;
                     });
    return result;
  }
};

}  // namespace supra2::latency
