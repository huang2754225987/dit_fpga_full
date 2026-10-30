#include "axi_ddr_model.h"
#include "compute_units.h"
#include "event_simulator.h"
#include "onchip_memory_model.h"

#include <cassert>
#include <stdexcept>

using namespace supra2::latency;

int main() {
  Workload graph;
  graph.tasks = {
      {"a", "input", {}, {"dma"}, 10, 1024, 0},
      {"b", "compute", {"a"}, {"array"}, 20, 0, 0},
      {"c", "compute", {}, {"other"}, 15, 0, 0},
      {"d", "output", {"b", "c"}, {"dma"}, 5, 0, 256},
  };
  const auto result = EventSimulator{}.run(graph);
  assert(result.makespan_cycles == 35);
  assert(result.stage_cycles.at("compute") == 35);

  ModelConfig config;
  config.axi_data_width_bits = 512;
  config.max_burst_beats = 4;
  config.first_response_latency_cycles = 7;
  config.cycles_per_beat = 2;
  AxiDdrModel ddr(config);
  const auto bursts = ddr.split(0xFC0, 160);
  assert(bursts.size() == 2);  // 4 KiB boundary and maximum burst length.
  assert(bursts[0].beats == 1);
  assert(bursts[1].beats == 2);
  assert(ddr.service_cycles(2) == 11);

  SystolicArrayModel array;
  assert(array.estimate_cycles({32, 32, 576}) == 642);
  OnChipMemoryModel tile_buffer({256, 2, 64, 1, 1, 64});
  const auto miss = tile_buffer.access(0x1000, 64, 0);
  const auto hit = tile_buffer.access(0x1000, 64, miss.ready_cycle);
  assert(!miss.hit && hit.hit);
  assert(tile_buffer.stats().misses == 1 && tile_buffer.stats().hits == 1);

  bool cycle_rejected = false;
  Workload cyclic;
  cyclic.tasks = {{"x", "s", {"y"}, {"r"}, 1, 0, 0},
                  {"y", "s", {"x"}, {"r"}, 1, 0, 0}};
  try { (void)EventSimulator{}.run(cyclic); }
  catch (const std::invalid_argument&) { cycle_rejected = true; }
  assert(cycle_rejected);
  return 0;
}
