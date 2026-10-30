#include "compute_units.h"
#include "npu_pipeline_model.h"

#include <iostream>
#include <stdexcept>

using namespace supra2::latency;

namespace {
void require(bool condition, const char* reason) {
  if (!condition) throw std::runtime_error(reason);
}

const PipelineTiming& timing(const PipelineResult& result, const std::string& id) {
  for (const auto& item : result.operations) if (item.id == id) return item;
  throw std::runtime_error("missing operation timing");
}

void buffer_lifetime() {
  OnChipTileBuffer memory({"BUF", 64, 1, 64, 1});
  memory.reserve(64);
  require(!memory.can_store(1), "DMA output reservation must protect capacity");
  require(memory.accept_dma_beat("x", 0, 64, true), "reserved DMA fill failed");
  const BufferSlice consuming{"BUF", "x", 64, true};
  memory.begin_read(consuming);
  require(memory.free_bytes() == 0, "compute input was released before execution ended");
  require(!memory.has("x", 64), "DMA must not consume an active compute input");
  memory.end_read(consuming);
  require(memory.free_bytes() == 64, "input capacity was not released at compute finish");
  memory.begin_ui();
  require(memory.accept_dma_beat("w", 0, 64), "second tile fill failed");
  const BufferSlice retained{"BUF", "w", 64, false};
  memory.begin_read(retained);
  memory.begin_read(retained);
  memory.end_read(retained);
  require(!memory.has("w", 64), "remaining shared reader must keep the tile pinned");
  memory.end_read(retained);
  require(memory.has("w", 64), "retained weight must survive all readers");
}

void dma_boundary() {
  NpuPipelineModel model({}, {{"INPUT", 4224, 2, 64, 1}}, {});
  const auto result = model.run({
      {"load", PipelineKind::DmaRead, {}, 0xfc0, 4224, 0,
       {}, {"INPUT", "x", 4224, true}, "", 0}});
  require(result.dma_stats.bursts == 3, "burst split must be 64 + 4096 + 64 bytes");
  require(result.dma_stats.read_beats_accepted == 66, "DMA lost or duplicated a read beat");
  require(result.mig_stats.read_bytes == 4224, "DDR byte count differs from descriptor");
}

void dependencies_and_resource_overlap() {
  NpuPipelineModel model({}, {}, {{"ARRAY", 1}, {"GELU", 1}});
  const auto result = model.run({
      {"array_a", PipelineKind::Compute, {}, 0, 0, 0, {}, {}, "ARRAY", 10},
      {"array_b", PipelineKind::Compute, {}, 0, 0, 0, {}, {}, "ARRAY", 3},
      {"gelu", PipelineKind::Compute, {}, 0, 0, 0, {}, {}, "GELU", 8},
      {"dependent", PipelineKind::Compute, {"array_b", "gelu"}, 0, 0, 0,
       {}, {}, "GELU", 2}});
  require(timing(result, "gelu").start_ui == timing(result, "array_a").start_ui,
          "independent physical resources must overlap");
  require(timing(result, "array_b").start_ui >= timing(result, "array_a").finish_ui,
          "one array cannot execute two tasks simultaneously");
  require(timing(result, "dependent").start_ui >= timing(result, "array_b").finish_ui,
          "task started before its dependency finished");
  require(result.makespan_ui == 15, "unexpected event-boundary scheduling");
}

void refill_waits_for_compute() {
  NpuPipelineModel model({}, {{"W", 64, 1, 64, 1}}, {{"ARRAY", 1}, {"INIT", 1}});
  const auto result = model.run({
      {"seed", PipelineKind::Compute, {}, 0, 0, 0, {}, {"W", "old", 64, true}, "INIT", 1},
      {"use", PipelineKind::Compute, {"seed"}, 0, 0, 0,
       {{"W", "old", 64, true}}, {}, "ARRAY", 20},
      {"refill", PipelineKind::DmaRead, {"seed"}, 0x1000, 64, 0,
       {}, {"W", "new", 64, true}, "", 0}});
  require(timing(result, "refill").start_ui >= timing(result, "use").finish_ui,
          "DMA overwrote the only buffer while compute still needed its input");
}

void tdm_idle_phase_and_profile() {
  MigAxiConfig config;
  config.tdm_read_write_arbitration = true;
  MigAxiModel mig(config);
  mig.step_ui();  // UI0 read slot passes while idle; UI1 must be a write slot.
  require(mig.submit({1, AxiDirection::Read, 0, 1}), "submit failed");
  mig.run_until_idle();
  require(mig.stats().queue_wait_ui == 1, "idle cycles must advance the fixed TDM phase");
  Cycle waits = 0;
  for (const auto& profile : kBlock13UnitProfiles) waits += profile.measured_weight_wait_cycles;
  require(waits == 385969, "recorded weight wait total changed");
  require(block13_cycles_excluding_weight_wait("QKV") == 459381,
          "QKV profile double counts weight wait");
}

void ddr_turnarounds() {
  Ddr4Device device;
  Ddr4Command command;
  command.kind = Ddr4CommandKind::Act;
  require(device.issue(command).accepted, "BG0 ACT failed");
  command.ck = 7;
  command.bank_group = 1;
  require(device.issue(command).accepted, "BG1 ACT failed");
  command.ck = 17;
  command.bank_group = 0;
  command.kind = Ddr4CommandKind::Read;
  require(device.issue(command).accepted, "first READ failed");
  command.ck = 30;
  command.kind = Ddr4CommandKind::Write;
  require(!device.issue(command).accepted, "tRTW was not enforced");
  command.ck = 31;
  require(device.issue(command).accepted, "WRITE at tRTW boundary failed");
  command.kind = Ddr4CommandKind::Read;
  command.bank_group = 1;
  command.ck = 50;
  require(!device.issue(command).accepted, "cross-BG tWTR_S was not enforced");
  command.ck = 51;
  require(device.issue(command).accepted, "cross-BG READ at tWTR_S boundary failed");
  command.bank_group = 0;
  command.ck = 55;
  require(!device.issue(command).accepted, "same-BG tWTR_L was lost across another BG");
  command.ck = 57;
  require(device.issue(command).accepted, "same-BG READ at tWTR_L boundary failed");
}
}  // namespace

int main() {
  try {
    buffer_lifetime();
    dma_boundary();
    dependencies_and_resource_overlap();
    refill_waits_for_compute();
    tdm_idle_phase_and_profile();
    ddr_turnarounds();
    std::cout << "PASS: buffer lifetime, burst splitting, overlap, refill, TDM, weight waits\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
