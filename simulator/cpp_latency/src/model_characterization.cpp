#include "npu_pipeline_model.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace supra2::latency;

namespace {
// Snapshot from rtl/pkg/block13_ddr_layout.sv, components 0..9.
constexpr std::uint64_t kInputBytes[] = {
    294912, 36864, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152};
constexpr std::uint64_t kInputTotal = 340992;
constexpr std::uint64_t kOutputTotal = 294912;
constexpr std::uint64_t kDdrBase = 0x100000;

PipelineOperation load(const std::string& id, std::uint64_t address,
                       std::uint64_t bytes, const std::string& destination) {
  PipelineOperation op;
  op.id = id;
  op.kind = PipelineKind::DmaRead;
  op.ddr_address = address;
  op.bytes = bytes;
  op.output = {destination, id, bytes, true};
  return op;
}

PipelineResult input_case() {
  std::vector<PipelineOperation> ops;
  std::uint64_t offset = 0;
  for (unsigned i = 0; i < 10; ++i) {
    ops.push_back(load("input_" + std::to_string(i), kDdrBase + offset,
                       kInputBytes[i], "INPUT"));
    offset += kInputBytes[i];
  }
  NpuPipelineModel model({}, {{"INPUT", kInputTotal, 2, 64, 1}}, {});
  return model.run(ops);
}

PipelineResult output_case() {
  PipelineOperation seed;
  seed.id = "resident_output";
  seed.kind = PipelineKind::Compute;
  seed.resource = "SEED";
  seed.compute_cycles = 1;
  seed.output = {"OUT", "result", kOutputTotal, true};
  PipelineOperation store;
  store.id = "write_output";
  store.kind = PipelineKind::DmaWrite;
  store.dependencies = {seed.id};
  store.ddr_address = 0x800000;
  store.bytes = kOutputTotal;
  store.inputs = {{"OUT", "result", kOutputTotal, true}};
  NpuPipelineModel model({}, {{"OUT", kOutputTotal, 2, 64, 1}}, {{"SEED", 1}});
  return model.run({seed, store});
}

PipelineResult qkv_case() {
  // QKV stage: 54 N tiles x 18 K tiles, each INT8 tile has 1024 bytes.
  // Scale and bias component sizes/offsets come from the same layout snapshot.
  std::vector<PipelineOperation> ops;
  for (unsigned tile = 0; tile < 972; ++tile)
    ops.push_back(load("weight_" + std::to_string(tile),
        kDdrBase + 340992 + static_cast<std::uint64_t>(tile) * 1024, 1024, "W"));
  ops.push_back(load("scale", kDdrBase + 1336320, 6912, "W"));
  ops.push_back(load("bias", kDdrBase + 1343232, 6912, "W"));
  NpuPipelineModel model({}, {{"W", 995328 + 2 * 6912, 2, 64, 1}}, {});
  return model.run(ops);
}

void row(std::ostream& out, const std::string& name, Cycle prediction,
          Cycle observed, Cycle local_floor, bool comma) {
  const double error = 100.0 * (static_cast<double>(prediction) / observed - 1.0);
  out << "    {\"case\":\"" << name << "\",\"predicted_ui\":" << prediction
      << ",\"rtl_reference_ui\":" << observed << ",\"signed_error_percent\":"
      << std::fixed << std::setprecision(4) << error
      << ",\"local_fsm_lower_bound_ui\":" << local_floor << "}"
      << (comma ? ",\n" : "\n");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("Usage: model_characterization OUTPUT.json");
    const auto input = input_case();
    const auto output = output_case();
    const auto qkv = qkv_case();
    Cycle output_duration = 0;
    for (const auto& operation : output.operations)
      if (operation.id == "write_output")
        output_duration = operation.finish_ui - operation.start_ui;
    std::ofstream report(argv[1]);
    if (!report) throw std::runtime_error("cannot write characterization report");
    report << "{\n  \"calibrated\": false,\n  \"scope\": "
           << "\"Independent traffic-only predictions; not full block inference\",\n"
           << "  \"ui_period_ps\":3332,\n  \"cases\":[\n";
    row(report, "input_dma", input.makespan_ui, 178086, (kInputTotal / 64) * 33, true);
    row(report, "output_dma", output_duration, 43145, (kOutputTotal / 64) * 9, true);
    row(report, "qkv_weight_load", qkv.makespan_ui, 71672,
         972 * (16 + 32) + (2 * 6912 / 64) * 17, false);
    report << "  ]\n}\n";
    std::cout << "input_dma_ui=" << input.makespan_ui
              << " output_dma_ui=" << output_duration
              << " qkv_weight_load_ui=" << qkv.makespan_ui
              << " report=" << argv[1] << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
