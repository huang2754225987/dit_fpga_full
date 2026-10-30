#include "event_simulator.h"
#include "model_config.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace supra2::latency;

namespace {
std::string read_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open file: " + path);
  std::ostringstream data;
  data << in.rdbuf();
  return data.str();
}

std::string field(const std::string& object, const std::string& name,
                  const std::string& fallback = "") {
  const std::regex re("\\\"" + name + "\\\"\\s*:\\s*(\\\"([^\\\"]*)\\\"|[0-9]+)");
  std::smatch match;
  if (!std::regex_search(object, match, re)) return fallback;
  return match[2].matched ? match[2].str() : match[1].str();
}

std::vector<std::string> string_array(const std::string& object,
                                      const std::string& name) {
  const std::regex re("\\\"" + name + "\\\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch match;
  if (!std::regex_search(object, match, re)) return {};
  const std::regex item("\\\"([^\\\"]*)\\\"");
  std::vector<std::string> values;
  for (auto it = std::sregex_iterator(match[1].first, match[1].second, item);
       it != std::sregex_iterator(); ++it) values.push_back((*it)[1].str());
  return values;
}

Workload parse_workload(const std::string& json) {
  std::smatch version_match;
  if (!std::regex_search(json, version_match,
                         std::regex("\\\"version\\\"\\s*:\\s*([0-9]+)")) ||
      version_match[1] != "1")
    throw std::invalid_argument("workload must declare version 1");
  std::smatch tasks_match;
  if (!std::regex_search(json, tasks_match,
                         std::regex("\\\"tasks\\\"\\s*:\\s*\\[([\\s\\S]*)\\]")))
    throw std::invalid_argument("workload is missing tasks array");
  Workload workload;
  const std::string body = tasks_match[1];
  const std::regex object_re("\\{([^{}]*)\\}");
  for (auto it = std::sregex_iterator(body.begin(), body.end(), object_re);
       it != std::sregex_iterator(); ++it) {
    const std::string object = (*it)[1].str();
    Task task;
    task.id = field(object, "id");
    task.stage = field(object, "stage", "unspecified");
    task.dependencies = string_array(object, "dependencies");
    task.resources = string_array(object, "resources");
    task.duration_cycles = std::stoull(field(object, "duration_cycles", "0"));
    task.bytes_read = std::stoull(field(object, "bytes_read", "0"));
    task.bytes_written = std::stoull(field(object, "bytes_written", "0"));
    workload.tasks.push_back(std::move(task));
  }
  if (workload.tasks.empty()) throw std::invalid_argument("tasks array is empty or malformed");
  return workload;
}

ModelConfig parse_config(const std::string& json) {
  std::smatch version_match;
  if (!std::regex_search(json, version_match,
                         std::regex("\\\"version\\\"\\s*:\\s*([0-9]+)")) ||
      version_match[1] != "1")
    throw std::invalid_argument("config must declare version 1");
  ModelConfig config;
  config.clock_period_ps = std::stoull(field(json, "clock_period_ps", "3332"));
  config.axi_data_width_bits = std::stoul(field(json, "axi_data_width_bits", "512"));
  config.max_burst_beats = std::stoul(field(json, "max_burst_beats", "256"));
  config.first_response_latency_cycles =
      std::stoull(field(json, "first_response_latency_cycles", "0"));
  config.cycles_per_beat = std::stoull(field(json, "cycles_per_beat", "1"));
  config.outstanding_capacity = std::stoul(field(json, "outstanding_capacity", "1"));
  config.memory_parameters_calibrated =
      std::regex_search(json, std::regex("\\\"memory_parameters_calibrated\\\"\\s*:\\s*true"));
  config.validate();
  return config;
}

void write_result(const SimulationResult& result, const ModelConfig& config,
                  const std::string& path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write output: " + path);
  out << "{\n  \"model_version\": 1,\n  \"makespan_cycles\": "
      << result.makespan_cycles << ",\n  \"clock_period_ps\": "
      << config.clock_period_ps << ",\n  \"makespan_ns\": "
      << std::fixed << std::setprecision(3)
      << static_cast<double>(result.makespan_cycles) * config.clock_period_ps / 1000.0
      << ",\n  \"memory_parameters_calibrated\": "
      << (config.memory_parameters_calibrated ? "true" : "false")
      << ",\n  \"bytes_read\": " << result.bytes_read
      << ",\n  \"bytes_written\": " << result.bytes_written
      << ",\n  \"tasks\": [\n";
  for (std::size_t i = 0; i < result.tasks.size(); ++i) {
    const auto& item = result.tasks[i];
    out << "    {\"id\": \"" << item.task.id << "\", \"stage\": \""
        << item.task.stage << "\", \"start_cycle\": " << item.start_cycle
        << ", \"finish_cycle\": " << item.finish_cycle << "}";
    out << (i + 1 == result.tasks.size() ? "\n" : ",\n");
  }
  out << "  ],\n  \"stage_busy_cycles\": {";
  bool first = true;
  for (const auto& [name, cycles] : result.stage_cycles) {
    out << (first ? "\n" : ",\n") << "    \"" << name << "\": " << cycles;
    first = false;
  }
  out << (first ? "},\n" : "\n  },\n") << "  \"resource_busy_cycles\": {";
  first = true;
  for (const auto& [name, cycles] : result.resource_busy_cycles) {
    out << (first ? "\n" : ",\n") << "    \"" << name << "\": " << cycles;
    first = false;
  }
  out << (first ? "}\n}\n" : "\n  }\n}\n");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::string workload_path;
    std::string config_path;
    std::string output_path = "latency_result.json";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::cout << "Usage: supra2_latency_model --workload FILE [--config FILE] [--output FILE]\n";
        return 0;
      }
      if ((arg == "--workload" || arg == "--config" || arg == "--output") && i + 1 >= argc)
        throw std::invalid_argument("missing value for " + arg);
      if (arg == "--workload") workload_path = argv[++i];
      else if (arg == "--config") config_path = argv[++i];
      else if (arg == "--output") output_path = argv[++i];
      else throw std::invalid_argument("unknown argument: " + arg);
    }
    if (workload_path.empty()) throw std::invalid_argument("--workload is required");
    const ModelConfig config = config_path.empty()
        ? ModelConfig{} : parse_config(read_file(config_path));
    const auto workload = parse_workload(read_file(workload_path));
    const auto result = EventSimulator{}.run(workload);
    write_result(result, config, output_path);
    std::cout << "makespan_cycles=" << result.makespan_cycles << " tasks="
              << result.tasks.size() << " clock_period_ps=" << config.clock_period_ps
              << " memory_parameters_calibrated="
              << (config.memory_parameters_calibrated ? "true" : "false")
              << " output=" << output_path << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
